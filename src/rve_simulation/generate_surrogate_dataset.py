#!/usr/bin/env python3
"""One independent hard-disk RVE per Slurm task; dry manifest/summary need no FE imports."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {"study_id", "L_over_D", "radius", "min_surface_gap",
            "main_vf_percent", "main_realizations",
            "interpolation_test_vf_percent", "interpolation_test_realizations",
            "main_base_seed", "interpolation_test_base_seed", "h_far_over_r",
            "h_interface_over_r", "e_matrix_Pa", "nu_matrix", "e_fibre_Pa",
            "nu_fibre", "mesh_area_rtol", "phase_fraction_tolerance",
            "maximum_relative_C_asymmetry", "max_vf_error", "save_meshes",
            "save_previews"}
LABELS = ("C11_Pa", "C12_Pa", "C13_Pa", "C22_Pa", "C23_Pa",
          "C33_Pa", "Exx_Pa", "Eyy_Pa", "Gxy_Pa", "nu_xy")


def load_config(path):
    raw = Path(path).read_bytes()
    c = json.loads(raw)
    missing = REQUIRED - c.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if not c["study_id"] or not all(ch.isalnum() or ch in "_-" for ch in c["study_id"]):
        raise ValueError("study_id must contain only letters, digits, '_' and '-'")
    if int(c["L_over_D"]) != 18 or not (0 < float(c["radius"]) and float(c["min_surface_gap"]) >= 0):
        raise ValueError("Check fixed L/D, radius, and minimum surface gap")
    if not (0 < float(c["max_vf_error"]) < .01):
        raise ValueError("max_vf_error is an absolute Vf fraction, not percent")
    for key in ("h_far_over_r", "h_interface_over_r", "e_matrix_Pa", "e_fibre_Pa",
                "mesh_area_rtol", "phase_fraction_tolerance",
                "maximum_relative_C_asymmetry"):
        if float(c[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if any(not (-1.0 < float(c[key]) < 0.5) for key in ("nu_matrix", "nu_fibre")):
        raise ValueError("Invalid Poisson ratio")
    for key in ("main_realizations", "interpolation_test_realizations"):
        if int(c[key]) < 1:
            raise ValueError(f"{key} must be positive")
    a, b = c["main_vf_percent"], c["interpolation_test_vf_percent"]
    if not a or not b or len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b):
        raise ValueError("Vf lists must be nonempty, unique and disjoint")
    if any(not (0 < float(v) < 100) for v in a + b):
        raise ValueError("All Vf percentages must be in (0, 100)")
    return c, hashlib.sha256(raw).hexdigest()


def manifest(c):
    tasks = []
    for split, levels, n, base_seed in (
        ("main", c["main_vf_percent"], c["main_realizations"], c["main_base_seed"]),
        ("interpolation_test", c["interpolation_test_vf_percent"],
         c["interpolation_test_realizations"], c["interpolation_test_base_seed"]),
    ):
        for vf in levels:
            for replicate in range(int(n)):
                task_id = len(tasks)
                tasks.append({"task_id": task_id,
                              "sample_id": f"sample_{task_id:04d}",
                              "split": split, "vf_target_percent": float(vf),
                              "replicate": replicate, "seed": int(base_seed) + task_id})
    if len({t["seed"] for t in tasks}) != len(tasks):
        raise ValueError("Seed ranges overlap; choose distinct base seeds")
    return tasks


def paths(c):
    root = ROOT / "results" / "surrogate_dataset" / c["study_id"]
    return root, {k: root / k for k in ("records", "geometry", "meshes", "previews", "failures")}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def provenance(c, digest):
    parts = [Path(__file__).resolve(), ROOT / "src/rve_simulation/packing_study.py",
             ROOT / "src/rve_simulation/rve_size_convergence_pipeline.py"]
    versions = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parts}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return {"config_sha256": digest, "code_sha256": versions,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "study_id": c["study_id"]}


def run(c, digest, row):
    root, dirs = paths(c)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    name = row["sample_id"]
    record = dirs["records"] / f"{name}.json"
    geomfile = dirs["geometry"] / f"{name}.csv"
    failure = dirs["failures"] / f"{name}.txt"
    meta = provenance(c, digest)
    if record.exists():
        old = json.loads(record.read_text())
        if any(old.get(k) != meta[k] for k in ("config_sha256", "code_sha256")):
            raise RuntimeError(f"Code/config changed since {record} was created; use a new study_id")
        if old.get("sample_id") != name or old.get("split") != row["split"]:
            raise RuntimeError("Existing record does not match manifest")
        if geomfile.exists() and all(k in old for k in LABELS):
            print(f"SKIP {record}", flush=True)
            return
        raise RuntimeError(f"Incomplete saved record/geometry for {name}; inspect before resubmitting")
    if geomfile.exists():
        raise RuntimeError(f"Geometry without complete record: {geomfile}; inspect before rerunning")
    try:
        import numpy as np
        import packing_study as pack
        import rve_size_convergence_pipeline as base
        r = float(c["radius"])
        L = float(c["L_over_D"]) * 2 * r
        n = max(1, round((row["vf_target_percent"] / 100) * L * L / (math.pi * r * r)))
        centers, achieved_r = base.generate_centres_LS(
            np.random.default_rng(row["seed"]), n, r, float(c["min_surface_gap"]), L)
        if abs(float(achieved_r) - r) > 1e-9:
            raise ValueError(f"Target fibre radius not reached: {achieved_r} != {r}")
        actual = n * math.pi * r * r / (L * L)
        target = row["vf_target_percent"] / 100
        if abs(actual - target) > float(c["max_vf_error"]):
            raise ValueError(f"Achieved Vf={actual:.6f}; nominal={target:.6f}")
        desc, gaps = pack.descriptors(centers, r, L, float(c["min_surface_gap"]))
        pack.save_geometry(geomfile, centers, gaps, r)
        geometry = {"L": L, "radius": r, "n_fibers": n, "vf_achieved": actual}
        mesh = dirs["meshes"] / f"{name}.msh"
        preview = dirs["previews"] / f"{name}.png"
        result = pack.solve(c, centers, geometry, mesh, preview, name)
        if not all(math.isfinite(float(result[k])) for k in LABELS):
            raise ValueError("Non-finite homogenized properties")
        c_mat = np.array([[result["C11_Pa"], result["C12_Pa"], result["C13_Pa"]],
                          [result["C12_Pa"], result["C22_Pa"], result["C23_Pa"]],
                          [result["C13_Pa"], result["C23_Pa"], result["C33_Pa"]]])
        if np.linalg.eigvalsh(c_mat).min() <= 0 or any(result[k] <= 0 for k in ("Exx_Pa", "Eyy_Pa", "Gxy_Pa")):
            raise ValueError("Elastic tensor not positive definite")
        if not geomfile.is_file() or geomfile.stat().st_size == 0:
            raise RuntimeError("Missing saved fibre coordinates")
        full = dict(row, **geometry, **desc, **result, **meta,
                    family="hard_disk", generation_method="rsa_seed_ls_growth",
                    L_over_D=int(c["L_over_D"]),
                    e_matrix_Pa=c["e_matrix_Pa"], nu_matrix=c["nu_matrix"],
                    e_fibre_Pa=c["e_fibre_Pa"], nu_fibre=c["nu_fibre"],
                    h_far_over_r=c["h_far_over_r"],
                    h_interface_over_r=c["h_interface_over_r"],
                    geometry_file=str(geomfile.relative_to(ROOT)),
                    mesh_file=str(mesh.relative_to(ROOT)) if c["save_meshes"] else None)
        atomic_json(record, full)
        if not c["save_meshes"]:
            mesh.unlink(missing_ok=True)
        failure.unlink(missing_ok=True)
        print(f"SUCCESS {name} {row['split']} Vf={actual:.6f} N={n}", flush=True)
    except Exception:
        failure.write_text(traceback.format_exc())
        raise


def summary(c, digest):
    root, dirs = paths(c)
    tasks = manifest(c)
    counts = {}
    rows = []
    bad = []
    for t in tasks:
        p = dirs["records"] / (t["sample_id"] + ".json")
        if not p.exists():
            bad.append(t["sample_id"])
            continue
        rec = json.loads(p.read_text())
        if rec.get("config_sha256") != digest or any(rec.get(k) != t[k] for k in t) or not all(k in rec for k in LABELS):
            raise RuntimeError(f"Mismatched or incomplete record: {p}")
        if not (ROOT / rec["geometry_file"]).exists():
            raise RuntimeError(f"Missing geometry for {p}")
        key = (rec["split"], rec["vf_target_percent"])
        counts[key] = counts.get(key, 0) + 1
        rows.append(rec)
    print(f"Complete {len(rows)}/{len(tasks)}; missing: {len(bad)}; failure logs: {len(list(dirs['failures'].glob('*.txt')))}")
    for (split, vf), number in sorted(counts.items()):
        print(f"{split}: nominal Vf={vf:g}% completed={number}")
    if bad:
        print("Missing sample IDs:", ", ".join(bad))
    if len(rows) != len(tasks) or list(dirs["failures"].glob("*.txt")):
        print("NOT COMPLETE: no consolidated table written")
        return 1
    table = root / "dataset.csv"
    fields = list(dict.fromkeys(k for rec in rows for k in rec.keys()))
    tmp = table.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, table)
    print(f"COMPLETE: {table} ({len(rows)} rows)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/surrogate_dataset_v1.json"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--task-id", type=int)
    group.add_argument("--summary", action="store_true")
    args = p.parse_args()
    c, digest = load_config(args.config)
    jobs = manifest(c)
    if args.list:
        for t in jobs:
            print(json.dumps(t, sort_keys=True))
        print(f"TASK_COUNT={len(jobs)}")
        return 0
    if args.summary:
        return summary(c, digest)
    if not 0 <= args.task_id < len(jobs):
        p.error(f"--task-id must be in 0..{len(jobs)-1}")
    run(c, digest, jobs[args.task_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
