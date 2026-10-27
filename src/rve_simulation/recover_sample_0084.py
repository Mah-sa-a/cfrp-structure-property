#!/usr/bin/env python3
"""Recover one failed surrogate RVE without changing the frozen 239-result study."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

import generate_surrogate_dataset as original

ROOT = Path(__file__).resolve().parents[2]
FAILED_ID = "sample_0084"
DEFAULT_CONFIG = ROOT / "configs/surrogate_dataset_v1.json"


def atomic_json(path, obj):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def context(config):
    c, digest = original.load_config(config)
    tasks = original.manifest(c)
    planned = tasks[84]
    if planned["sample_id"] != FAILED_ID or planned["vf_target_percent"] != 50.0 or planned["split"] != "main":
        raise RuntimeError("Frozen manifest does not match expected missing 50% sample")
    root, dirs = original.paths(c)
    meta = original.provenance(c, digest)
    if not (dirs["geometry"] / f"{FAILED_ID}.csv").is_file():
        raise RuntimeError("Original failed geometry missing: preserve audit trail")
    if not (dirs["failures"] / f"{FAILED_ID}.txt").is_file():
        raise RuntimeError("Original failure log missing: preserve audit trail")
    if (dirs["records"] / f"{FAILED_ID}.json").exists():
        raise RuntimeError("Original sample now has a result: do not add replacement")
    return c, digest, tasks, planned, root, dirs, meta


def run(config, seed):
    c, digest, tasks, planned, root, dirs, meta = context(config)
    if seed in {t["seed"] for t in tasks}:
        raise ValueError("Replacement seed must differ from all 240 planned seeds")
    folder = root / "replacements" / f"{FAILED_ID}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "record.json"
    coords = folder / "geometry.csv"
    failure = folder / "failure.txt"
    mesh = folder / "mesh.msh"
    preview = folder / "preview.png"
    if record.exists() or coords.exists():
        raise RuntimeError(f"Existing attempt at {folder}; inspect it, do not overwrite. Use a new seed if needed")
    try:
        import numpy as np
        import packing_study as pack
        import rve_size_convergence_pipeline as base
        r = float(c["radius"])
        L = float(c["L_over_D"]) * 2 * r
        vf_target = planned["vf_target_percent"] / 100.0
        n = max(1, round(vf_target * L * L / (math.pi * r * r)))
        centers, achieved_r = base.generate_centres_LS(
            np.random.default_rng(seed), n, r, float(c["min_surface_gap"]), L)
        if abs(float(achieved_r) - r) > 1e-9:
            raise ValueError(f"Target radius not achieved: {achieved_r}")
        actual = n * math.pi * r * r / (L * L)
        if abs(actual - vf_target) > float(c["max_vf_error"]):
            raise ValueError(f"Achieved Vf={actual:.6f} outside nominal tolerance")
        desc, gaps = pack.descriptors(centers, r, L, float(c["min_surface_gap"]))
        pack.save_geometry(coords, centers, gaps, r)
        geom = {"L": L, "radius": r, "n_fibers": n, "vf_achieved": actual}
        result = pack.solve(c, centers, geom, mesh, preview, f"replacement_{FAILED_ID}_{seed}")
        if not all(math.isfinite(float(result[k])) for k in original.LABELS):
            raise ValueError("Non-finite output")
        C = np.array([[result["C11_Pa"], result["C12_Pa"], result["C13_Pa"]],
                      [result["C12_Pa"], result["C22_Pa"], result["C23_Pa"]],
                      [result["C13_Pa"], result["C23_Pa"], result["C33_Pa"]]])
        if np.linalg.eigvalsh(C).min() <= 0 or any(result[k] <= 0 for k in ("Exx_Pa", "Eyy_Pa", "Gxy_Pa")):
            raise ValueError("Elastic tensor not positive definite")
        row = dict(planned, **geom, **desc, **result, **meta,
                   sample_id=f"{FAILED_ID}_replacement_seed{seed}",
                   replaces_sample_id=FAILED_ID,
                   original_failed_seed=planned["seed"],
                   seed=seed,
                   family="hard_disk", generation_method="rsa_seed_ls_growth",
                   L_over_D=int(c["L_over_D"]),
                   e_matrix_Pa=c["e_matrix_Pa"], nu_matrix=c["nu_matrix"],
                   e_fibre_Pa=c["e_fibre_Pa"], nu_fibre=c["nu_fibre"],
                   h_far_over_r=c["h_far_over_r"], h_interface_over_r=c["h_interface_over_r"],
                   geometry_file=str(coords.relative_to(ROOT)),
                   mesh_file=str(mesh.relative_to(ROOT)) if c["save_meshes"] else None,
                   recovery_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   exclusion_reason="Original sample failed periodic boundary curve count check")
        atomic_json(record, row)
        if not c["save_meshes"]:
            mesh.unlink(missing_ok=True)
        failure.unlink(missing_ok=True)
        print(f"SUCCESS {row['sample_id']} Vf={actual:.6f}; record={record}", flush=True)
    except Exception:
        failure.write_text(traceback.format_exc())
        raise


def assemble(config, seed):
    c, digest, tasks, planned, root, dirs, meta = context(config)
    replacement_file = root / "replacements" / f"{FAILED_ID}_seed{seed}" / "record.json"
    replacement = json.loads(replacement_file.read_text())
    if (replacement.get("replaces_sample_id") != FAILED_ID or replacement.get("seed") != seed or
        replacement.get("original_failed_seed") != planned["seed"] or
        replacement.get("split") != "main" or replacement.get("vf_target_percent") != 50.0 or
        replacement.get("config_sha256") != digest or replacement.get("code_sha256") != meta["code_sha256"]):
        raise RuntimeError("Replacement metadata/config/code do not match original study")
    if not (ROOT / replacement["geometry_file"]).is_file() or not all(k in replacement for k in original.LABELS):
        raise RuntimeError("Replacement geometry or FE labels missing")
    rows = []
    for task in tasks:
        if task["sample_id"] == FAILED_ID:
            rows.append(replacement)
            continue
        path = dirs["records"] / f"{task['sample_id']}.json"
        row = json.loads(path.read_text())
        if any(row.get(k) != task[k] for k in task):
            raise RuntimeError(f"Manifest mismatch: {path}")
        if row.get("config_sha256") != digest or row.get("code_sha256") != meta["code_sha256"]:
            raise RuntimeError(f"Code/config mismatch: {path}")
        if not (ROOT / row["geometry_file"]).is_file() or not all(k in row for k in original.LABELS):
            raise RuntimeError(f"Missing geometry/FE fields: {path}")
        rows.append(row)
    if len(rows) != len(tasks) or len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate or missing samples")
    from collections import Counter
    counts = Counter((r["split"], r["vf_target_percent"]) for r in rows)
    for vf in c["main_vf_percent"]:
        if counts[("main", float(vf))] != int(c["main_realizations"]):
            raise RuntimeError(f"Unexpected main count at Vf={vf}")
    for vf in c["interpolation_test_vf_percent"]:
        if counts[("interpolation_test", float(vf))] != int(c["interpolation_test_realizations"]):
            raise RuntimeError(f"Unexpected test count at Vf={vf}")
    target = root / "dataset_with_replacement.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = target.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, target)
    print(f"COMPLETE: {len(rows)}/{len(tasks)} accepted; one documented replacement")
    for (split, vf), n in sorted(counts.items()):
        print(f"{split} Vf={vf:g}%: {n}")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=30084)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.config, args.seed)
    else:
        assemble(args.config, args.seed)


if __name__ == "__main__":
    main()
