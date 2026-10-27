#!/usr/bin/env python3
"""Add main-fraction RVEs without modifying the original generator or dataset.

--list and --summary do not import FE libraries. --summary writes an expanded
440-row dataset only after all 200 new records pass provenance/geometry checks.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "surrogate_dataset_extension_v1.json"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def original_generator():
    path = SCRIPT_DIR / "generate_surrogate_dataset.py"
    if not path.exists():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("original_surrogate_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest(spec, base, old_rows, wrapper_sha):
    n = int(spec["new_realizations_per_fraction"])
    if n != 40:
        raise ValueError("This extension is defined for 40 new realizations per main fraction.")
    if len(old_rows) != 240:
        raise ValueError(f"Expected 240 accepted source records, found {len(old_rows)}.")
    old_ids = [r["sample_id"] for r in old_rows]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("Duplicate source sample IDs.")
    old_seeds = {int(float(r["seed"])) for r in old_rows}
    for r in old_rows:
        failed_seed = r.get("original_failed_seed", "")
        if failed_seed and failed_seed.lower() != "nan":
            old_seeds.add(int(float(failed_seed)))
    for f in base["main_vf_percent"]:
        group = [r for r in old_rows if r["split"] == "main" and float(r["vf_target_percent"]) == float(f)]
        if len(group) != 40:
            raise ValueError(f"Expected 40 original main records at {f}%, found {len(group)}.")
    if sum(r["split"] == "interpolation_test" for r in old_rows) != 40:
        raise ValueError("Expected 40 original interpolation-test records.")
    settings = ["L_over_D", "radius", "e_matrix_Pa", "nu_matrix", "e_fibre_Pa", "nu_fibre", "h_far_over_r", "h_interface_over_r"]
    for r in old_rows:
        for key in settings:
            if key not in r or not math.isclose(float(r[key]), float(base[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Source settings mismatch: {r['sample_id']} {key}.")
    jobs = []
    for vf in base["main_vf_percent"]:
        for k in range(n):
            task = len(jobs)
            row = {
                "task_id": task, "sample_id": f"sample_{int(spec['sample_id_start'])+task:04d}",
                "split": "main", "vf_target_percent": float(vf),
                "replicate": int(spec["replicate_start"])+k, "seed": int(spec["new_base_seed"])+task,
                "extension_wrapper_sha256": wrapper_sha,
                "source_study_id": base["study_id"],
            }
            if row["sample_id"] in set(old_ids) or row["seed"] in old_seeds:
                raise ValueError(f"Extension ID or seed collides with source data: {row}")
            old_replicates = {int(float(r["replicate"])) for r in old_rows
                              if r["split"] == "main" and float(r["vf_target_percent"]) == float(vf)}
            if row["replicate"] in old_replicates:
                raise ValueError("Extension replicate index overlaps an original replicate.")
            jobs.append(row)
    if len(jobs) != 200 or len({r["seed"] for r in jobs}) != 200 or len({r["sample_id"] for r in jobs}) != 200:
        raise ValueError("Invalid extension manifest.")
    return jobs


def load_extension(config_file, gen):
    config_file = Path(config_file).resolve()
    spec = json.loads(config_file.read_text())
    required = {"study_id", "source_config", "source_dataset", "new_realizations_per_fraction", "new_base_seed", "sample_id_start", "replicate_start"}
    if required - set(spec):
        raise ValueError(f"Missing extension fields: {sorted(required-set(spec))}")
    study = spec["study_id"]
    if not study or not all(c.isalnum() or c in "_-" for c in study):
        raise ValueError("Invalid extension study_id.")
    base_file, dataset_file = resolve(spec["source_config"]), resolve(spec["source_dataset"])
    base, base_digest = gen.load_config(base_file)
    if study == base["study_id"]:
        raise ValueError("Extension study_id must differ from the original.")
    rows = read_csv(dataset_file)
    wrapper_sha = sha256(Path(__file__))
    jobs = build_manifest(spec, base, rows, wrapper_sha)
    source_sha = sha256(dataset_file)
    for job in jobs:
        job["source_dataset_sha256"] = source_sha
    effective = dict(base)
    effective["study_id"] = study
    effective["extension_specification"] = spec
    effective["source_config_sha256"] = base_digest
    effective["source_dataset_sha256"] = source_sha
    effective["extension_wrapper_sha256"] = wrapper_sha
    # The original generator is used only for run(); its own manifest/summary are not invoked.
    effective["main_realizations"] = spec["new_realizations_per_fraction"]
    effective["main_base_seed"] = spec["new_base_seed"]
    effective["extension_generation_split"] = "main_only"
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    return effective, digest, jobs, rows, dataset_file


def geometry_signature(path):
    p = resolve(path)
    rows = read_csv(p)
    xyz = sorted((round(float(r["x_center"]), 12), round(float(r["y_center"]), 12), round(float(r["radius"]), 12)) for r in rows)
    if not xyz:
        raise ValueError(f"Empty geometry: {p}")
    return hashlib.sha256(json.dumps(xyz, separators=(",", ":")).encode()).hexdigest()


def write_csv_atomic(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def summarize(gen, effective, digest, jobs, original_rows, source_file):
    root, dirs = gen.paths(effective)
    new_rows, missing = [], []
    expected_provenance = gen.provenance(effective, digest)
    for job in jobs:
        file = dirs["records"] / f"{job['sample_id']}.json"
        if not file.exists():
            missing.append(job["task_id"])
            continue
        row = json.loads(file.read_text())
        if any(row.get(k) != v for k, v in job.items()):
            raise ValueError(f"Record differs from extension manifest: {file}")
        for k in ["config_sha256", "code_sha256", "study_id"]:
            if row.get(k) != expected_provenance[k]:
                raise ValueError(f"Provenance mismatch for {file}: {k}")
        if not all(k in row and math.isfinite(float(row[k])) for k in gen.LABELS):
            raise ValueError(f"Invalid mechanical record: {file}")
        if not resolve(row["geometry_file"]).is_file():
            raise ValueError(f"Missing geometry: {row['sample_id']}")
        new_rows.append(row)
    failures = list(dirs["failures"].glob("*.txt"))
    print(f"Accepted extension records: {len(new_rows)}/200; failure logs: {len(failures)}")
    if missing or failures:
        print(f"Missing task IDs: {missing}")
        print("Incomplete: no extension/expanded dataset written. Inspect failure logs and retained geometry before recovery.")
        return 1
    combined = original_rows + new_rows
    if len(combined) != 440 or len({r['sample_id'] for r in combined}) != 440:
        raise ValueError("Expanded dataset must contain 440 unique sample IDs.")
    seeds = [int(float(r["seed"])) for r in combined]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated accepted seeds in expanded dataset.")
    signatures = {}
    for row in combined:
        signature = geometry_signature(row["geometry_file"])
        if signature in signatures:
            raise ValueError(f"Repeated coordinate geometry: {signatures[signature]} and {row['sample_id']}")
        signatures[signature] = row["sample_id"]
    root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(root / "extension_dataset.csv", new_rows)
    write_csv_atomic(root / "expanded_dataset.csv", combined)
    gen.atomic_json(root / "extension_manifest.json", jobs)
    gen.atomic_json(root / "effective_configuration.json", effective)
    gen.atomic_json(root / "merge_metadata.json", {
        "source_dataset": str(source_file), "source_dataset_sha256": sha256(source_file),
        "original_accepted_records": 240, "new_accepted_records": 200,
        "expanded_records": 440, "development_records": 400, "existing_interpolation_records": 40,
        "new_interpolation_records": 0,
        "geometry_duplicate_check": "Sorted coordinates/radii rounded to 12 decimals; does not test translated/rotated equivalence",
        "original_dataset_modified": False,
        "original_test_already_evaluated": True,
    })
    print(f"Complete: {root / 'expanded_dataset.csv'}")
    print("Development: 400 (80 per principal fraction); existing interpolation records: 40.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--task-id", type=int)
    group.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    gen = original_generator()
    effective, digest, jobs, original_rows, source_file = load_extension(args.config, gen)
    if args.list:
        for row in jobs:
            print(json.dumps(row, sort_keys=True))
        print(f"TASK_COUNT={len(jobs)}")
        return 0
    if args.summary:
        return summarize(gen, effective, digest, jobs, original_rows, source_file)
    if args.task_id < 0 or args.task_id >= len(jobs):
        parser.error("--task-id must be in 0..199")
    gen.run(effective, digest, jobs[args.task_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
