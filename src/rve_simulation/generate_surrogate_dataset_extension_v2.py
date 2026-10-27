#!/usr/bin/env python3
"""Add 600 development and 200 fresh interpolation RVEs to the accepted 440 rows.
Reuse the original FE run() unchanged. --list/--prepare/--summary need no FE imports.
The fresh interpolation targets are reserved for evaluation after model decisions.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "surrogate_dataset_extension_v2.json"
MAIN_FRACTIONS = [40.0, 45.0, 50.0, 55.0, 60.0]
INTERPOLATION_FRACTIONS = [42.5, 47.5, 52.5, 57.5]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def integer(value):
    x = float(value)
    if not math.isfinite(x) or x != int(x):
        raise ValueError(f"Expected an integer, found {value!r}")
    return int(x)


def counts(rows):
    result = {}
    for row in rows:
        key = (row["split"], float(row["vf_target_percent"]))
        result[key] = result.get(key, 0)+1
    return result


def hash_map(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid stored code hashes.")
    return parsed


def load_original_generator():
    path = SCRIPTS / "generate_surrogate_dataset.py"
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("original_surrogate_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source(rows, spec, base, labels):
    expected = {("main", f):80 for f in MAIN_FRACTIONS}
    expected.update({("interpolation_test", f):10 for f in INTERPOLATION_FRACTIONS})
    if len(rows) != 440 or counts(rows) != expected:
        raise ValueError(f"Expected original accepted 440-row pool; found {len(rows)} rows, counts={counts(rows)}")
    ids = [row["sample_id"] for row in rows]
    seeds = [integer(row["seed"]) for row in rows]
    if len(set(ids)) != 440 or len(set(seeds)) != 440:
        raise ValueError("Duplicate source IDs or accepted seeds.")
    settings = ["L_over_D", "radius", "e_matrix_Pa", "nu_matrix", "e_fibre_Pa", "nu_fibre", "h_far_over_r", "h_interface_over_r"]
    for row in rows:
        for key in settings:
            if not math.isclose(float(row[key]), float(base[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Source/config mismatch: {row['sample_id']} {key}")
        if not math.isclose(float(row["L"]), 2*float(base["radius"])*float(base["L_over_D"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Source/config domain-size mismatch.")
        if row["family"] != "hard_disk" or hash_map(row["code_sha256"]) != spec["expected_code_sha256"]:
            raise ValueError("Source geometry family/code provenance differs from verified baseline.")
        if not all(math.isfinite(float(row[k])) for k in labels):
            raise ValueError(f"Non-finite source response: {row['sample_id']}")
    for (split, vf), n in expected.items():
        reps = [integer(row["replicate"]) for row in rows if row["split"] == split and float(row["vf_target_percent"]) == vf]
        if sorted(reps) != list(range(n)):
            raise ValueError(f"Expected source replicates 0..{n-1}: {split}, {vf}%")


def manifest(spec, base, source_rows, wrapper_hash):
    prior_ids = {row["sample_id"] for row in source_rows}
    prior_seeds = {integer(row["seed"]) for row in source_rows}
    for row in source_rows:
        failed = row.get("original_failed_seed", "")
        if failed and failed.lower() != "nan":
            prior_seeds.add(integer(failed))
    prior_replicates = {(float(row["vf_target_percent"]), integer(row["replicate"])) for row in source_rows}
    jobs = []
    for split, levels, n, seed_base, rep_start, role in (
        ("main", base["main_vf_percent"], integer(spec["new_main_per_fraction"]), integer(spec["main_base_seed"]), integer(spec["main_replicate_start"]), "development"),
        (spec["fresh_interpolation_split"], base["interpolation_test_vf_percent"], integer(spec["new_fresh_interpolation_per_fraction"]), integer(spec["fresh_interpolation_base_seed"]), integer(spec["fresh_interpolation_replicate_start"]), "fresh_interpolation_test"),
    ):
        block_start = len(jobs)
        for vf in levels:
            for k in range(n):
                task = len(jobs)
                row = {
                    "task_id":task,
                    "sample_id":f"sample_{integer(spec['sample_id_start'])+task:04d}",
                    "split":split, "vf_target_percent":float(vf),
                    "replicate":rep_start+k, "seed":seed_base+task-block_start,
                    "generation_batch_id":spec["study_id"], "dataset_role":role,
                    "source_dataset_sha256":spec["source_dataset_sha256"],
                    "extension_wrapper_sha256":wrapper_hash,
                }
                if row["sample_id"] in prior_ids or row["seed"] in prior_seeds:
                    raise ValueError(f"ID/seed overlaps prior accepted/failed record: {row}")
                if (row["vf_target_percent"], row["replicate"]) in prior_replicates:
                    raise ValueError(f"Replicate index overlaps an existing same-fraction record: {row}")
                jobs.append(row)
    expected = {("main", f):120 for f in MAIN_FRACTIONS}
    expected.update({("interpolation_test_fresh", f):50 for f in INTERPOLATION_FRACTIONS})
    if len(jobs) != 800 or counts(jobs) != expected:
        raise ValueError("This study must contain exactly 600 new main and 200 fresh interpolation tasks.")
    if len({r["sample_id"] for r in jobs}) != 800 or len({r["seed"] for r in jobs}) != 800:
        raise ValueError("Extension ID/seed ranges overlap internally.")
    return jobs


def load_context(config_path, gen):
    path = resolve(config_path)
    spec = json.loads(path.read_text())
    required = {
        "study_id", "source_config", "source_config_sha256", "source_dataset", "source_dataset_sha256", "expected_code_sha256",
        "expected_source_records", "existing_main_per_fraction", "existing_interpolation_per_fraction",
        "new_main_per_fraction", "new_fresh_interpolation_per_fraction", "main_base_seed", "fresh_interpolation_base_seed",
        "sample_id_start", "main_replicate_start", "fresh_interpolation_replicate_start", "fresh_interpolation_split",
    }
    if required-set(spec):
        raise ValueError(f"Missing extension fields: {sorted(required-set(spec))}")
    study = spec["study_id"]
    if not study or not all(c.isalnum() or c in "_-" for c in study):
        raise ValueError("Invalid study_id.")
    fixed = {"expected_source_records":440, "existing_main_per_fraction":80, "existing_interpolation_per_fraction":10,
             "new_main_per_fraction":120, "new_fresh_interpolation_per_fraction":50, "sample_id_start":440,
             "main_replicate_start":80, "fresh_interpolation_replicate_start":10}
    if any(integer(spec[k]) != v for k,v in fixed.items()) or spec["fresh_interpolation_split"] != "interpolation_test_fresh":
        raise ValueError("Configuration no longer describes the agreed 1000-development + 200-fresh + 40-existing design.")
    source_file = resolve(spec["source_dataset"])
    if sha256(source_file) != spec["source_dataset_sha256"]:
        raise ValueError("Source dataset bytes differ from the verified upload. Do not edit the source CSV; investigate before proceeding.")
    base, base_digest = gen.load_config(resolve(spec["source_config"]))
    if base_digest != spec["source_config_sha256"]:
        raise ValueError("Original v1 configuration differs from the verified baseline.")
    if [float(f) for f in base["main_vf_percent"]] != MAIN_FRACTIONS or [float(f) for f in base["interpolation_test_vf_percent"]] != INTERPOLATION_FRACTIONS:
        raise ValueError("Unexpected fractions/order in base configuration.")
    source_rows = read_csv(source_file)
    if study in {row["study_id"] for row in source_rows}:
        raise ValueError("Use a new study_id, not an existing source study.")
    validate_source(source_rows, spec, base, gen.LABELS)
    observed = gen.provenance(base, base_digest)["code_sha256"]
    if observed != spec["expected_code_sha256"]:
        raise ValueError("Original generator/packing/FE source changed since baseline. Review before running; do not disable this guard.")
    wrapper_hash = sha256(Path(__file__))
    jobs = manifest(spec, base, source_rows, wrapper_hash)
    effective = dict(base)
    effective.update({
        "study_id":study, "main_realizations":120, "interpolation_test_realizations":50,
        "main_base_seed":spec["main_base_seed"], "interpolation_test_base_seed":spec["fresh_interpolation_base_seed"],
        "extension_specification":spec, "extension_config_sha256":sha256(path),
        "extension_wrapper_sha256":wrapper_hash, "source_config_sha256":base_digest,
        "manifest_sha256":hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
        "manifest_policy":"Use this wrapper's manifest only. Main seeds 50000..50599; fresh seeds 60000..60199 for supplied configuration.",
        "fresh_test_policy":"Do not use fresh test responses for learning curves, feature selection, tuning, or stopping decisions.",
    })
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    return effective, digest, jobs, source_rows, source_file


def geometry_signature(path):
    p = resolve(path)
    rows = read_csv(p)
    coordinates = sorted((round(float(r["x_center"]),12), round(float(r["y_center"]),12), round(float(r["radius"]),12)) for r in rows)
    if not coordinates or not all(all(math.isfinite(v) for v in r) for r in coordinates):
        raise ValueError(f"Empty/non-finite geometry: {p}")
    return hashlib.sha256(json.dumps(coordinates, separators=(",",":")).encode()).hexdigest()


def validate_geometries(rows):
    seen = {}
    for row in rows:
        fingerprint = geometry_signature(row["geometry_file"])
        if fingerprint in seen:
            raise ValueError(f"Repeated sorted coordinate geometry: {seen[fingerprint]} and {row['sample_id']}")
        seen[fingerprint] = row["sample_id"]
    return seen


def write_csv_atomic(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def preparation_metadata(c, digest, jobs):
    return {"effective_config_sha256":digest, "study_id":c["study_id"],
            "extension_wrapper_sha256":c["extension_wrapper_sha256"],
            "source_dataset_sha256":c["extension_specification"]["source_dataset_sha256"],
            "source_records":440, "new_tasks":len(jobs), "new_development":600, "new_fresh_interpolation":200,
            "final_development":1000, "final_fresh_interpolation":200, "existing_interpolation":40,
            "final_records":1240, "fresh_split":"interpolation_test_fresh"}


def require_prepared(gen,c,digest,jobs):
    root,_ = gen.paths(c)
    p = root / "preparation_metadata.json"
    expected = preparation_metadata(c,digest,jobs)
    if not p.exists() or json.loads(p.read_text()) != expected:
        raise RuntimeError("Run this wrapper with --prepare first; if already prepared, code/config/source may have changed.")
    if not (root/"source_dataset_snapshot.csv").exists() or sha256(root/"source_dataset_snapshot.csv") != expected["source_dataset_sha256"]:
        raise RuntimeError("Prepared source snapshot is missing or changed.")
    if not (root/"extension_manifest.json").exists() or json.loads((root/"extension_manifest.json").read_text()) != jobs:
        raise RuntimeError("Prepared manifest is missing or changed.")


def prepare(gen,c,digest,jobs,source_rows,source_file):
    root,dirs = gen.paths(c)
    marker = root/"preparation_metadata.json"
    expected = preparation_metadata(c,digest,jobs)
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise RuntimeError("Existing prepared study has different provenance. Use a new study_id; do not overwrite it.")
    if not marker.exists() and (list(dirs["records"].glob("*.json")) or list(dirs["geometry"].glob("*.csv"))):
        raise RuntimeError("Existing records/geometry without matching preparation metadata; inspect before proceeding.")
    validate_geometries(source_rows)
    for folder in dirs.values():
        folder.mkdir(parents=True,exist_ok=True)
    snapshot = root/"source_dataset_snapshot.csv"
    if snapshot.exists() and sha256(snapshot) != expected["source_dataset_sha256"]:
        raise RuntimeError("Existing source snapshot differs. Inspect; it will not be overwritten.")
    if not snapshot.exists():
        temp = snapshot.with_suffix(f".csv.{os.getpid()}.tmp")
        try:
            temp.write_bytes(source_file.read_bytes())
            os.replace(temp,snapshot)
        finally:
            temp.unlink(missing_ok=True)
    gen.atomic_json(root/"extension_manifest.json",jobs)
    write_csv_atomic(root/"extension_manifest.csv",jobs)
    gen.atomic_json(root/"effective_configuration.json",c)
    gen.atomic_json(marker,expected)
    print(f"PREPARED {root}")
    print("TASK_COUNT=800; new main=600; fresh interpolation=200; no FE simulations run.")


def summarize(gen,c,digest,jobs,source_rows):
    require_prepared(gen,c,digest,jobs)
    root,dirs = gen.paths(c)
    expected = gen.provenance(c,digest)
    new_rows,missing = [],[]
    for job in jobs:
        record = dirs["records"]/f"{job['sample_id']}.json"
        if not record.exists():
            missing.append(job["task_id"])
            continue
        row = json.loads(record.read_text())
        if any(row.get(k) != v for k,v in job.items()):
            raise ValueError(f"Record/manifest mismatch: {record}")
        if any(row.get(k) != expected[k] for k in ["config_sha256","code_sha256","study_id"]):
            raise ValueError(f"Record provenance mismatch: {record}")
        if not all(k in row and math.isfinite(float(row[k])) for k in gen.LABELS):
            raise ValueError(f"Incomplete/non-finite response: {record}")
        if not resolve(row["geometry_file"]).is_file():
            raise ValueError(f"Missing geometry: {row['sample_id']}")
        new_rows.append(row)
    failure_logs = list(dirs["failures"].glob("*.txt"))
    print(f"Accepted extension records: {len(new_rows)}/800; failure logs: {len(failure_logs)}")
    if missing or failure_logs:
        print("Missing task IDs:", ",".join(map(str,missing)))
        print("Incomplete: no consolidated tables written. Inspect failure logs and retained geometries before recovery.")
        return 1
    combined = source_rows+new_rows
    expected_counts = {("main",f):200 for f in MAIN_FRACTIONS}
    expected_counts.update({("interpolation_test",f):10 for f in INTERPOLATION_FRACTIONS})
    expected_counts.update({("interpolation_test_fresh",f):50 for f in INTERPOLATION_FRACTIONS})
    if len(combined) != 1240 or counts(combined) != expected_counts:
        raise ValueError("Final dataset does not match the agreed split/fraction counts.")
    if len({r["sample_id"] for r in combined}) != 1240 or len({integer(r["seed"]) for r in combined}) != 1240:
        raise ValueError("Repeated IDs/seeds in combined table.")
    validate_geometries(combined)
    # Remove completion marker while publishing tables; consumers must wait for --summary success.
    completion = root/"completion_metadata.json"
    completion.unlink(missing_ok=True)
    tables = {
        "extension_dataset.csv":new_rows,
        "expanded_dataset.csv":combined,
        "development_dataset.csv":[r for r in combined if r["split"] == "main"],
        "fresh_interpolation_test.csv":[r for r in combined if r["split"] == "interpolation_test_fresh"],
        "legacy_interpolation_test.csv":[r for r in combined if r["split"] == "interpolation_test"],
    }
    for filename,rows in tables.items():
        write_csv_atomic(root/filename,rows)
    gen.atomic_json(completion,{
        **preparation_metadata(c,digest,jobs), "completed":True,
        "output_sha256":{filename:sha256(root/filename) for filename in tables},
        "source_tables_modified":False,
        "geometry_check":"Sorted centres/radii rounded to 12 decimals; not a test of translated or rotated equivalence.",
        "fresh_test_policy":c["fresh_test_policy"],
    })
    print(f"COMPLETE {root/'expanded_dataset.csv'}")
    print("Development: 1000 (200 per principal fraction); fresh interpolation: 200 (50 per intermediate fraction); existing interpolation: 40.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",action="store_true")
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--task-id",type=int)
    group.add_argument("--summary",action="store_true")
    args = parser.parse_args()
    gen = load_original_generator()
    c,digest,jobs,source_rows,source_file = load_context(args.config,gen)
    if args.list:
        for row in jobs:
            print(json.dumps(row,sort_keys=True))
        print("TASK_COUNT=800")
        return 0
    if args.prepare:
        prepare(gen,c,digest,jobs,source_rows,source_file)
        return 0
    if args.summary:
        return summarize(gen,c,digest,jobs,source_rows)
    if not 0 <= args.task_id < 800:
        parser.error("--task-id must be in 0..799")
    require_prepared(gen,c,digest,jobs)
    gen.run(c,digest,jobs[args.task_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
