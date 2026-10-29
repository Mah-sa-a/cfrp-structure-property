#!/usr/bin/env python3
"""Predict stiffness in GPa using saved extended-study surrogate models."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_surrogate_extended_v2 import build_inputs, predict_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--normalized-s2", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_file = args.model_dir / "final_models.joblib"
    metadata = json.loads(
        (args.model_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if digest != metadata["saved_models_sha256"]:
        raise ValueError("Model checksum differs from training metadata.")

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    # Load only a trusted model artifact.
    payload = joblib.load(model_file)
    names = args.models
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique.")
    unknown = set(names) - set(payload["model_names"])
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    descriptors = pd.read_csv(
        args.descriptors, dtype={"sample_id": str}
    )
    scalar_columns = sorted({
        column
        for columns in payload["scalar_representations"].values()
        for column in columns
    })
    required = {"sample_id", "vf_achieved", *scalar_columns}
    missing = required - set(descriptors.columns)
    if missing:
        raise ValueError(f"Missing descriptor columns: {sorted(missing)}")
    if descriptors.empty:
        raise ValueError("Descriptor table is empty.")
    if descriptors["sample_id"].isna().any():
        raise ValueError("Missing descriptor sample IDs.")
    if descriptors["sample_id"].duplicated().any():
        raise ValueError("Duplicate descriptor sample IDs.")
    if not np.isfinite(
        descriptors[["vf_achieved", *scalar_columns]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite descriptor values.")

    with np.load(args.normalized_s2, allow_pickle=False) as archive:
        ids = archive["sample_id"].astype(str)
        fields = archive["S2"]
        if ids.ndim != 1 or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicate S2 sample IDs.")
        if fields.shape != (len(ids), 16384):
            raise ValueError("Expected S2 shape (number_of_samples, 16384).")
        positions = {sample_id: i for i, sample_id in enumerate(ids)}
        requested = descriptors["sample_id"].tolist()
        absent = set(requested) - set(positions)
        if absent:
            raise ValueError(f"{len(absent)} descriptor IDs are missing in S2.")
        order = [positions[sample_id] for sample_id in requested]
        s2 = fields[order].astype(np.float32)

        if "vf_achieved" in archive:
            if not np.allclose(
                archive["vf_achieved"][order],
                descriptors["vf_achieved"].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError("Descriptor and S2 volume fractions differ.")

    if not np.isfinite(s2).all():
        raise ValueError("Non-finite normalized S2 values.")

    inputs = build_inputs(descriptors, s2)
    predictions = predict_models(
        payload["models"], inputs, payload["config"], names
    )

    tables = []
    for name in names:
        values = predictions[name]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions for {name}.")
        for index, target in enumerate(payload["targets"]):
            tables.append(pd.DataFrame({
                "sample_id": descriptors["sample_id"].to_numpy(),
                "model": name,
                "target": target,
                "prediction_GPa": values[:, index],
            }))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.output, index=False)
    print(f"Predictions written to: {args.output}")
    print("Units: GPa. No fitting or tuning performed.")


if __name__ == "__main__":
    main()