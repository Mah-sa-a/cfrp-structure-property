# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:16:55 2026

@author: 43020593
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT
    / "results"
    / "surrogate_dataset"
    / "surrogate_LD18_hard_disk_extension_v2"
)

INPUT_DIR = (
    DATASET_DIR
    / "two_point_statistics"
)

INPUT_S2 = (
    INPUT_DIR
    / "periodic_fibre_S2_128x128.npz"
)

INPUT_METADATA = (
    INPUT_DIR
    / "two_point_statistics_metadata.csv"
)

OUTPUT_S2 = (
    INPUT_DIR
    / "periodic_fibre_normalized_S2_128x128.npz"
)

OUTPUT_METADATA = (
    INPUT_DIR
    / "normalized_two_point_statistics_metadata.json"
)

FIGURES = INPUT_DIR / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)



def validate_v2_table(df):
    required = {"sample_id", "split", "vf_target_percent"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing identity columns: {sorted(required-set(df.columns))}")
    if len(df) != 1240 or df["sample_id"].isna().any() or df["sample_id"].astype(str).duplicated().any():
        raise ValueError("Expected 1240 unique accepted records.")
    expected = {("main", f):200 for f in [40.,45.,50.,55.,60.]}
    expected.update({("interpolation_test", f):10 for f in [42.5,47.5,52.5,57.5]})
    expected.update({("interpolation_test_fresh", f):50 for f in [42.5,47.5,52.5,57.5]})
    observed = df.groupby(["split", "vf_target_percent"], dropna=False).size().to_dict()
    if observed != expected:
        raise ValueError(f"Unexpected v2 split/fraction counts: {observed}")


def verify_v2_completion():
    import hashlib
    marker = DATASET_DIR / "completion_metadata.json"
    metadata = json.loads(marker.read_text())
    expected_counts = {"final_development":1000, "final_fresh_interpolation":200, "existing_interpolation":40, "final_records":1240}
    if metadata.get("completed") is not True or any(metadata.get(k) != v for k,v in expected_counts.items()):
        raise ValueError("Dataset completion metadata is invalid.")
    dataset = DATASET_DIR / "expanded_dataset.csv"
    observed = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if observed != metadata["output_sha256"]["expanded_dataset.csv"]:
        raise ValueError("Expanded CSV no longer matches its completion checksum.")

def main():
    verify_v2_completion()
    data = np.load(
        INPUT_S2,
        allow_pickle=False,
    )

    sample_id = data["sample_id"].astype(str)
    split = data["split"].astype(str)
    vf_achieved = data["vf_achieved"].astype(np.float64)
    vf_target_percent = data[
        "vf_target_percent"
    ].astype(np.float64)

    raw_s2 = data["S2"].astype(np.float32)

    image_resolution = int(
        data["image_resolution"][0]
    )

    expected_features = (
        image_resolution * image_resolution
    )

    if raw_s2.shape != (
        len(sample_id),
        expected_features,
    ):
        raise ValueError(
            "Unexpected raw S2 shape: "
            f"{raw_s2.shape}"
        )

    metadata = pd.read_csv(INPUT_METADATA)

    validate_v2_table(metadata)
    if not np.array_equal(metadata["split"].astype(str).to_numpy(),split):
        raise ValueError("Metadata/archive split labels differ.")
    for column,values in [("vf_achieved",vf_achieved),("vf_target_percent",vf_target_percent)]:
        if not np.allclose(metadata[column].to_numpy(dtype=float),values,rtol=0,atol=1e-12):
            raise ValueError(f"Metadata/archive {column} values differ.")
    if not np.isfinite(raw_s2).all():
        raise ValueError("Non-finite raw S2.")


    if len(metadata) != len(sample_id):
        raise ValueError(
            "Metadata and S2 row counts differ."
        )

    if not np.array_equal(
        metadata["sample_id"].astype(str).to_numpy(),
        sample_id,
    ):
        raise ValueError(
            "Sample order differs between metadata and S2 archive."
        )

    raster_vf = metadata[
        "raster_vf"
    ].to_numpy(dtype=np.float64)

    if np.any(raster_vf <= 0.0) or np.any(raster_vf >= 1.0):
        raise ValueError(
            "Raster fibre fractions must lie strictly between 0 and 1."
        )

    denominator = (
        raster_vf * (1.0 - raster_vf)
    )

    normalized_s2 = (
        raw_s2.astype(np.float64)
        - raster_vf[:, None] ** 2
    ) / denominator[:, None]

    normalized_s2 = normalized_s2.astype(np.float32)

    if not np.isfinite(normalized_s2).all():
        raise ValueError(
            "Non-finite values found in normalized S2."
        )


    if not np.allclose(normalized_s2[:,0],1.0,rtol=0,atol=1e-5):
        raise ValueError("Normalized S2 at zero offset must be approximately one.")
    if not np.allclose(normalized_s2.mean(axis=1),0.0,rtol=0,atol=1e-5):
        raise ValueError("Periodic normalized S2 spatial mean must be approximately zero.")

    np.savez_compressed(
        OUTPUT_S2,
        sample_id=sample_id.astype("<U64"),
        split=split.astype("<U32"),
        vf_achieved=vf_achieved,
        vf_target_percent=vf_target_percent,
        S2=normalized_s2,
        image_resolution=np.array(
            [image_resolution],
            dtype=np.int32,
        ),
    )

    example_index = 0

    raw_example = raw_s2[
        example_index
    ].reshape(
        image_resolution,
        image_resolution,
    )

    normalized_example = normalized_s2[
        example_index
    ].reshape(
        image_resolution,
        image_resolution,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    raw_image = axes[0].imshow(
        np.fft.fftshift(raw_example),
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )

    axes[0].set_title(
        "Raw periodic fibre-phase two-point statistic"
    )

    axes[0].set_xlabel(r"$\Delta x$ pixel offset")
    axes[0].set_ylabel(r"$\Delta y$ pixel offset")

    fig.colorbar(
        raw_image,
        ax=axes[0],
        label=r"$S_2(\Delta x,\Delta y)$",
    )

    normalized_image = axes[1].imshow(
        np.fft.fftshift(normalized_example),
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
    )

    axes[1].set_title(
        "Normalized periodic two-point autocovariance"
    )

    axes[1].set_xlabel(r"$\Delta x$ pixel offset")
    axes[1].set_ylabel(r"$\Delta y$ pixel offset")

    fig.colorbar(
        normalized_image,
        ax=axes[1],
        label=r"$\widetilde{S}_2(\Delta x,\Delta y)$",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "raw_and_normalized_S2_example.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    central_values = normalized_s2[
        :,
        0,
    ]

    far_offset_index = (
        image_resolution // 2
    ) * image_resolution + (
        image_resolution // 2
    )

    far_offset_values = normalized_s2[
        :,
        far_offset_index,
    ]

    output_metadata = {
        "input_s2_file": str(INPUT_S2),
        "input_metadata_file": str(INPUT_METADATA),
        "output_s2_file": str(OUTPUT_S2),
        "number_of_samples": int(len(sample_id)),
        "image_resolution": image_resolution,
        "representation": (
            "normalized periodic fibre-phase "
            "two-point autocovariance"
        ),
        "formula": (
            "(S2 - vf_raster^2) / "
            "(vf_raster * (1 - vf_raster))"
        ),
        "mean_normalized_S2_at_zero_offset": float(
            central_values.mean()
        ),
        "minimum_normalized_S2_at_zero_offset": float(
            central_values.min()
        ),
        "maximum_normalized_S2_at_zero_offset": float(
            central_values.max()
        ),
        "mean_normalized_S2_at_half_domain_offset": float(
            far_offset_values.mean()
        ),
        "minimum_normalized_S2": float(
            normalized_s2.min()
        ),
        "maximum_normalized_S2": float(
            normalized_s2.max()
        ),
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_metadata,
            file,
            indent=2,
        )

    print("\nNormalized two-point statistics completed.")
    print(
        f"Normalized S2 matrix shape: "
        f"{normalized_s2.shape}"
    )
    print(
        "Mean normalized S2 at zero offset: "
        f"{central_values.mean():.8f}"
    )
    print(
        "Mean normalized S2 at half-domain offset: "
        f"{far_offset_values.mean():.8f}"
    )
    print(
        "Minimum normalized S2 value: "
        f"{normalized_s2.min():.8f}"
    )
    print(
        "Maximum normalized S2 value: "
        f"{normalized_s2.max():.8f}"
    )
    print(f"Output file: {OUTPUT_S2}")


if __name__ == "__main__":
    main()