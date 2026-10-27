# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:57:09 2026

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

INPUT = (
    DATASET_DIR
    / "descriptor_audit"
    / "microstructure_descriptors.csv"
)

OUTPUT = (
    DATASET_DIR
    / "two_point_statistics"
)

FIGURES = OUTPUT / "figures"

OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


IMAGE_RESOLUTION = 128


def resolve_geometry_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return ROOT / path


def rasterize_periodic_fibres(centres, radius, L, resolution):
    dx = L / resolution

    image = np.zeros(
        (resolution, resolution),
        dtype=bool,
    )

    for x_center, y_center in centres:
        x_indices = np.arange(
            int(np.floor((x_center - radius) / dx)),
            int(np.floor((x_center + radius) / dx)) + 1,
        )

        y_indices = np.arange(
            int(np.floor((y_center - radius) / dx)),
            int(np.floor((y_center + radius) / dx)) + 1,
        )

        x_positions = (x_indices + 0.5) * dx
        y_positions = (y_indices + 0.5) * dx

        delta_x = x_positions - x_center
        delta_y = y_positions - y_center

        delta_x -= L * np.round(delta_x / L)
        delta_y -= L * np.round(delta_y / L)

        distance_squared = (
            delta_y[:, None] ** 2
            + delta_x[None, :] ** 2
        )

        fibre_mask = distance_squared <= radius**2

        wrapped_x = np.mod(x_indices, resolution)
        wrapped_y = np.mod(y_indices, resolution)

        image[
            np.ix_(wrapped_y, wrapped_x)
        ] |= fibre_mask

    return image.astype(np.float32)


def periodic_two_point_autocorrelation(indicator):
    fft_indicator = np.fft.fft2(indicator)

    autocorrelation = np.fft.ifft2(
        np.abs(fft_indicator) ** 2
    ).real

    autocorrelation /= indicator.size

    return autocorrelation.astype(np.float32)


def process_sample(row):
    geometry_path = resolve_geometry_path(
        row["geometry_path_resolved"]
    )

    if not geometry_path.exists():
        raise FileNotFoundError(
            f"Geometry file does not exist: {geometry_path}"
        )

    geometry = pd.read_csv(geometry_path)

    required_columns = [
        "x_center",
        "y_center",
        "radius",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in geometry.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{row['sample_id']} is missing columns: "
            f"{missing_columns}"
        )

    centres = geometry[
        ["x_center", "y_center"]
    ].to_numpy(dtype=float)

    radii = geometry["radius"].to_numpy(dtype=float)

    L = float(row["L"])
    radius = float(row["radius"])

    if not np.allclose(
        radii,
        radius,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{row['sample_id']}: geometry radius mismatch."
        )

    indicator = rasterize_periodic_fibres(
        centres=centres,
        radius=radius,
        L=L,
        resolution=IMAGE_RESOLUTION,
    )

    s2 = periodic_two_point_autocorrelation(indicator)

    raster_vf = float(indicator.mean())
    geometry_vf = float(row["vf_achieved"])

    if not np.isfinite(s2).all():
        raise ValueError(f"{row['sample_id']}: non-finite S2.")
    if not np.isclose(float(s2[0,0]),raster_vf,rtol=0,atol=1e-6):
        raise ValueError(f"{row['sample_id']}: zero-offset S2 does not equal raster fraction.")
    if not np.isclose(float(s2.mean()),raster_vf**2,rtol=0,atol=1e-6):
        raise ValueError(f"{row['sample_id']}: periodic S2 mean does not equal raster fraction squared.")


    return indicator, s2, raster_vf, geometry_vf


def plot_example_fields(indicator, s2, sample_id):
    shifted_s2 = np.fft.fftshift(s2)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
    )

    axes[0].imshow(
        indicator,
        cmap="gray_r",
        interpolation="nearest",
        origin="lower",
    )

    axes[0].set_title(
        f"Binary fibre indicator: {sample_id}"
    )

    axes[0].set_xlabel("Pixel index")
    axes[0].set_ylabel("Pixel index")

    image = axes[1].imshow(
        shifted_s2,
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
    )

    axes[1].set_title(
        "Periodic fibre-phase two-point statistic"
    )

    axes[1].set_xlabel(r"$\Delta x$ pixel offset")
    axes[1].set_ylabel(r"$\Delta y$ pixel offset")

    fig.colorbar(
        image,
        ax=axes[1],
        label=r"$S_2(\Delta x,\Delta y)$",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "example_binary_microstructure_and_S2.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)



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
    df = pd.read_csv(INPUT)
    validate_v2_table(df)

    if len(df) != 1240:
        raise ValueError(
            f"Expected 1240 RVEs, found {len(df)}."
        )

    required_dataset_columns = [
        "sample_id",
        "split",
        "vf_achieved",
        "vf_target_percent",
        "L",
        "radius",
        "geometry_path_resolved",
    ]

    missing_dataset_columns = [
        column
        for column in required_dataset_columns
        if column not in df.columns
    ]

    if missing_dataset_columns:
        raise ValueError(
            "Missing required dataset columns: "
            f"{missing_dataset_columns}"
        )

    all_s2 = []
    metadata_rows = []

    example_indicator = None
    example_s2 = None
    example_sample_id = None

    for index, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        indicator, s2, raster_vf, geometry_vf = process_sample(row)

        all_s2.append(
            s2.reshape(-1)
        )

        metadata_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "vf_target_percent": row[
                    "vf_target_percent"
                ],
                "vf_achieved": geometry_vf,
                "raster_vf": raster_vf,
                "raster_minus_geometry_vf": (
                    raster_vf - geometry_vf
                ),
            }
        )

        if example_indicator is None:
            example_indicator = indicator
            example_s2 = s2
            example_sample_id = row["sample_id"]

        print(
            f"[{index:3d}/{len(df)}] "
            f"{row['sample_id']} processed"
        )

    s2_matrix = np.vstack(all_s2).astype(np.float32)

    expected_shape = (
        len(df),
        IMAGE_RESOLUTION * IMAGE_RESOLUTION,
    )

    if s2_matrix.shape != expected_shape:
        raise RuntimeError(
            "Unexpected two-point-statistics matrix shape: "
            f"{s2_matrix.shape}; expected {expected_shape}."
        )

    metadata = pd.DataFrame(metadata_rows)

    metadata.to_csv(
        OUTPUT / "two_point_statistics_metadata.csv",
        index=False,
    )

    sample_id_array = (
        df["sample_id"]
        .astype(str)
        .to_numpy(dtype="<U64")
    )

    split_array = (
        df["split"]
        .astype(str)
        .to_numpy(dtype="<U32")
    )

    np.savez_compressed(
        OUTPUT / "periodic_fibre_S2_128x128.npz",
        sample_id=sample_id_array,
        split=split_array,
        vf_achieved=df["vf_achieved"].to_numpy(
            dtype=np.float64
        ),
        vf_target_percent=df[
            "vf_target_percent"
        ].to_numpy(dtype=np.float64),
        S2=s2_matrix,
        image_resolution=np.array(
            [IMAGE_RESOLUTION],
            dtype=np.int32,
        ),
    )

    plot_example_fields(
        indicator=example_indicator,
        s2=example_s2,
        sample_id=example_sample_id,
    )

    max_raster_error = float(
        np.abs(
            metadata["raster_minus_geometry_vf"]
        ).max()
    )

    mean_raster_error = float(
        metadata["raster_minus_geometry_vf"].mean()
    )

    run_metadata = {
        "input_file": str(INPUT),
        "number_of_samples": int(len(df)),
        "image_resolution": int(IMAGE_RESOLUTION),
        "two_point_representation": (
            "periodic fibre-phase autocorrelation "
            "computed by FFT"
        ),
        "S2_matrix_shape": [
            int(s2_matrix.shape[0]),
            int(s2_matrix.shape[1]),
        ],
        "sample_id_dtype": str(sample_id_array.dtype),
        "split_dtype": str(split_array.dtype),
        "maximum_absolute_raster_vf_error": max_raster_error,
        "mean_raster_vf_error": mean_raster_error,
    }

    with open(
        OUTPUT / "two_point_statistics_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            indent=2,
        )

    print("\nTwo-point-statistics generation completed.")
    print(
        f"S2 matrix shape: {s2_matrix.shape}"
    )
    print(
        "Maximum |raster Vf - geometry Vf|: "
        f"{max_raster_error:.6e}"
    )
    print(
        "Mean raster Vf - geometry Vf: "
        f"{mean_raster_error:.6e}"
    )
    print(
        "sample_id dtype: "
        f"{sample_id_array.dtype}"
    )
    print(
        "split dtype: "
        f"{split_array.dtype}"
    )
    print(
        "Compressed S2 file: "
        f"{OUTPUT / 'periodic_fibre_S2_128x128.npz'}"
    )


if __name__ == "__main__":
    main()