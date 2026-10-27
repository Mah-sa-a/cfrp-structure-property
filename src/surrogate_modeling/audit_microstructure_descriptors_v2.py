# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 00:47:55 2026

@author: 43020593
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT
    / "results"
    / "surrogate_dataset"
    / "surrogate_LD18_hard_disk_extension_v2"
)

DATASET_FILE = DATASET_DIR / "expanded_dataset.csv"

OUTPUT_DIR = DATASET_DIR / "descriptor_audit"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLUMNS = [
    "C11_Pa",
    "C12_Pa",
    "C22_Pa",
    "C33_Pa",
]

DESCRIPTOR_COLUMNS = [
    "vf_achieved",
    "nn_distance_mean_over_D",
    "nn_distance_std_over_D",
    "nn_distance_min_over_D",
    "ligament_mean_over_D",
    "ligament_std_over_D",
    "ligament_min_over_D",
    "local_vf_cv",
    "local_vf_range",
    "centre_anisotropy",
]


def periodic_pairwise_distances(centres, L):
    delta = centres[:, None, :] - centres[None, :, :]
    delta -= L * np.round(delta / L)

    distances = np.sqrt(np.sum(delta**2, axis=2))
    np.fill_diagonal(distances, np.inf)

    return distances


def nearest_neighbour_descriptors(centres, radius, L):
    distances = periodic_pairwise_distances(centres, L)

    nearest_distances = distances.min(axis=1)
    nearest_ligaments = nearest_distances - 2.0 * radius

    upper = distances[np.triu_indices_from(distances, k=1)]
    pair_ligaments = upper - 2.0 * radius

    return {
        "nn_distance_mean": float(np.mean(nearest_distances)),
        "nn_distance_std": float(np.std(nearest_distances, ddof=1)),
        "nn_distance_min": float(np.min(nearest_distances)),
        "ligament_mean": float(np.mean(nearest_ligaments)),
        "ligament_std": float(np.std(nearest_ligaments, ddof=1)),
        "ligament_min": float(np.min(nearest_ligaments)),
        "pair_ligament_mean": float(np.mean(pair_ligaments)),
        "pair_ligament_std": float(np.std(pair_ligaments, ddof=1)),
    }


def local_fibre_fraction_descriptors(centres, radius, L, n_cells=6):
    cell_size = L / n_cells
    counts = np.zeros((n_cells, n_cells), dtype=float)

    indices = np.floor(centres / cell_size).astype(int)
    indices = np.clip(indices, 0, n_cells - 1)

    for ix, iy in indices:
        counts[iy, ix] += 1.0

    fibre_area = np.pi * radius**2
    cell_area = cell_size**2
    local_vf = counts * fibre_area / cell_area

    mean_vf = float(np.mean(local_vf))

    if mean_vf <= 0.0:
        local_vf_cv = np.nan
    else:
        local_vf_cv = float(np.std(local_vf, ddof=1) / mean_vf)

    return {
        "local_vf_cv": local_vf_cv,
        "local_vf_range": float(np.max(local_vf) - np.min(local_vf)),
        "local_vf_std": float(np.std(local_vf, ddof=1)),
    }


def centre_anisotropy_descriptor(centres, L):
    angles = 2.0 * np.pi * centres / L

    mean_cos_x = np.mean(np.cos(angles[:, 0]))
    mean_sin_x = np.mean(np.sin(angles[:, 0]))
    mean_cos_y = np.mean(np.cos(angles[:, 1]))
    mean_sin_y = np.mean(np.sin(angles[:, 1]))

    circular_resultant_x = np.sqrt(mean_cos_x**2 + mean_sin_x**2)
    circular_resultant_y = np.sqrt(mean_cos_y**2 + mean_sin_y**2)

    return float(np.abs(circular_resultant_x - circular_resultant_y))


def resolve_geometry_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return ROOT / path


def process_sample(row):
    geometry_path = resolve_geometry_path(row["geometry_file"])

    if not geometry_path.exists():
        raise FileNotFoundError(
            f"Geometry file not found for {row['sample_id']}: {geometry_path}"
        )

    geom = pd.read_csv(geometry_path)

    required_columns = {
        "fibre_id",
        "x_center",
        "y_center",
        "radius",
        "nn_surface_gap",
    }

    missing = required_columns - set(geom.columns)

    if missing:
        raise ValueError(
            f"{row['sample_id']} is missing geometry columns: {sorted(missing)}"
        )

    centres = geom[["x_center", "y_center"]].to_numpy(dtype=float)
    radii = geom["radius"].to_numpy(dtype=float)

    L = float(row["L"])
    radius = float(row["radius"])

    if len(geom) != int(row["n_fibers"]):
        raise ValueError(
            f"{row['sample_id']}: geometry has {len(geom)} fibres, "
            f"but dataset reports {int(row['n_fibers'])}."
        )

    if not np.allclose(radii, radius, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{row['sample_id']}: geometry radii do not match dataset radius."
        )

    if np.any(centres < 0.0) or np.any(centres >= L):
        raise ValueError(
            f"{row['sample_id']}: at least one fibre centre is outside [0, L)."
        )

    vf_from_geometry = len(geom) * np.pi * radius**2 / L**2

    distances = periodic_pairwise_distances(centres, L)
    nearest_distances = distances.min(axis=1)
    nearest_ligaments = nearest_distances - 2.0 * radius

    if np.any(nearest_ligaments < -1.0e-10):
        raise ValueError(
            f"{row['sample_id']}: periodic fibre overlap detected."
        )

    stored_nn_gaps = geom["nn_surface_gap"].to_numpy(dtype=float)

    max_nn_gap_difference = float(
        np.max(np.abs(stored_nn_gaps - nearest_ligaments))
    )

    if max_nn_gap_difference > 1.0e-8:
        raise ValueError(
            f"{row['sample_id']}: stored nn_surface_gap differs from "
            f"recomputed periodic value by {max_nn_gap_difference:.3e}."
        )

    nn = nearest_neighbour_descriptors(centres, radius, L)
    local = local_fibre_fraction_descriptors(centres, radius, L)
    anisotropy = centre_anisotropy_descriptor(centres, L)

    D = 2.0 * radius

    result = row.to_dict()

    result.update(
        {
            "geometry_path_resolved": str(geometry_path),
            "vf_from_geometry": vf_from_geometry,
            "vf_geometry_minus_recorded": (
                vf_from_geometry - float(row["vf_achieved"])
            ),
            "max_nn_gap_difference": max_nn_gap_difference,
            "nn_distance_mean": nn["nn_distance_mean"],
            "nn_distance_std": nn["nn_distance_std"],
            "nn_distance_min": nn["nn_distance_min"],
            "ligament_mean": nn["ligament_mean"],
            "ligament_std": nn["ligament_std"],
            "ligament_min": nn["ligament_min"],
            "pair_ligament_mean": nn["pair_ligament_mean"],
            "pair_ligament_std": nn["pair_ligament_std"],
            "local_vf_cv": local["local_vf_cv"],
            "local_vf_range": local["local_vf_range"],
            "local_vf_std": local["local_vf_std"],
            "centre_anisotropy": anisotropy,
            "nn_distance_mean_over_D": nn["nn_distance_mean"] / D,
            "nn_distance_std_over_D": nn["nn_distance_std"] / D,
            "nn_distance_min_over_D": nn["nn_distance_min"] / D,
            "ligament_mean_over_D": nn["ligament_mean"] / D,
            "ligament_std_over_D": nn["ligament_std"] / D,
            "ligament_min_over_D": nn["ligament_min"] / D,
        }
    )

    return result


def plot_descriptor_distributions(df):
    descriptors = [
        "nn_distance_mean_over_D",
        "nn_distance_std_over_D",
        "ligament_min_over_D",
        "local_vf_cv",
        "local_vf_range",
        "centre_anisotropy",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    fractions = sorted(df["vf_target_percent"].unique())

    for ax, descriptor in zip(axes, descriptors):
        values = [
            df.loc[
                df["vf_target_percent"] == vf,
                descriptor,
            ].to_numpy()
            for vf in fractions
        ]

        ax.boxplot(
            values,
            positions=fractions,
            widths=1.3,
            showfliers=False,
        )

        for split, color, marker in [
            ("main", "black", "o"),
            ("interpolation_test", "#d62728", "s"),
        ]:
            subset = df[df["split"] == split]

            ax.scatter(
                subset["vf_target_percent"],
                subset[descriptor],
                s=18,
                color=color,
                marker=marker,
                alpha=0.75,
                label=split.replace("_", " "),
                zorder=3,
            )

        ax.set_title(descriptor)
        ax.set_xlabel(r"Target fibre area fraction, $V_f$ (%)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
    )

    fig.suptitle(
        "Packing-descriptor distributions by fibre fraction",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(
        FIGURES_DIR / "descriptor_distributions_by_vf.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_descriptor_vs_vf(df):
    descriptors = [
        "nn_distance_mean_over_D",
        "nn_distance_std_over_D",
        "ligament_min_over_D",
        "local_vf_cv",
        "local_vf_range",
        "centre_anisotropy",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, descriptor in zip(axes, descriptors):
        for split, color, marker in [
            ("main", "black", "o"),
            ("interpolation_test", "#d62728", "s"),
        ]:
            subset = df[df["split"] == split]

            ax.scatter(
                100.0 * subset["vf_achieved"],
                subset[descriptor],
                s=26,
                color=color,
                marker=marker,
                alpha=0.8,
                label=split.replace("_", " "),
            )

        ax.set_title(descriptor)
        ax.set_xlabel(r"Achieved fibre area fraction, $V_f$ (%)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
    )

    fig.suptitle(
        "Packing descriptors versus achieved fibre area fraction",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(
        FIGURES_DIR / "descriptors_vs_achieved_vf.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_correlation_heatmap(df):
    columns = [
        "vf_achieved",
        "nn_distance_mean_over_D",
        "nn_distance_std_over_D",
        "nn_distance_min_over_D",
        "ligament_mean_over_D",
        "ligament_std_over_D",
        "ligament_min_over_D",
        "local_vf_cv",
        "local_vf_range",
        "centre_anisotropy",
    ] + TARGET_COLUMNS

    correlation = df[columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(
        correlation.to_numpy(),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            value = correlation.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(value) < 0.65 else "white",
            )

    fig.colorbar(image, ax=ax, label="Pearson correlation coefficient")

    ax.set_title("Correlation between packing descriptors and stiffness outputs")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "descriptor_target_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    correlation.to_csv(
        OUTPUT_DIR / "descriptor_target_correlation.csv"
    )


def plot_descriptor_pca(df):
    feature_data = df[DESCRIPTOR_COLUMNS].copy()

    if feature_data.isna().any().any():
        raise ValueError("NaN values found in descriptor PCA input.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_data)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    pca_table = df[
        [
            "sample_id",
            "split",
            "vf_target_percent",
            "vf_achieved",
        ]
    ].copy()

    pca_table["PC1"] = coordinates[:, 0]
    pca_table["PC2"] = coordinates[:, 1]

    pca_table.to_csv(
        OUTPUT_DIR / "descriptor_pca_coordinates.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))

    fractions = sorted(df["vf_target_percent"].unique())
    cmap = plt.get_cmap("viridis")

    fraction_to_color = {
        vf: cmap(index / max(1, len(fractions) - 1))
        for index, vf in enumerate(fractions)
    }

    for vf in fractions:
        for split, marker in [
            ("main", "o"),
            ("interpolation_test", "s"),
        ]:
            mask = (
                (df["vf_target_percent"] == vf)
                & (df["split"] == split)
            )

            if mask.sum() == 0:
                continue

            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                color=fraction_to_color[vf],
                marker=marker,
                edgecolor="black",
                linewidth=0.35,
                s=42,
                alpha=0.85,
                label=f"{vf:g}% ({split})",
            )

    ax.set_xlabel(
        f"PC1 ({100.0 * pca.explained_variance_ratio_[0]:.1f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 ({100.0 * pca.explained_variance_ratio_[1]:.1f}% variance)"
    )

    ax.set_title("PCA of physical packing descriptors")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()

    unique_handles = []
    unique_labels = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=7,
        ncol=2,
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "descriptor_space_pca.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(
        OUTPUT_DIR / "descriptor_pca_explained_variance.csv",
        index=False,
    )



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
    df = pd.read_csv(DATASET_FILE)
    validate_v2_table(df)

    if len(df) != 1240:
        raise ValueError(
            f"Expected 1240 samples in dataset, found {len(df)}."
        )

    if set(df["split"].unique()) != {"main", "interpolation_test", "interpolation_test_fresh"}:
        raise ValueError(
            "Expected main, interpolation_test, and interpolation_test_fresh splits."
        )

    results = []

    for index, row in df.iterrows():
        result = process_sample(row)
        results.append(result)

        print(
            f"[{index + 1:3d}/{len(df)}] "
            f"{row['sample_id']} processed successfully"
        )

    audit = pd.DataFrame(results)

    if len(audit) != len(df):
        raise RuntimeError(
            "Descriptor table row count does not match dataset row count."
        )

    if audit["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id detected in descriptor table.")

    if audit[DESCRIPTOR_COLUMNS].isna().any().any():
        nan_columns = audit[DESCRIPTOR_COLUMNS].columns[
            audit[DESCRIPTOR_COLUMNS].isna().any()
        ].tolist()

        raise ValueError(
            f"NaN values found in descriptor columns: {nan_columns}"
        )

    max_vf_difference = float(
        np.max(np.abs(audit["vf_geometry_minus_recorded"]))
    )

    max_nn_gap_difference = float(
        audit["max_nn_gap_difference"].max()
    )

    audit.to_csv(
        OUTPUT_DIR / "microstructure_descriptors.csv",
        index=False,
    )

    summary_columns = [
        "vf_achieved",
        "nn_distance_mean_over_D",
        "nn_distance_std_over_D",
        "nn_distance_min_over_D",
        "ligament_mean_over_D",
        "ligament_std_over_D",
        "ligament_min_over_D",
        "local_vf_cv",
        "local_vf_range",
        "centre_anisotropy",
    ]

    summary = (
        audit.groupby(["split", "vf_target_percent"])[summary_columns]
        .agg(["mean", "std", "min", "max"])
    )

    summary.to_csv(
        OUTPUT_DIR / "descriptor_summary_by_split_and_vf.csv"
    )

    plot_descriptor_distributions(audit.loc[audit["split"].eq("main")].copy())
    plot_descriptor_vs_vf(audit.loc[audit["split"].eq("main")].copy())
    plot_correlation_heatmap(audit.loc[audit["split"].eq("main")].copy())
    plot_descriptor_pca(audit.loc[audit["split"].eq("main")].copy())

    metadata = {
        "number_of_fresh_interpolation_samples": int(audit["split"].eq("interpolation_test_fresh").sum()),
        "exploratory_plots_fit_and_display": "development records only; no fresh-test target analysis",
        "dataset_file": str(DATASET_FILE),
        "number_of_samples": int(len(audit)),
        "number_of_main_samples": int((audit["split"] == "main").sum()),
        "number_of_existing_interpolation_samples": int(
            (audit["split"] == "interpolation_test").sum()
        ),
        "max_absolute_vf_geometry_minus_recorded": max_vf_difference,
        "max_nn_surface_gap_difference": max_nn_gap_difference,
        "fixed_L": sorted(audit["L"].unique().tolist()),
        "fixed_radius": sorted(audit["radius"].unique().tolist()),
        "fixed_L_over_D": sorted(audit["L_over_D"].unique().tolist()),
    }

    with open(
        OUTPUT_DIR / "descriptor_audit_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nDescriptor audit completed successfully.")
    print("Fresh interpolation samples: " + str(int(audit["split"].eq("interpolation_test_fresh").sum())))
    print(f"Samples processed: {len(audit)}")
    print(f"Main samples: {(audit['split'] == 'main').sum()}")
    print(
        "Existing interpolation samples: "
        f"{(audit['split'] == 'interpolation_test').sum()}"
    )
    print(
        "Maximum |Vf_geometry - Vf_recorded|: "
        f"{max_vf_difference:.6e}"
    )
    print(
        "Maximum stored/recomputed nearest-neighbour-gap difference: "
        f"{max_nn_gap_difference:.6e}"
    )
    print(f"Descriptor table: {OUTPUT_DIR / 'microstructure_descriptors.csv'}")
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()