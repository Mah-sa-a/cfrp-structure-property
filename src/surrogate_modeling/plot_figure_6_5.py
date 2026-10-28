# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:50:13 2026

@author: 43020593
"""

#!/usr/bin/env python3
"""Replot learning curves for the three retained surrogate configurations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "A_quadratic_Vf": {
        "label": "Model A: quadratic fibre fraction",
        "color": "#1f77b4",
        "marker": "o",
    },
    "B1_mean_std_spacing": {
        "label": "Model B1: mean + SD",
        "color": "#2ca02c",
        "marker": "s",
    },
    "C_PCA100": {
        "label": "Model C: 100 PCA components",
        "color": "#d62728",
        "marker": "^",
    },
}

TARGETS = [
    ("C11_Pa", r"$C_{11}$"),
    ("C12_Pa", r"$C_{12}$"),
    ("C22_Pa", r"$C_{22}$"),
    ("C33_Pa", r"$C_{66}$"),
]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "surrogate_supplement_v2.json",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Optional explicit path to learning_curve_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: a figure_6_5 subfolder beside the input CSV.",
    )
    args = parser.parse_args()

    if args.input_csv:
        csv_path = resolve_path(args.input_csv)
    else:
        config_path = resolve_path(args.config)
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        csv_path = (
            resolve_path(config["output_dir"])
            / "learning"
            / "learning_curve_summary.csv"
        )

    if not csv_path.is_file():
        raise FileNotFoundError(f"Learning-curve summary not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "model",
        "target",
        "role",
        "n_per_fraction",
        "n_repetitions",
        "mae_GPa_mean",
        "mae_GPa_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    target_names = [target for target, _ in TARGETS]
    selected = df[
        df["model"].isin(MODELS)
        & df["target"].isin(target_names)
        & df["role"].isin(["train", "validation"])
    ].copy()

    keys = ["model", "target", "role", "n_per_fraction"]
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate model/target/role/training-size rows.")

    numeric = ["n_per_fraction", "mae_GPa_mean", "mae_GPa_std"]
    if not np.isfinite(selected[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Selected learning-curve rows contain non-finite values.")
    if (selected["mae_GPa_std"] < 0).any():
        raise ValueError("Standard deviations must be non-negative.")

    for model in MODELS:
        for target in target_names:
            for role in ["train", "validation"]:
                rows = selected[
                    (selected["model"] == model)
                    & (selected["target"] == target)
                    & (selected["role"] == role)
                ]
                if rows.empty:
                    raise ValueError(
                        f"Missing learning-curve results: {model}, {target}, {role}"
                    )

    repetitions = selected["n_repetitions"].unique()
    if len(repetitions) != 1:
        raise ValueError("Selected results use different repetition counts.")
    n_repetitions = int(repetitions[0])

    out = (
        resolve_path(args.output_dir)
        if args.output_dir
        else csv_path.parent / "figure_6_5"
    )
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.5, 8.5),
        sharex=True,
    )

    sizes = sorted(selected["n_per_fraction"].unique())

    for ax, (target, title) in zip(axes.ravel(), TARGETS):
        for model, style in MODELS.items():
            rows = selected[
                (selected["model"] == model)
                & (selected["target"] == target)
            ]

            valid = rows[
                rows["role"] == "validation"
            ].sort_values("n_per_fraction")

            train = rows[
                rows["role"] == "train"
            ].sort_values("n_per_fraction")

            ax.errorbar(
                valid["n_per_fraction"],
                valid["mae_GPa_mean"],
                yerr=valid["mae_GPa_std"],
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.8,
                elinewidth=1.0,
                capsize=3,
                linestyle="-",
                zorder=3,
            )

            ax.plot(
                train["n_per_fraction"],
                train["mae_GPa_mean"],
                color=style["color"],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                zorder=2,
            )

        ax.set_title(title)
        ax.set_xlabel("Training RVEs per fibre-fraction level")
        ax.set_ylabel("MAE (GPa)")
        ax.set_xticks(sizes)
        ax.tick_params(axis="x", labelbottom=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, alpha=0.22)
        ax.margins(x=0.035)

    model_handles = [
        Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=style["label"],
        )
        for style in MODELS.values()
    ]

    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
    )

    fig.text(
        0.5, 0.025,
        (
            "Solid: validation mean ± repetition SD; "
            "dashed: training mean. "
            "Total training RVEs = 5 × horizontal-axis value."
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    stem = "figure_6_5_retained_models_learning_curves"
    fig.savefig(out / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    selected.sort_values(keys).to_csv(
        out / "figure_6_5_source_data.csv",
        index=False,
    )

    caption = (
        "Figure 6.5. Learning curves for the retained surrogate configurations: "
        "Model A, Model B1 with the mean-plus-SD representation, and Model C "
        "with 100 PCA components. Panels show the four predicted stiffness "
        "components. Solid curves represent mean validation MAE, error bars "
        f"indicate the sample standard deviation across {n_repetitions} "
        "repetitions of five-fold cross-validation, and dashed curves "
        "represent mean training MAE. Training subsets contained equal "
        "numbers of RVEs from each of the five target fibre-fraction levels; "
        "the total training-set size is therefore five times the "
        "horizontal-axis value. Model C is shown only at training sizes "
        "supporting its prescribed PCA dimension."
    )
    (out / "figure_6_5_caption.txt").write_text(
        caption + "\n",
        encoding="utf-8",
    )

    print(f"Input: {csv_path}")
    print(f"Output folder: {out}")
    print(f"Repetitions: {n_repetitions}")
    print("Replotting complete. No models fitted or predictions changed.")


if __name__ == "__main__":
    main()