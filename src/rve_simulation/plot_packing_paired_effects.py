# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:24:53 2026

@author: 43020593
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = ROOT / "results" / "packing_study" / "packing_LD18_v2_40_50"
INPUT_CSV = STUDY_DIR / "paired_differences.csv"
OUTPUT_DIR = STUDY_DIR / "figures"

PROPERTIES = [
    ("Exx_Pa_percent_change", r"$E_{xx}$"),
    ("Eyy_Pa_percent_change", r"$E_{yy}$"),
    ("Gxy_Pa_percent_change", r"$G_{xy}$"),
]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"N", "replicate", "vf_achieved"}
    required.update(column for column, _ in PROPERTIES)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    expected_counts = {165: 12, 206: 12}
    output_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)

    for ax, (column, label) in zip(axes, PROPERTIES):
        for x_position, N in enumerate((165, 206)):
            group = df.loc[df["N"] == N].sort_values("replicate")

            if len(group) != expected_counts[N]:
                raise ValueError(
                    f"N={N}: expected {expected_counts[N]} pairs, "
                    f"found {len(group)}"
                )

            if group["replicate"].duplicated().any():
                raise ValueError(f"N={N}: duplicate replicate labels")

            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"N={N}: non-finite values in {column}")

            n = len(values)
            mean = values.mean()
            sample_sd = values.std(ddof=1)
            ci_half_width = t.ppf(0.975, df=n - 1) * sample_sd / np.sqrt(n)
            achieved_vf = group["vf_achieved"].iloc[0]

            # Deterministic horizontal offsets make all individual pairs visible.
            offsets = (
                np.arange(n) - (n - 1) / 2
            ) * 0.018

            ax.scatter(
                x_position + offsets,
                values,
                s=22,
                alpha=0.75,
                color="#597692",
                zorder=2,
            )

            ax.errorbar(
                x_position,
                mean,
                yerr=ci_half_width,
                fmt="o",
                markersize=8,
                capsize=5,
                color="#b44327",
                zorder=3,
            )

            output_rows.append({
                "N": N,
                "achieved_vf_percent": 100 * achieved_vf,
                "property": column,
                "n_pairs": n,
                "mean_paired_change_percent": mean,
                "sample_sd_percentage_points": sample_sd,
                "ci95_lower_percent": mean - ci_half_width,
                "ci95_upper_percent": mean + ci_half_width,
                "minimum_pair_percent": values.min(),
                "maximum_pair_percent": values.max(),
            })

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1], ["40%", "50%"])
        ax.set_xlabel("Achieved fiber area fraction")

    axes[0].set_ylabel(
        "Clustered − hard-disc (% of paired hard-disc value)"
    )

    fig.suptitle(
        "Effect of fiber packing on homogenized properties "
        r"($L/D=18$)"
    )
    fig.text(
        0.5,
        0.005,
        "Small points: matched pairs; red points: paired mean ± 95% CI",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_DIR / "packing_paired_effects.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "packing_paired_effects.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(output_rows).to_csv(
        OUTPUT_DIR / "packing_paired_effects_table.csv",
        index=False,
    )

    print(f"Figure and table saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()