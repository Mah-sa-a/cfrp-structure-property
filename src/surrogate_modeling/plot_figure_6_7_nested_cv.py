#!/usr/bin/env python3
"""Plot Figure 6.7 from repeated nested-cross-validation summary results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("A_quadratic_Vf",      "Model A",  "#1f77b4"),  # Blue
    ("B1_mean_std_spacing", "Model B1", "#ff7f0e"),  # Orange
    ("C_PCA100",            "Model C",  "#2ca02c"),  # Green
]
TARGETS = ["C11_Pa", "C12_Pa", "C22_Pa", "C33_Pa"]
TITLES = [r"$C_{11}$", r"$C_{12}$", r"$C_{22}$", r"$C_{66}$"]

def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "surrogate_training_extended_v2.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.input_csv:
        csv_path = resolve(args.input_csv)
    else:
        with resolve(args.config).open(encoding="utf-8") as f:
            config = json.load(f)
        csv_path = resolve(config["output_dir"]) / "nested_cv_metric_summary.csv"
    df = pd.read_csv(csv_path)
    required = {"model", "target", "n_repetitions", "mae_GPa_mean", "mae_GPa_std"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    selected = df[df.model.isin([m[0] for m in MODELS]) & df.target.isin(TARGETS)].copy()
    if selected.duplicated(["model", "target"]).any():
        raise ValueError("Duplicate model/target summary rows.")
    expected = {(m[0], t) for m in MODELS for t in TARGETS}
    if set(zip(selected.model, selected.target)) != expected:
        raise ValueError("Summary must contain all four targets for A, B1 mean+SD, and C_PCA100.")
    if not selected.n_repetitions.eq(10).all():
        raise ValueError("Figure 6.7 requires summaries across ten repetitions.")
    values = selected[["mae_GPa_mean", "mae_GPa_std"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("MAE means and standard deviations must be finite and non-negative.")
    out = resolve(args.output_dir) if args.output_dir else csv_path.parent / "figure_6_7"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TARGETS))
    width = 0.24
    for i, (name, label, color) in enumerate(MODELS):
        rows = selected[selected.model.eq(name)].set_index("target").reindex(TARGETS)
        ax.bar(x + (i - 1) * width, rows.mae_GPa_mean, width,
               yerr=rows.mae_GPa_std, capsize=4, color=color, label=label,
               error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(TITLES)
    ax.set_xlabel("Homogenized stiffness component")
    ax.set_ylabel("Repeated nested-CV MAE (GPa)")
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    fig.text(0.5, 0.015, "Error bars: ±1 sample SD across ten CV repetitions.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    stem = "figure_6_7_nested_cv_mae_comparison"
    fig.savefig(out / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(out / "figure_6_7_source_data.csv", index=False)
    caption = (
        "Figure 6.7. Repeated nested-cross-validation MAE for Model A, Model B1 "
        "with the mean-plus-SD representation, and Model C with 100 PCA components. "
        "Bars represent mean MAE, and error bars indicate the sample standard deviation "
        "of repetition-level MAE across ten repetitions of five-fold outer cross-validation. "
        "Each repetition-level MAE was calculated from 1,000 pooled out-of-fold predictions."
    )
    (out / "figure_6_7_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Input: {csv_path}")
    print(f"Output: {out}")
    print("Used mae_GPa_std directly: no division by sqrt(N), no interpolation data, no retraining.")

if __name__ == "__main__":
    main()
