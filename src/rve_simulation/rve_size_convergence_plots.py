"""
Manuscript figures for the RVE-size convergence study.

Reads the long-format CSV produced by rve_size_convergence_pipeline.py
(one row per Vf x L_over_D x realization) and produces four figures
commonly used in RVE-size-determination chapters (Kanit et al. 2003;
Gitman et al. 2007 style).

OUTPUT LOCATION: figures are written to
<project_root>/results/rve_size_study_figures/, where <project_root>
is the parent of the folder this script lives in (i.e. if this file is
in .../RVE_Simulation_HPC/src/, figures go to
.../RVE_Simulation_HPC/results/rve_size_study_figures/). This matches
where rve_size_convergence_pipeline.py writes its results CSV, and
works no matter what directory you run this script from.

Size axis: L_over_D = domain edge length / fiber diameter, i.e. how
many fiber diameters fit across the RVE. This is the physically
correct "size" variable when the fiber radius is held fixed across the
whole study (same magnification, growing field of view) -- N_fibers is
also available in the CSV/plots as secondary information, but it is a
*result* of (Vf, L_over_D), not the independent variable.

  fig1_mean_property_vs_size.png/.pdf
      Mean effective property vs L_over_D, one curve per Vf, with a
      shaded 95% confidence band across realizations.

  fig2_coefficient_of_variation_vs_size.png/.pdf
      Coefficient of variation (%) vs L_over_D (log-log), one curve per
      Vf, with a 1/L reference slope (in 2D, variance ~ 1/Area ~
      1/L_over_D^2, so CoV ~ 1/L_over_D) to check the classical scaling
      of realization scatter with RVE size.

  fig3_required_realizations_vs_size.png/.pdf
      Kanit's formula n = 4*D_Z(V)^2 / (mean^2 * eps_rel^2): required
      number of realizations to reach 1% and 5% relative error, vs
      L_over_D, one curve per Vf.

  fig4_boxplot_scatter_vs_size.png/.pdf
      Boxplots of the raw realization values at a few representative
      sizes, for one chosen Vf, to visualise scatter narrowing directly.

Only needs: numpy, pandas, matplotlib. No dolfinx/gmsh required, so
this can run on a laptop after copying the results CSV off the HPC.

Usage:
    python rve_size_convergence_plots.py path/to/rve_size_study_results.csv
    (or with no argument, defaults to the standard pipeline output location)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROPERTY_COLUMN = "Exx_Pa"     # change to "Gxy_Pa", "Eyy_Pa", etc. as needed
PROPERTY_LABEL = r"$E_{xx}$"
PROPERTY_UNIT_SCALE = 1e9      # Pa -> GPa for plotting
PROPERTY_UNIT_LABEL = "GPa"

SIZE_COLUMN = "L_over_D"       # the physically correct size axis
SIZE_LABEL = r"Domain size, $L/D$ (fiber diameters)"

TARGET_RELATIVE_ERRORS = [0.01, 0.05]   # 1% and 5%, Kanit-style
BOXPLOT_VF = None                        # None -> auto-pick the middle Vf
BOXPLOT_SIZE_PICKS = [4, 8, 12, 18, 20, 24]   # None -> auto-pick 4 sizes spanning the range

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "RVE_size_study" / "rve_size_study_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rve_size_study_figures"


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Vf_target", SIZE_COLUMN, "realization", PROPERTY_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    return df


def summarize(df, value_col):
    rows = []
    for (Vf, size), g in df.groupby(["Vf_target", SIZE_COLUMN]):
        vals = g[value_col].values / PROPERTY_UNIT_SCALE
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        ci95 = 1.96 * sem
        cov_pct = 100 * std / mean if mean != 0 else np.nan
        mean_N = g["N_fibers"].mean() if "N_fibers" in g else np.nan
        rows.append(dict(Vf=Vf, size=size, n=n, mean=mean, std=std,
                          ci95=ci95, cov_pct=cov_pct, mean_N_fibers=mean_N))
    return pd.DataFrame(rows).sort_values(["Vf", "size"]).reset_index(drop=True)


def n_required(std, mean, eps_rel):
    """Kanit et al. (2003): n = 4 D_Z(V)^2 / (mean^2 * eps_rel^2)."""
    return 4.0 * std ** 2 / (mean ** 2 * eps_rel ** 2)


def make_color_map(vf_values):
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(vf_values)))
    return {vf: palette[i] for i, vf in enumerate(sorted(vf_values))}


def fig_mean_vs_size(summary, colors, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for Vf, g in summary.groupby("Vf"):
        ax.plot(g["size"], g["mean"], "-o", color=colors[Vf],
                 label=f"Vf = {int(round(Vf * 100))}%")
        ax.fill_between(g["size"], g["mean"] - g["ci95"], g["mean"] + g["ci95"],
                         color=colors[Vf], alpha=0.2)
    ax.set_xlabel(SIZE_LABEL)
    ax.set_ylabel(f"Homogenized {PROPERTY_LABEL} ({PROPERTY_UNIT_LABEL})")
    ax.set_title("Convergence of effective stiffness with RVE size\n"
                  "(shaded band = 95% CI across realizations)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_mean_property_vs_size.{ext}", dpi=300)
    plt.close(fig)


def fig_cov_vs_size(summary, colors, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for Vf, g in summary.groupby("Vf"):
        ax.loglog(g["size"], g["cov_pct"], "-o", color=colors[Vf],
                   label=f"Vf = {int(round(Vf * 100))}%")

    size_all = np.sort(summary["size"].unique())
    size_min = size_all.min()
    ref_level = summary.loc[summary["size"] == size_min, "cov_pct"].mean()
    # In 2D, variance ~ 1/Area ~ 1/(L_over_D)^2, so CoV ~ 1/L_over_D.
    ax.loglog(size_all, ref_level * (size_min / size_all), "k--", lw=1,
               label=r"$1/(L/D)$ reference")

    ax.set_xlabel(SIZE_LABEL + " (log scale)")
    ax.set_ylabel("Coefficient of variation (%) (log scale)")
    ax.set_title("Realization-to-realization scatter vs RVE size")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_coefficient_of_variation_vs_size.{ext}", dpi=300)
    plt.close(fig)


def fig_required_realizations(summary, colors, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    linestyles = ["-", "--", ":", "-."]
    for Vf, g in summary.groupby("Vf"):
        for eps, ls in zip(TARGET_RELATIVE_ERRORS, linestyles):
            n_req = n_required(g["std"].values, g["mean"].values, eps)
            ax.semilogy(g["size"], n_req, ls, marker="o", color=colors[Vf],
                         label=f"Vf={int(round(Vf*100))}%, target {int(eps*100)}%")
    ax.axhline(1.0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xlabel(SIZE_LABEL)
    ax.set_ylabel("Required number of realizations, n (log scale)")
    ax.set_title("Kanit's required-realizations criterion vs RVE size")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_required_realizations_vs_size.{ext}", dpi=300)
    plt.close(fig)


def fig_boxplot_scatter(df, out_dir):
    Vf_pick = BOXPLOT_VF if BOXPLOT_VF is not None else sorted(df["Vf_target"].unique())[
        len(df["Vf_target"].unique()) // 2]
    sub = df[np.isclose(df["Vf_target"], Vf_pick)]
    size_all = np.sort(sub[SIZE_COLUMN].unique())

    if BOXPLOT_SIZE_PICKS is not None:
        size_picks = BOXPLOT_SIZE_PICKS
    else:
        idx = np.linspace(0, len(size_all) - 1, min(4, len(size_all))).astype(int)
        size_picks = size_all[idx]

    data = [sub.loc[sub[SIZE_COLUMN] == s, PROPERTY_COLUMN].values / PROPERTY_UNIT_SCALE
            for s in size_picks]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.boxplot(data, tick_labels=[str(s) for s in size_picks])
    ax.set_xlabel(SIZE_LABEL)
    ax.set_ylabel(f"Homogenized {PROPERTY_LABEL} ({PROPERTY_UNIT_LABEL})")
    ax.set_title(f"Realization scatter narrows as RVE grows (Vf = {int(round(Vf_pick*100))}%)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_boxplot_scatter_vs_size.{ext}", dpi=300)
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_CSV
    if not csv_path.exists():
        print(f"Results CSV not found: {csv_path}")
        print("Usage: python rve_size_convergence_plots.py <results.csv>")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    summary = summarize(df, PROPERTY_COLUMN)
    summary.to_csv(OUTPUT_DIR / "rve_size_study_summary.csv", index=False)

    colors = make_color_map(summary["Vf"].unique())

    fig_mean_vs_size(summary, colors, OUTPUT_DIR)
    fig_cov_vs_size(summary, colors, OUTPUT_DIR)
    fig_required_realizations(summary, colors, OUTPUT_DIR)
    fig_boxplot_scatter(df, OUTPUT_DIR)

    print(f"Figures and summary CSV written to: {OUTPUT_DIR.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
