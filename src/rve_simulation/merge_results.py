"""
Concatenate all per-task CSVs from an array-job sweep
(<project_root>/results/RVE_size_study/partial_results/task_*.csv)
into the single rve_size_study_results.csv expected by
rve_size_convergence_plots.py.

Uses the same <project_root>/results/ output location as
rve_size_convergence_pipeline.py, resolved relative to this file's own
location so it works no matter what directory you run it from.

Usage:
    python merge_results.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = PROJECT_ROOT / "results" / "RVE_size_study"
PARTIAL_DIR = WORK_DIR / "partial_results"
MERGED_CSV = WORK_DIR / "rve_size_study_results.csv"


def main():
    partial_files = sorted(PARTIAL_DIR.glob("task_*.csv"))
    if not partial_files:
        raise FileNotFoundError(f"No partial result files found in {PARTIAL_DIR}.")

    frames = [pd.read_csv(f) for f in partial_files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["Vf_target", "L_over_D", "realization"]).reset_index(drop=True)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"Merged {len(partial_files)} partial files -> {MERGED_CSV}")
    print(f"Total rows: {len(merged)}")
    print(merged.groupby(["Vf_target", "L_over_D"]).size().rename("n_realizations"))


if __name__ == "__main__":
    main()
