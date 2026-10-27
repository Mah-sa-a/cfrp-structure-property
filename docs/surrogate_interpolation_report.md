# Pooled interpolation report

## Purpose

This workflow combines existing predictions from:

- 200 fresh interpolation RVEs
- 40 legacy, previously evaluated interpolation RVEs

The pooled report contains 240 RVEs, with 60 cases at each target
fibre area fraction: 42.5%, 47.5%, 52.5%, and 57.5%.

Aggregation performs no model fitting, hyperparameter tuning, or new
prediction generation. It does not make the legacy cases newly untouched.

## Implementation

- `src/surrogate_modeling/combine_interpolation_240.py`
- `scripts/slurm/run_combine_interpolation_240.slurm`

Required libraries include NumPy, Pandas, scikit-learn, and Matplotlib.

## Required inputs

The default input directory is:

`results/surrogate_models/extended_study_v2/`

It must contain:

- `fresh_interpolation_test_predictions.csv`
- `legacy_interpolation_test_predictions.csv`
- `evaluation_started.json`
- `evaluation_complete.json`
- `training_complete.json`
- `final_models.joblib`

The script checks training and evaluation completion metadata and
compares the current model-file hash with the recorded hashes.

It also checks prediction identities, cohort sizes, target/fraction
coverage, model-name consistency, and agreement of FE labels between
model outputs.

## Recorded model selection

The inspected evaluation metadata and pooled metrics identify:

- `A_quadratic_Vf`
- `B1_mean_std_spacing`
- `C_PCA_tuned`

The final development-data refit selected 100 PCA components for every
target of `C_PCA_tuned`. Preserve its recorded model ID; do not relabel
these predictions as `C_PCA100`.

## Input check

From the repository root:

```bash
python src/surrogate_modeling/combine_interpolation_240.py --check-only
```

This validates the inputs and constructs metric tables in memory
without writing report files, fitting models, or generating predictions.

## Report generation

For a new report, run from the repository root:

```bash
python src/surrogate_modeling/combine_interpolation_240.py
```

Alternatively, submit the HPCassiopee launcher:

```bash
sbatch scripts/slurm/run_combine_interpolation_240.slurm
```

The default output directory is:

`results/surrogate_models/extended_study_v2/combined_interpolation_240/`

The script refuses to overwrite a nonempty report directory. Preserve
accepted historical reports. To intentionally create another report,
supply a different `--output-dir`; do not delete accepted results merely
to bypass the guard.

## Outputs

Principal outputs include:

- `combined_interpolation_predictions.csv`
- `combined_interpolation_metrics.csv`
- `combined_interpolation_metrics_by_fraction.csv`
- `cohort_metrics_reference.csv`
- `combined_report_metadata.json`
- `figure_caption.txt`
- MAE and MAPE comparison figures
- Predicted-versus-FE figures
- MAE-by-fraction figures

Metrics are recomputed from the pooled predictions. RMSE and R2 are
not obtained by averaging the two cohorts' metric values.

The pooled prediction CSV retains `evaluation_cohort`, and separate
cohort metrics remain available for traceability.

## Plotting-version limitation

`plot_interpolation_240_retained.py` requires `C_PCA100` and does not
accept `C_PCA_tuned` as a substitute. It is not part of the documented
workflow for the inspected pooled report.

Do not rename model IDs in accepted prediction or metric files to make
an incompatible plotting script run.