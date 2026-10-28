# Surrogate model training and evaluation

## Main implementation

The final workflow uses:

- `src/surrogate_modeling/train_surrogate_extended_v2.py`
- `configs/surrogate_training_extended_v2.json`
- `scripts/slurm/run_surrogate_training_extended_v2.slurm`

The Python driver implements separate training and evaluation stages.

The older `evaluate_final_models_on_reserved_test.py` targets the
initial 200-development/40-interpolation study. It is not the evaluator
for this final workflow.

## Required inputs and software

Training requires the completed final dataset, preprocessing outputs,
and `preprocessing_validation_v2.json`.

Input checksums must match the preprocessing validation report.

Required scientific libraries include NumPy, Pandas, scikit-learn,
Joblib, and Matplotlib. No FE simulations are performed during training.

The configured output directory is:

`results/surrogate_models/extended_study_v2/`

## Targets and units

The predicted stiffness entries are:

- `C11_Pa`
- `C12_Pa`
- `C22_Pa`
- `C33_Pa`

The stored name `C33_Pa` denotes the third diagonal entry of the reduced
two-dimensional stiffness matrix, labeled C66 in the manuscript.

The driver converts target values from Pa to GPa for regression,
predictions, and error metrics.

## Model families

### Model A

Quadratic regression using achieved fibre area fraction.

### Model B1

Ridge regression using achieved fibre fraction, its square, and scalar
packing descriptors.

Three representations are compared:

- `B1_std_spacing`
- `B1_mean_spacing`
- `B1_mean_std_spacing`

### Model C

Ridge regression using fibre-fraction terms and PCA scores from normalized
periodic two-point statistics.

The configured PCA dimensions are 20, 35, 50, 75, and 100.

Fixed-dimension variants tune Ridge regularization. `C_PCA_tuned`
selects both PCA dimension and Ridge regularization using inner
cross-validation, separately for each stiffness target.

## Development-only validation

The training stage uses exactly 1,000 development RVEs.

The configured protocol uses:

- Five outer folds, stratified by nominal fibre fraction
- Ten outer repetitions
- Four shuffled inner folds within each outer-training subset
- Inner mean absolute error for hyperparameter selection

Each outer fold contains 800 training and 200 validation RVEs.

Scaling and spatial PCA are fitted on the relevant training subsets,
not on the full dataset before cross-validation.

After validation, final hyperparameters are selected using development
data, and the models are refitted on all 1,000 development records.

Neither fresh nor legacy interpolation cases are scored by the
training stage.

## Training execution

The supplied HPC launcher runs:

```bash
python -u src/surrogate_modeling/train_surrogate_extended_v2.py \
  --config configs/surrogate_training_extended_v2.json \
  --stage train
```

It can be submitted from the repository root:

```bash
sbatch scripts/slurm/run_surrogate_training_extended_v2.slurm
```

For an input check without model fitting:

```bash
python src/surrogate_modeling/train_surrogate_extended_v2.py \
  --config configs/surrogate_training_extended_v2.json \
  --stage train --check-only
```

Do not rerun training over accepted historical results. The driver
guards against replacing an existing training study.

## Recorded final interpolation evaluation

The saved evaluation metadata identifies these evaluated models:

- `A_quadratic_Vf`
- `B1_mean_std_spacing`
- `C_PCA_tuned`

For the final development-data refit, `C_PCA_tuned` selected 100 PCA
components for every target.

Its Ridge alpha values were 0.1 for C11, C12, and C66, and 0.01 for C22.

This final selection does not mean that every outer-CV fold necessarily
selected 100 components.

## Separate evaluation stage

The evaluation stage loads the saved fitted models and performs no
fitting or tuning.

The command corresponding to the recorded model selection is:

```bash
python src/surrogate_modeling/train_surrogate_extended_v2.py \
  --config configs/surrogate_training_extended_v2.json \
  --stage evaluate \
  --confirm-frozen \
  --evaluate-models A_quadratic_Vf B1_mean_std_spacing C_PCA_tuned
```

This documents the procedure; it is not an instruction to repeat an
already completed evaluation.

The driver checks training completion, configuration, model hashes,
input hashes, and separation of training and evaluation sample IDs.

It writes separate predictions and metrics for:

- 200 fresh interpolation cases
- 40 legacy interpolation cases, untouched until the final evaluation

These cohorts retain their distinct histories. Aggregating their
predictions into a 240-case report is a separate downstream step.

The evaluation-start marker prevents an accidental automatic repeat.
Do not delete it simply to rerun or retune against the evaluation cases.

## Principal output artifacts

Training produces:

- Nested-CV predictions, metrics, split manifests, and tuning records
- `final_selected_parameters.json`
- `final_models.joblib`
- `training_metadata.json`
- `training_complete.json`

Evaluation produces:

- `evaluation_started.json`
- `evaluation_complete.json`
- Separate fresh and legacy interpolation prediction and metric tables

Preserve configurations, metadata, and accepted results together.
A successful syntax check alone does not validate the saved models,
predictions, or numerical results.

## Supplementary model comparisons and learning curves

### Implementation and prerequisites

The supplementary workflow uses:

- `src/surrogate_modeling/run_surrogate_supplement_v2.py`
- `configs/surrogate_supplement_v2.json`
- `scripts/slurm/run_nonlinear_cv_v2.slurm`
- `scripts/slurm/run_learning_curves_full_v2.slurm`

It imports the shared implementation in
`train_surrogate_extended_v2.py` and requires the completed primary
study under:

`results/surrogate_models/extended_study_v2/`

It checks primary-study completion, input and source hashes, and
reference split manifests. Preserve the source and inputs associated
with accepted results; do not modify them simply to bypass a
provenance mismatch.

Additional dependencies include XGBoost and threadpoolctl.

### Additional scalar regression methods

The supplementary comparison evaluates:

- B2: Elastic Net
- B3: random forest
- B4: radial-basis-function support-vector regression
- B5: XGBoost

All four use the five-feature `B1_std_spacing` representation.
Their matched-input Ridge reference is therefore B1 spacing-SD,
not the six-feature mean-plus-SD configuration retained for the
primary interpolation comparison.

The configured assessment uses ten repetitions of five outer folds
and four-fold inner tuning, producing 50 worker tasks. Outer
partitions are checked against the primary study.

### Balanced learning curves

Learning curves use three repetitions of five outer folds and nine
training sizes: 5, 10, 20, 32, 48, 64, 96, 128, and 160 RVEs per
fibre-fraction level.

These correspond to 25–800 training RVEs and 135 worker tasks.
Validation samples remain unchanged across training sizes within
each outer split, and smaller training subsets are nested within
larger ones.

The workflow covers the quadratic baseline, scalar Ridge
representations, fixed spatial PCA dimensions of 20, 50, and 100,
and the additional B2–B5 regressors.

Infeasible PCA dimensions are explicitly skipped rather than reduced.
Validation metrics are pooled across out-of-fold predictions within
each repetition. Reported error bars are sample standard deviations
across repetitions, not confidence intervals.

### Execution sequence

The following commands document the procedure for a new study.
They are not instructions to rerun accepted historical results.

From the repository root, preparation creates task manifests and
provenance metadata without fitting estimators:

```bash
python src/surrogate_modeling/run_surrogate_supplement_v2.py --prepare
```

Submit the worker arrays on the configured HPC system:

```bash
sbatch scripts/slurm/run_nonlinear_cv_v2.slurm
sbatch scripts/slurm/run_learning_curves_full_v2.slurm
```

After all tasks for a mode have completed, generate its summary:

```bash
python src/surrogate_modeling/run_surrogate_supplement_v2.py --mode cv --summary
python src/surrogate_modeling/run_surrogate_supplement_v2.py --mode learning --summary
```

The supplied launchers use the CPU_Compute partition,
fenicsx-env-312 environment, and at most two simultaneous tasks
per array. Adapt scheduler settings for another system.

### Outputs and evaluation boundaries

Outputs are stored under:

`results/surrogate_models/supplementary_study_v2/`

They include task-level predictions, metrics, tuning records,
split manifests, completion markers, aggregated summaries, and
comparison figures.

All supplementary fitting and prediction use development records.
The workflow does not score fresh or legacy interpolation cases.

The optional `--fit-final` operation fits additional B2–B5 models
on development data; it is not required for generating the CV
or learning-curve summaries and does not perform interpolation
evaluation.

## Manuscript figures from saved development summaries

The following scripts generate figures from existing summary tables.
They do not fit models or generate new predictions.

### Figure 6.5: retained-model learning curves

Script:

`src/surrogate_modeling/plot_figure_6_5.py`

Default input:

`results/surrogate_models/supplementary_study_v2/learning/learning_curve_summary.csv`

Selected models are A_quadratic_Vf, B1_mean_std_spacing, and C_PCA100.

The figure shows validation MAE with repetition-level sample standard
deviations and dashed training-MAE curves. The horizontal axis gives
training RVEs per fibre-fraction level; total training size is five
times that value.

### Figure 6.7: repeated nested-CV comparison

Script:

`src/surrogate_modeling/plot_figure_6_7_nested_cv.py`

Default input:

`results/surrogate_models/extended_study_v2/nested_cv_metric_summary.csv`

Selected models are A_quadratic_Vf, B1_mean_std_spacing, and C_PCA100.

Bars show mean repetition-level MAE; error bars use the recorded
sample standard deviations across ten repetitions directly.
They are not standard errors or confidence intervals.

### Model identity and outputs

Both figures use the fixed C_PCA100 candidate. This should be
distinguished from C_PCA_tuned used in the recorded final
interpolation evaluation.

Each script writes PNG and PDF figures, a selected source-data CSV,
and caption text. Default output subfolders are figure_6_5 and
figure_6_7 beside their respective input tables.

Both scripts support --input-csv and --output-dir overrides.
Their default configuration files are surrogate_supplement_v2.json
and surrogate_training_extended_v2.json, respectively.

Regenerating figures requires the saved summary tables. Including
these scripts does not imply that figures or numerical analyses
were rerun during repository preparation.