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
- 40 legacy, previously evaluated interpolation cases

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