# CFRP Structure–Property Modelling

This repository contains computational workflows for studying the relationship between carbon-fibre-reinforced polymer (CFRP) microstructure and effective elastic properties.

The workflows connect synthetic periodic fibre-matrix microstructures
to apparent homogenized in-plane elastic stiffness through
finite-element simulation, and use the resulting microstructure-property
dataset to develop and evaluate surrogate models. The surrogate study
examines whether scalar fibre-packing descriptors and normalized
two-point spatial statistics improve prediction beyond an
achieved-fibre-area-fraction baseline.

The numerical scope is restricted to two-dimensional, small-strain,
plane-strain linear elasticity with perfectly bonded fibre and matrix
phases. Explicit porosity, experimentally reconstructed geometries,
interfacial debonding, progressive damage, and failure are not included
in the current workflows and remain directions for future research.

The project combines:

- Generation of periodic two-dimensional fibre-matrix representative volume elements (RVEs)
- Gmsh mesh generation with periodic boundaries
- FEniCSx-based computational homogenization
- Mesh-convergence verification
- RVE-size-convergence analysis across multiple random realizations
- Fibre-packing sensitivity studies using paired hard-disk and clustered arrangements
- Simulation-dataset generation and microstructure-descriptor preprocessing
- Surrogate modeling of effective elastic stiffness components using fibre-fraction, scalar packing, and two-point-statistics descriptors
- Model assessment using repeated nested cross-validation
- Evaluation of frozen surrogate models on fresh and legacy interpolation cohorts
- Pooled interpolation reporting with cohort traceability

## Current workflow

The computational workflow connects synthetic periodic microstructures
to homogenized elastic stiffness and descriptor-based surrogate
prediction through the following stages:

1. Periodic RVE generation and meshing:
   Generate synthetic two-phase fibre-matrix geometries and construct
   conforming finite-element meshes using Gmsh.

2. Periodic computational homogenization:
   Solve the small-strain, plane-strain elasticity problem using
   FEniCSx and periodic boundary conditions. Compute the apparent
   homogenized in-plane stiffness matrix and derived engineering
   constants.

3. Numerical verification and controlled microstructure studies:
   Assess mesh convergence and statistical RVE-size convergence.
   Separately examine packing sensitivity using matched baseline
   hard-disk and modified clustered configurations.

4. Surrogate-dataset construction and descriptor extraction:
   Assemble paired microstructure-stiffness records and extract
   achieved fibre area fraction, scalar packing descriptors, and
   normalized periodic two-point statistics. Preserve the separation
   between development and reserved interpolation-test records.

5. Surrogate-model development and assessment:
   Compare a quadratic fibre-fraction baseline with scalar
   packing-descriptor Ridge models and spatial models combining
   principal component analysis (PCA) with Ridge regression.
   Assess predictive performance using repeated nested
   cross-validation on the development dataset, with learned
   scaling and PCA transformations fitted within the relevant
   training partitions.

6. Independent interpolation evaluation and reporting:
   Apply the final fitted models to reserved RVEs at intermediate
   fibre-area-fraction levels without refitting or tuning. Combine
   saved predictions for pooled reporting while preserving
   fresh and legacy cohort labels for traceability.

The paired packing-sensitivity study is a separate controlled
investigation; its simulations are not included in the
1,240-record surrogate dataset.

### Periodic RVE generation

The fibre-packing algorithm combines:

1. Random sequential adsorption (RSA) at a reduced initial fibre radius
2. Adaptive fibre-radius growth
3. Periodic collision relaxation inspired by the
   Lubachevsky–Stillinger approach
4. Final damping to obtain a settled fibre configuration

The procedure enforces periodic non-overlap and the prescribed minimum
fibre surface gap. Growth and relaxation permit collective rearrangement
of fibre centres, supporting the generation of denser configurations
where direct sequential placement becomes inefficient.

## Repository structure

The following overview shows the configuration, documentation,
environment records, and primary and supplementary workflow scripts.

```text
cfrp-structure-property/
├── LICENSE
├── README.md
├── environment-hpc-2026-10-28.yml
├── configs/
│   ├── packing_study_v2.json
│   ├── surrogate_dataset_v1.json
│   ├── surrogate_dataset_extension_v1.json
│   ├── surrogate_dataset_extension_v2.json
│   ├── surrogate_training_extended_v2.json
│   └── surrogate_supplement_v2.json
├── data/
│   └── rve_reference/
│       └── RVE_01_fibers.csv
├── docs/
│   ├── packing_study_v2_README.md
│   ├── surrogate_dataset_generation.md
│   ├── surrogate_preprocessing.md
│   ├── surrogate_training.md
│   └── surrogate_interpolation_report.md
├── hpc/
│   ├── README.md
│   ├── environment-hpcassiopee.yml
│   ├── environment-hpcassiopee-py312.yml
│   └── setup_hpcassiopee.md
├── scripts/
│   └── slurm/
│       ├── recover_sample_0084.slurm
│       ├── run_mesh_convergence.slurm
│       ├── run_packing_mesh_refinement.slurm
│       ├── run_packing_study_v2_array.slurm
│       ├── run_rve_size_convergence_array.slurm
│       ├── run_surrogate_dataset_v1.slurm
│       ├── run_surrogate_dataset_extension_v1.slurm
│       ├── run_surrogate_dataset_extension_v2.slurm
│       ├── run_surrogate_preprocess_v2.slurm
│       ├── run_surrogate_training_extended_v2.slurm
│       ├── run_combine_interpolation_240.slurm
│       ├── run_nonlinear_cv_v2.slurm
│       └── run_learning_curves_full_v2.slurm
└── src/
    ├── rve_simulation/
    │   ├── generate_periodic_fiber_rves_gmsh.py
    │   ├── generate_surrogate_dataset.py
    │   ├── generate_surrogate_dataset_extension.py
    │   ├── generate_surrogate_dataset_extension_v2.py
    │   ├── homogenize_fibre_RVEs_fenicsx.py
    │   ├── merge_results.py
    │   ├── mesh_convergence_RVE01.py
    │   ├── packing_mesh_refinement.py
    │   ├── packing_study.py
    │   ├── packing_study_v2.py
    │   ├── packing_study_v2_analysis.py
    │   ├── plot_packing_paired_effects.py
    │   ├── recover_sample_0084.py
    │   ├── run_single_combo.py
    │   ├── rve_size_convergence_pipeline.py
    │   └── rve_size_convergence_plots.py
    └── surrogate_modeling/
        ├── audit_microstructure_descriptors_v2.py
        ├── check_surrogate_preprocessing_v2.py
        ├── combine_interpolation_240.py
        ├── generate_two_point_statistics_v2.py
        ├── normalize_two_point_statistics_v2.py
        ├── plot_figure_6_5.py
        ├── plot_figure_6_7_nested_cv.py
        ├── run_surrogate_supplement_v2.py
        └── train_surrogate_extended_v2.py
```

## RVE studies

### Mesh convergence

The mesh-convergence study uses a fixed 30-fibre geometry to isolate
discretization effects from variability caused by different random
fibre arrangements.

Three mesh levels—coarse, medium, and fine—are compared while keeping
the geometry, constituent properties, periodic boundary conditions,
and finite-element formulation unchanged. Refinement reduces the
target element sizes both near the fibre-matrix interfaces and in
the surrounding regions.

The homogenized plane-strain stiffness coefficients are compared
against the fine-mesh results to assess sensitivity to further
refinement and select a practical mesh density.

This check concerns the homogenized elastic response of the selected
geometry; it does not, by itself, establish statistical
representativeness or convergence of local stress concentrations.

### Statistical RVE-size convergence

The RVE-size study changes the square domain size while keeping the
fibre radius fixed. The normalized size is L/D, where L is the domain
edge length and D is the fibre diameter.

The study considers:

- Normalized sizes: L/D = 4, 8, 12, 16, 18, 20, 24, and 28
- Nominal fibre area fractions: 40%, 50%, and 60%
- Random realizations: 25 per size and nominal fibre fraction

The achieved fibre area fraction is recorded for each realization
because the integer fibre count can produce small differences from
the nominal target, particularly for small domains.

The analysis examines ensemble means, realization-to-realization
scatter, coefficients of variation, and 95% confidence intervals.
A Kanit-style required-realization estimate assesses sampling
precision; it does not independently establish RVE-size convergence.

The largest tested domain, L/D = 28, serves as a finite reference
rather than a proven infinite-domain limit. Both L/D = 16 and 18
satisfy the adopted 1% comparison criterion for the ensemble means
of Exx, Eyy, Gxy, and the in-plane Poisson ratio at all three nominal
fibre fractions.

L/D = 18 was selected as a practical compromise between mean-response
agreement, realization-to-realization variability, and computational
cost. It is used for the subsequent packing-sensitivity study and
surrogate-dataset generation.

This selection applies to the investigated synthetic microstructures,
constituent properties, and volume-averaged linear-elastic outputs.
It does not establish a universally sufficient domain size for local
fields, damage, failure, or other microstructure populations.

## Fibre-packing sensitivity study

This controlled study examines whether a deliberate change in fibre
spatial arrangement affects the apparent homogenized elastic response
when fibre area fraction, domain size, and constituent properties are
held fixed.

Statistical RVE-size convergence and packing sensitivity address
different questions. Reduced scatter among realizations generated by
one packing procedure does not establish that a different spatial
organization produces the same response.

The study compares baseline periodic hard-disk configurations with
modified clustered counterparts at L/D = 18 and approximately 40%
and 50% fibre area fraction.

For each fibre fraction, 12 baseline configurations are generated.
A clustered counterpart is constructed from each baseline by proposing
fibre displacements toward attraction points. Displacements are
accepted only when periodic non-overlap and the prescribed minimum
surface-gap constraints remain satisfied.

Each matched pair therefore retains the same fibre count, fibre
radius, domain size, and constituent properties while differing in
fibre positions. Geometry statistics quantify the imposed packing
changes before the paired mechanical responses are compared.

Within the investigated configurations, the manuscript reports no
statistically resolved change in the homogenized elastic properties
examined. This does not demonstrate equivalence of the arrangements
or their local stress fields, nor does it establish insensitivity
of damage or failure to fibre packing.

The packing study is separate from the production dataset used for
surrogate modelling.

### Study configuration

The main study is defined in `configs/packing_study_v2.json`:

- Fibre counts: 165 and 206, corresponding to approximately 40% and 50% fibre area fraction
- Packing families: `hard_disk` and `clustered`
- Realizations: 12 per family and fibre count, giving 48 FE simulations
- Fibre radius: 0.07 in the model's length units
- Minimum fibre surface gap: 0.005 in the same units
- Base random seed: 4200
- Study identifier: `packing_LD18_v2_40_50`

The numerical model uses two-dimensional, small-strain, plane-strain
linear elasticity with periodic boundary conditions.

### Implementation

- `packing_study_v2.py`: main study driver
- `packing_study.py`: shared clustering, geometry-checking, geometry-export, and FE routines; also contains the original pilot driver
- `rve_size_convergence_pipeline.py`: shared packing and meshing utilities
- `packing_study_v2_analysis.py`: geometry statistics, group summaries, and paired mechanical differences
- `plot_packing_paired_effects.py`: paired-effects figure and summary table

The main V2 workflow requires `packing_study.py`; it is not an unused
backup file.

### Running the study on HPC

From the repository root, submit:

```bash
sbatch scripts/slurm/run_packing_study_v2_array.slurm
```

The supplied launcher uses the HPCassiopee `CPU_Compute` partition and
the `fenicsx-env-312` Conda environment. Adapt these settings for another
system. The array covers tasks 0–47, with at most four simultaneous tasks.

### Post-processing

After the simulations have completed, run these commands in order
from the repository root:

```bash
python src/rve_simulation/packing_study_v2_analysis.py
python src/rve_simulation/plot_packing_paired_effects.py
```

Post-processing requires NumPy, Matplotlib, Pandas, and SciPy.

Study outputs are written under:

```text
results/packing_study/packing_LD18_v2_40_50/
```

The paired-effects plot shows individual matched-pair percentage
changes and their mean with a Student-t-based 95% confidence interval.

Preserve the configuration associated with an existing study.
The driver checks its configuration hash against saved records;
use a new study identifier when changing the configuration.

### Mesh-refinement check for saved packing geometries

`packing_mesh_refinement.py` checks mesh sensitivity without regenerating
the fibre arrangements. For a selected fibre count and replicate, it
reads the saved geometry and original FE record for both the hard-disk
and clustered configurations.

The script reuses the meshing and homogenization functions in
`rve_size_convergence_pipeline.py`. Its material constants must match
those used for the original packing study.

Available target mesh sizes, in the model's length units, are:

| Level | Far-field mesh size | Interface mesh size |
|---|---:|---:|
| Medium | 0.0100 | 0.0030 |
| Fine | 0.0050 | 0.0015 |

The original V2 records and geometry CSV files must already exist under
`results/packing_study/packing_LD18_v2_40_50/`.

For example, submit a medium-mesh check from the repository root on HPC:

```bash
sbatch --export=ALL,N=165,REP=0 scripts/slurm/run_packing_mesh_refinement.slurm
```

Here, `N` is 165 or 206, and `REP` selects a saved replicate from 0 to 11.
Each job processes both packing families for that selected pair.

The supplied launcher requests the medium level. The Python script also
supports `--level fine`.

Refined meshes and comparison records are saved separately under:

```text
results/packing_study/packing_LD18_v2_40_50/mesh_refinement/
```

The comparison records contain the original and refined values of
Exx, Eyy, and Gxy. Original simulation records and meshes are preserved.

For the configured study design, implementation dependencies,
execution procedure, analysis outputs, and interpretation limits,
see [Packing study V2](docs/packing_study_v2_README.md).

## Surrogate dataset generation

The surrogate dataset contains 1,240 paired microstructure-response
records generated using synthetic periodic two-phase fibre-matrix
RVEs at L/D = 18.

The constituent properties, fibre radius, packing-generation procedure,
and plane-strain linear-elastic homogenization framework are held fixed.
Each record links a stored fibre geometry to its apparent homogenized
stiffness matrix and derived engineering constants.

The surrogate targets are the four in-plane stiffness components
C11, C12, C22, and C66.

### Scientific dataset partition

| Partition | Target fibre area fractions | RVEs per target fraction | Total |
|---|---|---:|---:|
| Development | 40%, 45%, 50%, 55%, 60% | 200 | 1,000 |
| Reserved interpolation test | 42.5%, 47.5%, 52.5%, 57.5% | 60 | 240 |

The development set supports descriptor audit, representation
comparisons, model fitting, hyperparameter selection, repeated nested
cross-validation, and learning-curve analysis.

The intermediate target fractions are withheld from model development.
Their role is to assess interpolation across fibre area fraction,
rather than only prediction of additional realizations at target
fractions already represented during training.

Achieved fibre area fraction, rather than nominal target fraction,
is used as the principal scalar model input.

### Construction history and provenance

The dataset was assembled through an initial generation campaign,
a documented sample replacement, and two successive extensions.

The reserved interpolation set comprises 200 fresh and 40 legacy
RVEs. These cohort labels preserve construction history within the
240-record scientific test partition.

The paired hard-disk and clustered packing-sensitivity study is
separate and is not included in the 1,240-record total.

See [Surrogate dataset generation](docs/surrogate_dataset_generation.md)
for the construction history, required scripts and configurations,
HPC launchers, output tables, and provenance requirements.

## Surrogate dataset preprocessing

The V2 preprocessing workflow extracts microstructural representations
from the stored periodic fibre geometries while preserving the
development, fresh interpolation, and legacy interpolation labels.

### Scalar microstructure descriptors

The descriptor audit uses the 1,000-RVE development set to examine
achieved fibre area fraction, periodic nearest-neighbour spacing,
matrix-ligament distances, local fibre occupancy, and directional
organization of fibre centres.

Distance-based descriptors are normalized by the fibre diameter.
Candidate scalar representations compare mean nearest-neighbour
spacing, spacing variability, and their combination, together with
local occupancy variability and directional imbalance.

Descriptor correlations are used to examine dependence and redundancy;
they do not, by themselves, establish predictive value beyond fibre
area fraction.

### Normalized periodic two-point statistics

Binary fibre-phase fields are used to calculate periodic two-point
statistics. These are converted to normalized autocovariances to
reduce the dominant contribution of global fibre fraction and
represent spatial correlation in the fibre arrangement.

The normalization uses the fibre fraction measured from the same
rasterized phase field as the two-point statistic.

### Separation of extraction and model fitting

Geometry-based descriptors and normalized spatial fields can be
calculated for all stored records without fitting a predictive model.
However, the reserved interpolation records must not influence
descriptor selection, representation comparisons, or model tuning.

Learned transformations—including feature standardization and the
PCA basis—are fitted within the relevant training partitions during
cross-validation. For final interpolation evaluation, the reserved
records are transformed using pipelines fitted on development
data only.

See [Surrogate dataset preprocessing](docs/surrogate_preprocessing.md)
for required inputs, software dependencies, execution order, outputs,
and validation checks.

## Surrogate model training and evaluation

The surrogate study examines whether fibre-arrangement information
improves prediction of C11, C12, C22, and C66 beyond an
achieved-fibre-area-fraction baseline.

### Primary model comparison

The manuscript retains three configurations for the final comparison:

- Model A: quadratic regression using achieved fibre area fraction.
- Model B1: Ridge regression using achieved fibre area fraction,
  its quadratic term, normalized mean nearest-neighbour spacing,
  normalized nearest-neighbour-spacing standard deviation,
  local occupancy coefficient of variation, and directional imbalance.
- Model C: Ridge regression using achieved fibre area fraction,
  its quadratic term, and PCA scores obtained from normalized
  periodic two-point statistics. Development comparisons include
  fixed PCA dimensions and an inner-CV dimension-selection procedure.

Model B1 uses six input features. The recorded final interpolation
evaluation uses C_PCA_tuned, whose development-data refit selected
100 PCA components for every target, giving 102 input features.

This final refit does not imply that every outer cross-validation
fold selected 100 components. Fixed-100-component results and
dimension-tuned cross-validation results should be distinguished.

### Development-stage assessment

Model development uses only the 1,000-RVE development set.

The repeated nested-cross-validation protocol uses:

- Five outer folds, stratified by target fibre area fraction
- Ten repetitions of the outer partitioning
- Four-fold inner cross-validation for hyperparameter selection
- Identical outer partitions for the compared models

Learned preprocessing transformations are fitted within the relevant
training partitions. Performance metrics are calculated from the
pooled out-of-fold predictions within each repetition and summarized
across repetitions.

The primary training driver compares scalar representations and
fixed or inner-CV-selected PCA dimensions.

The supplementary driver, run_surrogate_supplement_v2.py, implements
matched-input comparisons of Elastic Net, random forest, RBF-SVR,
and XGBoost, together with balanced learning curves for the baseline,
scalar, and spatial representations. These analyses use development
records only.

See the training documentation for the supplementary configuration,
HPC launchers, preparation and summary stages, dependencies, and outputs.

### Final interpolation evaluation

The retained configurations are fixed before evaluation on the
reserved interpolation dataset.

Final model fitting and any remaining hyperparameter selection use
development data only. The resulting fitted pipelines are applied
to 200 fresh and 40 legacy interpolation RVEs without refitting
or tuning on those records.

This test assesses prediction at intermediate fibre-area-fraction
levels within the studied synthetic simulation domain. It does not
establish extrapolation capability or transfer to experimental
microstructures, different constituent systems, or damage-related
responses.

See [Surrogate model training and evaluation](docs/surrogate_training.md)
for implementation details, validation settings, execution instructions,
recorded model selection, and output artifacts.

## Pooled interpolation reporting

The reporting workflow combines saved predictions for 200 fresh and
40 legacy interpolation RVEs into a 240-record evaluation table.

This is a reporting-only stage. It does not fit models, select
descriptors, tune hyperparameters, change PCA dimensions, or generate
new predictions.

### Scientific interpretation

The combined test set covers the withheld target fibre area fractions
42.5%, 47.5%, 52.5%, and 57.5%, with 60 RVEs per level.

Pooled metrics summarize interpolation performance across these
intermediate fractions for the retained Model A, Model B1, and
Model C configurations. They do not represent an additional
independent test beyond the records being combined.

The comparison uses predictions from models whose configurations
were fixed before interpolation-test evaluation.

### Cohort traceability

The pooled prediction table retains fresh and legacy cohort labels,
and separate cohort metrics remain available. These labels distinguish
dataset-generation cohorts, not prior use in model development or
evaluation. The 40 legacy cases remained untouched until the final
interpolation evaluation.

The reporting script, combine_interpolation_240.py, produces pooled
metrics and figures from the saved prediction artifacts.

See [Pooled interpolation report](docs/surrogate_interpolation_report.md)
for required inputs, validation checks, recorded model IDs, execution
instructions, and output artifacts.

## Environment and reproducibility

The primary HPC environment is `fenicsx-env-312` on HPCassiopee.

A snapshot of that environment, exported on 2026-10-28, is available
in [environment-hpc-2026-10-28.yml](environment-hpc-2026-10-28.yml).
It records package versions and channel order, omits build strings,
and excludes the machine-specific installation prefix.

Python 3.12.14 and successful imports of the main simulation and
scientific Python packages were confirmed in the existing HPC
environment. The surrogate workflows also require packages including
SciPy, scikit-learn, and Joblib, in addition to those listed above.

This snapshot records the environment observed on that date. It does
not establish the exact historical environment used for every result.
Environment recreation, multi-node execution, saved-model compatibility,
and reproduction of accepted numerical results were not tested as
part of the import check.

See [HPC environment records](hpc/README.md) and
[HPCassiopee setup notes](hpc/setup_hpcassiopee.md) for details.

No simulations, model training, or evaluations were rerun to prepare
the snapshot or update this documentation.

## Software

The simulation, preprocessing, and primary surrogate-modelling
workflows use the following software.

### RVE generation and finite-element homogenization

- Python
- NumPy
- Gmsh
- FEniCSx / DOLFINx
- dolfinx_mpc
- MPI / mpi4py
- PETSc / petsc4py
- UFL

### Data processing, statistical analysis, and surrogate modelling

- NumPy
- Pandas
- SciPy
- scikit-learn
- Joblib
- Matplotlib
- XGBoost (supplementary regression comparisons)
- threadpoolctl (used by the supplementary driver)

Requirements depend on the workflow being executed. Consult the
linked workflow documentation for execution-specific dependencies
and the environment records above for recorded package versions.

The environment snapshot documents the observed HPC installation;
it is not a guarantee that every workflow has been reproduced in a
newly created environment.

## Status

This repository is under active development as part of my doctoral
research. It documents the periodic homogenization and
descriptor-based surrogate-modelling work presented in Chapters 5
and 6 of the thesis.

The tracked code includes periodic RVE generation and homogenization,
convergence and packing-sensitivity studies, staged surrogate-dataset
generation, descriptor preprocessing, primary surrogate-model training,
and pooled interpolation reporting.

Final frozen-model interpolation evaluation is implemented as a
separate stage of the primary training driver. Additional scalar
regression comparisons and balanced learning curves are implemented
in the supplementary driver.

The workflow documentation distinguishes primary model development,
supplementary analyses, frozen-model evaluation, and pooled reporting.
Source inclusion and documentation do not, by themselves, establish
that every manuscript result has been independently reproduced.

The current numerical scope is synthetic two-phase, two-dimensional,
plane-strain linear elasticity. Extensions toward experimentally
informed microstructures, explicit porosity, three-dimensional
modelling, and nonlinear damage or failure remain future research
directions.

## Author

Mahsa Arabkoohichetabi