# Surrogate dataset preprocessing

## Purpose

This workflow extracts scalar microstructural descriptors and periodic
fibre-phase two-point statistics from the completed Chapter 6 dataset.

It processes 1,240 records while preserving three split labels:

- `main`: 1,000 development RVEs
- `interpolation_test_fresh`: 200 fresh interpolation RVEs
- `interpolation_test`: 40 legacy interpolation RVEs

No surrogate regressors are trained or evaluated during preprocessing.

## Required inputs

The completed dataset directory is:

`results/surrogate_dataset/surrogate_LD18_hard_disk_extension_v2/`

Required inputs include:

- `expanded_dataset.csv`
- `completion_metadata.json`
- Every geometry CSV referenced by the dataset

Geometry files may belong to earlier generation stages or the documented
replacement. Preserve their repository-relative locations.

The scripts check the completed dataset's recorded checksum. Do not
edit the CSV or replace its expected hash merely to bypass a mismatch.

## Software requirements

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

These preprocessing scripts do not require FEniCSx or Gmsh imports.
The supplied HPC launcher uses the existing `fenicsx-env-312` environment.

## Execution order

Run from the repository root:

```bash
python src/surrogate_modeling/audit_microstructure_descriptors_v2.py
python src/surrogate_modeling/generate_two_point_statistics_v2.py
python src/surrogate_modeling/normalize_two_point_statistics_v2.py
python src/surrogate_modeling/check_surrogate_preprocessing_v2.py
```

Alternatively, on HPCassiopee, submit the launcher that runs these
four steps in sequence:

```bash
sbatch scripts/slurm/run_surrogate_preprocess_v2.slurm
```

Adapt the partition and Conda environment settings for another system.

The preprocessing steps write to their configured output locations
and may overwrite existing outputs. Preserve accepted historical
artifacts before rerunning them.

## Scalar descriptors

`audit_microstructure_descriptors_v2.py` reads the saved fibre geometry
and computes:

- Periodic nearest-neighbour centre distances
- Nearest-neighbour matrix-ligament distances
- Distance statistics normalized by fibre diameter
- Fibre-centre-count-based local occupancy statistics on a 6 × 6 grid
- A directional-imbalance descriptor based on periodic centre coordinates

The local occupancy calculation counts fibre centres; it does not
calculate exact fibre area intersecting each grid cell.

The descriptor table includes all three splits. The audit's distribution,
correlation, and exploratory descriptor-PCA plots use development records
only.

This exploratory descriptor PCA is separate from the two-point-statistics
PCA used later in surrogate modeling.

## Periodic two-point statistics

`generate_two_point_statistics_v2.py` rasterizes each periodic geometry
into a 128 × 128 binary fibre-phase image and computes its periodic
autocorrelation using FFTs.

Each flattened field contains 16,384 values. The complete raw matrix
therefore has shape `(1240, 16384)`.

`normalize_two_point_statistics_v2.py` computes:

`normalized_S2 = (raw_S2 - raster_vf**2) / (raster_vf * (1 - raster_vf))`

Here, `raster_vf` is the fibre fraction of the binary image, not the
nominal target fraction.

This normalization is performed separately for each RVE. Fitting the
spatial PCA representation is a separate downstream operation.

## Main outputs

Under the completed dataset directory:

- `descriptor_audit/microstructure_descriptors.csv`
- `descriptor_audit/descriptor_audit_metadata.json`
- `two_point_statistics/two_point_statistics_metadata.csv`
- `two_point_statistics/periodic_fibre_S2_128x128.npz`
- `two_point_statistics/periodic_fibre_normalized_S2_128x128.npz`
- `preprocessing_validation_v2.json`

Additional diagnostic figures and metadata are also generated.

The descriptor table contains resolved geometry paths. If moving the
workflow between computers, regenerate the descriptor audit from the
dataset and available geometry rather than relying on stale absolute
paths from another machine.

## Validation

The final checker verifies:

- Dataset counts and split labels
- Sample identities and row ordering across tables and arrays
- Agreement of fibre-fraction metadata
- Preservation of mechanical target labels
- Finite values and expected array dimensions
- Raw and normalized two-point-statistics consistency

A successful syntax check alone does not perform these data checks.
Run the final checker on the actual preprocessing outputs to validate them.

## Earlier workflow versions

The `_expanded` scripts target the intermediate 440-record dataset.

The `prepare_...` scripts generated preprocessing source files during
development. They are not required to execute the published final V2
scripts and should not be rerun over existing source files as part of
normal preprocessing.