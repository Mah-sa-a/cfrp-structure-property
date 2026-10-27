# Surrogate dataset generation

## Scope

The Chapter 6 dataset contains synthetic periodic two-dimensional
hard-disk fibre–matrix RVEs at L/D = 18, with fixed constituent properties
and small-strain, plane-strain linear elasticity.

The dataset was constructed in three generation stages, with one
documented replacement in the initial stage.

## Dataset construction history

| Stage | New development RVEs | New interpolation RVEs | Combined records |
|---|---:|---:|---:|
| Initial dataset | 200 | 40 | 240 |
| Extension V1 | 200 | 0 | 440 |
| Extension V2 | 600 | 200 | 1,240 |

The final development dataset contains 200 realizations at each principal
target fibre area fraction: 40%, 45%, 50%, 55%, and 60%.

The interpolation records comprise 10 legacy and 50 fresh realizations
at each intermediate target fraction: 42.5%, 47.5%, 52.5%, and 57.5%.

Thus, the final dataset contains:

- 1,000 development RVEs
- 200 fresh interpolation RVEs
- 40 legacy interpolation RVEs

The legacy interpolation cases had already been evaluated during an
earlier stage. Their history must remain distinct from that of the fresh
cases when describing model evaluation.

## Source files and dependencies

The generation scripts are located in `src/rve_simulation/`:

- `generate_surrogate_dataset.py`: original generator and shared FE execution
- `recover_sample_0084.py`: documented replacement and initial-table assembly
- `generate_surrogate_dataset_extension.py`: development-only Extension V1
- `generate_surrogate_dataset_extension_v2.py`: Extension V2 and final-table assembly

Both extension scripts reuse the original generator. The generation
and recovery workflows also require:

- `packing_study.py`
- `rve_size_convergence_pipeline.py`

Keep these dependencies together. The extension filenames describe
successive dataset-construction stages, not interchangeable alternatives.

## Configurations

The corresponding configurations are:

- `configs/surrogate_dataset_v1.json`
- `configs/surrogate_dataset_extension_v1.json`
- `configs/surrogate_dataset_extension_v2.json`

Material properties, geometry settings, and mesh ratios are defined in
the original configuration. The extensions inherit those settings.

## Historical execution sequence

1. Run the initial 240-task generation campaign.
2. Inspect completed records and failures.
3. For the documented failed `sample_0084`, generate the replacement
   using seed 30084, preserving the original failed geometry and log.
4. Assemble the accepted initial table as `dataset_with_replacement.csv`.
5. Run the 200-task Extension V1 campaign.
6. Use the Extension V1 summary to validate and assemble
   `expanded_dataset.csv`, containing 440 accepted records.
7. Prepare Extension V2 using its `--prepare` mode.
8. Run the 800-task Extension V2 campaign.
9. Use its `--summary` mode to validate and assemble the final tables.

The replacement step records a specific historical failure. Do not
apply it automatically to a new campaign in which that sample succeeds.

## HPC launchers

The launchers are located in `scripts/slurm/`:

| Launcher | Task range or purpose |
|---|---|
| `run_surrogate_dataset_v1.slurm` | Tasks 0–239 |
| `recover_sample_0084.slurm` | Replacement using seed 30084 |
| `run_surrogate_dataset_extension_v1.slurm` | Tasks 0–199 |
| `run_surrogate_dataset_extension_v2.slurm` | Tasks 0–799 |

Submit from the repository root. The array launchers limit concurrency
to four tasks.

These scripts target HPCassiopee. The recovery launcher additionally
contains absolute Conda installation and environment paths that must
be adapted for another account or system.

## Final generation outputs

Extension V2 writes its outputs under:

`results/surrogate_dataset/surrogate_LD18_hard_disk_extension_v2/`

Its summary produces:

- `expanded_dataset.csv`: all 1,240 records
- `development_dataset.csv`: 1,000 development records
- `fresh_interpolation_test.csv`: 200 fresh interpolation records
- `legacy_interpolation_test.csv`: 40 legacy interpolation records
- `extension_dataset.csv`: the 800 newly generated records
- `completion_metadata.json`: completion information and output hashes

Model fitting, prediction, and aggregation of interpolation metrics
are separate downstream workflows.

## Provenance and reproduction requirements

The scripts record and validate configuration and source-code hashes.
Extension V2 also requires the exact accepted 440-row source CSV and
its referenced geometry files.

Matching row counts alone is insufficient. Editing a source CSV,
changing line endings, or regenerating records with different metadata
may change its hash.

The repository uses LF line endings for Python, JSON, and shell scripts.
Do not change expected hashes merely to bypass a failed validation.

Reproducing the frozen historical extension requires its original
source artifacts. A new simulation campaign requires separately
defined provenance and configurations; it must not be silently mixed
with the historical dataset.