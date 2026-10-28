# Packing study V2

## Purpose and scope

This workflow compares hard-disk and clustered periodic fibre
arrangements at fixed domain-size-to-diameter ratio L/D = 18.

The clustered arrangement is generated from the hard-disk starting
configuration using the same realization seed. The study supports
paired comparisons between packing families.

The configured study covers approximately 40% and 50% fibre area
fraction. It does not include an ordered-lattice comparison or a
60% fibre-fraction case.

This guide documents the implementation and configuration. It does
not establish that all simulations or publication checks have passed.

## Implementation

- `configs/packing_study_v2.json`
- `src/rve_simulation/packing_study_v2.py`
- `src/rve_simulation/packing_study_v2_analysis.py`
- `scripts/slurm/run_packing_study_v2_array.slurm`

The V2 worker imports functionality from:

- `src/rve_simulation/packing_study.py`
- `src/rve_simulation/rve_size_convergence_pipeline.py`

These dependencies must remain available even though they also
support earlier studies.

## Configured study design

The configuration identifies the study as:

`packing_LD18_v2_40_50`

| Setting | Configured value |
|---|---|
| Domain size / fibre diameter | 18 |
| Fibre radius | 0.07 |
| Minimum surface gap | 0.005 |
| Fibre counts | 165 and 206 |
| Packing families | `hard_disk`, `clustered` |
| Realizations per count and family | 12 |
| Base seed | 4200 |
| Total tasks | 48 |
| Retain meshes | true |
| Retain previews | true |
| Include 60% case | false |

For fibre count N, radius r, and domain side L, the geometric fibre
area fraction is N*pi*r^2/L^2, with L = 18*(2*r).

The two counts correspond to approximately 39.997% and 49.936%.
Use recorded achieved fractions when reporting results.

The same realization index uses the same starting seed across
families and fibre counts. Do not describe all 48 tasks as
independently seeded observations.

## Materials and mesh settings

Configured elastic properties are:

| Phase | Young's modulus | Poisson ratio |
|---|---|---|
| Matrix | 3 GPa | 0.35 |
| Fibre | 15 GPa | 0.30 |

Configured mesh-size ratios are:

- `h_far_over_r`: 0.30
- `h_interface_over_r`: 0.09

The configuration also supplies:

- `mesh_area_rtol`: 0.002
- `phase_fraction_tolerance`: 0.005
- `maximum_relative_C_asymmetry`: 0.002

These are configured settings and acceptance thresholds, not evidence
that a particular run passed them. Inspect the worker implementation
and saved records when interpreting the checks.

Do not describe this mesh as converged solely because these settings
are present. Mesh adequacy requires the relevant refinement evidence.

## Execution and preservation of accepted results

Run commands from the repository root.

The commands below describe the workflow. They are not instructions
to repeat an already completed and accepted study.

To list tasks without performing FE solves:

```bash
python src/rve_simulation/packing_study_v2.py --list
```

For this configuration, the expected task count is 48.

For a deliberately initiated study on the configured HPC system:

```bash
sbatch scripts/slurm/run_packing_study_v2_array.slurm
```

Before submission, check the launcher's environment, partition,
resource requests, and array bounds. The configured task IDs are
0 through 47.

Do not change the configuration while jobs are running. For a changed
experiment, use a distinct study ID and preserve accepted historical
records.

The worker checks configuration hashes when encountering an existing
task record. Successful full tasks can be skipped on resubmission.

## Output locations

The study writes under:

`results/packing_study/packing_LD18_v2_40_50/`

Subdirectories include:

- `records/`: per-task JSON records
- `geometry/`: saved fibre-centre coordinates
- `meshes/`: retained meshes
- `previews/`: geometry previews
- `failures/`: task failure tracebacks
- `figures/`: analysis figures

The current configuration retains meshes. Earlier pilot settings may
have deleted meshes after successful FE solves; inspect each study's
configuration rather than assuming identical retention behaviour.

Keep large generated artifacts separate from the source-code release.
Preserve the configuration, records, and provenance needed to identify
accepted results.

## Analysis of saved results

The analysis script reads saved geometry and task records. It does
not launch new FE simulations.

```bash
python src/rve_simulation/packing_study_v2_analysis.py
```

Principal tabular outputs are:

- `all_realizations.csv`
- `group_summary.csv`
- `paired_differences.csv`

The analysis also writes figures under `figures/` and reports the
number of completed FE tasks.

It computes periodic geometric descriptors, including pair-correlation
estimates, local sampled fibre-fraction statistics, and random-point
clearance statistics.

The configured phase-mask grid is 128 by 128. Window side lengths are
3, 4.5, and 6 fibre diameters, and the clearance calculation uses
20,000 random probe points per analyzed realization.

Analysis writes output tables and figures. Preserve accepted analysis
artifacts before intentionally regenerating them.

## Interpretation and limitations

- Report completion counts and inspect failed or missing tasks.
- Analyze packing effects as paired comparisons where appropriate.
- Sample standard deviation describes realization scatter; it is not
  automatically a confidence interval or an equivalence test.
- Random-point clearance is zero inside fibres. Its distribution is
  not a measurement of guaranteed fibre-free circle radii.
- Rasterized local-fraction statistics are approximate. Check grid
  sensitivity before using them for numerical conclusions.
- A small measured packing contrast does not establish insensitivity
  to arbitrary clustering.
- Baseline RVE-size convergence does not automatically establish
  convergence for clustered arrangements.
- Experimental representativeness requires separate evidence; these
  descriptors alone do not establish it.
- This workflow does not establish microscopic stress or damage
  predictions.

Preserve the distinction between configured checks, completed
validation, and conclusions supported by the actual results.
