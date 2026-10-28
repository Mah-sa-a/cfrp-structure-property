# HPC setup and environment records

The final project workflows used the Python 3.12 Conda environment
named `fenicsx-env-312` on HPCassiopee.

The older `fenicsx-env` specification is retained as a development
environment record. It is not the primary environment for the final
workflow.

## Software used

The FE simulation workflows use FEniCSx, Gmsh, MPI, PETSc, and
dolfinx_mpc.

The surrogate training, evaluation, and reporting workflows use
scientific Python packages including NumPy, Pandas, scikit-learn,
Joblib, and Matplotlib.

## Environment records and setup files

- [environment-hpc-2026-10-28.yml](../environment-hpc-2026-10-28.yml):
  dated snapshot of the current `fenicsx-env-312` environment,
  stored at the repository root.
- `environment-hpcassiopee-py312.yml`: earlier specification for
  the primary Python 3.12 environment, `fenicsx-env-312`.
  Retained unchanged.
- `environment-hpcassiopee.yml`: specification for the older
  development environment, `fenicsx-env`. Retained unchanged.
- `setup_hpcassiopee.md`: environment activation and setup notes.

## Primary environment activation

On HPCassiopee, in a shell where Conda is available:

```bash
conda activate fenicsx-env-312
```

Individual Slurm launchers may include their own environment
activation commands. Review the relevant launcher before submission.

## Current environment snapshot and verification

On 2026-10-28, the current HPC environment was exported using:

```bash
conda env export -n fenicsx-env-312 --no-builds
```

The machine-specific `prefix:` entry was removed before committing
the export as
[environment-hpc-2026-10-28.yml](../environment-hpc-2026-10-28.yml).

The snapshot records package versions and channel order.
Build strings were omitted.

A separate check confirmed that the active interpreter belonged to
`fenicsx-env-312`, used Python 3.12.14, and successfully imported:

- `dolfinx`, `dolfinx_mpc`, and `gmsh`;
- `mpi4py` and `petsc4py`;
- `numpy`, `pandas`, and `scipy`;
- `sklearn`, `joblib`, and `matplotlib`.

No simulations, model training, or evaluations were rerun for
this check.

## Current reproducibility limitations

The dated snapshot records the environment observed on 2026-10-28.
It does not establish the exact historical environment used to
produce every accepted numerical result.

Because build strings were omitted, the snapshot is not an exact
build-level lockfile. Recreating an environment from this export
has not been tested.

Successful imports do not validate multi-node MPI execution,
complete workflow execution, compatibility with previously saved
models, or reproduction of accepted numerical results.

The two earlier specifications in this folder remain unchanged.
Most package versions in those files are not pinned, and they do
not explicitly list every dependency of the final surrogate
workflow, including scikit-learn and Joblib. Both also retain
account-specific installation prefixes.

The dated snapshot includes scikit-learn and Joblib with their
versions and does not contain the machine-specific prefix.

## Preservation of accepted results

Preserve the working environment, configurations, metadata, and
accepted numerical results together.

Do not upgrade or recreate the working environment merely to update
this documentation. Updating environment records does not require
rerunning simulations, model training, or interpolation evaluation.

## Security and portability

Do not commit passwords, SSH keys, access tokens, or other credentials.

Review environment exports and launch scripts for account-specific
paths before sharing them. Document cluster-specific assumptions
rather than presenting them as portable defaults.