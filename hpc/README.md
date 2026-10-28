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

## Files in this folder

- `environment-hpcassiopee-py312.yml`: environment specification
  for the primary Python 3.12 environment, `fenicsx-env-312`.
- `environment-hpcassiopee.yml`: environment specification for
  the older development environment, `fenicsx-env`.
- `setup_hpcassiopee.md`: environment activation and setup notes.

## Primary environment activation

On HPCassiopee, in a shell where Conda is available:

```bash
conda activate fenicsx-env-312
```

Individual Slurm launchers may include their own environment
activation commands. Review the relevant launcher before submission.

## Current reproducibility limitations

The Python 3.12 specification records the Python minor version,
but most other package versions are not pinned.

The checked-in YAML files also do not explicitly list every package
required by the final surrogate workflow, including scikit-learn
and Joblib. This does not establish that those packages were absent
from the working environment.

These files should therefore not be treated as complete records of
the software versions used for the accepted numerical results.

Both specifications currently contain an account-specific `prefix`
path. This identifies an installation location on the original HPC
system; it is not a portable installation path for other users.

Verifying and recording the complete working environment remains a
follow-up reproducibility task.

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