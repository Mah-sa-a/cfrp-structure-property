# HPCassiopee setup notes

## Primary environment

The final project workflows used the Python 3.12 Conda environment:

`fenicsx-env-312`

The current environment snapshot, recorded on 2026-10-28, is:

[environment-hpc-2026-10-28.yml](../environment-hpc-2026-10-28.yml)

The earlier specification is retained unchanged at:

`hpc/environment-hpcassiopee-py312.yml`

The older `fenicsx-env` environment was used during development.
Its specification is retained in `hpc/environment-hpcassiopee.yml`.

## Activate the existing environment

On HPCassiopee, in a shell where Conda is available:

```bash
conda activate fenicsx-env-312
```

This command activates an existing environment. It does not install
the environment or verify that all workflow dependencies are present.

## Job submission

Submit workflow launchers from the repository root unless the
individual workflow guide specifies otherwise.

Before submitting, review the launcher's environment activation,
working directory, partition, resource requests, and any
account-specific paths.

Workflow instructions are provided in:

- `docs/packing_study_v2_README.md`
- `docs/surrogate_dataset_generation.md`
- `docs/surrogate_preprocessing.md`
- `docs/surrogate_training.md`
- `docs/surrogate_interpolation_report.md`

Do not repeat accepted simulations, training, or interpolation
evaluation merely to check or update these setup notes.

## Current environment verification

On 2026-10-28, the active interpreter was confirmed to belong to
`fenicsx-env-312` and to use Python 3.12.14.

The following packages imported successfully:

- `dolfinx`, `dolfinx_mpc`, and `gmsh`;
- `mpi4py` and `petsc4py`;
- `numpy`, `pandas`, and `scipy`;
- `sklearn`, `joblib`, and `matplotlib`.

No simulations, model training, or evaluations were rerun for
this check.

## Environment-record limitations

The dated snapshot records package versions and channel order
from the current HPC environment. Build strings were omitted,
and the machine-specific installation prefix was removed.

This is a current environment record, not proof of the exact
historical environment used for every accepted result. It is not
an exact build-level lockfile or a tested portable installation
recipe.

Environment recreation, multi-node MPI execution, complete
workflow execution, saved-model compatibility, and reproduction
of numerical results were not tested as part of this check.

The earlier specifications in `hpc/` remain unchanged. They do
not pin most package versions or explicitly list every dependency
of the final surrogate workflow, and they retain installation
prefixes specific to the original HPC account.

See [the HPC environment documentation](README.md) for the
snapshot details, import-check scope, and remaining limitations.

## Security

Do not store passwords, SSH keys, access tokens, or other credentials
in the repository.

Review exported files for account-specific information before
sharing them.