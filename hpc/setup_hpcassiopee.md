# HPCassiopee setup notes

## Primary environment

The final project workflows used the Python 3.12 Conda environment:

`fenicsx-env-312`

The corresponding checked-in specification is:

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

## Environment-record limitations

The checked-in specifications do not pin most package versions or
explicitly list every dependency of the final surrogate workflow.

They also contain installation prefixes specific to the original
HPC account.

See `hpc/README.md` for the current limitations and the outstanding
environment-verification task. These files are not yet a verified,
portable installation recipe.

## Security

Do not store passwords, SSH keys, access tokens, or other credentials
in the repository.

Review exported files for account-specific information before
sharing them.