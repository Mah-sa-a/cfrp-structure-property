# HPCassiopee setup notes

The RVE simulation workflow was run on the HPCassiopee cluster using Conda environments with FEniCSx and related scientific-computing packages.

## Available Conda environments

Two FEniCSx environments were used during project development:

- `fenicsx-env`
- `fenicsx-env-312` — Python 3.12 environment


## Environment activation

Example activation command:

```bash
conda activate fenicsx-env
```

For the Python 3.12 environment:

```bash
conda activate fenicsx-env-312
```

## Reproducibility files

The following exported environment specifications are stored in this folder:

- `environment-hpcassiopee.yml`
- `environment-hpcassiopee-py312.yml`

## Security

Do not store passwords, SSH keys, personal tokens, allocation identifiers, or private filesystem paths in this repository.