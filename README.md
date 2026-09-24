# CFRP Structure–Property Modelling

This repository contains computational workflows for studying the relationship between carbon-fibre-reinforced polymer (CFRP) microstructure and effective elastic properties.

The project combines:

- Generation of periodic two-dimensional fibre–matrix representative volume elements (RVEs)
- Gmsh mesh generation with periodic boundaries
- FEniCSx-based computational homogenization
- Mesh-convergence verification
- RVE-size-convergence analysis across multiple random realizations
- Preparation of simulation data for future surrogate-model development

## Current workflow

The current RVE workflow uses a periodic fibre-packing algorithm based on:

1. Random sequential adsorption (RSA) at a small initial fibre radius
2. Adaptive fibre-radius growth
3. Periodic collision relaxation inspired by the Lubachevsky–Stillinger approach
4. Final damping to obtain a settled fibre configuration

This approach is designed to generate random periodic fibre configurations at moderate to relatively high fibre area fractions, where direct RSA placement alone may become impractical.

## Repository structure

```text
cfrp-structure-property/
├── data/
│   └── rve_reference/
│       └── RVE_01_fibers.csv
├── docs/
├── examples/
├── scripts/
│   └── slurm/
├── src/
│   └── rve_simulation/
│       ├── generate_periodic_fiber_rves_gmsh.py
│       ├── homogenize_fibre_RVEs_fenicsx.py
│       ├── mesh_convergence_RVE01.py
│       ├── rve_size_convergence_pipeline.py
│       ├── run_single_combo.py
│       ├── merge_results.py
│       └── rve_size_convergence_plots.py
└── tests/
```

## RVE studies

### Mesh convergence

A fixed fibre geometry is meshed at coarse, medium, and fine resolutions while keeping the material parameters, geometry, boundary conditions, and finite-element formulation unchanged.

The effective elastic stiffness and engineering constants are compared against the fine-mesh solution to select an appropriate mesh density.

### RVE-size convergence

The RVE-size study changes domain size while keeping fibre radius fixed. The primary size variable is:

\[
L/D
\]

where \(L\) is the square RVE edge length and \(D\) is the fibre diameter.

For each selected fibre area fraction and RVE size, multiple random realizations are generated, homogenized, and analyzed statistically using mean values, confidence intervals, coefficient of variation, and a Kanit-style estimate of the required number of realizations.

## Software

The complete simulation workflow requires:

- Python
- NumPy
- Gmsh
- Matplotlib
- FEniCSx / DOLFINx
- dolfinx_mpc
- MPI / mpi4py
- PETSc / petsc4py
- UFL
- Pandas

## Status

This repository is under active development as part of doctoral research. The initial public version focuses on the RVE generation, mesh verification, and homogenization workflow. Surrogate-model training workflows will be added later.

## Author

Mahsa Arabkoohichetabi