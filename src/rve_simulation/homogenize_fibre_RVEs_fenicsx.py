"""Periodic linear-elastic homogenization of circular-fibre RVEs in FEniCSx.

Adapted to .msh files created by generate_periodic_fiber_rves_gmsh.py.
Method: periodic displacement fluctuation formulation, following the workflow
of Jeremy Bleyer's FEniCSx periodic-elasticity numerical tour.

Required packages (versions must be compatible):
    dolfinx, dolfinx_mpc, gmsh, mpi4py, petsc4py, numpy

Run, for example:
    python homogenize_fibre_RVEs_fenicsx.py
or in parallel:
    mpirun -n 4 python homogenize_fibre_RVEs_fenicsx.py

The code assumes 2D SMALL-STRAIN PLANE STRAIN elasticity.
Replace the four material constants below by your own measured/literature values.
"""

from pathlib import Path
import csv
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from dolfinx_mpc import MultiPointConstraint, LinearProblem

# ------------------------------ User settings ------------------------------
RVE_DIRECTORY = Path("RVE_10_realizations")
RVE_IDS = range(1, 11)       # Use range(1, 2) first to test only RVE_01
RVE_SIZE = 1.0               # Must be the same value used to make the meshes

# Gmsh physical tags from the mesh-generation code
MATRIX_TAG = 1
FIBRE_TAG = 2
LEFT_TAG = 11
RIGHT_TAG = 12
BOTTOM_TAG = 13
TOP_TAG = 14

# EXAMPLE elastic properties in Pa: replace with your actual phase properties.
# These parameters are used as 2D plane-strain isotropic phase properties.
E_MATRIX = 3.0e9
NU_MATRIX = 0.35
E_FIBRE = 15.0e9
NU_FIBRE = 0.25

OUTPUT_DIRECTORY = RVE_DIRECTORY / "fenicsx_homogenization_results"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.rank


def create_piecewise_constant_field(domain, cell_tags, values_by_tag, name):
    """Return a scalar DG0 function with one value per material cell tag."""
    V0 = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(V0, name=name)

    for tag, value in values_by_tag.items():
        tagged_cells = cell_tags.find(tag)
        if tagged_cells.size == 0:
            raise RuntimeError(
                f"No cells with physical tag {tag} were found in this .msh file."
            )
        field.x.array[tagged_cells] = value

    field.x.scatter_forward()
    return field


def periodic_right_to_left(x):
    """Map a point on x=L to its periodic partner on x=0."""
    mapped = np.zeros_like(x)
    mapped[0] = x[0] - RVE_SIZE
    mapped[1] = x[1]
    mapped[2] = x[2]
    return mapped


def periodic_top_to_bottom(x):
    """Map a point on y=L to its periodic partner on y=0."""
    mapped = np.zeros_like(x)
    mapped[0] = x[0]
    mapped[1] = x[1] - RVE_SIZE
    mapped[2] = x[2]
    return mapped


def assemble_global_scalar(form):
    """Assemble a scalar form and sum it over MPI ranks."""
    local_value = fem.assemble_scalar(form)
    return comm.allreduce(local_value, op=MPI.SUM)


def read_rve_mesh(mesh_file):
    """Read a Gmsh mesh plus physical cell/facet tags."""
    mesh_data = gmshio.read_from_msh(mesh_file, comm, rank=0, gdim=2)

    # DOLFINx >= 0.10 returns a MeshData object.
    # The fallback supports some older releases returning a tuple.
    if hasattr(mesh_data, "mesh"):
        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        facet_tags = mesh_data.facet_tags
    else:
        domain, cell_tags, facet_tags = mesh_data

    if cell_tags is None:
        raise RuntimeError("No cell tags were imported from the .msh file.")
    if facet_tags is None:
        raise RuntimeError("No facet tags were imported from the .msh file.")

    return domain, cell_tags, facet_tags


def homogenize_one_rve(rve_id):
    """Compute the 2D homogenized stiffness matrix for one RVE mesh."""
    mesh_file = RVE_DIRECTORY / f"RVE_{rve_id:02d}.msh"

    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")

    domain, cell_tags, facet_tags = read_rve_mesh(mesh_file)
    gdim = domain.geometry.dim

    # Verify that the mesh contains the expected material and boundary tags.
    for tag, label in [(MATRIX_TAG, "matrix"), (FIBRE_TAG, "fibre")]:
        if cell_tags.find(tag).size == 0:
            raise RuntimeError(f"RVE {rve_id:02d}: missing {label} cell tag {tag}.")

    for tag, label in [(LEFT_TAG, "left"), (RIGHT_TAG, "right"),
                       (BOTTOM_TAG, "bottom"), (TOP_TAG, "top")]:
        if facet_tags.find(tag).size == 0:
            raise RuntimeError(f"RVE {rve_id:02d}: missing {label} facet tag {tag}.")

    # Elastic material fields: piecewise constant matrix/fibre values.
    young_modulus = create_piecewise_constant_field(
        domain,
        cell_tags,
        {MATRIX_TAG: E_MATRIX, FIBRE_TAG: E_FIBRE},
        "Young_modulus"
    )
    poisson_ratio = create_piecewise_constant_field(
        domain,
        cell_tags,
        {MATRIX_TAG: NU_MATRIX, FIBRE_TAG: NU_FIBRE},
        "Poisson_ratio"
    )

    # Plane-strain Lamé coefficients.
    lame_lambda = (
        young_modulus * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))

    # Unknown: periodic displacement fluctuation v.
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    v_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    macro_strain = fem.Constant(
        domain,
        np.zeros((gdim, gdim), dtype=default_scalar_type)
    )
    test_macro_strain = fem.Constant(
        domain,
        np.zeros((gdim, gdim), dtype=default_scalar_type)
    )

    def epsilon(displacement):
        return ufl.sym(ufl.grad(displacement))

    def sigma(fluctuation):
        total_strain = macro_strain + epsilon(fluctuation)
        return (
            lame_lambda * ufl.tr(total_strain) * ufl.Identity(gdim)
            + 2.0 * shear_modulus * total_strain
        )

    # ufl.system splits the macro-strain-dependent expression into a(v, w)=L(w).
    bilinear_form, linear_form = ufl.system(
        ufl.inner(sigma(v_trial), epsilon(v_test)) * ufl.dx
    )

    # Fix v(0, 0) = 0 to remove the rigid translation mode.
    corner_dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if corner_dofs.size == 0:
        raise RuntimeError(
            "No degree of freedom was found at (0, 0). Check RVE_SIZE and mesh boundaries."
        )

    zero_displacement = np.zeros(gdim, dtype=default_scalar_type)
    bcs = [fem.dirichletbc(zero_displacement, corner_dofs, V)]

    # Periodicity of the fluctuation v: right -> left and top -> bottom.
    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(
        V, facet_tags, RIGHT_TAG, periodic_right_to_left, bcs
    )
    mpc.create_periodic_constraint_topological(
        V, facet_tags, TOP_TAG, periodic_top_to_bottom, bcs
    )
    mpc.finalize()

    fluctuation = fem.Function(mpc.function_space, name="periodic_fluctuation")

    problem = LinearProblem(
        bilinear_form,
        linear_form,
        mpc,
        bcs=bcs,
        u=fluctuation,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )

    # Mesh area is used to calculate volume/area averages.
    rve_area = assemble_global_scalar(fem.form(1.0 * ufl.dx(domain=domain)))

    if not np.isclose(rve_area, RVE_SIZE**2, rtol=2.0e-3):
        raise RuntimeError(
            f"RVE {rve_id:02d}: mesh area is {rve_area:.8g}, but expected "
            f"{RVE_SIZE**2:.8g}. Check that no exterior surfaces remain."
        )

    # Voigt conventions used here:
    # strain vector: [Exx, Eyy, gamma_xy], gamma_xy = 2 Exy
    # stress vector: [Sxx, Syy, Sxy]
    elementary_strains = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=default_scalar_type),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=default_scalar_type),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=default_scalar_type),
    ]
    load_labels = ["Exx", "Eyy", "gamma_xy"]

    C_hom = np.zeros((3, 3), dtype=np.float64)

    for load_id, applied_strain in enumerate(elementary_strains):
        macro_strain.value = applied_strain

        # The same LinearProblem is reused; only the Constant macro_strain changes.
        problem.solve()
        fluctuation.x.scatter_forward()

        # Average microscopic stress for this unit macroscopic-strain load case.
        for component_id, virtual_macro_strain in enumerate(elementary_strains):
            test_macro_strain.value = virtual_macro_strain
            stress_average = assemble_global_scalar(
                fem.form(
                    ufl.inner(sigma(fluctuation), test_macro_strain)
                    * ufl.dx
                )
            ) / rve_area
            C_hom[component_id, load_id] = stress_average

        if rank == 0:
            print(f"RVE {rve_id:02d}: solved {load_labels[load_id]}")

    # Numerical asymmetry should be small for a sufficiently converged mesh/solve.
    C_hom_symmetric = 0.5 * (C_hom + C_hom.T)
    asymmetry = np.max(np.abs(C_hom - C_hom.T))

    # Compliance-derived engineering quantities for the 2D in-plane response.
    S_hom = np.linalg.inv(C_hom_symmetric)
    Exx_hom = 1.0 / S_hom[0, 0]
    Eyy_hom = 1.0 / S_hom[1, 1]
    nu_xy_hom = -S_hom[0, 1] / S_hom[0, 0]
    nu_yx_hom = -S_hom[1, 0] / S_hom[1, 1]
    Gxy_hom = 1.0 / S_hom[2, 2]

    if rank == 0:
        stiffness_file = OUTPUT_DIRECTORY / f"RVE_{rve_id:02d}_C_hom.csv"
        np.savetxt(
            stiffness_file,
            C_hom_symmetric,
            delimiter=",",
            header="Columns: Exx, Eyy, gamma_xy; Rows: Sxx, Syy, Sxy; units: Pa",
            comments=""
        )

    return {
        "rve_id": rve_id,
        "area": rve_area,
        "C11_Pa": C_hom_symmetric[0, 0],
        "C12_Pa": C_hom_symmetric[0, 1],
        "C13_Pa": C_hom_symmetric[0, 2],
        "C22_Pa": C_hom_symmetric[1, 1],
        "C23_Pa": C_hom_symmetric[1, 2],
        "C33_Pa": C_hom_symmetric[2, 2],
        "Exx_Pa": Exx_hom,
        "Eyy_Pa": Eyy_hom,
        "nu_xy": nu_xy_hom,
        "nu_yx": nu_yx_hom,
        "Gxy_Pa": Gxy_hom,
        "maximum_C_asymmetry_Pa": asymmetry,
    }


def main():
    all_results = []

    for rve_id in RVE_IDS:
        result = homogenize_one_rve(rve_id)
        all_results.append(result)

        if rank == 0:
            print(f"\nRVE {rve_id:02d} homogenized stiffness [Pa]:")
            print(
                np.array([
                    [result["C11_Pa"], result["C12_Pa"], result["C13_Pa"]],
                    [result["C12_Pa"], result["C22_Pa"], result["C23_Pa"]],
                    [result["C13_Pa"], result["C23_Pa"], result["C33_Pa"]],
                ])
            )
            print(
                f"Exx = {result['Exx_Pa'] / 1e9:.4f} GPa, "
                f"Eyy = {result['Eyy_Pa'] / 1e9:.4f} GPa, "
                f"Gxy = {result['Gxy_Pa'] / 1e9:.4f} GPa\n"
            )

    if rank == 0:
        summary_file = OUTPUT_DIRECTORY / "homogenization_summary.csv"
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        C11_values = np.array([row["C11_Pa"] for row in all_results])
        C22_values = np.array([row["C22_Pa"] for row in all_results])
        Gxy_values = np.array([row["Gxy_Pa"] for row in all_results])

        print("=" * 72)
        print(f"Results saved in: {OUTPUT_DIRECTORY}")
        print(f"Number of RVEs: {len(all_results)}")
        print(
            f"C11 mean ± sample std: {C11_values.mean() / 1e9:.4f} ± "
            f"{C11_values.std(ddof=1) / 1e9:.4f} GPa"
        )
        print(
            f"C22 mean ± sample std: {C22_values.mean() / 1e9:.4f} ± "
            f"{C22_values.std(ddof=1) / 1e9:.4f} GPa"
        )
        print(
            f"Gxy mean ± sample std: {Gxy_values.mean() / 1e9:.4f} ± "
            f"{Gxy_values.std(ddof=1) / 1e9:.4f} GPa"
        )
        print("=" * 72)


if __name__ == "__main__":
    main()
