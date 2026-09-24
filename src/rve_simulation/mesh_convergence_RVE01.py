"""Mesh-convergence study for the fixed geometry RVE_01.

This script uses the fixed reference fibre geometry stored in:
    data/rve_reference/RVE_01_fibers.csv

It creates coarse, medium, and fine meshes for exactly the same
microstructure, homogenizes each mesh with FEniCSx, and writes one
comparison table.

Run from the project root:
    python src/rve_simulation/mesh_convergence_RVE01.py

Required source modules in this directory:
    generate_periodic_fiber_rves_gmsh.py
    homogenize_fibre_RVEs_fenicsx.py

Generated meshes and convergence results are written to:
    results/mesh_convergence_RVE01/
"""


# ------------------------------ User settings ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FIBRE_CSV = (
    PROJECT_ROOT / "data" / "rve_reference" / "RVE_01_fibers.csv"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "mesh_convergence_RVE01"


# Keep polynomial degree fixed. Only mesh size changes between these cases.
MESH_LEVELS = {
    "coarse": {"H_FAR": 0.0200, "H_INTERFACE": 0.0060},
    "medium": {"H_FAR": 0.0100, "H_INTERFACE": 0.0030},
    "fine":   {"H_FAR": 0.0050, "H_INTERFACE": 0.0015},
}

# Effective stiffness entries evaluated for convergence.
CHECKED_QUANTITIES = [
    "C11_Pa", "C12_Pa", "C22_Pa", "C33_Pa",
    "Exx_Pa", "Eyy_Pa", "Gxy_Pa"
]


def read_fibre_centres(csv_file):
    """Read the already-generated RVE_01 fibre geometry without changing it."""
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_file}. Generate RVE_01_fibers.csv first."
        )

    centres = []
    radii = []

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            centres.append([
                float(row["x_center"]),
                float(row["y_center"])
            ])
            radii.append(float(row["radius"]))

    centres = np.asarray(centres, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if centres.shape != (rvegen.N_FIBERS, 2):
        raise RuntimeError(
            f"Expected {rvegen.N_FIBERS} fibre centres, but found {len(centres)}."
        )

    if not np.allclose(radii, rvegen.RADIUS):
        raise RuntimeError(
            "The radius in RVE_01_fibers.csv does not match RADIUS in the "
            "mesh-generation script. Use the same geometry settings."
        )

    return centres


def percent_change(value, reference_value):
    """Absolute percentage difference relative to the fine-mesh result."""
    return 100.0 * abs(value - reference_value) / abs(reference_value)


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # The geometry is read ONCE. It is identical for coarse, medium and fine.
    centres = read_fibre_centres(SOURCE_FIBRE_CSV)
    minimum_gap = rvegen.minimum_surface_gap(centres)

    print("=" * 72)
    print("MESH-CONVERGENCE STUDY: FIXED RVE_01 GEOMETRY")
    print(f"Source fibre geometry: {SOURCE_FIBRE_CSV}")
    print(f"Number of fibres: {len(centres)}")
    print(f"Minimum fibre surface gap: {minimum_gap:.8f}")
    print("=" * 72)

    results = []

    for level_name, mesh_settings in MESH_LEVELS.items():
        case_directory = OUTPUT_DIRECTORY / level_name
        case_directory.mkdir(parents=True, exist_ok=True)

        msh_file = case_directory / "RVE_01.msh"
        png_file = case_directory / "RVE_01.png"

        # Modify only mesh density. Fibre positions, radius, RVE size,
        # material constants and boundary conditions remain unchanged.
        rvegen.H_FAR = mesh_settings["H_FAR"]
        rvegen.H_INTERFACE = mesh_settings["H_INTERFACE"]

        print(f"\nGenerating {level_name} mesh")
        print(f"  H_FAR       = {rvegen.H_FAR}")
        print(f"  H_INTERFACE = {rvegen.H_INTERFACE}")

        # Save a geometry preview; it is identical for all mesh levels.
        rvegen.save_rve_png(
   		centres,
		rvegen.RADIUS,
   		rvegen.RVE_SIZE,
   		png_file,
  		rve_id=1,
			)
        # Create tagged periodic mesh with the fixed fibre geometry.
        rvegen.build_and_write_mesh(
            centres,
            msh_file,
            model_name=f"RVE_01_{level_name}"
        )

        # Point the existing homogenization module to this particular mesh.
        homogenization.RVE_DIRECTORY = case_directory
        homogenization.OUTPUT_DIRECTORY = case_directory / "homogenization_results"
        homogenization.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        print(f"Solving {level_name} mesh in FEniCSx")
        result = homogenization.homogenize_one_rve(1)
        result["mesh_level"] = level_name
        result["H_FAR"] = mesh_settings["H_FAR"]
        result["H_INTERFACE"] = mesh_settings["H_INTERFACE"]
        result["msh_file"] = str(msh_file)

        results.append(result)

    # Use the fine mesh as the reference solution.
    fine_result = next(row for row in results if row["mesh_level"] == "fine")

    for row in results:
        for quantity in CHECKED_QUANTITIES:
            row[f"error_vs_fine_{quantity}_percent"] = percent_change(
                row[quantity],
                fine_result[quantity]
            )

    # Write all data into one CSV file.
    summary_file = OUTPUT_DIRECTORY / "mesh_convergence_summary.csv"

    fieldnames = [
        "mesh_level", "H_FAR", "H_INTERFACE", "area",
        "C11_Pa", "C12_Pa", "C13_Pa", "C22_Pa", "C23_Pa", "C33_Pa",
        "Exx_Pa", "Eyy_Pa", "nu_xy", "nu_yx", "Gxy_Pa",
        "maximum_C_asymmetry_Pa", "msh_file"
    ] + [f"error_vs_fine_{q}_percent" for q in CHECKED_QUANTITIES]

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if homogenization.rank == 0:
        print("\n" + "=" * 72)
        print("MESH-CONVERGENCE RESULTS")
        print("=" * 72)
        print(
            "Level      H_FAR    H_INTERFACE      C11 [GPa]    "
            "C22 [GPa]    C66 [GPa]"
        )

        for row in results:
            print(
                f"{row['mesh_level']:<8} "
                f"{row['H_FAR']:<8.4f} "
                f"{row['H_INTERFACE']:<12.4f} "
                f"{row['C11_Pa'] / 1e9:<12.5f} "
                f"{row['C22_Pa'] / 1e9:<12.5f} "
                f"{row['C33_Pa'] / 1e9:<12.5f}"
            )

        medium_result = next(
            row for row in results if row["mesh_level"] == "medium"
        )

        print("\nMedium-to-fine relative differences:")

        for quantity in ["C11_Pa", "C12_Pa", "C22_Pa", "C33_Pa"]:
            error = medium_result[f"error_vs_fine_{quantity}_percent"]
            print(f"  {quantity}: {error:.4f}%")

        max_medium_error = max(
            medium_result[f"error_vs_fine_{q}_percent"]
            for q in ["C11_Pa", "C12_Pa", "C22_Pa", "C33_Pa"]
        )

        print(f"\nLargest medium-to-fine stiffness difference: {max_medium_error:.4f}%")

        if max_medium_error < 1.0:
            print(
                "Conclusion: the MEDIUM mesh is converged to within 1% "
                "relative to the fine mesh for the checked stiffness terms."
            )
        else:
            print(
                "Conclusion: medium and fine meshes differ by 1% or more. "
                "Use the fine mesh or add a finer mesh level."
            )

        print(f"\nCSV summary saved to: {summary_file}")
        print("=" * 72)


if __name__ == "__main__":
    main()
