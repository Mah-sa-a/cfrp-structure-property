"""Generate periodic 2D circular-fibre RVEs and Gmsh meshes.

This module uses RSA seeding followed by adaptive fibre growth and
periodic collision relaxation. It is intended for moderate to high fibre
area fractions where direct random sequential adsorption alone may jam.

Outputs for each realization:
    RVE_XX_fibers.csv : fibre centres and achieved radius
    RVE_XX.png        : periodic geometry preview
    RVE_XX.msh        : tagged periodic Gmsh mesh

The mesh physical tags are:
    1  matrix cells
    2  fibre cells
    11 left boundary facets
    12 right boundary facets
    13 bottom boundary facets
    14 top boundary facets

Run inside an environment with numpy, gmsh, and matplotlib:
    python generate_periodic_fiber_rves_gmsh.py
"""

from pathlib import Path
import csv

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================== User parameters ==============================
RVE_SIZE = 1.0
N_FIBERS = 30
RADIUS = 0.07
MIN_SURFACE_GAP = 0.005
N_RVES = 10
BASE_SEED = 42
OUTPUT_DIR = Path("RVE_realizations")

# Mesh sizes in the same normalized units as RVE_SIZE.
H_FAR = 0.020
H_INTERFACE = 0.006

# RSA seed + adaptive growth / collision-relaxation parameters.
R_START_FRACTION = 0.25
INITIAL_STEP_FRACTION = 0.05
MIN_STEP_FRACTION = 0.0008
MAX_TRIALS_PER_FIBER = 20000
LS_DT = 0.01
LS_INITIAL_SPEED = 0.02
RELAX_STEPS_PER_INCREMENT = 3000
FINAL_SETTLE_STEPS = 8000
LS_POSITIONAL_CORRECTION = 0.5
LS_DAMPING = 0.98

# Gmsh physical tags.
MATRIX_TAG = 1
FIBRE_TAG = 2
LEFT_TAG = 11
RIGHT_TAG = 12
BOTTOM_TAG = 13
TOP_TAG = 14
TOL = 1.0e-9


def minimum_image_delta(delta, length):
    """Return the minimum-image periodic displacement."""
    return delta - length * np.round(delta / length)


def periodic_distance(point_a, point_b, length):
    """Distance between two points under square periodic boundary conditions."""
    return np.linalg.norm(minimum_image_delta(point_b - point_a, length))


def periodic_pairwise_deltas(centres, length):
    """Pairwise minimum-image displacement array for all fibre centres."""
    differences = centres[:, None, :] - centres[None, :, :]
    return minimum_image_delta(differences, length)


def minimum_surface_gap(centres, radius=None, length=None):
    """Smallest periodic surface-to-surface gap between fibres."""
    radius = RADIUS if radius is None else radius
    length = RVE_SIZE if length is None else length
    if len(centres) < 2:
        return np.inf

    smallest_gap = np.inf
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            gap = periodic_distance(centres[i], centres[j], length) - 2.0 * radius
            smallest_gap = min(smallest_gap, gap)
    return smallest_gap


def rsa_seed(rng, n_fibers, start_radius, min_gap, length):
    """Generate a feasible low-radius initial configuration by RSA."""
    required_distance = 2.0 * start_radius + min_gap
    centres = []

    for fibre_id in range(n_fibers):
        for _ in range(MAX_TRIALS_PER_FIBER):
            candidate = rng.uniform(0.0, length, size=2)
            if all(
                periodic_distance(candidate, centre, length) >= required_distance
                for centre in centres
            ):
                centres.append(candidate)
                break
        else:
            raise RuntimeError(
                f"RSA seeding failed at fibre {fibre_id + 1} of {n_fibers}. "
                f"Try fewer fibres, a smaller target radius, or a smaller gap."
            )

    return np.asarray(centres, dtype=float)


def resolve_collisions(centres, velocities, collision_radius, length, rng):
    """Perform one periodic hard-disk collision/overlap-correction pass."""
    differences = periodic_pairwise_deltas(centres, length)
    distances = np.linalg.norm(differences, axis=-1)
    np.fill_diagonal(distances, np.inf)

    required_distance = 2.0 * collision_radius
    row_ids, column_ids = np.triu_indices(len(centres), k=1)
    pair_distances = distances[row_ids, column_ids]
    overlap_mask = pair_distances < required_distance

    if not overlap_mask.any():
        return centres, velocities, False

    updated_centres = centres.copy()
    updated_velocities = velocities.copy()

    for index in np.nonzero(overlap_mask)[0]:
        i = row_ids[index]
        j = column_ids[index]
        distance = pair_distances[index]

        if distance < 1.0e-12:
            normal = rng.normal(size=2)
            normal /= np.linalg.norm(normal) + 1.0e-12
        else:
            normal = -differences[i, j] / distance

        relative_velocity = updated_velocities[i] - updated_velocities[j]
        normal_velocity = np.dot(relative_velocity, normal)

        if normal_velocity > 0.0:
            updated_velocities[i] -= normal_velocity * normal
            updated_velocities[j] += normal_velocity * normal

        overlap = required_distance - distance
        correction = LS_POSITIONAL_CORRECTION * 0.5 * overlap
        updated_centres[i] -= correction * normal
        updated_centres[j] += correction * normal

    updated_centres %= length
    return updated_centres, updated_velocities, True


def relax_via_collisions(centres, velocities, radius, min_gap, length, max_steps, rng, damping=1.0):
    """Relax periodic fibre overlaps at a fixed radius."""
    required_distance = 2.0 * radius + min_gap

    for _ in range(max_steps):
        if damping != 1.0:
            velocities *= damping

        centres = (centres + velocities * LS_DT) % length
        centres, velocities, _ = resolve_collisions(
            centres, velocities, radius + 0.5 * min_gap, length, rng
        )

        distances = np.linalg.norm(periodic_pairwise_deltas(centres, length), axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() >= required_distance - TOL:
            return centres, velocities, True

    distances = np.linalg.norm(periodic_pairwise_deltas(centres, length), axis=-1)
    np.fill_diagonal(distances, np.inf)
    converged = distances.min() >= required_distance - TOL
    return centres, velocities, converged


def generate_centres_LS(rng, n_fibers, target_radius, min_gap, length, verbose=False):
    """Generate periodic fibre centres using RSA seed plus adaptive growth.

    Returns the fibre centres and the radius actually reached. The returned
    radius can be slightly smaller than target_radius if the configuration
    cannot be relaxed at the requested packing density.
    """
    start_radius = target_radius * R_START_FRACTION
    centres = rsa_seed(rng, n_fibers, start_radius, min_gap, length)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_fibers)
    velocities = LS_INITIAL_SPEED * np.column_stack((np.cos(angles), np.sin(angles)))

    current_radius = start_radius
    step = target_radius * INITIAL_STEP_FRACTION
    minimum_step = target_radius * MIN_STEP_FRACTION

    while current_radius < target_radius - TOL:
        trial_radius = min(current_radius + step, target_radius)
        trial_centres, trial_velocities, converged = relax_via_collisions(
            centres=centres,
            velocities=velocities,
            radius=trial_radius,
            min_gap=min_gap,
            length=length,
            max_steps=RELAX_STEPS_PER_INCREMENT,
            rng=rng,
            damping=1.0,
        )

        if converged:
            centres = trial_centres
            velocities = trial_velocities
            current_radius = trial_radius
            step = min(step * 1.3, target_radius * INITIAL_STEP_FRACTION)
            if verbose:
                print(f"Growth accepted: radius={current_radius:.6f}, step={step:.6f}")
        else:
            step *= 0.5
            if verbose:
                print(f"Growth rejected: radius={trial_radius:.6f}, step={step:.6f}")
            if step < minimum_step:
                break

    centres, velocities, converged = relax_via_collisions(
        centres=centres,
        velocities=velocities,
        radius=current_radius,
        min_gap=min_gap,
        length=length,
        max_steps=FINAL_SETTLE_STEPS,
        rng=rng,
        damping=LS_DAMPING,
    )

    if not converged:
        current_radius *= 0.995

    return centres, current_radius


def write_fibre_csv(centres, radius, csv_filename):
    """Write fibre centres and achieved radius to a CSV file."""
    with open(csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fibre_id", "x_center", "y_center", "radius"])
        for fibre_id, (x_center, y_center) in enumerate(centres, start=1):
            writer.writerow([fibre_id, x_center, y_center, radius])


def save_rve_png(centres, radius, length, png_filename, rve_id=None):
    """Save a periodic RVE preview clipped to the physical square domain."""
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.set_facecolor("#D6EAF8")
    axis.set_xlim(0.0, length)
    axis.set_ylim(0.0, length)
    axis.set_aspect("equal")

    for x_center, y_center in centres:
        for x_shift in (-length, 0.0, length):
            for y_shift in (-length, 0.0, length):
                x_plot = x_center + x_shift
                y_plot = y_center + y_shift
                if -radius <= x_plot <= length + radius and -radius <= y_plot <= length + radius:
                    fibre = plt.Circle(
                        (x_plot, y_plot),
                        radius,
                        facecolor="#E8633A",
                        edgecolor="#7D2B0B",
                        linewidth=0.7,
                    )
                    fibre.set_clip_on(True)
                    axis.add_patch(fibre)

    boundary = plt.Rectangle((0.0, 0.0), length, length, fill=False, edgecolor="black", linewidth=1.2)
    axis.add_patch(boundary)

    vf = len(centres) * np.pi * radius ** 2 / length ** 2
    gap = minimum_surface_gap(centres, radius, length)
    title = f"N = {len(centres)}, Vf = {vf * 100:.2f}%"
    if rve_id is not None:
        title = f"RVE {rve_id:02d} | {title}"
    axis.set_title(f"{title}\nradius = {radius:.5f}, min surface gap = {gap:.5f}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def disks_intersecting_rve(centres, radius, length):
    """Create disks and periodic images that intersect the square RVE."""
    disks = []
    for x_center, y_center in centres:
        for x_shift in (-length, 0.0, length):
            for y_shift in (-length, 0.0, length):
                x_disk = x_center + x_shift
                y_disk = y_center + y_shift
                if -radius < x_disk < length + radius and -radius < y_disk < length + radius:
                    disks.append((2, gmsh.model.occ.addDisk(x_disk, y_disk, 0.0, radius, radius)))
    return disks


def boundary_curve_sets(length):
    """Identify and consistently sort external boundary fragments."""
    edge_tolerance = 1.0e-6
    left, right, bottom, top = [], [], [], []

    for _, curve_tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve_tag)
        if max(abs(xmin), abs(xmax)) < edge_tolerance:
            left.append(curve_tag)
        elif max(abs(xmin - length), abs(xmax - length)) < edge_tolerance:
            right.append(curve_tag)
        elif max(abs(ymin), abs(ymax)) < edge_tolerance:
            bottom.append(curve_tag)
        elif max(abs(ymin - length), abs(ymax - length)) < edge_tolerance:
            top.append(curve_tag)

    left.sort(key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[1])
    right.sort(key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[1])
    bottom.sort(key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[0])
    top.sort(key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[0])

    if not (left and right and bottom and top):
        raise RuntimeError("Could not identify all four external RVE boundaries.")
    if len(left) != len(right) or len(bottom) != len(top):
        raise RuntimeError("Opposite boundary fragment counts do not match.")

    return left, right, bottom, top


def build_and_write_mesh(centres, msh_filename, model_name, radius=None, length=None):
    """Build and save a periodic, tagged Gmsh mesh from fibre centres."""
    radius = RADIUS if radius is None else radius
    length = RVE_SIZE if length is None else length
    h_far = H_FAR
    h_interface = H_INTERFACE

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.model.add(model_name)

    try:
        square = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, length, length)
        clip_tool = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, length, length)
        gmsh.model.occ.synchronize()

        raw_disks = disks_intersecting_rve(centres, radius, length)
        gmsh.model.occ.synchronize()

        clipped_disks, _ = gmsh.model.occ.intersect(
            raw_disks,
            [(2, clip_tool)],
            removeObject=True,
            removeTool=False,
        )
        gmsh.model.occ.remove([(2, clip_tool)], recursive=True)
        gmsh.model.occ.synchronize()

        _, fragment_map = gmsh.model.occ.fragment(
            [(2, square)],
            clipped_disks,
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        remaining_surfaces = {tag for dimension, tag in gmsh.model.getEntities(2)}
        fibre_surfaces = set()
        for descendants in fragment_map[1:]:
            for dimension, surface_tag in descendants:
                if dimension != 2 or surface_tag not in remaining_surfaces:
                    continue
                x_center, y_center, _ = gmsh.model.occ.getCenterOfMass(dimension, surface_tag)
                if -TOL <= x_center <= length + TOL and -TOL <= y_center <= length + TOL:
                    fibre_surfaces.add(surface_tag)

        fibre_surfaces = sorted(fibre_surfaces)
        matrix_surfaces = sorted(remaining_surfaces - set(fibre_surfaces))
        if not matrix_surfaces or not fibre_surfaces:
            raise RuntimeError("Matrix or fibre surfaces are missing after geometry fragmentation.")

        gmsh.model.addPhysicalGroup(2, matrix_surfaces, MATRIX_TAG)
        gmsh.model.setPhysicalName(2, MATRIX_TAG, "matrix")
        gmsh.model.addPhysicalGroup(2, fibre_surfaces, FIBRE_TAG)
        gmsh.model.setPhysicalName(2, FIBRE_TAG, "fibre")

        left, right, bottom, top = boundary_curve_sets(length)
        gmsh.model.addPhysicalGroup(1, left, LEFT_TAG)
        gmsh.model.setPhysicalName(1, LEFT_TAG, "left")
        gmsh.model.addPhysicalGroup(1, right, RIGHT_TAG)
        gmsh.model.setPhysicalName(1, RIGHT_TAG, "right")
        gmsh.model.addPhysicalGroup(1, bottom, BOTTOM_TAG)
        gmsh.model.setPhysicalName(1, BOTTOM_TAG, "bottom")
        gmsh.model.addPhysicalGroup(1, top, TOP_TAG)
        gmsh.model.setPhysicalName(1, TOP_TAG, "top")

        affine_x = [1, 0, 0, length, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        affine_y = [1, 0, 0, 0, 0, 1, 0, length, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, right, left, affine_x)
        gmsh.model.mesh.setPeriodic(1, top, bottom, affine_y)

        fibre_boundary_curves = []
        for surface_tag in fibre_surfaces:
            boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            fibre_boundary_curves.extend(tag for dimension, tag in boundary if dimension == 1)
        fibre_boundary_curves = sorted(set(fibre_boundary_curves))

        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", fibre_boundary_curves)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", h_interface)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 3.0 * radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_count = gmsh.model.mesh.getNodes()[0].size
        _, element_tags, _ = gmsh.model.mesh.getElements(2)
        element_count = sum(len(tags) for tags in element_tags)
        if node_count == 0 or element_count == 0:
            raise RuntimeError("Gmsh created an empty mesh.")

        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(msh_filename))
        print(f"Mesh saved: {msh_filename} | nodes={node_count}, elements={element_count}")
    finally:
        gmsh.finalize()


def main():
    """Generate N_RVES periodic RVEs, previews, meshes, and a summary table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for rve_id in range(1, N_RVES + 1):
        seed = BASE_SEED + rve_id - 1
        rng = np.random.default_rng(seed)
        centres, achieved_radius = generate_centres_LS(
            rng=rng,
            n_fibers=N_FIBERS,
            target_radius=RADIUS,
            min_gap=MIN_SURFACE_GAP,
            length=RVE_SIZE,
        )

        csv_file = OUTPUT_DIR / f"RVE_{rve_id:02d}_fibers.csv"
        png_file = OUTPUT_DIR / f"RVE_{rve_id:02d}.png"
        msh_file = OUTPUT_DIR / f"RVE_{rve_id:02d}.msh"

        write_fibre_csv(centres, achieved_radius, csv_file)
        save_rve_png(centres, achieved_radius, RVE_SIZE, png_file, rve_id=rve_id)
        build_and_write_mesh(
            centres=centres,
            msh_filename=msh_file,
            model_name=f"RVE_{rve_id:02d}",
            radius=achieved_radius,
            length=RVE_SIZE,
        )

        achieved_vf = N_FIBERS * np.pi * achieved_radius ** 2 / RVE_SIZE ** 2
        actual_gap = minimum_surface_gap(centres, achieved_radius, RVE_SIZE)
        summary_rows.append(
            {
                "rve_id": rve_id,
                "seed": seed,
                "n_fibres": N_FIBERS,
                "target_radius": RADIUS,
                "achieved_radius": achieved_radius,
                "reached_target_radius": achieved_radius >= RADIUS - TOL,
                "rve_size": RVE_SIZE,
                "vf_target": N_FIBERS * np.pi * RADIUS ** 2 / RVE_SIZE ** 2,
                "vf_achieved": achieved_vf,
                "requested_min_surface_gap": MIN_SURFACE_GAP,
                "actual_min_surface_gap": actual_gap,
                "csv_file": csv_file.name,
                "png_file": png_file.name,
                "msh_file": msh_file.name,
            }
        )
        print(
            f"RVE {rve_id:02d}: Vf_achieved={achieved_vf * 100:.2f}%, "
            f"radius={achieved_radius:.6f}, min_gap={actual_gap:.6f}"
        )

    summary_file = OUTPUT_DIR / "RVE_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Generated {N_RVES} RVEs in: {OUTPUT_DIR.resolve()}")
    print(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
