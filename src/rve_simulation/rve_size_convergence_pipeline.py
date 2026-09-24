"""
End-to-end RVE-size convergence study (size = domain size at FIXED
fiber radius).

FIBER PACKING: starts from a valid RSA-seeded configuration, then
grows to the target radius using adaptive step-size bisection, with
overlap resolution at each step via Lubachevsky-Stillinger-style
elastic hard-disk collisions.

DAMPING FIX (this version): the FINAL settle phase now applies velocity
damping (LS_DAMPING < 1.0 each step), so the system actually comes to
rest instead of jiggling forever under perfectly elastic collisions.
Without damping, the final convergence check took an arbitrary
snapshot mid-motion, occasionally reporting "not converged" even for
easy, non-jamming cases (e.g. Vf=40%), triggering an unnecessary small
safety back-off (achieved ~39% instead of ~40%). Growth-stage
relaxation (while still incrementing radius) is left undamped so the
system can keep exploring/rearranging while growing; damping is only
applied once we've reached the target radius (or the practical ceiling)
and just need the configuration to genuinely settle.

Also includes:
1. MESH-GEOMETRY FIX: exact clipping of fiber disks to the square
   domain before fragmentation.
2. MEMORY-SAFETY FIX: each realization runs in its own subprocess.
3. RESUMABILITY FIX: run_combo() skips already-completed seeds and
   appends new results to disk immediately.
4. PNG divide-by-zero guard in save_rve_png().
5. FILENAME FIX: PNG/mesh tags use r01..r25 (derived from seed),
   matching the original convention.
6. COLLISION SIGN FIX: elastic collision trigger condition corrected
   (vn > 0 for approaching pairs).

Grid (must match your Slurm array's --array=0-N-1 upper bound):
    VF_LIST = [0.40, 0.50, 0.60]                -> 3 values
    L_OVER_D_LIST = [4, 8, 12, 18, 20, 24]        -> 6 values
    3 * 6 = 18 combinations -> --array=0-17

Required packages: numpy, gmsh, matplotlib, dolfinx, dolfinx_mpc,
mpi4py, petsc4py, ufl.
"""

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import numpy as np
import gmsh
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from dolfinx_mpc import MultiPointConstraint, LinearProblem

# ============================== Sweep grid ==================================
VF_LIST = [0.40, 0.50, 0.60]
L_OVER_D_LIST = [4, 8, 12, 18, 20, 24]
N_REALIZATIONS = 25
BASE_SEED = 42

R_FIXED = 0.07
MIN_SURFACE_GAP = 0.005

# ----------------------------- LS + bisection packing parameters -----------------------------
R_START_FRACTION = 0.25
INITIAL_STEP_FRACTION = 0.05
MIN_STEP_FRACTION = 0.0008
MAX_TRIALS_PER_FIBER = 20000

LS_DT = 0.01
LS_INITIAL_SPEED = 0.02
RELAX_STEPS_PER_INCREMENT = 3000
FINAL_SETTLE_STEPS = 8000
LS_POSITIONAL_CORRECTION = 0.5
LS_DAMPING = 0.98  # applied only during the final settle phase

H_FAR_FRACTION = 0.30
H_INTERFACE_FRACTION = 0.09
MESH_AREA_RTOL = 2.0e-2

MATRIX_TAG, FIBRE_TAG = 1, 2
LEFT_TAG, RIGHT_TAG, BOTTOM_TAG, TOP_TAG = 11, 12, 13, 14
TOL = 1.0e-9

E_MATRIX, NU_MATRIX = 3.0e9, 0.35
E_FIBRE, NU_FIBRE = 15.0e9, 0.30

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = PROJECT_ROOT / "results" / "RVE_size_study"
MESH_DIR = WORK_DIR / "meshes"
PNG_DIR = WORK_DIR / "png_previews"
PARTIAL_DIR = WORK_DIR / "partial_results"

comm = MPI.COMM_WORLD
rank = comm.rank

# =============================================================================
# 1. Periodic fiber packing: RSA seed + LS-style collision-based growth
# =============================================================================

def minimum_image_delta(delta, L):
    return delta - L * np.round(delta / L)


def periodic_pairwise_deltas(centres, L):
    diff = centres[:, None, :] - centres[None, :, :]
    return minimum_image_delta(diff, L)


def periodic_distance(a, b, L):
    return np.linalg.norm(minimum_image_delta(b - a, L))


def minimum_surface_gap_for(centres, radius, L):
    smallest = np.inf
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            gap = periodic_distance(centres[i], centres[j], L) - 2.0 * radius
            smallest = min(smallest, gap)
    return smallest


def rsa_seed(rng, n_fibers, r_start, min_gap, L, max_trials=MAX_TRIALS_PER_FIBER):
    """Cheap RSA at a small starting radius -- always feasible at low Vf."""
    required = 2.0 * r_start + min_gap
    centres = []
    for fibre_number in range(n_fibers):
        for _try in range(max_trials):
            cand = rng.uniform(0.0, L, size=2)
            if all(
                np.linalg.norm(minimum_image_delta(cand - c, L)) >= required
                for c in centres
            ):
                centres.append(cand)
                break
        else:
            raise RuntimeError(
                f"Seeding failed at fibre {fibre_number + 1} "
                f"(n_fibers={n_fibers}, r_start={r_start:.5f}, L={L:.5f})."
            )
    return np.asarray(centres)


def _resolve_collisions(centres, velocities, r, L, rng):
    """
    One pass of elastic hard-disk collision resolution at FIXED
    collision radius r (LS-style), plus a small positional correction
    to remove residual penetration.

    normal = -diff[i, j] / d points FROM fiber i TOWARD fiber j.
    A pair is APPROACHING (needs a collision response) when
    vn = dot(velocities[i]-velocities[j], normal) > 0.
    """
    diff = centres[:, None, :] - centres[None, :, :]
    diff = minimum_image_delta(diff, L)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    required = 2.0 * r
    n = len(centres)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dist = dist[i_idx, j_idx]
    overlapping = pair_dist < required

    if not overlapping.any():
        return centres, velocities, False

    centres = centres.copy()
    velocities = velocities.copy()

    for k in np.nonzero(overlapping)[0]:
        i, j = i_idx[k], j_idx[k]
        d = pair_dist[k]
        if d < 1e-12:
            normal = rng.normal(size=2)
            normal /= np.linalg.norm(normal) + 1e-12
        else:
            normal = -diff[i, j] / d

        rel_v = velocities[i] - velocities[j]
        vn = np.dot(rel_v, normal)
        if vn > 0.0:
            velocities[i] = velocities[i] - vn * normal
            velocities[j] = velocities[j] + vn * normal

        overlap_amount = required - d
        correction = LS_POSITIONAL_CORRECTION * 0.5 * overlap_amount
        centres[i] = centres[i] - correction * normal
        centres[j] = centres[j] + correction * normal

    centres = centres % L
    return centres, velocities, True


def relax_via_collisions(centres, velocities, r, min_gap, L, max_steps, rng, damping=1.0):
    """
    Run collision-resolution MD steps at a FIXED target collision
    radius r until no pair overlaps or max_steps is exhausted.
    damping < 1.0 shrinks velocities each step so the system can
    actually come to rest (used only in the final settle phase, so
    growth-stage exploration is unaffected). Returns
    (centres, velocities, converged).
    """
    required = 2.0 * r + min_gap
    for _ in range(max_steps):
        if damping != 1.0:
            velocities = velocities * damping
        centres = (centres + velocities * LS_DT) % L
        centres, velocities, _ = _resolve_collisions(
            centres, velocities, r + 0.5 * min_gap, L, rng)

        diff = periodic_pairwise_deltas(centres, L)
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        if dist.min() >= required - 1e-9:
            return centres, velocities, True

    diff = periodic_pairwise_deltas(centres, L)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    converged = dist.min() >= required - 1e-9
    return centres, velocities, converged


def generate_centres_LS(rng, n_fibers, target_radius, min_gap, L, verbose=False):
    """
    Grow N fibers from a valid RSA-seeded starting radius up to
    target_radius using adaptive-step bisection, resolving overlaps at
    each step via Lubachevsky-Stillinger-style elastic collisions. The
    final settle phase applies damping so the system genuinely comes
    to rest before achieved_radius is reported.
    """
    r_start = target_radius * R_START_FRACTION
    centres = rsa_seed(rng, n_fibers, r_start, min_gap, L)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_fibers)
    velocities = LS_INITIAL_SPEED * np.column_stack([np.cos(angles), np.sin(angles)])

    r_current = r_start
    step = target_radius * INITIAL_STEP_FRACTION
    min_step = target_radius * MIN_STEP_FRACTION

    while r_current < target_radius - 1e-12:
        r_trial = min(r_current + step, target_radius)
        trial_centres, trial_velocities, converged = relax_via_collisions(
            centres, velocities, r_trial, min_gap, L,
            max_steps=RELAX_STEPS_PER_INCREMENT, rng=rng, damping=1.0)
        if converged:
            centres, velocities = trial_centres, trial_velocities
            r_current = r_trial
            step = min(step * 1.3, target_radius * INITIAL_STEP_FRACTION)
            if verbose:
                print(f"  LS: r={r_current:.5f} OK, step={step:.5f}")
        else:
            step *= 0.5
            if verbose:
                print(f"  LS: r={r_trial:.5f} FAILED, halving step to {step:.5f}")
            if step < min_step:
                break

    # Final settle WITH damping: let the system actually come to rest
    # before judging whether r_current is truly achieved, instead of
    # taking an arbitrary snapshot mid-perpetual-motion.
    centres, velocities, converged = relax_via_collisions(
        centres, velocities, r_current, min_gap, L,
        max_steps=FINAL_SETTLE_STEPS, rng=rng, damping=LS_DAMPING)
    if not converged:
        r_current *= 0.995

    return centres, r_current


# =============================================================================
# 2. PNG preview
# =============================================================================

def save_rve_png(centres, radius, L, png_filename, title_extra=""):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#D6EAF8")
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_aspect("equal")

    shifts = [-L, 0.0, L]
    for x_center, y_center in centres:
        for x_shift in shifts:
            for y_shift in shifts:
                x_plot = x_center + x_shift
                y_plot = y_center + y_shift
                if (-radius <= x_plot <= L + radius and
                        -radius <= y_plot <= L + radius):
                    fibre = plt.Circle(
                        (x_plot, y_plot), radius,
                        facecolor="#E8633A", edgecolor="#7D2B0B", linewidth=0.7)
                    fibre.set_clip_on(True)
                    ax.add_patch(fibre)

    rve_boundary = plt.Rectangle(
        (0.0, 0.0), L, L,
        fill=False, edgecolor="black", linewidth=1.2)
    ax.add_patch(rve_boundary)

    vf = len(centres) * np.pi * radius ** 2 / L ** 2
    actual_gap = minimum_surface_gap_for(centres, radius, L)
    l_over_d_label = f"{L / (2 * radius):.1f}" if radius > 1e-9 else "undefined (radius~0)"

    ax.set_title(
        f"N = {len(centres)}, Vf = {vf * 100:.2f}%, L/D = {l_over_d_label}\n"
        f"radius = {radius:.5f}, min surface gap = {actual_gap:.5f}{title_extra}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(png_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 3. Gmsh periodic mesh construction (exact clipping before fragment)
# =============================================================================

def disks_intersecting_rve(centres, radius, L):
    disks = []
    shifts = (-L, 0.0, L)
    for x, y in centres:
        for sx in shifts:
            for sy in shifts:
                xc, yc = x + sx, y + sy
                if (-radius < xc < L + radius and -radius < yc < L + radius):
                    disks.append((2, gmsh.model.occ.addDisk(xc, yc, 0.0, radius, radius)))
    return disks


def boundary_curve_sets(L):
    EDGE_TOL = 1.0e-6
    left, right, bottom, top = [], [], [], []
    for _, curve_tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, curve_tag)
        if max(abs(xmin - 0.0), abs(xmax - 0.0)) < EDGE_TOL:
            left.append(curve_tag)
        elif max(abs(xmin - L), abs(xmax - L)) < EDGE_TOL:
            right.append(curve_tag)
        elif max(abs(ymin - 0.0), abs(ymax - 0.0)) < EDGE_TOL:
            bottom.append(curve_tag)
        elif max(abs(ymin - L), abs(ymax - L)) < EDGE_TOL:
            top.append(curve_tag)

    left.sort(key=lambda t: gmsh.model.occ.getCenterOfMass(1, t)[1])
    right.sort(key=lambda t: gmsh.model.occ.getCenterOfMass(1, t)[1])
    bottom.sort(key=lambda t: gmsh.model.occ.getCenterOfMass(1, t)[0])
    top.sort(key=lambda t: gmsh.model.occ.getCenterOfMass(1, t)[0])

    if not (left and right and bottom and top):
        raise RuntimeError("Could not identify all four external RVE boundaries.")
    if len(left) != len(right) or len(bottom) != len(top):
        raise RuntimeError("Boundary curve count mismatch.")
    return left, right, bottom, top


def build_mesh(centres, radius, L, msh_filename, model_name):
    h_far = H_FAR_FRACTION * radius
    h_interface = H_INTERFACE_FRACTION * radius

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.model.add(model_name)

    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)

    try:
        square = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)

        clip_tool = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
        gmsh.model.occ.synchronize()

        raw_disks = disks_intersecting_rve(centres, radius, L)
        gmsh.model.occ.synchronize()

        clipped_disks, _ = gmsh.model.occ.intersect(
            raw_disks, [(2, clip_tool)], removeObject=True, removeTool=False)
        gmsh.model.occ.remove([(2, clip_tool)], recursive=True)
        gmsh.model.occ.synchronize()

        _, fragment_map = gmsh.model.occ.fragment(
            [(2, square)], clipped_disks, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        remaining_surfaces = {tag for dim, tag in gmsh.model.getEntities(2)}
        fibre_surfaces = set()
        for descendants in fragment_map[1:]:
            for dim, surface_tag in descendants:
                if dim != 2 or surface_tag not in remaining_surfaces:
                    continue
                x_com, y_com, _ = gmsh.model.occ.getCenterOfMass(dim, surface_tag)
                if -TOL <= x_com <= L + TOL and -TOL <= y_com <= L + TOL:
                    fibre_surfaces.add(surface_tag)

        fibre_surfaces = sorted(fibre_surfaces)
        matrix_surfaces = sorted(remaining_surfaces - set(fibre_surfaces))
        if not matrix_surfaces or not fibre_surfaces:
            raise RuntimeError("Matrix or fibre surfaces missing after fragmentation.")

        gmsh.model.addPhysicalGroup(2, matrix_surfaces, MATRIX_TAG)
        gmsh.model.setPhysicalName(2, MATRIX_TAG, "matrix")
        gmsh.model.addPhysicalGroup(2, fibre_surfaces, FIBRE_TAG)
        gmsh.model.setPhysicalName(2, FIBRE_TAG, "fibre")

        left, right, bottom, top = boundary_curve_sets(L)
        gmsh.model.addPhysicalGroup(1, left, LEFT_TAG)
        gmsh.model.setPhysicalName(1, LEFT_TAG, "left")
        gmsh.model.addPhysicalGroup(1, right, RIGHT_TAG)
        gmsh.model.setPhysicalName(1, RIGHT_TAG, "right")
        gmsh.model.addPhysicalGroup(1, bottom, BOTTOM_TAG)
        gmsh.model.setPhysicalName(1, BOTTOM_TAG, "bottom")
        gmsh.model.addPhysicalGroup(1, top, TOP_TAG)
        gmsh.model.setPhysicalName(1, TOP_TAG, "top")

        affine_x = [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        affine_y = [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, right, left, affine_x)
        gmsh.model.mesh.setPeriodic(1, top, bottom, affine_y)

        fibre_boundary_curves = []
        for surface_tag in fibre_surfaces:
            boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            fibre_boundary_curves.extend(t for d, t in boundary if d == 1)
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

        num_nodes = gmsh.model.mesh.getNodes()[0].size
        _, element_tags, _ = gmsh.model.mesh.getElements(2)
        num_elements = sum(len(t) for t in element_tags)
        if num_nodes == 0 or num_elements == 0:
            raise RuntimeError("Gmsh created an empty mesh.")

        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(msh_filename))
    finally:
        gmsh.finalize()


# =============================================================================
# 4. FEniCSx periodic homogenization
# =============================================================================

def create_piecewise_constant_field(domain, cell_tags, values_by_tag, name):
    V0 = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(V0, name=name)
    for tag, value in values_by_tag.items():
        tagged_cells = cell_tags.find(tag)
        if tagged_cells.size == 0:
            raise RuntimeError(f"No cells with physical tag {tag}.")
        field.x.array[tagged_cells] = value
    field.x.scatter_forward()
    return field


def periodic_right_to_left(L):
    def _map(x):
        mapped = np.zeros_like(x)
        mapped[0] = x[0] - L
        mapped[1] = x[1]
        mapped[2] = x[2]
        return mapped
    return _map


def periodic_top_to_bottom(L):
    def _map(x):
        mapped = np.zeros_like(x)
        mapped[0] = x[0]
        mapped[1] = x[1] - L
        mapped[2] = x[2]
        return mapped
    return _map


def assemble_global_scalar(form):
    local_value = fem.assemble_scalar(form)
    return comm.allreduce(local_value, op=MPI.SUM)


def homogenize_mesh(mesh_file, L):
    mesh_data = gmshio.read_from_msh(str(mesh_file), comm, rank=0, gdim=2)
    if hasattr(mesh_data, "mesh"):
        domain, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    else:
        domain, cell_tags, facet_tags = mesh_data

    gdim = domain.geometry.dim

    young_modulus = create_piecewise_constant_field(
        domain, cell_tags, {MATRIX_TAG: E_MATRIX, FIBRE_TAG: E_FIBRE}, "E")
    poisson_ratio = create_piecewise_constant_field(
        domain, cell_tags, {MATRIX_TAG: NU_MATRIX, FIBRE_TAG: NU_FIBRE}, "nu")

    lame_lambda = (young_modulus * poisson_ratio /
                   ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)))
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))

    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    v_trial, v_test = ufl.TrialFunction(V), ufl.TestFunction(V)

    macro_strain = fem.Constant(domain, np.zeros((gdim, gdim), dtype=default_scalar_type))
    test_macro_strain = fem.Constant(domain, np.zeros((gdim, gdim), dtype=default_scalar_type))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(v):
        total_strain = macro_strain + epsilon(v)
        return (lame_lambda * ufl.tr(total_strain) * ufl.Identity(gdim) +
                2.0 * shear_modulus * total_strain)

    bilinear_form, linear_form = ufl.system(
        ufl.inner(sigma(v_trial), epsilon(v_test)) * ufl.dx)

    corner_dofs = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if corner_dofs.size == 0:
        raise RuntimeError("No DOF found at (0, 0).")
    bcs = [fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), corner_dofs, V)]

    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(
        V, facet_tags, RIGHT_TAG, periodic_right_to_left(L), bcs)
    mpc.create_periodic_constraint_topological(
        V, facet_tags, TOP_TAG, periodic_top_to_bottom(L), bcs)
    mpc.finalize()

    fluctuation = fem.Function(mpc.function_space, name="fluctuation")
    problem = LinearProblem(
        bilinear_form, linear_form, mpc, bcs=bcs, u=fluctuation,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    rve_area = assemble_global_scalar(fem.form(1.0 * ufl.dx(domain=domain)))
    if not np.isclose(rve_area, L ** 2, rtol=MESH_AREA_RTOL):
        raise RuntimeError(f"Mesh area {rve_area:.6g} != expected {L**2:.6g}.")

    elementary_strains = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=default_scalar_type),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=default_scalar_type),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=default_scalar_type),
    ]
    C_hom = np.zeros((3, 3), dtype=np.float64)

    for load_id, applied_strain in enumerate(elementary_strains):
        macro_strain.value = applied_strain
        problem.solve()
        fluctuation.x.scatter_forward()
        for component_id, virtual_strain in enumerate(elementary_strains):
            test_macro_strain.value = virtual_strain
            stress_avg = assemble_global_scalar(
                fem.form(ufl.inner(sigma(fluctuation), test_macro_strain) * ufl.dx)
            ) / rve_area
            C_hom[component_id, load_id] = stress_avg

    C_hom = 0.5 * (C_hom + C_hom.T)
    S_hom = np.linalg.inv(C_hom)
    Exx = 1.0 / S_hom[0, 0]
    Eyy = 1.0 / S_hom[1, 1]
    nu_xy = -S_hom[0, 1] / S_hom[0, 0]
    Gxy = 1.0 / S_hom[2, 2]

    return dict(Exx_Pa=Exx, Eyy_Pa=Eyy, nu_xy=nu_xy, Gxy_Pa=Gxy, area=rve_area)


# =============================================================================
# 5. One realization (pack + mesh + homogenize)
# =============================================================================

def run_one_realization(Vf_target, L_over_D, seed, mesh_dir, png_dir):
    mesh_dir = Path(mesh_dir)
    png_dir = Path(png_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    L = L_over_D * 2.0 * R_FIXED
    N = max(1, round(Vf_target * L ** 2 / (np.pi * R_FIXED ** 2)))
    rng = np.random.default_rng(seed)

    centres, achieved_radius = generate_centres_LS(
        rng, N, R_FIXED, MIN_SURFACE_GAP, L)

    realization_index = seed - BASE_SEED + 1  # 1-based, matches old r01..r25 convention
    tag = f"Vf{int(Vf_target*100):02d}_LD{L_over_D:03d}_r{realization_index:02d}"
    png_file = png_dir / f"{tag}.png"
    save_rve_png(centres, achieved_radius, L, png_file)

    msh_file = mesh_dir / f"{tag}.msh"
    build_mesh(centres, achieved_radius, L, msh_file, tag)

    result = homogenize_mesh(msh_file, L)

    row = {
        "Vf_target": Vf_target,
        "L_over_D": L_over_D,
        "L": L,
        "N_fibers": N,
        "realization": realization_index,
        "seed": seed,
        "R_fixed": R_FIXED,
        "achieved_radius": achieved_radius,
        "reached_target_radius": achieved_radius >= R_FIXED - 1e-9,
        "Vf_achieved": N * np.pi * achieved_radius ** 2 / L ** 2,
        "Exx_Pa": result["Exx_Pa"],
        "Eyy_Pa": result["Eyy_Pa"],
        "Gxy_Pa": result["Gxy_Pa"],
        "nu_xy": result["nu_xy"],
        "mesh_area": result["area"],
        "png_file": png_file.name,
    }
    return row


# =============================================================================
# 6. Resumable, memory-safe combo driver
# =============================================================================

def _load_existing_seeds(task_csv):
    if not task_csv.exists():
        return set(), None
    with open(task_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        seeds = {int(row["seed"]) for row in reader}
    return seeds, fieldnames


def _append_row(task_csv, row, fieldnames_hint):
    file_exists = task_csv.exists()
    fieldnames = fieldnames_hint or list(row.keys())
    with open(task_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_combo(Vf_target, L_over_D, task_id, output_dir=None,
              n_realizations=None, base_seed=None):
    """
    Run all realizations for one (Vf_target, L_over_D) combination.
    Resumable: skips seeds already present in task_<id>.csv. Each new
    realization runs in its own subprocess and is appended to disk
    immediately upon success.
    """
    output_dir = Path(output_dir) if output_dir else WORK_DIR
    mesh_dir = output_dir / "meshes"
    png_dir = output_dir / "png_previews"
    partial_dir = output_dir / "partial_results"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    n_realizations = n_realizations or N_REALIZATIONS
    base_seed = base_seed if base_seed is not None else BASE_SEED

    task_csv = partial_dir / f"task_{task_id:04d}.csv"
    already_seeds, fieldnames_hint = _load_existing_seeds(task_csv)
    if already_seeds:
        print(f"[task {task_id}] found {len(already_seeds)} existing realizations "
              f"in {task_csv.name} -- will skip those seeds")

    this_script = Path(__file__).resolve()
    n_done, n_skipped, n_failed = 0, 0, 0

    for realization in range(n_realizations):
        seed = base_seed + realization
        if seed in already_seeds:
            n_skipped += 1
            continue

        print(f"[task {task_id}] Vf={Vf_target}, L/D={L_over_D}, "
              f"realization {realization + 1}/{n_realizations} (seed={seed}) "
              f"-- launching subprocess")

        cmd = [
            sys.executable, str(this_script), "--_worker",
            "--Vf", str(Vf_target), "--L_over_D", str(L_over_D),
            "--seed", str(seed),
            "--mesh_dir", str(mesh_dir), "--png_dir", str(png_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"[task {task_id}] SKIPPING seed={seed}: subprocess exited "
                  f"with code {proc.returncode}")
            print(proc.stderr[-4000:])
            n_failed += 1
            continue

        result_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:")),
            None)
        if result_line is None:
            print(f"[task {task_id}] SKIPPING seed={seed}: no RESULT_JSON line found.")
            n_failed += 1
            continue

        row = json.loads(result_line[len("RESULT_JSON:"):])
        _append_row(task_csv, row, fieldnames_hint)
        if fieldnames_hint is None:
            fieldnames_hint = list(row.keys())
        n_done += 1
        print(f"[task {task_id}] seed={seed} DONE and saved -> {task_csv.name} "
              f"(Vf_achieved={row['Vf_achieved']*100:.2f}%, "
              f"reached_target={row['reached_target_radius']})")

    print(f"[task {task_id}] finished this run: {n_done} newly completed, "
          f"{n_skipped} already done, {n_failed} failed/skipped this pass.")


def _run_realization_worker(Vf_target, L_over_D, seed, mesh_dir, png_dir):
    """Executed inside a fresh subprocess for exactly ONE realization."""
    import traceback
    try:
        row = run_one_realization(Vf_target, L_over_D, seed, mesh_dir, png_dir)
        print("RESULT_JSON:" + json.dumps(row))
    except Exception:
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# 7. Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RVE size-convergence pipeline.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--Vf", type=float)
    parser.add_argument("--L_over_D", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--mesh_dir", type=str, default=str(MESH_DIR))
    parser.add_argument("--png_dir", type=str, default=str(PNG_DIR))
    parser.add_argument("--output_dir", type=str, default=str(WORK_DIR))
    args = parser.parse_args()

    if args._worker:
        _run_realization_worker(args.Vf, args.L_over_D, args.seed, args.mesh_dir, args.png_dir)
    elif args.Vf is not None and args.L_over_D is not None and args.task_id is not None:
        run_combo(args.Vf, args.L_over_D, args.task_id, args.output_dir)
    else:
        parser.error("Provide either --_worker (internal use) or "
                      "--Vf/--L_over_D/--task_id to run one combination.")


if __name__ == "__main__":
    main()
