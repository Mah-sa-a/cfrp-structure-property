"""Single-task packing sensitivity study, compatible with existing RVE size pipeline.
Run from repository root: python src/rve_simulation/packing_study.py --config configs/packing_study.json --task-id 0
Generate geometry-only preflight: add --geometry-only. For a manifest use --list.
Single MPI rank only. Does not modify the existing size-study pipeline or outputs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    c = json.loads(raw)
    if int(c["L_over_D"]) != 18:
        raise ValueError("This experiment is defined at L/D=18.")
    if not c["vf_targets"] or not set(c["families"]).issubset({"hard_disk", "ordered_square", "clustered"}):
        raise ValueError("Invalid families or empty vf_targets")
    if not (0 < c["radius"] and c["min_surface_gap"] >= 0 and 0 < c["max_vf_error"] < 1):
        raise ValueError("Invalid radius/gap/vf tolerance")
    if int(c["n_realizations"]) < 1:
        raise ValueError("n_realizations must be >= 1")
    return c, hashlib.sha256(raw).hexdigest(), path


def tasks(c):
    out = []
    for vf in c["vf_targets"]:
        for family in c["families"]:
            for rep in range(1 if family == "ordered_square" else int(c["n_realizations"])):
                out.append(dict(task_id=len(out), family=family, vf_target=float(vf), replicate=rep,
                                seed=int(c["base_seed"]) + rep))
    return out


def geometry(c, row):
    import rve_size_convergence_pipeline as base
    r = float(c["radius"])
    L = float(c["L_over_D"]) * 2 * r
    n = max(1, round(float(row["vf_target"]) * L * L / (np.pi * r * r)))
    rng = np.random.default_rng(row["seed"])
    family = row["family"]
    if family == "ordered_square":
        # Commensurate square lattice: m^2 fibres and a fixed radius; achieved Vf is checked below.
        m = round(np.sqrt(n))
        n = m * m
        a = L / m
        xy = (np.arange(m) + 0.5) * a
        centers = np.array([(x, y) for x in xy for y in xy], dtype=float)
        if a < 2*r + float(c["min_surface_gap"]) - 1e-9:
            raise ValueError("Ordered lattice violates minimum gap")
    else:
        centers, achieved_r = base.generate_centres_LS(rng, n, r, float(c["min_surface_gap"]), L)
        if abs(achieved_r - r) > 1e-9:
            raise ValueError(f"Hard-disc generator reached radius {achieved_r:.7g}, not {r}; no radius shrink permitted")
        if family == "clustered":
            centers = cluster_centers(centers, r, L, float(c["min_surface_gap"]), c, rng)
    vf_actual = len(centers) * np.pi * r*r / (L*L)
    if abs(vf_actual - float(row["vf_target"])) > float(c["max_vf_error"]):
        raise ValueError(f"Actual Vf={vf_actual:.6f}, target={row['vf_target']:.6f}. Commensurate lattice or N choice fails tolerance; report as infeasible, do not adjust radius")
    desc, gaps = descriptors(centers, r, L, float(c["min_surface_gap"]))
    return centers, gaps, desc, dict(L=L, n_fibers=len(centers), radius=r, vf_achieved=vf_actual)


def distances(centers, L):
    v = centers[:, None, :] - centers[None, :, :]
    v -= L * np.round(v / L)
    d = np.linalg.norm(v, axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def descriptors(centers, r, L, gap_required):
    d = distances(centers, L)
    gaps = np.min(d, axis=1) - 2*r
    if not np.isfinite(gaps).all() or gaps.min() < gap_required - 1e-7:
        raise ValueError(f"Geometry has overlap or insufficient gap: min={gaps.min():.8g}")
    ng = gaps / (2*r)
    # Periodic square windows: center-count variation is a screening descriptor, not phase area Vf.
    m = 4
    hist = np.histogram2d(centers[:, 0], centers[:, 1], bins=m,
                          range=[[0, L], [0, L]])[0]
    return dict(min_gap=float(gaps.min()), mean_gap_over_D=float(ng.mean()),
                p10_gap_over_D=float(np.quantile(ng, .1)),
                p50_gap_over_D=float(np.quantile(ng, .5)),
                p90_gap_over_D=float(np.quantile(ng, .9)),
                close_gap_fraction=float(np.mean(gaps < 0.25*r)),
                window_count_cv=float(hist.std() / hist.mean())), gaps


def cluster_centers(centers, r, L, min_gap, c, rng):
    """Conservative Monte-Carlo displacement toward periodic cluster anchors.
    An anchor-attraction heuristic, NOT a calibrated physical point process.
    """
    centers = centers.copy()
    anchors = rng.uniform(0, L, (max(2, round(L / (float(c["cluster_background_radius_in_radii"])*r))**2 // 4), 2))
    cutoff = 2*r + min_gap
    step = float(c["cluster_step_in_radii"]) * r
    strength = float(c["cluster_strength"])
    for _ in range(int(c["cluster_iterations"])):
        i = int(rng.integers(len(centers)))
        delta = anchors - centers[i]
        delta -= L*np.round(delta/L)
        target = delta[np.argmin(np.sum(delta*delta, axis=1))]
        proposal = (centers[i] + strength * step * target / (np.linalg.norm(target) + 1e-12)
                    + rng.normal(scale=0.15*step, size=2)) % L
        other = np.concatenate((centers[:i], centers[i+1:]))
        diff = other - proposal
        diff -= L*np.round(diff/L)
        if np.min(np.linalg.norm(diff, axis=1)) >= cutoff + 1e-9:
            centers[i] = proposal
    return centers


def save_geometry(path, centers, gaps, r):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fibre_id", "x_center", "y_center", "radius", "nn_surface_gap"])
        for i, ((x, y), gap) in enumerate(zip(centers, gaps), 1):
            w.writerow([i, x, y, r, gap])


def solve(c, centers, geom, mesh_path, preview_path, tag):
    # Existing checked mesh implementation is reused, but mesh size changes are local
    # to this single task's process; size-study source files are never edited.
    import rve_size_convergence_pipeline as base
    import gmsh
    from dolfinx import fem, default_scalar_type
    from dolfinx.io import gmshio
    from dolfinx_mpc import MultiPointConstraint, LinearProblem
    from mpi4py import MPI
    import ufl

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run one MPI rank per array task; no mpirun")
    base.H_FAR_FRACTION = float(c["h_far_over_r"])
    base.H_INTERFACE_FRACTION = float(c["h_interface_over_r"])
    base.build_mesh(centers, geom["radius"], geom["L"], mesh_path, tag)
    if bool(c["save_previews"]):
        base.save_rve_png(centers, geom["radius"], geom["L"], preview_path)

    mesh_data = gmshio.read_from_msh(str(mesh_path), MPI.COMM_WORLD, rank=0, gdim=2)
    if hasattr(mesh_data, "mesh"):
        domain, ct, ft = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    else:
        domain, ct, ft = mesh_data
    if ct is None or ft is None:
        raise RuntimeError("Mesh has no physical tags")
    for label, t in [("matrix", 1), ("fibre", 2)]:
        if ct.find(t).size == 0:
            raise RuntimeError(f"Missing phase {label}")
    for t in (11, 12, 13, 14):
        if ft.find(t).size == 0:
            raise RuntimeError(f"Missing periodic facet tag {t}")
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    area = float(fem.assemble_scalar(fem.form(1.0*ufl.dx(domain=domain))))
    fibre_area = float(fem.assemble_scalar(fem.form(1.0*dx(2))))
    if abs(area / geom["L"]**2 - 1) > float(c["mesh_area_rtol"]):
        raise ValueError(f"Mesh area discrepancy: {area}")
    if abs(fibre_area / area - geom["vf_achieved"]) > float(c["phase_fraction_tolerance"]):
        raise ValueError(f"Mesh phase fraction={fibre_area/area}, geometry Vf={geom['vf_achieved']}")
    E = base.create_piecewise_constant_field(domain, ct,
        {1: float(c["e_matrix_Pa"]), 2: float(c["e_fibre_Pa"])}, "E")
    nu = base.create_piecewise_constant_field(domain, ct,
        {1: float(c["nu_matrix"]), 2: float(c["nu_fibre"])}, "nu")
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    macro = fem.Constant(domain, np.zeros((2, 2), dtype=default_scalar_type))
    unit = fem.Constant(domain, np.zeros((2, 2), dtype=default_scalar_type))
    def stress(v):
        e = macro + ufl.sym(ufl.grad(v))
        return lam*ufl.tr(e)*ufl.Identity(2) + 2*mu*e
    a, b = ufl.system(ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx)
    corner = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    if corner.size == 0:
        raise RuntimeError("Missing origin dof")
    bcs = [fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), corner, V)]
    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, ft, 12, base.periodic_right_to_left(geom["L"]), bcs)
    mpc.create_periodic_constraint_topological(V, ft, 14, base.periodic_top_to_bottom(geom["L"]), bcs)
    mpc.finalize()
    v = fem.Function(mpc.function_space)
    problem = LinearProblem(a, b, mpc, bcs=bcs, u=v,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    strains = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]]),
               np.array([[0., .5], [.5, 0.]])]
    C = np.zeros((3, 3))
    for j, eps in enumerate(strains):
        macro.value = np.asarray(eps, dtype=default_scalar_type)
        problem.solve()
        v.x.scatter_forward()
        for i, virtual in enumerate(strains):
            unit.value = np.asarray(virtual, dtype=default_scalar_type)
            C[i, j] = float(fem.assemble_scalar(fem.form(ufl.inner(stress(v), unit)*ufl.dx))) / area
    asym = float(np.max(np.abs(C - C.T)) / np.max(np.abs(C)))
    if asym > float(c["maximum_relative_C_asymmetry"]):
        raise ValueError(f"Relative tensor asymmetry too large: {asym}")
    C = (C+C.T)/2
    S = np.linalg.inv(C)
    return dict(C11_Pa=C[0, 0], C12_Pa=C[0, 1], C13_Pa=C[0, 2],
                C22_Pa=C[1, 1], C23_Pa=C[1, 2], C33_Pa=C[2, 2],
                Exx_Pa=1/S[0, 0], Eyy_Pa=1/S[1, 1], Gxy_Pa=1/S[2, 2],
                nu_xy=-S[0, 1]/S[0, 0],
                relative_C_asymmetry=asym, mesh_area=area, mesh_vf=fibre_area/area)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "packing_study.json"))
    p.add_argument("--task-id", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--geometry-only", action="store_true")
    args = p.parse_args()
    c, digest, path = load_config(args.config)
    manifest = tasks(c)
    if args.list:
        for row in manifest:
            print(json.dumps(row))
        print(f"TASK_COUNT={len(manifest)}")
        return
    if args.task_id is None or args.task_id < 0 or args.task_id >= len(manifest):
        p.error(f"--task-id must be in 0..{len(manifest)-1}")
    row = manifest[args.task_id]
    tag = f"vf{round(row['vf_target']*100):02d}_{row['family']}_r{row['replicate']:02d}"
    out = ROOT / "results" / "packing_study" / c["study_id"]
    for folder in ("records", "geometry", "meshes", "previews", "failures"):
        (out/folder).mkdir(parents=True, exist_ok=True)
    record_path = out/"records"/f"{tag}.json"
    failure_path = out/"failures"/f"{tag}.txt"
    if record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing.get("config_sha256") != digest:
            raise RuntimeError("Existing result was made with a different config. Use a new study_id.")
        if not args.geometry_only or existing.get("mode") == "full":
            print(f"SKIP valid completed result: {record_path}")
            return
    try:
        centers, gaps, desc, geom = geometry(c, row)
        csv_path = out/"geometry"/f"{tag}.csv"
        save_geometry(csv_path, centers, gaps, geom["radius"])
        result = dict(**row, **geom, **desc, config_sha256=digest, study_id=c["study_id"],
                      geometry_file=str(csv_path.relative_to(ROOT)), mode="geometry_only" if args.geometry_only else "full")
        if not args.geometry_only:
            msh = out/"meshes"/f"{tag}.msh"
            png = out/"previews"/f"{tag}.png"
            result.update(solve(c, centers, geom, msh, png, tag))
            if not c["save_meshes"]:
                msh.unlink(missing_ok=True)
        tmp = record_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, default=lambda x: float(x)))
        os.replace(tmp, record_path)
        failure_path.unlink(missing_ok=True)
        print(f"SUCCESS {tag} Vf={geom['vf_achieved']:.6f} -> {record_path}")
    except Exception:
        failure_path.write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
