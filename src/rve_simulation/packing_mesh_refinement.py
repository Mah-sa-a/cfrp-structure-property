"""Re-mesh SAVED V2 geometry, solve at medium, preserve original results/meshes.
Run from repository root on ONE Slurm rank per case. This does not import the V2
worker (which would regenerate coordinates); uses the existing V1 FE solver.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'results' / 'packing_study' / 'packing_LD18_v2_40_50'
H = {'medium': (0.0100, 0.0030), 'fine': (0.0050, 0.0015)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, choices=(165, 206), required=True)
    p.add_argument('--replicate', type=int, required=True)
    p.add_argument('--level', choices=H, default='medium')
    a = p.parse_args()
    import packing_study as pilot
    import rve_size_convergence_pipeline as base
    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError('Use one MPI rank, not mpirun')
    out = STUDY / 'mesh_refinement'
    out.mkdir(parents=True, exist_ok=True)
    for family in ('hard_disk', 'clustered'):
        tag = f"n{a.N}_{family}_r{a.replicate:02d}"
        old_path = STUDY / 'records' / f'{tag}.json'
        old = json.loads(old_path.read_text())
        if old['mode'] != 'full' or old['N'] != a.N:
            raise ValueError(f'Invalid full record: {tag}')
        with (ROOT / old['geometry_file']).open(newline='') as f:
            rows = list(csv.DictReader(f))
        centers = np.array([[float(x['x_center']), float(x['y_center'])] for x in rows])
        radius = float(rows[0]['radius'])
        L = float(old['L'])
        if len(centers) != a.N or abs(radius - 0.07) > 1e-9:
            raise ValueError(f'Geometry mismatch in {tag}')
        h_far, h_interface = H[a.level]
        result_path = out / f'{tag}_{a.level}.json'
        if result_path.exists():
            print(f'SKIP completed {result_path}', flush=True)
            continue
        mesh_path = out / f'{tag}_{a.level}.msh'
        base.H_FAR_FRACTION = h_far / radius
        base.H_INTERFACE_FRACTION = h_interface / radius
        base.build_mesh(centers, radius, L, mesh_path, f'{tag}_{a.level}')
        new = base.homogenize_mesh(mesh_path, L)
        if abs(new['area'] - L*L) > 0.002*L*L:
            raise ValueError('Mesh domain area mismatch')
        record = dict(N=a.N, family=family, replicate=a.replicate, level=a.level,
                      h_far=h_far, h_interface=h_interface, L=L, radius=radius,
                      source_geometry=str(old['geometry_file']), mesh=str(mesh_path.relative_to(ROOT)),
                      coarse_Exx_Pa=old['Exx_Pa'], coarse_Eyy_Pa=old['Eyy_Pa'],
                      coarse_Gxy_Pa=old['Gxy_Pa'], refined_Exx_Pa=new['Exx_Pa'],
                      refined_Eyy_Pa=new['Eyy_Pa'], refined_Gxy_Pa=new['Gxy_Pa'])
        result_path.write_text(json.dumps(record, indent=2))
        print(f'SUCCESS {tag} {a.level}: {result_path}', flush=True)
    print(f'Pair complete: N={a.N}, replicate={a.replicate}, level={a.level}', flush=True)


if __name__ == '__main__':
    main()
