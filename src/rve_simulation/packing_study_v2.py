"""V2 packing study, fixed L/D=18 and N-matched hard-disc versus clustered.
Uses the tested FE solve of packing_study.py and existing RVE-size generator.
Run from repository root: python src/rve_simulation/packing_study_v2.py --list
One rank per task; pilot data remain in an untouched separate study_id.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import packing_study as v1

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    raw = Path(path).resolve().read_bytes()
    c = json.loads(raw)
    if c["L_over_D"] != 18:
        raise ValueError("Selected L/D=18; clustered size dependence requires a separate study")
    if c.get("run_vf60"):
        raise ValueError("60% requires a separate packing-feasibility screen")
    if not c["fibre_counts"] or min(c["fibre_counts"]) < 2:
        raise ValueError("Invalid fibre counts")
    if set(c["families"]) != {"hard_disk", "clustered"}:
        raise ValueError("V2 main comparison requires hard_disk and clustered only")
    if int(c["n_realizations"]) < 1:
        raise ValueError("n_realizations must be positive")
    return c, hashlib.sha256(raw).hexdigest()


def tasks(c):
    rows=[]
    for N in c["fibre_counts"]:
        for family in c["families"]:
            for rep in range(int(c["n_realizations"])):
                rows.append(dict(task_id=len(rows), N=int(N), family=family, replicate=rep,
                                 seed=int(c["base_seed"])+rep))
    return rows


def generate(c,row):
    import rve_size_convergence_pipeline as base
    r=float(c["radius"])
    L=float(c["L_over_D"])*2*r
    N=int(row["N"])
    rng=np.random.default_rng(row["seed"])
    centres,achieved=base.generate_centres_LS(rng,N,r,float(c["min_surface_gap"]),L)
    if abs(achieved-r)>1e-9:
        raise ValueError(f"LS only achieved radius {achieved:.8g} versus {r}; do not compare unequal Vf")
    if row["family"]=="clustered":
        centres=v1.cluster_centers(centres,r,L,float(c["min_surface_gap"]),c,rng)
    desc,gaps=v1.descriptors(centres,r,L,float(c["min_surface_gap"]))
    vf=N*np.pi*r*r/L**2
    return centres,gaps,desc,dict(L=L,radius=r,n_fibers=N,vf_achieved=vf,vf_target=vf)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(ROOT/"configs"/"packing_study_v2.json"))
    p.add_argument("--task-id",type=int)
    p.add_argument("--list",action="store_true")
    p.add_argument("--geometry-only",action="store_true")
    a=p.parse_args()
    c,digest=load(a.config)
    manifest=tasks(c)
    if a.list:
        for r in manifest:
            print(json.dumps(r))
        print(f"TASK_COUNT={len(manifest)}")
        return
    if a.task_id is None or not 0<=a.task_id<len(manifest):
        p.error(f"task id must be in [0,{len(manifest)-1}]")
    row=manifest[a.task_id]
    tag=f"n{row['N']}_{row['family']}_r{row['replicate']:02d}"
    root=ROOT/"results"/"packing_study"/c["study_id"]
    for sub in ("records","geometry","meshes","previews","failures","figures"):
        (root/sub).mkdir(parents=True,exist_ok=True)
    rec=root/"records"/f"{tag}.json"
    failure=root/"failures"/f"{tag}.txt"
    if rec.exists():
        old=json.loads(rec.read_text())
        if old.get("config_sha256")!=digest:
            raise RuntimeError("Config changed; start with a new study_id")
        if a.geometry_only or old.get("mode")=="full":
            print(f"SKIP {rec}")
            return
    try:
        centres,gaps,desc,geom=generate(c,row)
        csv_path=root/"geometry"/f"{tag}.csv"
        v1.save_geometry(csv_path,centres,gaps,geom["radius"])
        result=dict(**row,**geom,**desc,config_sha256=digest,study_id=c["study_id"],
                    mode="geometry_only" if a.geometry_only else "full",
                    geometry_file=str(csv_path.relative_to(ROOT)))
        if not a.geometry_only:
            msh=root/"meshes"/f"{tag}.msh"
            png=root/"previews"/f"{tag}.png"
            result.update(v1.solve(c,centres,geom,msh,png,tag))
            if not bool(c["save_meshes"]):
                msh.unlink(missing_ok=True)
        tmp=rec.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result,indent=2,default=lambda x:float(x)))
        os.replace(tmp,rec)
        failure.unlink(missing_ok=True)
        print(f"SUCCESS {tag} Vf={geom['vf_achieved']:.6f} {rec}")
    except Exception:
        failure.write_text(traceback.format_exc())
        raise


if __name__=="__main__":
    main()
