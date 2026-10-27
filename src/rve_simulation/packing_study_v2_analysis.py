"""Analyze saved V2 geometry/results only; no FE/Gmsh imports or HPC jobs.
Deterministic periodic phase-mask and probe statistics; check grid convergence for publication.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FIELDS=["Exx_Pa","Eyy_Pa","Gxy_Pa","nu_xy","C11_Pa","C12_Pa","C13_Pa","C22_Pa","C23_Pa","C33_Pa",
        "mean_gap_over_D","p10_gap_over_D","close_gap_fraction","void_probe_p95_over_D","void_probe_p99_over_D"]


def write(path,rows):
    if not rows:
        return
    keys=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def coords(file):
    with file.open(newline="") as f:
        rows=list(csv.DictReader(f))
    return np.array([[float(x["x_center"]),float(x["y_center"])] for x in rows]),float(rows[0]["radius"])


def nearest(centres,query,L,chunk=512):
    arr=[]
    for i in range(0,len(query),chunk):
        d=query[i:i+chunk,None,:]-centres[None,:,:]
        d-=L*np.round(d/L)
        arr.append(np.sqrt(np.sum(d*d,axis=2)).min(axis=1))
    return np.concatenate(arr)


def descriptors(xy,r,L,c,seed):
    D=2*r;n=int(c["analysis_grid"])
    rng=np.random.default_rng(seed)
    probe=rng.uniform(0,L,(int(c["random_points_for_void_probe"]),2))
    clear=np.maximum(0,nearest(xy,probe,L)-r)/D
    ans=dict(void_probe_mean_over_D=float(clear.mean()),
             void_probe_p95_over_D=float(np.quantile(clear,.95)),
             void_probe_p99_over_D=float(np.quantile(clear,.99)))
    axis=(np.arange(n)+.5)*L/n
    points=np.array(np.meshgrid(axis,axis,indexing="ij")).reshape(2,-1).T
    mask=(nearest(xy,points,L)<=r).reshape(n,n).astype(float)
    ans["sampled_phase_vf"]=float(mask.mean())
    for width in c["window_sides_over_D"]:
        side=float(width)*D
        if side>=L:
            raise ValueError("Window wider than periodic cell")
        half=int(round(side/L*n/2))
        idx=np.arange(-half,half+1)
        kernel=np.zeros((n,n))
        kernel[np.ix_(idx%n,idx%n)]=1
        local=np.fft.ifft2(np.fft.fft2(mask)*np.fft.fft2(kernel)).real/kernel.sum()
        label=(f"window_{width:g}D").replace(".","p")
        ans[label+"_std_vf"]=float(local.std())
        ans[label+"_p05_vf"]=float(np.quantile(local,.05))
        ans[label+"_p95_vf"]=float(np.quantile(local,.95))
    delta=xy[:,None,:]-xy[None,:,:]
    delta-=L*np.round(delta/L)
    dist=np.linalg.norm(delta,axis=2)[np.triu_indices(len(xy),k=1)]/D
    edges=np.asarray(c["radial_bin_edges_over_D"],float)
    if not np.all(np.diff(edges)>0) or edges[-1]*D>=L/2:
        raise ValueError("Invalid radial bins")
    counts=np.histogram(dist,bins=edges)[0]
    rho=len(xy)/L**2
    for count,a,b in zip(counts,edges[:-1],edges[1:]):
        expected=len(xy)*rho*np.pi*D**2*(b*b-a*a)/2
        ans[(f"pair_g_{a:g}_{b:g}D").replace(".","p")]=float(count/expected)
    return ans,clear,mask


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(ROOT/"configs"/"packing_study_v2.json"))
    a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    out=ROOT/"results"/"packing_study"/c["study_id"]
    files=sorted((out/"records").glob("*.json"))
    if not files:
        raise SystemExit("No records yet")
    if len({json.loads(f.read_text())["config_sha256"] for f in files})!=1:
        raise SystemExit("Different config hashes in same study: stop")
    figdir=out/"figures";figdir.mkdir(exist_ok=True,parents=True)
    rows=[];gaps={};masks={}
    for f in files:
        row=json.loads(f.read_text())
        xy,r=coords(ROOT/row["geometry_file"])
        extra,gap,mask=descriptors(xy,r,float(row["L"]),c,int(row["seed"])+100*int(row["N"]))
        row.update(extra)
        k=(row["N"],row["family"],row["replicate"])
        gaps[k]=gap;masks[k]=mask;rows.append(row)
    write(out/"all_realizations.csv",rows)
    full=[r for r in rows if r["mode"]=="full"]
    grouped=defaultdict(list)
    for r in full:
        grouped[(r["N"],r["family"])].append(r)
    names=FIELDS+[(f"window_{w:g}D_{s}").replace(".","p")
                  for w in c["window_sides_over_D"] for s in ("std_vf","p05_vf","p95_vf")]
    names += [k for k in rows[0] if k.startswith("pair_g_")]
    summary=[]
    for (N,family),items in sorted(grouped.items()):
        d=dict(N=N,family=family,n=len(items),vf_achieved=items[0]["vf_achieved"])
        for name in names:
            values=np.array([z[name] for z in items],float)
            d[name+"_mean"]=float(values.mean())
            d[name+"_sample_sd"]=float(values.std(ddof=1)) if len(values)>1 else ""
        summary.append(d)
    write(out/"group_summary.csv",summary)
    paired=[]
    for N in c["fibre_counts"]:
        for rep in range(int(c["n_realizations"])):
            f={r["family"]:r for r in full if r["N"]==N and r["replicate"]==rep}
            if "hard_disk" in f and "clustered" in f:
                h,cl=f["hard_disk"],f["clustered"]
                d=dict(N=N,replicate=rep,vf_achieved=h["vf_achieved"])
                for field in names:
                    d[field+"_cluster_minus_hard"]=float(cl[field]-h[field])
                for field in ("Exx_Pa","Eyy_Pa","Gxy_Pa"):
                    d[field+"_percent_change"]=float(100*(cl[field]-h[field])/h[field])
                paired.append(d)
    write(out/"paired_differences.csv",paired)
    for N in c["fibre_counts"]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for fam in c["families"]:
            arrays=[gaps[(r["N"],r["family"],r["replicate"])] for r in rows if r["N"]==N and r["family"]==fam]
            if arrays:
                x=np.sort(np.concatenate(arrays))
                ax.plot(x,np.arange(1,len(x)+1)/len(x),label=fam)
            mask=masks.get((N,fam,0))
            if mask is not None:
                f2,a2=plt.subplots(figsize=(5,5))
                a2.imshow(mask.T,origin="lower",interpolation="nearest",extent=(0,18,0,18),cmap="gray_r",vmin=0,vmax=1)
                a2.set(xlabel="x/D",ylabel="y/D",title=f"{fam}, N={N}: sampled phase mask")
                f2.tight_layout();f2.savefig(figdir/f"phase_mask_N{N}_{fam}.png",dpi=220);plt.close(f2)
        ax.set(xlabel="Matrix clearance from random location / D",ylabel="Empirical CDF",title=f"N={N}; L/D=18")
        ax.legend();fig.tight_layout();fig.savefig(figdir/f"void_clearance_CDF_N{N}.png",dpi=220);plt.close(fig)
    metric=[("Exx_Pa",1e9,"Exx (GPa)"),("Eyy_Pa",1e9,"Eyy (GPa)"),("Gxy_Pa",1e9,"Gxy (GPa)"),
            ("void_probe_p95_over_D",1,"Void clearance 95th percentile / D"),
            ((f"window_{c['window_sides_over_D'][1]:g}D_std_vf").replace(".","p"),1,"Local sampled Vf SD")]
    for field,scale,label in metric:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for fam in c["families"]:
            items=[x for x in summary if x["family"]==fam]
            if not items:
                continue
            ax.errorbar([x["vf_achieved"]*100 for x in items],
                        [x[field+"_mean"]/scale for x in items],
                        yerr=[x[field+"_sample_sd"]/scale if x[field+"_sample_sd"]!="" else 0 for x in items],
                        label=fam,marker="o",capsize=3)
        ax.set(xlabel="Achieved fibre area fraction (%)",ylabel=label,title="L/D=18; bars: realization SD")
        ax.legend();fig.tight_layout();fig.savefig(figdir/f"{field}_by_packing.png",dpi=220);plt.close(fig)
    expected=len(c["fibre_counts"])*len(c["families"])*int(c["n_realizations"])
    print(f"Completed FE {len(full)}/{expected}; saved geometry {len(rows)}; failures {len(list((out/'failures').glob('*.txt')))}")
    print("Grid phase Vf is approximate: check analysis_grid convergence; SD is not a confidence interval.")
    print(f"Results: {out}")


if __name__=="__main__":
    main()
