#!/usr/bin/env python3
"""Pool existing fresh-200 and legacy-40 predictions without fitting or changing models.

Outputs: one 240-case metric table, metrics by fraction, pooled prediction CSV,
MAE/MAPE comparisons, and one 4-panel predicted-versus-FE figure per model.
Figures pool both cohorts; prediction CSV retains evaluation_cohort for traceability.
Requires completed evaluation in the same directory, from the same saved pipelines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[2]
TARGETS=["C11_Pa","C12_Pa","C22_Pa","C33_Pa"]
TITLES=[r"$C_{11}$",r"$C_{12}$",r"$C_{22}$",r"$C_{66}$"]
LEVELS=[42.5,47.5,52.5,57.5]
LABELS={"A_quadratic_Vf":"A: quadratic Vf", "B1_std_spacing":"B1: spacing SD", "B1_mean_spacing":"B1: mean spacing",
        "B1_mean_std_spacing":"B1: mean + SD", "C_PCA_tuned":"C: tuned PCA size"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,df):
    temp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    try:
        df.to_csv(temp,index=False);os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path,value):
    temp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def load_predictions(directory):
    paths={"fresh_200":directory/"fresh_interpolation_test_predictions.csv",
           "legacy_40":directory/"legacy_interpolation_test_predictions.csv"}
    required_files=[*paths.values(),directory/"evaluation_complete.json",directory/"evaluation_started.json",
                    directory/"training_complete.json",directory/"final_models.joblib"]
    missing=[str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Evaluation is not complete or inputs are missing. Finish the separate evaluation stage first; do not repeat a completed evaluation. Missing: "+", ".join(missing))
    complete=json.loads((directory/"evaluation_complete.json").read_text())
    started=json.loads((directory/"evaluation_started.json").read_text())
    trained=json.loads((directory/"training_complete.json").read_text())
    if complete.get("completed") is not True or complete.get("models_changed") is not False or trained.get("completed") is not True:
        raise ValueError("Evaluation/training completion metadata is invalid.")
    model_hash=sha(directory/"final_models.joblib")
    if model_hash!=started["saved_models_sha256"] or model_hash!=trained["saved_models_sha256"]:
        raise ValueError("The model file changed since evaluation; do not combine different fitted model versions.")
    required={"model","target","sample_id","vf_target_percent","y_true_GPa","y_pred_GPa"}
    tables=[];ids={};model_sets=[]
    for cohort,n,per_level in [("fresh_200",200,50),("legacy_40",40,10)]:
        df=pd.read_csv(paths[cohort])
        if required-set(df):
            raise ValueError(f"Missing prediction columns: {sorted(required-set(df))}")
        df["sample_id"]=df["sample_id"].astype(str)
        if df[["sample_id","model","target"]].isna().any().any() or df.duplicated(["model","target","sample_id"]).any():
            raise ValueError(f"Missing/duplicate prediction identities in {cohort}.")
        if not np.isfinite(df[["vf_target_percent","y_true_GPa","y_pred_GPa"]].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite predictions or labels.")
        if set(df.target)!=set(TARGETS) or set(df.vf_target_percent)!=set(LEVELS):
            raise ValueError(f"Unexpected targets/fractions in {cohort}.")
        models=sorted(df.model.unique())
        expected={(m,t,f):per_level for m in models for t in TARGETS for f in LEVELS}
        got=df.groupby(["model","target","vf_target_percent"]).size().to_dict()
        if got!=expected or df.sample_id.nunique()!=n:
            raise ValueError(f"Expected {n} independent cases, {per_level} per fraction in {cohort}.")
        ids[cohort]=set(df.sample_id)
        model_sets.append(set(models))
        df["evaluation_cohort"]=cohort
        tables.append(df)
    if model_sets[0]!=model_sets[1] or ids["fresh_200"]&ids["legacy_40"]:
        raise ValueError("Cohorts overlap or model names differ.")
    if "models_evaluated" in complete and set(complete["models_evaluated"])!=model_sets[0]:
        raise ValueError("Evaluation model list and predictions differ.")
    if "models_evaluated" in started and set(started["models_evaluated"])!=model_sets[0]:
        raise ValueError("Started evaluation model list and predictions differ.")
    combined=pd.concat(tables,ignore_index=True)
    identity=combined.groupby("sample_id")["vf_target_percent"].nunique()
    truth=combined.groupby(["sample_id","target"])["y_true_GPa"].agg(["min","max"])
    if not identity.eq(1).all() or np.any((truth["max"]-truth["min"]).to_numpy()>1e-10):
        raise ValueError("Fractions or FE labels disagree between model outputs.")
    combined["signed_error_GPa"]=combined.y_pred_GPa-combined.y_true_GPa
    combined["absolute_error_GPa"]=combined.signed_error_GPa.abs()
    if np.any(combined.y_true_GPa==0):
        raise ValueError("MAPE is undefined for zero FE labels.")
    combined["absolute_percentage_error"]=100*combined.absolute_error_GPa/combined.y_true_GPa.abs()
    hashes={name:sha(path) for name,path in paths.items()}
    hashes["saved_models"]=model_hash
    return combined,sorted(model_sets[0]),hashes


def metrics(group):
    y=group.y_true_GPa.to_numpy(dtype=float);p=group.y_pred_GPa.to_numpy(dtype=float)
    return {"samples":len(group),"mae_GPa":float(mean_absolute_error(y,p)),"rmse_GPa":float(np.sqrt(mean_squared_error(y,p))),
            "bias_GPa":float(np.mean(p-y)),"mape_percent":float(100*np.mean(np.abs(p-y)/np.abs(y))),"r2":float(r2_score(y,p))}


def build_tables(predictions):
    overall=[];by_fraction=[];by_cohort=[]
    for (model,target),g in predictions.groupby(["model","target"]):
        if len(g)!=240 or g.sample_id.nunique()!=240:
            raise ValueError("Each model/target must have 240 unique interpolation cases.")
        overall.append({"model":model,"target":target,**metrics(g)})
        for vf,part in g.groupby("vf_target_percent"):
            if len(part)!=60:
                raise ValueError("Each fraction must have 60 pooled cases.")
            by_fraction.append({"model":model,"target":target,"vf_target_percent":float(vf),**metrics(part)})
        for cohort,part in g.groupby("evaluation_cohort"):
            by_cohort.append({"model":model,"target":target,"evaluation_cohort":cohort,**metrics(part)})
    return pd.DataFrame(overall),pd.DataFrame(by_fraction),pd.DataFrame(by_cohort)


def plot_comparison(summary,models,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.arange(len(TARGETS));width=.8/len(models)
    for metric,ylabel,stem in [("mae_GPa","MAE (GPa)","combined_240_mae_comparison"),
                               ("mape_percent","MAPE (%)","combined_240_mape_comparison")]:
        fig,ax=plt.subplots(figsize=(10,6))
        for i,model in enumerate(models):
            values=summary[summary.model.eq(model)].set_index("target").reindex(TARGETS)[metric]
            ax.bar(x+(i-(len(models)-1)/2)*width,values,width,label=LABELS.get(model,model))
        ax.set_xticks(x);ax.set_xticklabels(TITLES)
        ax.set_ylabel(ylabel);ax.set_title("Pooled interpolation evaluation: 240 RVEs")
        ax.grid(axis="y",alpha=.25);ax.legend(fontsize=9)
        fig.tight_layout();fig.savefig(out/(stem+".png"),dpi=300,bbox_inches="tight")
        fig.savefig(out/(stem+".pdf"),bbox_inches="tight");plt.close(fig)


def plot_parity(predictions,models,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=plt.get_cmap("viridis")(np.linspace(.05,.95,len(LEVELS)))
    for model in models:
        fig,axes=plt.subplots(2,2,figsize=(11,9))
        for ax,t,title in zip(axes.ravel(),TARGETS,TITLES):
            g=predictions[(predictions.model==model)&(predictions.target==t)]
            low=min(g.y_true_GPa.min(),g.y_pred_GPa.min());high=max(g.y_true_GPa.max(),g.y_pred_GPa.max())
            pad=.04*(high-low) if high>low else .01
            for vf,color in zip(LEVELS,colors):
                part=g[g.vf_target_percent.eq(vf)]
                ax.scatter(part.y_true_GPa,part.y_pred_GPa,s=23,color=color,alpha=.65,label=f"{vf:g}%",linewidths=.2,edgecolors="black")
            ax.plot([low-pad,high+pad],[low-pad,high+pad],"k--",linewidth=1,label="1:1")
            ax.set_xlim(low-pad,high+pad);ax.set_ylim(low-pad,high+pad)
            ax.set_aspect("equal",adjustable="box")
            ax.set_title(title);ax.set_xlabel("FE stiffness (GPa)");ax.set_ylabel("Predicted stiffness (GPa)");ax.grid(alpha=.2)
            s=metrics(g)
            ax.text(.04,.96,f"N=240\nMAE={s['mae_GPa']:.5f} GPa\nMAPE={s['mape_percent']:.3f}%",transform=ax.transAxes,va="top",fontsize=8,
                    bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        handles,labels=axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=5,title="Target fibre fraction",fontsize=9)
        fig.suptitle(f"{LABELS.get(model,model)} — all 240 interpolation RVEs")
        fig.tight_layout(rect=[0,.09,1,.95])
        stem="combined_240_predicted_vs_FE_"+model
        fig.savefig(out/(stem+".png"),dpi=300,bbox_inches="tight")
        fig.savefig(out/(stem+".pdf"),bbox_inches="tight");plt.close(fig)


def plot_fraction_errors(table,models,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,t,title in zip(axes.ravel(),TARGETS,TITLES):
        for model in models:
            g=table[(table.model==model)&(table.target==t)].sort_values("vf_target_percent")
            ax.plot(g.vf_target_percent,g.mae_GPa,marker="o",label=LABELS.get(model,model))
        ax.set_title(title);ax.set_xlabel("Target fibre fraction (%)");ax.set_ylabel("MAE (GPa)");ax.set_xticks(LEVELS);ax.grid(alpha=.25)
    handles,labels=axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=min(3,len(models)))
    fig.suptitle("Pooled interpolation errors: 60 RVEs per fraction")
    fig.tight_layout(rect=[0,.07,1,.96]);fig.savefig(out/"combined_240_mae_by_fraction.png",dpi=300,bbox_inches="tight")
    fig.savefig(out/"combined_240_mae_by_fraction.pdf",bbox_inches="tight");plt.close(fig)


def main():
    default=ROOT/"results"/"surrogate_models"/"extended_study_v2"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=default)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    directory=args.input_dir.resolve();out=args.output_dir.resolve() if args.output_dir else directory/"combined_interpolation_240"
    predictions,models,hashes=load_predictions(directory)
    overall,fractions,cohorts=build_tables(predictions)
    if args.check_only:
        print(f"COMBINATION CHECK PASSED: 240 unique RVEs; 60 per fraction; models={models}.")
        print("No files written and no models fitted.")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Combined-report directory already contains files. Choose a new --output-dir; existing results are not overwritten.")
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"combined_interpolation_predictions.csv",predictions)
    write_csv(out/"combined_interpolation_metrics.csv",overall)
    write_csv(out/"combined_interpolation_metrics_by_fraction.csv",fractions)
    write_csv(out/"cohort_metrics_reference.csv",cohorts)
    plot_comparison(overall,models,out);plot_parity(predictions,models,out);plot_fraction_errors(fractions,models,out)
    caption=("Pooled interpolation evaluation of the frozen surrogate predictors on 240 RVEs at four intermediate target fibre fractions "
             "(42.5%, 47.5%, 52.5%, and 57.5%), with 60 realizations per fraction. The pooled cohort comprises 200 newly reserved "
             "evaluation cases and 40 previously evaluated cases. All cases are predicted using the same final fitted models; "
             "no refitting or tuning is performed during aggregation. Parity-plot colours identify fibre fraction, not evaluation cohort.")
    (out/"figure_caption.txt").write_text(caption+"\n")
    write_json(out/"combined_report_metadata.json",{"completed":True,"unique_RVEs":240,"per_fraction":60,"fresh_records":200,"previously_evaluated_records":40,
                "models":models,"models_refitted":False,"input_sha256":hashes,"source_directory":str(directory),
                "metric_policy":"All metrics recomputed from pooled predictions; RMSE and R2 are not averages of cohort metrics.",
                "figure_policy":"One pooled cohort; colour by fibre fraction. CSV preserves evaluation_cohort."})
    print(f"COMBINED REPORT COMPLETE: {out}")
    print(f"Unique RVEs: 240; 60 per fraction; models: {models}")
    print("Metrics and figures pooled across all 240 cases; original prediction files and fitted models unchanged.")
    print(overall[["model","target","samples","mae_GPa","mape_percent","r2"]].to_string(index=False))


if __name__=="__main__":
    main()
