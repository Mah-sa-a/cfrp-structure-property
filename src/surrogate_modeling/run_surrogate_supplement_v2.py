#!/usr/bin/env python3
"""Development-only B2-B5 comparison and balanced learning curves on 1000 RVEs.

Requires train_surrogate_extended_v2.py and its completed extended_study_v2.
--prepare checks inputs/dependencies and saves task manifests; no estimators fitted.
CV: 50 tasks (10 repetitions x 5 folds). Learning: 135 tasks (3 x 5 x 9 sizes).
Workers checkpoint independently and can be safely resubmitted. No fresh test scoring.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent / "train_surrogate_extended_v2.py"
DEFAULT_CONFIG=ROOT / "configs" / "surrogate_supplement_v2.json"
ALGORITHMS=["B2_elasticnet_packing","B3_random_forest_packing","B4_rbf_svr_packing","B5_xgboost_packing"]
LABELS={"A_quadratic_Vf":"A: quadratic Vf","B1_std_spacing":"B1: spacing SD","B1_mean_spacing":"B1: mean spacing",
        "B1_mean_std_spacing":"B1: mean + SD","B2_elasticnet_packing":"B2: Elastic Net","B3_random_forest_packing":"B3: random forest",
        "B4_rbf_svr_packing":"B4: RBF-SVR","B5_xgboost_packing":"B5: XGBoost","C_PCA_tuned":"C: tuned PCA"}


def resolve(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core():
    spec=importlib.util.spec_from_file_location("extended_core",SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path):
    path=resolve(path)
    c=json.loads(path.read_text())
    required={"dataset_dir","reference_study","output_dir","random_state","outer_folds","cv_repetitions","learning_repetitions",
              "inner_folds","learning_sizes_per_fraction","learning_pca_components","ridge_alphas","algorithm_grids","scalar_representation"}
    if required-set(c):
        raise ValueError(f"Missing configuration: {sorted(required-set(c))}")
    if c["outer_folds"]!=5 or c["inner_folds"]<2 or c["cv_repetitions"]<2 or c["learning_repetitions"]<2:
        raise ValueError("Use five outer folds, >=2 inner folds, >=2 repetitions.")
    if c["scalar_representation"]!="B1_std_spacing":
        raise ValueError("B2-B5 must use original B1 spacing-SD representation for this matched-input comparison.")
    c["learning_sizes_per_fraction"]=sorted(set(int(n) for n in c["learning_sizes_per_fraction"]))
    if not c["learning_sizes_per_fraction"] or min(c["learning_sizes_per_fraction"])<2 or max(c["learning_sizes_per_fraction"])>160:
        raise ValueError("Learning sizes must be 2..160 per fraction.")
    c["learning_pca_components"]=sorted(set(int(k) for k in c["learning_pca_components"]))
    if not c["learning_pca_components"] or min(c["learning_pca_components"])<1:
        raise ValueError("Invalid PCA counts.")
    if set(c["algorithm_grids"])!=set(ALGORITHMS):
        raise ValueError("Define all four B2-B5 parameter grids; no candidates silently skipped.")
    return c,sha(path)


def manifests(c):
    cv=[{"task_id":len(range((r-1)*5+f-1)),"repetition":r,"fold":f,"mode":"cv"}
        for r in range(1,int(c["cv_repetitions"])+1) for f in range(1,6)]
    learning=[]
    for r in range(1,int(c["learning_repetitions"])+1):
        for f in range(1,6):
            for n in c["learning_sizes_per_fraction"]:
                learning.append({"task_id":len(learning),"repetition":r,"fold":f,"mode":"learning","n_per_fraction":n})
    return {"cv":cv,"learning":learning}


def context(c,config_sha,core):
    # Core loads/validates all archives but returns only main rows and labels.
    main,X,y,input_hashes=core.load_inputs(c,include_test=False)
    if len(main)!=1000 or not main["split"].eq("main").all():
        raise ValueError("Only the 1000 main RVEs may enter this workflow.")
    ref=resolve(c["reference_study"])
    meta=json.loads((ref/"training_metadata.json").read_text())
    complete=json.loads((ref/"training_complete.json").read_text())
    if meta.get("status")!="complete" or complete.get("completed") is not True:
        raise ValueError("Extended study is not complete.")
    if meta["input_sha256"]!=input_hashes or meta["script_sha256"]!=sha(SOURCE):
        raise ValueError("Reference study/core/input provenance differs. Do not change shared source or inputs.")
    cfg=meta["config"]
    if int(cfg["random_state"])!=int(c["random_state"]) or int(cfg["inner_folds"])!=int(c["inner_folds"]) or int(cfg["outer_repetitions"])!=int(c["cv_repetitions"]):
        raise ValueError("Reference seed/folds/repetitions must match the supplement.")
    reference_files={k:ref/k for k in ["nested_cv_predictions.csv","nested_cv_split_manifest.csv","nested_cv_metric_summary.csv","training_metadata.json","training_complete.json"]}
    reference_hashes={k:sha(p) for k,p in reference_files.items()}
    definitions=manifests(c)
    meta={"config_sha256":config_sha,"script_sha256":sha(Path(__file__)),"core_sha256":sha(SOURCE),"input_sha256":input_hashes,
          "reference_sha256":reference_hashes,"config":c,"python":sys.version,"sklearn":sklearn.__version__,"xgboost":xgboost.__version__,
          "cv_tasks":len(definitions["cv"]),"learning_tasks":len(definitions["learning"]),"fresh_test_used":False,
          "B2_B5_scalar_features":["vf_achieved","vf_squared",*core.SCALAR_SETS[c["scalar_representation"]]],
          "inner_tuning":"all scaling fitted inside inner-training partitions; MAE in GPa",
          "learning_selection":"nested fraction-balanced subsets; same outer validation samples across sizes/models",
          "learning_PCA":"fixed-dimensional curves only; infeasible sizes skipped explicitly, never reduce k",
          "error_bars":"sample SD across repeated pooled-OOF metrics; not confidence intervals"}
    digest=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    return main,X,y,meta,digest,definitions


def prepared(c,meta,digest):
    out=resolve(c["output_dir"])
    path=out/"preparation_metadata.json"
    if not path.exists() or json.loads(path.read_text())!={"preparation_digest":digest,**meta}:
        raise ValueError("Run --prepare first. If already prepared, provenance changed; investigate rather than overwrite.")
    return out


def prepare(c,core,main,meta,digest,definitions):
    out=resolve(c["output_dir"])
    marker=out/"preparation_metadata.json"
    expected={"preparation_digest":digest,**meta}
    if marker.exists() and json.loads(marker.read_text())!=expected:
        raise ValueError("Existing supplementary study has different provenance; use another output directory.")
    if not marker.exists() and any((out/mode/"tasks").glob("*/complete.json") for mode in []):
        raise ValueError("Existing workers without preparation metadata.")
    for mode,tasks in definitions.items():
        (out/mode/"tasks").mkdir(parents=True,exist_ok=True)
        core.atomic_csv(out/(mode+"_task_manifest.csv"),tasks)
    core.atomic_json(marker,expected)
    print(f"PREPARED {out}")
    print(f"CV_TASK_COUNT={len(definitions['cv'])}; LEARNING_TASK_COUNT={len(definitions['learning'])}")
    print(f"B2-B5 use matched original B1 inputs. Learning sizes={c['learning_sizes_per_fraction']}; PCA curves={c['learning_pca_components']}.")
    print("No models fitted; fresh interpolation records not evaluated.")


def make_model(name,c):
    seed=int(c["random_state"])
    if name=="B2_elasticnet_packing":
        return Pipeline([("scale",StandardScaler()),("regression",ElasticNet(max_iter=100000,random_state=seed))])
    if name=="B3_random_forest_packing":
        return RandomForestRegressor(random_state=seed,n_jobs=1)
    if name=="B4_rbf_svr_packing":
        from sklearn.svm import SVR
        return Pipeline([("scale",StandardScaler()),("regression",SVR(kernel="rbf"))])
    if name=="B5_xgboost_packing":
        return XGBRegressor(objective="reg:squarederror",random_state=seed,n_jobs=1,tree_method="hist",verbosity=0)
    raise ValueError(name)


def folds(main,row,c):
    groups=main["vf_target_percent"].to_numpy()
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=int(c["random_state"])+row["repetition"]-1)
    train,valid=list(outer.split(np.zeros(len(main)),groups))[row["fold"]-1]
    if any(np.sum(groups[train]==f)!=160 or np.sum(groups[valid]==f)!=40 for f in [40.,45.,50.,55.,60.]):
        raise ValueError("Outer split is not fraction-balanced.")
    if row["mode"]=="learning":
        seed=int(c["random_state"])+10000*row["repetition"]+100*row["fold"]
        rng=np.random.default_rng(seed)
        orders={f:rng.permutation(train[groups[train]==f]) for f in [40.,45.,50.,55.,60.]}
        n=row["n_per_fraction"]
        train=np.sort(np.concatenate([orders[f][:n] for f in orders]))
    return train,valid


def verify_reference(main,row,train,valid,c):
    table=pd.read_csv(resolve(c["reference_study"])/"nested_cv_split_manifest.csv")
    expected=table[(table.repetition==row["repetition"])&(table.fold==row["fold"])]
    ids=main["sample_id"].to_numpy()
    actual_valid=set(ids[valid])
    ref_valid=set(expected.loc[expected.role.eq("validation"),"sample_id"])
    ref_train=set(expected.loc[expected.role.eq("train"),"sample_id"])
    if actual_valid!=ref_valid or not set(ids[train]).issubset(ref_train):
        raise ValueError("Worker split does not match the reference study.")


def local_scores(core,y,pred,levels):
    result=core.metrics(y,pred)
    centered=y.copy()
    for f in np.unique(levels):
        mask=levels==f
        centered[mask]-=y[mask].mean()
    denominator=float(np.sum(centered**2))
    result["within_fraction_r2"]=float(1-np.sum((pred-y)**2)/denominator) if denominator>0 else np.nan
    return result


def eligible_pca(c,n_train):
    smallest=n_train-int(np.ceil(n_train/int(c["inner_folds"])))
    return [k for k in c["learning_pca_components"] if k<smallest],smallest


def fit_scalar_and_spatial(core,X,y,c,inner,valid_ks,jobs):
    models,best,tuning={},{},[]
    polynomial=PolynomialFeatures(degree=2,include_bias=False)
    za=polynomial.fit_transform(X[core.MODEL_A])
    models[core.MODEL_A]={t:Pipeline([("polynomial",polynomial),("regression",LinearRegression().fit(za,y[:,ti]))]) for ti,t in enumerate(core.TARGETS)}
    best[core.MODEL_A]={t:{} for t in core.TARGETS}
    alphas=c["ridge_alphas"]
    for name in core.B_NAMES:
        scores=np.zeros((len(alphas),len(core.TARGETS)))
        for tr,va in inner:
            scale=StandardScaler();zt=scale.fit_transform(X[name][tr]);zv=scale.transform(X[name][va])
            for ai,a in enumerate(alphas):
                fitted=Ridge(alpha=a).fit(zt,y[tr])
                scores[ai]+=np.mean(np.abs(fitted.predict(zv)-y[va]),axis=0)
        scores/=len(inner)
        scale=StandardScaler();z=scale.fit_transform(X[name]);models[name]={};best[name]={}
        for ti,t in enumerate(core.TARGETS):
            ai=int(np.argmin(scores[:,ti]));choice={"alpha":float(alphas[ai]),"inner_mae_GPa":float(scores[ai,ti])}
            best[name][t]=choice
            models[name][t]=Pipeline([("scale",scale),("regression",Ridge(alpha=choice["alpha"]).fit(z,y[:,ti]))])
            tuning.extend({"model":name,"target":t,"alpha":a,"pca_components":0,"inner_mae_GPa":float(scores[j,ti])} for j,a in enumerate(alphas))
    if not valid_ks:
        return models,best,tuning
    scores=np.zeros((len(valid_ks),len(alphas),len(core.TARGETS)))
    with threadpool_limits(limits=jobs):
        for tr,va in inner:
            features,wt=core.make_spatial_features(X["S2"][tr],max(valid_ks),int(c["random_state"]))
            wv=features.transform(X["S2"][va])
            for ki,k in enumerate(valid_ks):
                scale=StandardScaler();zt=scale.fit_transform(wt[:,:k+2]);zv=scale.transform(wv[:,:k+2])
                for ai,a in enumerate(alphas):
                    fitted=Ridge(alpha=a).fit(zt,y[tr])
                    scores[ki,ai]+=np.mean(np.abs(fitted.predict(zv)-y[va]),axis=0)
        scores/=len(inner)
        features,w=core.make_spatial_features(X["S2"],max(valid_ks),int(c["random_state"]))
        for ki,k in enumerate(valid_ks):
            name=core.c_name(k);models[name]={};best[name]={}
            selector=core.make_selector(k);scale=StandardScaler();z=scale.fit_transform(selector.fit_transform(w))
            for ti,t in enumerate(core.TARGETS):
                ai=int(np.argmin(scores[ki,:,ti]));a=float(alphas[ai])
                best[name][t]={"alpha":a,"pca_components":k,"inner_mae_GPa":float(scores[ki,ai,ti])}
                models[name][t]=Pipeline([("features",features),("select_PCA_prefix",selector),("scale_final_features",scale),
                                          ("regression",Ridge(alpha=a).fit(z,y[:,ti]))])
                tuning.extend({"model":name,"target":t,"alpha":a0,"pca_components":k,"inner_mae_GPa":float(scores[ki,j,ti])} for j,a0 in enumerate(alphas))
    return models,best,tuning


def predict_fixed(core,models,X):
    # Fixed-learning models share X-only transformations just as in extended_core.
    names=list(models)
    # 'pca_components' is unused when an explicit name list is passed.
    return core.predict_models(models,X,{"pca_components":[]},names)


def worker(c,config_sha,core,main,X,y,meta,digest,definitions,mode,task_id,jobs):
    out=prepared(c,meta,digest)
    if not 0<=task_id<len(definitions[mode]):
        raise ValueError(f"Task ID must be 0..{len(definitions[mode])-1} for mode {mode}.")
    row=definitions[mode][task_id]
    path=out/mode/"tasks"/f"task_{task_id:04d}"
    marker=path/"complete.json"
    if marker.exists():
        done=json.loads(marker.read_text())
        if done["preparation_digest"]!=digest or done["task"]!=row:
            raise ValueError("Existing worker provenance differs.")
        for name,h in done["output_sha256"].items():
            if sha(path/name)!=h:
                raise ValueError("Existing worker output checksum differs.")
        print(f"SKIP complete {mode} task {task_id}",flush=True)
        return
    path.mkdir(parents=True,exist_ok=True)
    train,valid=folds(main,row,c)
    verify_reference(main,row,train,valid,c)
    inner_seed=int(c["random_state"])+1000*row["repetition"]+row["fold"]
    inner=list(KFold(n_splits=int(c["inner_folds"]),shuffle=True,random_state=inner_seed).split(train))
    xt={k:v[train] for k,v in X.items()};xv={k:v[valid] for k,v in X.items()}
    predictions,metric_rows,parameter_rows,tuning_rows,physical=[],[],[],[],[]
    meta_columns={k:v for k,v in row.items() if k!="mode"}
    ids=main["sample_id"].to_numpy();levels=main["vf_target_percent"].to_numpy()
    split_rows=[{**meta_columns,"role":role,"sample_id":ids[i],"vf_target_percent":float(levels[i])} for role,idx in [("train",train),("validation",valid)] for i in idx]
    skipped=[]
    print(f"BEGIN {mode} task={task_id}, repeat={row['repetition']}, fold={row['fold']}, n_train={len(train)}",flush=True)
    if mode=="learning":
        ks,smallest=eligible_pca(c,len(train))
        skipped=[{"model":core.c_name(k),"reason":f"{k} PCs exceeds centered inner-training rank; smallest inner training={smallest}"} for k in c["learning_pca_components"] if k not in ks]
        fixed,best,tuning=fit_scalar_and_spatial(core,xt,y[train],c,inner,ks,jobs)
        with threadpool_limits(limits=jobs):
            pv=predict_fixed(core,fixed,xv);pt=predict_fixed(core,fixed,xt)
        tuning_rows.extend({**meta_columns,**r} for r in tuning)
        for name in fixed:
            physical.append({**meta_columns,"model":name,"validation_samples":len(valid),"reduced_tensor_spd_violations":int((~core.validity(pv[name])).sum())})
            for ti,t in enumerate(core.TARGETS):
                choice=best[name][t]
                parameter_rows.append({**meta_columns,"model":name,"target":t,"best_params":json.dumps(choice,sort_keys=True)})
                for role,idx,pred in [("train",train,pt[name][:,ti]),("validation",valid,pv[name][:,ti])]:
                    metric_rows.append({**meta_columns,"model":name,"target":t,"role":role,"n_training":len(train),"n_evaluated":len(idx),
                                        **local_scores(core,y[idx,ti],pred,levels[idx])})
                predictions.append(pd.DataFrame({**meta_columns,"model":name,"target":t,"sample_id":ids[valid],"vf_target_percent":levels[valid],
                                                  "y_true_GPa":y[valid,ti],"y_pred_GPa":pv[name][:,ti]}))
    for name in ALGORITHMS:
        pv=np.zeros((len(valid),len(core.TARGETS)))
        for ti,t in enumerate(core.TARGETS):
            search=GridSearchCV(make_model(name,c),c["algorithm_grids"][name],scoring="neg_mean_absolute_error",cv=inner,
                                n_jobs=jobs,refit=True,error_score="raise",return_train_score=False)
            search.fit(xt[c["scalar_representation"]],y[train,ti])
            pt=search.predict(xt[c["scalar_representation"]])
            values=search.predict(xv[c["scalar_representation"]]);pv[:,ti]=values
            choice={"params":search.best_params_,"inner_mae_GPa":float(-search.best_score_)}
            parameter_rows.append({**meta_columns,"model":name,"target":t,"best_params":json.dumps(choice,sort_keys=True)})
            for role,idx,pred in [("train",train,pt),("validation",valid,values)]:
                metric_rows.append({**meta_columns,"model":name,"target":t,"role":role,"n_training":len(train),"n_evaluated":len(idx),
                                    **local_scores(core,y[idx,ti],pred,levels[idx])})
            predictions.append(pd.DataFrame({**meta_columns,"model":name,"target":t,"sample_id":ids[valid],"vf_target_percent":levels[valid],
                                              "y_true_GPa":y[valid,ti],"y_pred_GPa":values}))
            cv=search.cv_results_
            tuning_rows.extend({**meta_columns,"model":name,"target":t,"parameters":json.dumps(params,sort_keys=True),"inner_mae_GPa":float(-score)}
                               for params,score in zip(cv["params"],cv["mean_test_score"]))
            print(f"DONE {mode} task={task_id}, {name}, {t}",flush=True)
        physical.append({**meta_columns,"model":name,"validation_samples":len(valid),"reduced_tensor_spd_violations":int((~core.validity(pv)).sum())})
    tables={"predictions.csv":pd.concat(predictions,ignore_index=True),"metrics.csv":metric_rows,"parameters.csv":parameter_rows,
            "tuning_scores.csv":tuning_rows,"split_manifest.csv":split_rows,"physical_validity.csv":physical}
    for filename,values in tables.items():
        core.atomic_csv(path/filename,values)
    core.atomic_json(marker,{"completed":True,"preparation_digest":digest,"task":row,"skipped_models":skipped,
                             "output_sha256":{name:sha(path/name) for name in tables},"fresh_test_used":False})
    print(f"TASK COMPLETE {mode} {task_id}; main-only fits and predictions",flush=True)


def gather(c,core,digest,definitions,mode):
    out=resolve(c["output_dir"])/mode
    missing=[];frames={n:[] for n in ["predictions","metrics","parameters","physical_validity"]}
    skipped=[]
    for task in definitions[mode]:
        path=out/"tasks"/f"task_{task['task_id']:04d}"
        marker=path/"complete.json"
        if not marker.exists():
            missing.append(task["task_id"]);continue
        done=json.loads(marker.read_text())
        if done.get("completed") is not True or done["preparation_digest"]!=digest or done["task"]!=task:
            raise ValueError("Worker provenance mismatch.")
        for name,h in done["output_sha256"].items():
            if sha(path/name)!=h:
                raise ValueError(f"Changed worker table: {path/name}")
        for n in frames:
            frames[n].append(pd.read_csv(path/(n+".csv")))
        skipped.extend({**task,**r} for r in done["skipped_models"])
    print(f"Complete {mode} tasks: {len(definitions[mode])-len(missing)}/{len(definitions[mode])}")
    if missing:
        print("Missing task IDs:",','.join(map(str,missing)))
        return None,None,None
    return out,{n:pd.concat(v,ignore_index=True) for n,v in frames.items()},skipped


def pooled_metrics(core,predictions,learning=False):
    rows=[]
    keys=["model","target","repetition"]+(["n_per_fraction"] if learning else [])
    for key,g in predictions.groupby(keys):
        if len(g)!=1000 or g.sample_id.nunique()!=1000:
            raise ValueError(f"Incomplete OOF coverage: {key}")
        rows.append({**dict(zip(keys,key)),**local_scores(core,g.y_true_GPa.to_numpy(),g.y_pred_GPa.to_numpy(),g.vf_target_percent.to_numpy())})
    return pd.DataFrame(rows)


def summarize_metrics(table,keys):
    aggregations={"n_repetitions":("repetition","nunique")}
    for key in ["mae_GPa","rmse_GPa","bias_GPa","mape_percent","r2","within_fraction_r2"]:
        aggregations[key+"_mean"]=(key,"mean");aggregations[key+"_std"]=(key,"std")
    return table.groupby(keys,as_index=False).agg(**aggregations)


def plots_cv(summary,out,core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=[core.MODEL_A,"B1_std_spacing",*ALGORITHMS,"C_PCA_tuned"]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,t,title in zip(axes.ravel(),core.TARGETS,[r"$C_{11}$",r"$C_{12}$",r"$C_{22}$",r"$C_{66}$"]):
        selected=summary[summary.target.eq(t)].set_index("model").reindex(names)
        ax.bar(np.arange(len(names)),selected.mae_GPa_mean,yerr=selected.mae_GPa_std,capsize=2,color=plt.get_cmap("tab10")(np.arange(len(names))))
        ax.set_xticks(np.arange(len(names)));ax.set_xticklabels([LABELS.get(n,n) for n in names],rotation=35,ha="right",fontsize=8)
        ax.set_title(title);ax.set_ylabel("Nested-CV MAE (GPa)");ax.grid(axis="y",alpha=.25)
    fig.suptitle("Matched scalar inputs: B2-B5 versus baseline, Ridge and spatial C")
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/"algorithm_comparison_mae.png",dpi=300,bbox_inches="tight")
    fig.savefig(out/"algorithm_comparison_mae.pdf",bbox_inches="tight");plt.close(fig)


def plots_learning(summary,out,c,core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups={"representations":[core.MODEL_A,*core.B_NAMES,core.c_name(c["learning_pca_components"][0])],
            "algorithms":[core.MODEL_A,"B1_std_spacing",*ALGORITHMS],
            "PCA_sizes":[core.MODEL_A,*[core.c_name(k) for k in c["learning_pca_components"]]]}
    for group,names in groups.items():
        for metric,ylabel in [("mae_GPa","MAE (GPa)"),("mape_percent","MAPE (%)")]:
            fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True)
            for ax,t,title in zip(axes.ravel(),core.TARGETS,[r"$C_{11}$",r"$C_{12}$",r"$C_{22}$",r"$C_{66}$"]):
                for i,name in enumerate(names):
                    color=plt.get_cmap("tab10")(i)
                    valid=summary[(summary.model==name)&(summary.target==t)&(summary.role=="validation")].sort_values("n_per_fraction")
                    train=summary[(summary.model==name)&(summary.target==t)&(summary.role=="train")].sort_values("n_per_fraction")
                    ax.errorbar(valid.n_per_fraction,valid[metric+"_mean"],yerr=valid[metric+"_std"],color=color,marker="o",capsize=2,label=LABELS.get(name,name))
                    ax.plot(train.n_per_fraction,train[metric+"_mean"],color=color,linestyle="--",alpha=.6)
                ax.set_title(title);ax.set_xlabel("Training RVEs per fibre fraction");ax.set_ylabel(ylabel);ax.grid(alpha=.25)
                ax.set_xticks(c["learning_sizes_per_fraction"]);ax.tick_params(axis="x",labelbottom=True,labelsize=8)
            handles,labels=axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=8)
            fig.suptitle(f"{group}: solid validation ± repetition SD; dashed training")
            fig.tight_layout(rect=[0,.10,1,.96])
            stem=f"learning_curve_{group}_{metric}";fig.savefig(out/(stem+".png"),dpi=300,bbox_inches="tight")
            fig.savefig(out/(stem+".pdf"),bbox_inches="tight");plt.close(fig)


def summary(c,core,main,meta,digest,definitions,mode):
    prepared(c,meta,digest)
    out,tables,skipped=gather(c,core,digest,definitions,mode)
    if tables is None:
        return 1
    if mode=="cv":
        reference=pd.read_csv(resolve(c["reference_study"])/"nested_cv_predictions.csv")
        combined=pd.concat([reference,tables["predictions"]],ignore_index=True)
        by_repeat=pooled_metrics(core,combined)
        merged=summarize_metrics(by_repeat,["model","target"])
        core.atomic_csv(out/"combined_cv_predictions.csv",combined)
        core.atomic_csv(out/"combined_cv_metrics_by_repeat.csv",by_repeat)
        core.atomic_csv(out/"combined_cv_metric_summary.csv",merged)
        core.atomic_csv(out/"nonlinear_fold_metrics.csv",tables["metrics"])
        core.atomic_csv(out/"selected_parameters.csv",tables["parameters"])
        core.atomic_csv(out/"physical_validity.csv",tables["physical_validity"])
        wide=by_repeat.pivot_table(index=["target","repetition"],columns="model",values="mae_GPa")
        paired=[]
        for ref in [core.MODEL_A,"B1_std_spacing","C_PCA_tuned"]:
            for name in ALGORITHMS:
                for t in core.TARGETS:
                    change=100*(wide.loc[t][name]-wide.loc[t][ref])/wide.loc[t][ref]
                    paired.append({"model":name,"reference":ref,"target":t,"mae_change_percent_mean":float(change.mean()),
                                   "mae_change_percent_std":float(change.std()),"repetitions_with_lower_mae":int((change<0).sum()),"repetitions":len(change)})
        core.atomic_csv(out/"paired_mae_changes.csv",paired)
        plots_cv(merged,out,core)
        print(f"CV SUMMARY COMPLETE: {out}")
    else:
        by_repeat=pooled_metrics(core,tables["predictions"],learning=True)
        by_repeat["role"]="validation"
        # Training errors: fold averages per repetition; validation: pooled OOF.
        train=tables["metrics"][tables["metrics"].role.eq("train")]
        columns=["mae_GPa","rmse_GPa","bias_GPa","mape_percent","r2","within_fraction_r2"]
        training=train.groupby(["model","target","repetition","n_per_fraction"],as_index=False)[columns].mean()
        training["role"]="train"
        records=pd.concat([by_repeat,training],ignore_index=True)
        result=summarize_metrics(records,["model","target","n_per_fraction","role"])
        result["n_training"]=5*result.n_per_fraction
        core.atomic_csv(out/"learning_curve_predictions.csv",tables["predictions"])
        core.atomic_csv(out/"learning_curve_fold_metrics.csv",tables["metrics"])
        core.atomic_csv(out/"learning_curve_metrics_by_repeat.csv",records)
        core.atomic_csv(out/"learning_curve_summary.csv",result)
        core.atomic_csv(out/"selected_parameters.csv",tables["parameters"])
        core.atomic_json(out/"skipped_PCA_sizes.json",skipped)
        plots_learning(result,out,c,core)
        print(f"LEARNING-CURVE SUMMARY COMPLETE: {out}")
    core.atomic_json(out/"summary_complete.json",{"completed":True,"mode":mode,"preparation_digest":digest,
                    "tasks":len(definitions[mode]),"fresh_test_evaluated":False})
    print("Fresh interpolation test remains untouched.")
    return 0


def fit_final(c,core,main,X,y,meta,digest,jobs):
    out=prepared(c,meta,digest)/"cv"
    marker=json.loads((out/"summary_complete.json").read_text())
    if marker.get("completed") is not True or marker["preparation_digest"]!=digest:
        raise ValueError("Complete CV and inspect development results before final fitting.")
    model_file=out/"final_B2_B5_models.joblib"
    if model_file.exists():
        raise FileExistsError("Final nonlinear models already exist; do not overwrite.")
    inner=list(KFold(n_splits=c["inner_folds"],shuffle=True,random_state=c["random_state"]+999999).split(main))
    models,params={},[]
    for name in ALGORITHMS:
        models[name]={}
        for ti,t in enumerate(core.TARGETS):
            search=GridSearchCV(make_model(name,c),c["algorithm_grids"][name],scoring="neg_mean_absolute_error",cv=inner,n_jobs=jobs,refit=True,error_score="raise")
            search.fit(X[c["scalar_representation"]],y[:,ti])
            models[name][t]=search.best_estimator_
            params.append({"model":name,"target":t,"best_params":json.dumps(search.best_params_,sort_keys=True),"inner_mae_GPa":float(-search.best_score_)})
            print(f"FINAL FIT {name} {t}",flush=True)
    core.atomic_joblib(model_file,{"models":models,"targets":core.TARGETS,"prediction_units":"GPa","config":c,"preparation_digest":digest,
                      "training_sample_ids":main.sample_id.tolist(),"features":["vf_achieved","vf_squared",*core.SCALAR_SETS[c["scalar_representation"]]],"fresh_test_used":False})
    core.atomic_csv(out/"final_selected_parameters.csv",params)
    print("FINAL B2-B5 FIT COMPLETE on 1000 main RVEs; no interpolation predictions.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--mode",choices=["cv","learning"],default="cv")
    parser.add_argument("--n-jobs",type=int,default=4)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--check-only",action="store_true")
    group.add_argument("--task-id",type=int)
    group.add_argument("--summary",action="store_true")
    group.add_argument("--fit-final",action="store_true")
    args=parser.parse_args()
    if args.n_jobs<1:
        parser.error("Use a positive n-jobs.")
    c,config_sha=load_config(args.config)
    core=load_core()
    main,X,y,meta,digest,definitions=context(c,config_sha,core)
    if args.prepare:
        prepare(c,core,main,meta,digest,definitions)
    elif args.check_only:
        print(f"CHECK PASSED: 1000 development RVEs; sklearn={sklearn.__version__}; xgboost={xgboost.__version__}")
        print(f"CV tasks={len(definitions['cv'])}; learning tasks={len(definitions['learning'])}; no fits or fresh-test predictions.")
    elif args.summary:
        return summary(c,core,main,meta,digest,definitions,args.mode)
    elif args.fit_final:
        fit_final(c,core,main,X,y,meta,digest,args.n_jobs)
    else:
        worker(c,config_sha,core,main,X,y,meta,digest,definitions,args.mode,args.task_id,args.n_jobs)
    return 0


if __name__=="__main__":
    sys.exit(main())
