#!/usr/bin/env python3
"""Extended development-only study on the 1000-RVE pool.

Models: A quadratic Vf; three scalar-packing Ridge variants (spacing std, mean spacing,
mean + std); five fixed PCA-size Model C variants; and C with PCA size tuned inside CV.
Training never scores the fresh or legacy interpolation records. Evaluation is a separate,
explicit stage for models named before looking at those records. Outputs are in GPa.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["C11_Pa", "C12_Pa", "C22_Pa", "C33_Pa"]
FRACTIONS = [40., 45., 50., 55., 60.]
MODEL_A = "A_quadratic_Vf"
SCALAR_SETS = {
    "B1_std_spacing": ["nn_distance_std_over_D", "local_vf_cv", "centre_anisotropy"],
    "B1_mean_spacing": ["nn_distance_mean_over_D", "local_vf_cv", "centre_anisotropy"],
    "B1_mean_std_spacing": ["nn_distance_mean_over_D", "nn_distance_std_over_D", "local_vf_cv", "centre_anisotropy"],
}
B_NAMES = list(SCALAR_SETS)
MODEL_C_TUNED = "C_PCA_tuned"
LABELS = {MODEL_A: "A: quadratic Vf", "B1_std_spacing": "B1: spacing SD",
          "B1_mean_spacing": "B1: mean spacing", "B1_mean_std_spacing": "B1: mean + SD",
          MODEL_C_TUNED: "C: tuned PCA size"}


def c_name(k):
    return f"C_PCA{int(k)}"


def model_names(c):
    return [MODEL_A, *B_NAMES, *[c_name(k) for k in c["pca_components"]], MODEL_C_TUNED]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path, table):
    table = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        table.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_joblib(path, value):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        joblib.dump(value, tmp, compress=3)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_config(path):
    path = resolve(path)
    c = json.loads(path.read_text())
    required = {"dataset_dir", "output_dir", "random_state", "outer_folds", "outer_repetitions",
                "inner_folds", "ridge_alphas", "pca_components"}
    if required - set(c):
        raise ValueError(f"Missing training configuration: {sorted(required-set(c))}")
    if int(c["outer_folds"]) != 5 or int(c["inner_folds"]) < 2 or int(c["outer_repetitions"]) < 2:
        raise ValueError("Use five outer folds, at least two inner folds and two repetitions.")
    c["ridge_alphas"] = sorted(set(float(a) for a in c["ridge_alphas"]))
    c["pca_components"] = sorted(set(int(k) for k in c["pca_components"]))
    if not c["ridge_alphas"] or min(c["ridge_alphas"]) <= 0 or not c["pca_components"] or min(c["pca_components"]) < 1:
        raise ValueError("Invalid alpha or PCA grid.")
    return c, sha(path)


def build_inputs(selected, s2):
    vf = selected["vf_achieved"].to_numpy(dtype=float)
    terms = np.column_stack([vf, vf ** 2])
    X = {MODEL_A: vf[:, None]}
    for name, columns in SCALAR_SETS.items():
        X[name] = np.column_stack([terms, selected[columns].to_numpy(dtype=float)])
    X["S2"] = np.column_stack([terms, s2]).astype(np.float32)
    return X


def load_inputs(c, include_test=False):
    directory = resolve(c["dataset_dir"])
    validation = json.loads((directory / "preprocessing_validation_v2.json").read_text())
    expected_counts = {"records": 1240, "development": 1000, "fresh_interpolation_test": 200, "existing_interpolation": 40}
    if validation.get("completed") is not True or any(validation.get(k) != v for k, v in expected_counts.items()):
        raise ValueError("Missing or invalid successful v2 preprocessing validation.")
    files = {"dataset": directory / "expanded_dataset.csv",
             "descriptors": directory / "descriptor_audit" / "microstructure_descriptors.csv",
             "S2_metadata": directory / "two_point_statistics" / "two_point_statistics_metadata.csv",
             "raw": directory / "two_point_statistics" / "periodic_fibre_S2_128x128.npz",
             "normalized": directory / "two_point_statistics" / "periodic_fibre_normalized_S2_128x128.npz"}
    for name, path in files.items():
        if sha(path) != validation["sha256"][name]:
            raise ValueError(f"Preprocessed input changed since validation: {path}")
    df = pd.read_csv(files["descriptors"])
    scalar_columns = sorted({col for cols in SCALAR_SETS.values() for col in cols})
    required = {"sample_id", "split", "vf_target_percent", "vf_achieved", *scalar_columns, *TARGETS}
    if required - set(df):
        raise ValueError(f"Missing descriptor columns: {sorted(required-set(df))}")
    df["sample_id"] = df["sample_id"].astype(str)
    expected = {("main", f): 200 for f in FRACTIONS}
    expected.update({("interpolation_test", f): 10 for f in [42.5, 47.5, 52.5, 57.5]})
    expected.update({("interpolation_test_fresh", f): 50 for f in [42.5, 47.5, 52.5, 57.5]})
    if len(df) != 1240 or df["sample_id"].duplicated().any() or df.groupby(["split", "vf_target_percent"], dropna=False).size().to_dict() != expected:
        raise ValueError("Unexpected v2 descriptor identities or counts.")
    selected = (df.copy() if include_test else df.loc[df["split"].eq("main")].copy()).reset_index(drop=True)
    numeric = ["vf_achieved", "vf_target_percent", *scalar_columns, *TARGETS]
    if not np.isfinite(selected[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite selected inputs or labels.")
    with np.load(files["normalized"], allow_pickle=False) as archive:
        ids, splits = archive["sample_id"].astype(str), archive["split"].astype(str)
        if len(ids) != 1240 or len(set(ids)) != 1240 or archive["S2"].shape != (1240, 16384):
            raise ValueError("Invalid normalized S2 identities or shape.")
        positions = {sid: i for i, sid in enumerate(ids)}
        order = np.array([positions[sid] for sid in selected["sample_id"]])
        if not np.array_equal(splits[order], selected["split"].astype(str).to_numpy()):
            raise ValueError("Descriptor and S2 split labels differ.")
        for field in ["vf_achieved", "vf_target_percent"]:
            if not np.allclose(archive[field][order], selected[field].to_numpy(dtype=float), rtol=0, atol=1e-12):
                raise ValueError(f"Descriptor and S2 {field} differ.")
        s2 = archive["S2"][order].astype(np.float32)
    if not np.isfinite(s2).all():
        raise ValueError("Non-finite normalized fields.")
    return selected, build_inputs(selected, s2), selected[TARGETS].to_numpy(dtype=float) / 1e9, {n: sha(p) for n, p in files.items()}


def make_spatial_features(X_train, max_pca, seed):
    if max_pca >= min(len(X_train), X_train.shape[1] - 2):
        raise ValueError("PCA count must be below the training sample and feature rank limit.")
    features = ColumnTransformer([
        ("vf_terms", "passthrough", [0, 1]),
        ("normalized_S2_PCA", Pipeline([("scale_S2", StandardScaler()),
                                        ("pca", PCA(n_components=max_pca, svd_solver="randomized", random_state=seed))]), slice(2, None)),
    ], remainder="drop")
    return features, features.fit_transform(X_train)


def make_selector(k):
    return ColumnTransformer([("selected_features", "passthrough", list(range(k + 2)))], remainder="drop")


def tune(X, y, c, seed):
    alphas, ks, T = c["ridge_alphas"], c["pca_components"], len(TARGETS)
    inner = list(KFold(n_splits=int(c["inner_folds"]), shuffle=True, random_state=seed).split(y))
    if any(max(ks) >= len(train) for train, _ in inner):
        raise ValueError("PCA grid exceeds the smallest inner-training rank.")
    totals_B = {n: np.zeros((len(alphas), T)) for n in B_NAMES}
    totals_C = np.zeros((len(ks), len(alphas), T))
    for train, valid in inner:
        for name in B_NAMES:
            scaler = StandardScaler()
            zt, zv = scaler.fit_transform(X[name][train]), None
            zv = scaler.transform(X[name][valid])
            for ai, alpha in enumerate(alphas):
                estimator = Ridge(alpha=alpha).fit(zt, y[train])
                totals_B[name][ai] += np.mean(np.abs(estimator.predict(zv) - y[valid]), axis=0)
        features, wt = make_spatial_features(X["S2"][train], max(ks), int(c["random_state"]))
        wv = features.transform(X["S2"][valid])
        for ki, k in enumerate(ks):
            scaler = StandardScaler()
            zt = scaler.fit_transform(wt[:, :k + 2])
            zv = scaler.transform(wv[:, :k + 2])
            for ai, alpha in enumerate(alphas):
                estimator = Ridge(alpha=alpha).fit(zt, y[train])
                totals_C[ki, ai] += np.mean(np.abs(estimator.predict(zv) - y[valid]), axis=0)
    n = len(inner)
    best, records = {MODEL_A: {t: {} for t in TARGETS}}, []
    for name in B_NAMES:
        scores, best[name] = totals_B[name] / n, {}
        for ti, t in enumerate(TARGETS):
            ai = int(np.argmin(scores[:, ti]))
            best[name][t] = {"alpha": alphas[ai], "inner_mae_GPa": float(scores[ai, ti])}
            records.extend({"model": name, "target": t, "alpha": a, "pca_components": 0, "inner_mae_GPa": float(scores[j, ti])} for j, a in enumerate(alphas))
    scores_C = totals_C / n
    for ki, k in enumerate(ks):
        best[c_name(k)] = {}
        for ti, t in enumerate(TARGETS):
            ai = int(np.argmin(scores_C[ki, :, ti]))
            best[c_name(k)][t] = {"alpha": alphas[ai], "pca_components": k, "inner_mae_GPa": float(scores_C[ki, ai, ti])}
            records.extend({"model": c_name(k), "target": t, "alpha": a, "pca_components": k, "inner_mae_GPa": float(scores_C[ki, j, ti])} for j, a in enumerate(alphas))
    best[MODEL_C_TUNED] = {}
    for ti, t in enumerate(TARGETS):
        ki, ai = np.unravel_index(np.argmin(scores_C[:, :, ti]), scores_C[:, :, ti].shape)
        best[MODEL_C_TUNED][t] = {"alpha": alphas[int(ai)], "pca_components": ks[int(ki)], "inner_mae_GPa": float(scores_C[ki, ai, ti])}
    return best, records


def fit_models(X, y, best, c):
    ks, models = c["pca_components"], {}
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    a = polynomial.fit_transform(X[MODEL_A])
    models[MODEL_A] = {t: Pipeline([("polynomial", polynomial), ("regression", LinearRegression().fit(a, y[:, ti]))]) for ti, t in enumerate(TARGETS)}
    for name in B_NAMES:
        scaler = StandardScaler()
        z = scaler.fit_transform(X[name])
        models[name] = {t: Pipeline([("scale", scaler), ("regression", Ridge(alpha=best[name][t]["alpha"]).fit(z, y[:, ti]))]) for ti, t in enumerate(TARGETS)}
    spatial, w = make_spatial_features(X["S2"], max(ks), int(c["random_state"]))
    parts = {}
    for k in ks:
        selector = make_selector(k)
        scaler = StandardScaler()
        parts[k] = (selector, scaler, scaler.fit_transform(selector.fit_transform(w)))
    for name in [c_name(k) for k in ks] + [MODEL_C_TUNED]:
        models[name] = {}
        for ti, t in enumerate(TARGETS):
            choice = best[name][t]
            selector, scaler, z = parts[choice["pca_components"]]
            regression = Ridge(alpha=choice["alpha"]).fit(z, y[:, ti])
            models[name][t] = Pipeline([("features", spatial), ("select_PCA_prefix", selector),
                                        ("scale_final_features", scaler), ("regression", regression)])
    explained = spatial.named_transformers_["normalized_S2_PCA"].named_steps["pca"].explained_variance_ratio_
    return models, {str(k): float(explained[:k].sum()) for k in ks}


def predict_models(models, X, c, names=None):
    names = names or [MODEL_A, *B_NAMES, *[c_name(k) for k in c["pca_components"]], MODEL_C_TUNED]
    out = {}
    if MODEL_A in names:
        a = models[MODEL_A][TARGETS[0]].named_steps["polynomial"].transform(X[MODEL_A])
        out[MODEL_A] = np.column_stack([models[MODEL_A][t].named_steps["regression"].predict(a) for t in TARGETS])
    for name in [n for n in B_NAMES if n in names]:
        z = models[name][TARGETS[0]].named_steps["scale"].transform(X[name])
        out[name] = np.column_stack([models[name][t].named_steps["regression"].predict(z) for t in TARGETS])
    c_models = [n for n in names if n.startswith("C_")]
    if c_models:
        spatial = models[c_models[0]][TARGETS[0]].named_steps["features"]
        w = spatial.transform(X["S2"])
        for name in c_models:
            columns = []
            for t in TARGETS:
                pipe = models[name][t]
                values = w if pipe.named_steps["features"] is spatial else pipe.named_steps["features"].transform(X["S2"])
                z = pipe.named_steps["scale_final_features"].transform(pipe.named_steps["select_PCA_prefix"].transform(values))
                columns.append(pipe.named_steps["regression"].predict(z))
            out[name] = np.column_stack(columns)
    return out


def metrics(y, pred):
    if np.any(y == 0):
        raise ValueError("MAPE is undefined for zero-valued targets.")
    return {"mae_GPa": float(mean_absolute_error(y, pred)), "rmse_GPa": float(np.sqrt(mean_squared_error(y, pred))),
            "bias_GPa": float(np.mean(pred - y)), "mape_percent": float(100 * np.mean(np.abs(pred - y) / np.abs(y))),
            "r2": float(r2_score(y, pred))}


def validity(pred):
    c11, c12, c22, c66 = pred.T
    return np.isfinite(pred).all(axis=1) & (c11 > 0) & (c22 > 0) & (c66 > 0) & (c11 * c22 - c12 * c12 > 0)


def plot_bars(summary, out, names, filename, title, mean_col="mae_GPa_mean", std_col="mae_GPa_std"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(names)
    x = np.arange(len(TARGETS))
    for i, name in enumerate(names):
        s = summary[summary["model"].eq(name)].set_index("target").reindex(TARGETS)
        ax.bar(x + (i - (len(names) - 1) / 2) * width, s[mean_col], width=width,
               yerr=s[std_col] if std_col else None, capsize=2, label=LABELS.get(name, name))
    ax.set_xticks(x)
    ax.set_xticklabels([r"$C_{11}$", r"$C_{12}$", r"$C_{22}$", r"$C_{66}$"])
    ax.set_ylabel("MAE (GPa)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / (filename + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out / (filename + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pca_sensitivity(summary, out, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = c["pca_components"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, target, title in zip(axes.ravel(), TARGETS, [r"$C_{11}$", r"$C_{12}$", r"$C_{22}$", r"$C_{66}$"]):
        rows = [summary[(summary["model"] == c_name(k)) & (summary["target"] == target)].iloc[0] for k in ks]
        ax.errorbar(ks, [r["mae_GPa_mean"] for r in rows], yerr=[r["mae_GPa_std"] for r in rows], marker="o", capsize=3, label="C: fixed PCA size")
        for name, style in [(MODEL_A, "--"), ("B1_std_spacing", ":"), (MODEL_C_TUNED, "-.")]:
            value = summary[(summary["model"] == name) & (summary["target"] == target)].iloc[0]["mae_GPa_mean"]
            ax.axhline(value, linestyle=style, color="gray", label=LABELS[name])
        ax.set_title(title)
        ax.set_xlabel("PCA components")
        ax.set_ylabel("Nested-CV MAE (GPa)")
        ax.set_xticks(ks)
        ax.grid(alpha=.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Model C sensitivity to PCA size")
    fig.tight_layout(rect=[0, .07, 1, .96])
    fig.savefig(out / "pca_sensitivity_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "pca_sensitivity_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def train(c, config_sha, check_only):
    main, X, y, input_hashes = load_inputs(c, include_test=False)
    names = model_names(c)
    if len(main) != 1000 or not main["split"].eq("main").all():
        raise ValueError("Training must use exactly 1000 main records.")
    min_outer = len(main) - int(np.ceil(len(main) / int(c["outer_folds"])))
    min_inner = min_outer - int(np.ceil(min_outer / int(c["inner_folds"])))
    if max(c["pca_components"]) >= min_inner:
        raise ValueError("PCA grid exceeds the smallest inner-training rank.")
    if check_only:
        print(f"TRAINING INPUTS VALID: {len(main)} development records; outer train=800, validation=200; smallest inner training={min_inner}.")
        print(f"Models ({len(names)}): {names}")
        print(f"PCA grid={c['pca_components']}; Ridge alpha grid={c['ridge_alphas']}.")
        print("No models fitted and no interpolation prediction errors computed.")
        return
    out = resolve(c["output_dir"])
    if (out / "training_metadata.json").exists() or (out / "final_models.joblib").exists():
        raise FileExistsError("Output study already exists. Use a new output_dir; nothing was overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "running", "config": c, "config_sha256": config_sha, "script_sha256": sha(Path(__file__)),
                "input_sha256": input_hashes, "development_samples": 1000, "models": names,
                "scalar_representations": SCALAR_SETS, "fresh_test_samples_used_for_selection": 0,
                "legacy_test_samples_used_for_selection": 0, "targets": TARGETS, "prediction_units": "GPa",
                "outer_splitter": "StratifiedKFold by nominal fraction; 160 train + 40 validation per fraction",
                "inner_splitter": "shuffled KFold within outer-training data",
                "PCA_policy": "one max-grid PCA basis per training split; leading k scores used for each candidate; fixed-k models tune only alpha",
                "tuned_C_policy": "PCA size and alpha selected jointly by inner MAE per target",
                "caution": "Choosing the best fixed PCA size from outer-validation results is optimistic; the tuned C model is the nested estimate.",
                "tensor_policy": "four retained entries; C13/C23 couplings not predicted; no SPD constraint during regression",
                "fresh_test_evaluation": "not performed by training stage"}
    atomic_json(out / "training_metadata.json", metadata)
    groups = main["vf_target_percent"].to_numpy()
    pred_frames, fold_rows, split_rows, tuning_rows, choice_rows, pca_rows, validity_rows = [], [], [], [], [], [], []
    ids = main["sample_id"].to_numpy()
    for repeat in range(1, int(c["outer_repetitions"]) + 1):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(c["random_state"]) + repeat - 1)
        for fold, (train_idx, valid_idx) in enumerate(outer.split(np.zeros(len(main)), groups), 1):
            print(f"BEGIN repeat={repeat}/{c['outer_repetitions']}, fold={fold}/5", flush=True)
            if any(np.sum(groups[train_idx] == f) != 160 or np.sum(groups[valid_idx] == f) != 40 for f in FRACTIONS):
                raise ValueError("Outer fold is not fraction-balanced.")
            xt = {k: v[train_idx] for k, v in X.items()}
            xv = {k: v[valid_idx] for k, v in X.items()}
            best, scores = tune(xt, y[train_idx], c, int(c["random_state"]) + 1000 * repeat + fold)
            models, explained = fit_models(xt, y[train_idx], best, c)
            pred, train_pred = predict_models(models, xv, c), predict_models(models, xt, c)
            for role, idx in [("train", train_idx), ("validation", valid_idx)]:
                split_rows.extend({"repetition": repeat, "fold": fold, "role": role, "sample_id": ids[i], "vf_target_percent": float(groups[i])} for i in idx)
            tuning_rows.extend({"repetition": repeat, "fold": fold, **r} for r in scores)
            pca_rows.extend({"repetition": repeat, "fold": fold, "pca_components": int(k), "explained_variance_fraction": v} for k, v in explained.items())
            for name in names:
                validity_rows.append({"repetition": repeat, "fold": fold, "model": name, "validation_samples": len(valid_idx),
                                      "reduced_tensor_spd_violations": int((~validity(pred[name])).sum())})
                for ti, t in enumerate(TARGETS):
                    choice = best[name][t]
                    choice_rows.append({"repetition": repeat, "fold": fold, "model": name, "target": t, **choice})
                    for role, idx, values in [("train", train_idx, train_pred[name][:, ti]), ("validation", valid_idx, pred[name][:, ti])]:
                        fold_rows.append({"repetition": repeat, "fold": fold, "model": name, "target": t, "role": role,
                                          "n_training": 800, "n_evaluated": len(idx), "best_params": json.dumps(choice, sort_keys=True), **metrics(y[idx, ti], values)})
                    pred_frames.append(pd.DataFrame({"repetition": repeat, "fold": fold, "model": name, "target": t, "sample_id": ids[valid_idx],
                                                     "vf_target_percent": groups[valid_idx], "y_true_GPa": y[valid_idx, ti], "y_pred_GPa": pred[name][:, ti]}))
            print(f"DONE repeat={repeat}, fold={fold}", flush=True)
        for filename, rows in [("nested_cv_predictions.csv", pd.concat(pred_frames, ignore_index=True)), ("nested_cv_fold_metrics.csv", fold_rows),
                               ("nested_cv_split_manifest.csv", split_rows), ("nested_cv_tuning_scores.csv", tuning_rows),
                               ("nested_cv_selected_parameters.csv", choice_rows), ("nested_cv_pca_variance.csv", pca_rows),
                               ("nested_cv_physical_validity.csv", validity_rows)]:
            atomic_csv(out / filename, rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    repeat_rows = []
    for (name, target, repeat), g in predictions.groupby(["model", "target", "repetition"]):
        if len(g) != 1000 or g["sample_id"].nunique() != 1000:
            raise ValueError("Incomplete out-of-fold coverage.")
        repeat_rows.append({"model": name, "target": target, "repetition": repeat, **metrics(g.y_true_GPa.to_numpy(), g.y_pred_GPa.to_numpy())})
    repeat_df = pd.DataFrame(repeat_rows)
    atomic_csv(out / "nested_cv_metrics_by_repeat.csv", repeat_df)
    aggs = {"n_repetitions": ("repetition", "nunique")}
    for key in ["mae_GPa", "rmse_GPa", "bias_GPa", "mape_percent", "r2"]:
        aggs[key + "_mean"], aggs[key + "_std"] = (key, "mean"), (key, "std")
    summary = repeat_df.groupby(["model", "target"], as_index=False).agg(**aggs)
    atomic_csv(out / "nested_cv_metric_summary.csv", summary)
    wide = repeat_df.pivot_table(index=["target", "repetition"], columns="model", values="mae_GPa")
    paired = []
    for reference in [MODEL_A, "B1_std_spacing"]:
        for name in names:
            if name == reference:
                continue
            for t in TARGETS:
                change = 100 * (wide.loc[t][name] - wide.loc[t][reference]) / wide.loc[t][reference]
                paired.append({"model": name, "reference": reference, "target": t, "mae_change_percent_mean": float(change.mean()),
                               "mae_change_percent_std": float(change.std()), "repetitions_with_lower_mae": int((change < 0).sum()), "repetitions": int(len(change))})
    atomic_csv(out / "nested_cv_paired_mae_changes.csv", paired)
    print("FINAL DEVELOPMENT-ONLY TUNING AND REFIT ON ALL 1000 MAIN RVEs", flush=True)
    final_best, final_scores = tune(X, y, c, int(c["random_state"]) + 999999)
    final_models, final_variance = fit_models(X, y, final_best, c)
    atomic_csv(out / "final_tuning_scores.csv", final_scores)
    atomic_json(out / "final_selected_parameters.json", final_best)
    payload = {"models": final_models, "targets": TARGETS, "model_names": names, "selected_parameters": final_best, "config": c,
               "config_sha256": config_sha, "input_sha256": input_hashes, "prediction_units": "GPa", "training_sample_ids": main["sample_id"].tolist(),
               "scalar_representations": SCALAR_SETS}
    atomic_joblib(out / "final_models.joblib", payload)
    coefficient_rows = []
    for name in names:
        for t in TARGETS:
            regression = final_models[name][t].named_steps["regression"]
            coefficient_rows.append({"model": name, "target": t, "intercept_GPa": float(np.ravel(regression.intercept_)[0]),
                                     "coefficients": json.dumps(np.ravel(regression.coef_).tolist()),
                                     "coefficient_space": "fitted transformed features; use saved pipeline for prediction"})
    atomic_csv(out / "final_regression_coefficients.csv", coefficient_rows)
    metadata.update({"status": "complete", "final_parameters": final_best, "final_PCA_explained_variance_fraction": final_variance,
                     "saved_models_sha256": sha(out / "final_models.joblib"), "fresh_test_prediction_errors_computed": False})
    atomic_json(out / "training_metadata.json", metadata)
    plot_bars(summary, out, [MODEL_A, *B_NAMES, MODEL_C_TUNED], "nested_cv_mae_comparison", "Repeated nested-CV comparison: fibre fraction, scalar packing and spatial models")
    plot_pca_sensitivity(summary, out, c)
    atomic_json(out / "training_complete.json", {"completed": True, "saved_models_sha256": metadata["saved_models_sha256"], "config_sha256": config_sha})
    print(f"TRAINING COMPLETE: {out}", flush=True)
    print("All variants refitted from scratch; fresh interpolation test NOT evaluated.", flush=True)


def evaluate(c, config_sha, confirm_frozen, selected_models):
    if not confirm_frozen:
        raise ValueError("Evaluate only after model choices are fixed; supply --confirm-frozen.")
    out = resolve(c["output_dir"])
    names = model_names(c)
    if not selected_models or any(m not in names for m in selected_models) or len(set(selected_models)) != len(selected_models):
        raise ValueError(f"Supply unique --evaluate-models from: {names}")
    complete = json.loads((out / "training_complete.json").read_text())
    if complete.get("completed") is not True or complete["config_sha256"] != config_sha or sha(out / "final_models.joblib") != complete["saved_models_sha256"]:
        raise ValueError("Final model or configuration changed, or training is incomplete.")
    if (out / "evaluation_started.json").exists():
        raise FileExistsError("Fresh evaluation was already started. Inspect existing outputs; do not retune on this test.")
    rows, X_all, y_all, hashes = load_inputs(c, include_test=True)
    payload = joblib.load(out / "final_models.joblib")
    if payload["input_sha256"] != hashes:
        raise ValueError("Evaluation inputs differ from the training study.")
    reserved = rows.loc[~rows["split"].eq("main")]
    if set(payload["training_sample_ids"]) & set(reserved["sample_id"]):
        raise ValueError("Training and interpolation sample IDs overlap.")
    atomic_json(out / "evaluation_started.json", {"procedure_frozen_confirmed": True, "models_evaluated": selected_models, "saved_models_sha256": complete["saved_models_sha256"]})
    for split, stem, n in [("interpolation_test_fresh", "fresh_interpolation_test", 200), ("interpolation_test", "legacy_interpolation_test", 40)]:
        index = np.flatnonzero(rows["split"].eq(split).to_numpy())
        if len(index) != n:
            raise ValueError("Unexpected evaluation cohort size.")
        pred = predict_models(payload["models"], {k: v[index] for k, v in X_all.items()}, c, selected_models)
        pred_rows, summary, by_fraction, checks = [], [], [], []
        levels = rows.iloc[index]["vf_target_percent"].to_numpy()
        for name in selected_models:
            checks.append({"model": name, "cohort": split, "samples": n, "reduced_tensor_spd_violations": int((~validity(pred[name])).sum())})
            for ti, t in enumerate(TARGETS):
                summary.append({"model": name, "target": t, "cohort": split, "samples": n, **metrics(y_all[index, ti], pred[name][:, ti])})
                pred_rows.append(pd.DataFrame({"model": name, "target": t, "sample_id": rows.iloc[index]["sample_id"].to_numpy(), "vf_target_percent": levels,
                                               "y_true_GPa": y_all[index, ti], "y_pred_GPa": pred[name][:, ti], "absolute_error_GPa": np.abs(pred[name][:, ti] - y_all[index, ti])}))
                for f in sorted(set(levels)):
                    mask = levels == f
                    by_fraction.append({"model": name, "target": t, "vf_target_percent": float(f), "samples": int(mask.sum()), **metrics(y_all[index, ti][mask], pred[name][:, ti][mask])})
        atomic_csv(out / (stem + "_predictions.csv"), pd.concat(pred_rows, ignore_index=True))
        atomic_csv(out / (stem + "_metrics.csv"), summary)
        atomic_csv(out / (stem + "_metrics_by_fraction.csv"), by_fraction)
        atomic_csv(out / (stem + "_physical_validity.csv"), checks)
        plot_bars(pd.DataFrame(summary), out, selected_models, stem + "_mae_comparison", f"{n}-RVE {'fresh' if n == 200 else 'previously evaluated'} interpolation cohort", mean_col="mae_GPa", std_col=None)
    atomic_json(out / "evaluation_complete.json", {"completed": True, "models_evaluated": selected_models, "fresh_samples": 200, "legacy_samples": 40,
                                                   "models_changed": False, "no_tuning_or_fitting_performed": True})
    print(f"EVALUATION COMPLETE for {selected_models}: 200 fresh and 40 legacy cases reported separately; no fitting or tuning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "surrogate_training_extended_v2.json")
    parser.add_argument("--stage", choices=["train", "evaluate"], default="train")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("--evaluate-models", nargs="+")
    args = parser.parse_args()
    c, digest = load_config(args.config)
    if args.stage == "train":
        train(c, digest, args.check_only)
    else:
        if args.check_only:
            parser.error("--check-only applies to the training stage.")
        evaluate(c, digest, args.confirm_frozen, args.evaluate_models)


if __name__ == "__main__":
    main()
