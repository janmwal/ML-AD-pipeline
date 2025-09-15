#!/usr/bin/env python3
"""
Classify a single subject from a per-region segmentation CSV and output SHAP values.

Inputs
------
--input_csv:      Path to CSV with columns like ['name','index','GMvalues_unthresholded','GM_values_thresholded'].
--output_folder:  Output directory (created if missing).
--model:          'lgbm' or 'extratrees' (expects ./models/{model}/{model}_{thrs|unthrs}.joblib).
--thresholded:    'True' or 'False' -> chooses which GM values column to use.

Outputs (written to --output_folder)
------------------------------------
- prediction.json         : {"proba": float, "model_path": str, "thresholded": bool, ...}
- X_new.csv               : The single-row input aligned to the pipeline's expected raw columns.
- shap_transformed.csv    : 1 x n_transformed SHAP values (what the model actually sees).
- shap_original.csv       : 1 x n_original SHAP values (aggregated back to base columns), if applicable.
- merged_row.csv          : Original features (base cols) + "<col>_shap" + "proba".
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import joblib
import shap
import config

# --------------------------- CLI utils ---------------------------

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "yes", "y", "1"}:
        return True
    if v in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


# ------------------------ Model loading --------------------------

def load_pipeline(model: str = "lgbm", thresholded: bool = True):
    """
    Load a saved sklearn Pipeline from ./models/{model}/{model}_{thrs|unthrs}.joblib
    """
    suffix = "thrs" if thresholded else "unthrs"
    path = Path("./models") / model / f"{model}_{suffix}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found at {path}")
    pipeline = joblib.load(path)
    return pipeline, path


# --------------------- Preprocessing (CSV -> X) ------------------

def _pick_value_column(df: pd.DataFrame, thresholded: bool) -> str:
    candidates_thrs = ["GM_values_thresholded", "GMvalues_thresholded", "GM_value_thresholded"]
    candidates_unth = ["GMvalues_unthresholded", "GM_values_unthresholded", "GM_value_unthresholded"]
    pool = candidates_thrs if thresholded else candidates_unth
    for c in pool:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a value column for thresholded={thresholded}. Looked for: {pool}"
    )

def _expected_raw_columns_from_pipeline(pipeline) -> Optional[List[str]]:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    preproc = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
    if preproc is not None and hasattr(preproc, "feature_names_in_"):
        return list(preproc.feature_names_in_)
    return None

def preprocess_seg_df_for_pipeline(
    seg_df: pd.DataFrame,
    pipeline,
    *,
    thresholded: bool,
    key: str = "name",           # change to "index" if you trained on numeric IDs
    aggfunc: str = "mean",
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    Convert a per-region long df -> single row with columns matching the pipeline's expected raw inputs.

    Returns:
      X_aligned    : (1 x n_expected) aligned to training raw columns (NaNs for missing)
      missing_cols : expected but absent (left as NaN)
      extra_cols   : present in CSV but not expected (dropped)
      X_wide       : (1 x K) single row built from CSV before alignment
    """
    if key not in seg_df.columns:
        raise KeyError(f"seg_df missing key column '{key}'. Available: {list(seg_df.columns)}")

    val_col = _pick_value_column(seg_df, thresholded)

    df = seg_df[[key, val_col]].copy()
    if key == "index":
        # If your training used plain ints as column names, comment the next line:
        df[key] = df[key].astype(str)

    if aggfunc not in {"mean", "median", "first"}:
        raise ValueError("aggfunc must be one of {'mean','median','first'}")

    if aggfunc == "mean":
        s = df.groupby(key, dropna=False)[val_col].mean()
    elif aggfunc == "median":
        s = df.groupby(key, dropna=False)[val_col].median()
    else:
        s = df.groupby(key, dropna=False)[val_col].first()

    X_wide = s.T.to_frame().T
    X_wide.index = [0]

    expected_cols = _expected_raw_columns_from_pipeline(pipeline)
    # Fallback: if we can't get a sensible raw feature list (or it's suspiciously tiny),
    # just use the wide row as-is (ColumnTransformer will select by name).
    if (expected_cols is None) or (len(expected_cols) < 10):
        return X_wide.copy(), [], [], X_wide


    # Align to expected raw columns
    X_aligned = pd.DataFrame(index=[0], columns=expected_cols, dtype=float)
    intersect = list(set(expected_cols).intersection(X_wide.columns))
    if intersect:
        X_aligned.loc[0, intersect] = X_wide.loc[0, intersect]

    missing_cols = [c for c in expected_cols if c not in X_wide.columns]
    extra_cols   = [c for c in X_wide.columns if c not in expected_cols]

    return X_aligned, missing_cols, extra_cols, X_wide


# ------------------------ SHAP utilities -------------------------

def _coerce_expected_value(ev, pos_idx: int) -> float:
    """
    SHAP expected_value can be:
      - scalar (float)
      - list/array of length 1 (compact binary)
      - list/array of length 2 (binary, per class)
      - list/array of length K (multiclass)
    Pick pos_idx if available; otherwise fall back to 0.
    """
    if isinstance(ev, (list, tuple, np.ndarray)):
        arr = np.asarray(ev).squeeze()
        if arr.ndim == 0:
            return float(arr)
        # choose pos_idx if in range, else 0
        k = pos_idx if arr.shape[0] > pos_idx else 0
        return float(arr[k])
    # scalar
    return float(ev)

def _transform_to_model_input_and_names(pipeline, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified for our use-cases: run all steps except the final estimator.
    Prefer DataFrame outputs for names; otherwise try 'preprocessor' names or
    fall back to original columns. If lengths don't match, generate f{i} names.
    """
    # Ask sklearn to emit DataFrames if supported
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass

    # Ensure DataFrame input for consistent behavior
    if isinstance(X_new, pd.DataFrame):
        X_curr = X_new.copy()
        base_names = list(X_curr.columns)
    else:
        X_curr = pd.DataFrame(X_new)
        base_names = [f"col_{i}" for i in range(X_curr.shape[1])]

    # Apply all but the last step
    for name, step in list(pipeline.steps)[:-1]:
        if hasattr(step, "transform"):
            X_curr = step.transform(X_curr)

    # If we have a DataFrame, use its columns
    if isinstance(X_curr, pd.DataFrame):
        feat_names = np.array(list(X_curr.columns), dtype=object)
        return X_curr.to_numpy(), feat_names

    # Otherwise try to pull names from a named 'preprocessor'
    feat_names = None
    preproc = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
    if preproc is not None:
        if hasattr(preproc, "get_feature_names_out"):
            try:
                feat_names = np.asarray(preproc.get_feature_names_out(), dtype=object)
            except Exception:
                feat_names = None
        elif hasattr(preproc, "get_feature_names"):
            try:
                feat_names = np.asarray(preproc.get_feature_names(), dtype=object)
            except Exception:
                feat_names = None

    X_model = np.asarray(X_curr)
    if feat_names is None:
        feat_names = np.array(base_names, dtype=object)
    if len(feat_names) != X_model.shape[1]:
        feat_names = np.array([f"f{i}" for i in range(X_model.shape[1])], dtype=object)
    return X_model, feat_names


"""def _is_tree_estimator(est) -> bool:
    tree_like = {
        "LGBMClassifier", "LGBMRegressor",
        "XGBClassifier", "XGBRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    }
    return est.__class__.__name__ in tree_like"""

def _aggregate_to_original(feat_names: np.ndarray, shap_row: np.ndarray) -> pd.DataFrame:
    """
    Collapse transformed features back to base columns.
    Robust to underscores in base feature names.
    Strategy:
      - Parse '{prefix}__{rhs}'.
      - If prefix looks like one-hot (contains 'onehot' or 'ohe'), strip only the LAST '_category'.
      - Else keep full rhs intact.
    """
    base = []
    for full in map(str, feat_names):
        if "__" in full:
            prefix, rhs = full.split("__", 1)
            pfx = prefix.lower()
            if ("onehot" in pfx) or ("ohe" in pfx):
                # one-hot: remove only the *last* underscore suffix (category)
                if "_" in rhs:
                    rhs = rhs.rsplit("_", 1)[0]
                # if no underscore, leave as-is (rare)
            # non one-hot: numeric/ordinal passthrough → keep rhs intact
            base.append(rhs)
        else:
            # No transformer prefix present; keep whole name
            base.append(full)

    base = np.array(base, dtype=object)
    df = pd.DataFrame(shap_row.reshape(1, -1), columns=feat_names)
    agg = (
        df.T
        .assign(__base__=base)
        .groupby("__base__", sort=False)
        .sum()
        .T
    )
    return agg


# ----------------------------- HELPER ----------------------------

def _suffix_from_bool(thresholded_bool: bool) -> str:
    return "thrs" if thresholded_bool else "unthrs"

# ----------------------------- Main ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict + SHAP from per-region CSV")
    parser.add_argument(
        "--input_csv", 
        required=True, 
        type=str, 
        help="Path to subject per-region CSV")
    parser.add_argument(
        "--output_folder", 
        required=False, 
        type=str, 
        default="output_pred",
        help="Directory to write outputs (created as output_pred if missing)")
    parser.add_argument(
        "--model", 
        required=False, 
        default="lgbm",
        choices=["lgbm", "extratrees"], 
        help="Model family to load")
    parser.add_argument(
        "--GM_thrs", 
        required=False, 
        default=False,
        type=str2bool, 
        help="Use thresholded GM values (True/False)")
    parser.add_argument(
        "--thrs_target",
        required=False,
        default="youden",
        choices=["youden", "sensitivity", "f1"],
        help="Which precomputed classification threshold to use."
    )
    args = parser.parse_args()

    data_suffix = _suffix_from_bool(args.GM_thrs)
    # Safely fetch threshold from config
    try:
        THRESHOLD = float(config.PREDICTION_THRESHOLDS[args.model][data_suffix][args.thrs_target])
    except KeyError as e:
        raise KeyError(
            f"Missing threshold in config.PREDICTION_THRESHOLDS for "
            f"model='{args.model}', data_suffix='{data_suffix}', target='{args.thrs_target}'."
        ) from e

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load pipeline
    pipeline, model_path = load_pipeline(model=args.model, thresholded=args.GM_thrs)
    est = pipeline[-1]

    # Ensure intermediate transforms yield pandas DataFrames (preserves column names)
    """try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass"""

    # 2) Read & preprocess CSV -> X_new (single row aligned to training raw features)
    seg_df = pd.read_csv(input_csv, sep=None, engine="python")

    # Choose key: if you trained on region names, keep 'name'; if on numeric IDs, switch to key='index'
    X_new, missing_cols, extra_cols, X_wide = preprocess_seg_df_for_pipeline(
        seg_df, pipeline, thresholded=args.GM_thrs, key="name"
    )

    # Persist the actual single-row input
    X_new.to_csv(out_dir / "X_new.csv", index=False)

    # 3) Predict probability for positive class
    # Try to resolve positive class index intelligently
    # Resolve positive class index if possible
    pos_idx = 1
    if hasattr(est, "classes_"):
        classes = list(est.classes_)
        if len(classes) == 2:
            pos_idx = classes.index(1) if 1 in classes else 1

    proba_arr = pipeline.predict_proba(X_new)
    proba = float(np.asarray(proba_arr)[0, pos_idx])

    label = config.CLASS_LABELS[int(proba >= THRESHOLD)]

    # --- SHAP: always return LOG-ODDS (logit) φᵢ; robust for LGBM & ExtraTrees ---
    X_tr, feat_names = _transform_to_model_input_and_names(pipeline, X_new)
    X_df = pd.DataFrame(X_tr, columns=feat_names)

    # Positive-class index
    pos_idx = 1
    if hasattr(est, "classes_"):
        _cls = list(est.classes_)
        if len(_cls) == 2:
            pos_idx = _cls.index(1) if 1 in _cls else 1

    proba_row = np.asarray(pipeline.predict_proba(X_new))[0]

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _is_prob_baseline(bv) -> bool:
        bv = np.asarray(bv)
        return (bv.ndim == 2) and (bv.shape[1] >= 2) and np.allclose(bv.sum(axis=1), 1.0, atol=1e-6)

    def _extract_tree_raw(exp, pos_idx: int, n_features: int):
        v = np.asarray(exp.values)
        b = np.asarray(exp.base_values)
        if v.ndim == 3:
            if v.shape[2] == n_features:        # (n, n_outputs, n_features)
                vals = v[:, pos_idx, :]
            elif v.shape[1] == n_features:      # (n, n_features, n_outputs)
                vals = v[:, :, pos_idx]
            else:
                raise ValueError(f"Can't infer feature axis from {v.shape} vs n_features={n_features}")
            base = b[:, pos_idx] if b.ndim == 2 else b
        elif v.ndim == 2:
            vals = v
            base = b[:, pos_idx] if b.ndim == 2 else b
        else:
            raise ValueError(f"Unexpected Explanation.values ndim={v.ndim}")
        return np.asarray(vals), float(np.ravel(base)[0])

    use_logit_fallback = False
    shap_source = None

    # 1) Try TreeExplainer raw (true logits for boosted trees)
    try:
        tree_exp = shap.TreeExplainer(
            est,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )(X_df)

        if _is_prob_baseline(tree_exp.base_values):
            # ExtraTrees/RandomForest quirk → baseline is per-class probabilities, not logits
            use_logit_fallback = True
        else:
            vals_logit, base_logit = _extract_tree_raw(tree_exp, pos_idx, n_features=len(feat_names))
            shap_source = "tree_raw"
    except Exception:
        use_logit_fallback = True

    # 2) Fallback: model-agnostic with LOGIT link over predict_proba and a non-trivial background
    if use_logit_fallback:
        # (a) Try to load a saved background set colocated with the model (recommended to save at train-time)
        #     Supported filenames: background_transformed.npy (shape: [m, n_features]) or background_transformed.csv
        model_dir = Path(model_path).parent
        bg_np = model_dir / f"background_transformed_{_suffix_from_bool(args.GM_thrs)}.npy"
        bg_csv = model_dir / f"background_transformed_{_suffix_from_bool(args.GM_thrs)}.csv"

        X_bg = None
        if bg_np.exists():
            arr = np.load(bg_np)
            if arr.shape[1] == X_df.shape[1]:
                X_bg = pd.DataFrame(arr, columns=X_df.columns)
        elif bg_csv.exists():
            tmp = pd.read_csv(bg_csv)
            # Align/rename just in case
            if set(X_df.columns).issubset(tmp.columns):
                X_bg = tmp[X_df.columns].copy()

        # (b) If no background on disk, synthesize a small local background by jittering the instance
        if X_bg is None:
            rng = np.random.default_rng(12345)
            center = X_df.iloc[0].to_numpy(dtype=float, copy=True)
            # per-feature scale: 1% of |x| with a small absolute floor to escape zeros
            scale = 0.01 * np.maximum(np.abs(center), 1e-3)
            m = 64  # number of background samples
            jitter = rng.normal(loc=0.0, scale=scale, size=(m, center.size))
            X_bg = pd.DataFrame(center + jitter, columns=X_df.columns)

        # (c) Clipped predict_proba to avoid logit(0) / logit(1) during masking
        def _f_clipped(Z):
            P = est.predict_proba(pd.DataFrame(Z, columns=X_df.columns))[:, pos_idx]
            return np.clip(P, 1e-6, 1 - 1e-6)

        masker = shap.maskers.Independent(X_bg, max_samples=min(256, len(X_bg)))
        logit_exp = shap.Explainer(_f_clipped, masker, link=shap.links.logit)(X_df)

        vals_logit = np.atleast_2d(np.asarray(logit_exp.values))        # (1, n_features)
        base_logit = float(np.ravel(logit_exp.base_values)[0])          # scalar
        shap_source = "logit_link_model_agnostic"

    # Normalize to a single row
    if vals_logit.ndim == 1:
        vals_logit = vals_logit.reshape(1, -1)
    elif vals_logit.shape[0] != 1:
        vals_logit = vals_logit[:1, :]

    # 3) Transformed-space SHAP (LOG-ODDS)
    shap_transformed = pd.DataFrame(vals_logit, columns=feat_names, index=[0])
    shap_transformed.to_csv(out_dir / "shap_transformed.csv", index=False)

    # 4) Aggregate back to original columns (still LOG-ODDS)
    shap_original = _aggregate_to_original(feat_names, shap_transformed.iloc[0].to_numpy())
    shap_original.index = [0]
    shap_original.to_csv(out_dir / "shap_original.csv", index=False)

    # 5) Sanity: SHAP additivity in logit space
    delta_logit = float(shap_transformed.iloc[0].sum())
    pred_logit_from_shap = base_logit + delta_logit
    pred_proba_from_shap = _sigmoid(pred_logit_from_shap)
    base_proba = _sigmoid(base_logit)
    recon_error = abs(pred_proba_from_shap - float(proba_row[pos_idx]))
    shap_space = "raw_logit"


    # Build name -> index mapping from the CSV
    region_map = (seg_df[['name', 'index']]
              .drop_duplicates('name')
              .set_index('name')['index']
              .to_dict())

    base_vals_aligned = X_wide.reindex(columns=shap_original.columns)

    shap_long = pd.DataFrame({
        "region_name": shap_original.columns,
        "region_index": [region_map.get(c, np.nan) for c in shap_original.columns],
        "shap_value": shap_original.iloc[0].values,
        "feature_value": base_vals_aligned.iloc[0].values
    })
    shap_long.to_csv(out_dir / "shap_original_long.csv", index=False)

    unmatched_cols = [c for c in shap_original.columns if c not in region_map]
    if unmatched_cols:
        print(f"[WARN] The following original feature columns could not be mapped to region indices:")
        print(unmatched_cols)

    # 6) Build merged row (base columns + _shap + proba)
    # shap_original has one row; rename its columns to *_shap
    shap_cols = {c: f"{c}_shap" for c in shap_original.columns}
    shap_df = shap_original.rename(columns=shap_cols)

    # Select base feature values from X_wide (aligning on same columns if available)
    base_cols = list(shap_original.columns)
    base_vals = X_wide.reindex(columns=base_cols)

    # Concatenate base values + shap values side by side
    merged = pd.concat([base_vals, shap_df], axis=1)

    # Add proba as last column
    merged["proba"] = proba

    # Save
    merged.to_csv(out_dir / "merged_row.csv", index=False)


    # 7) Write a small JSON summary

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_folder": str(out_dir.resolve()),
        "model": args.model,
        "gm_thresholded": bool(args.GM_thrs),
        "model_path": str(model_path),
        "positive_class_index_used": int(pos_idx),
        "proba": proba,
        "predicted_label": label,
        "threshold_used": THRESHOLD,
        "threshold_target": args.thrs_target,
        "shap_space": shap_space,
        "shap_source": shap_source,  # "tree_raw" or "logit_link_model_agnostic"
        "shap_expected_value_logit": float(base_logit),
        "shap_expected_value_proba": float(base_proba),
        "shap_sum_logit": float(delta_logit),
        "pred_proba_from_shap": float(pred_proba_from_shap),
        "pred_proba_pipeline": float(proba_row[pos_idx]),
        "prob_reconstruction_error": float(recon_error),
        "n_transformed_features": int(shap_transformed.shape[1]),
        "n_original_features": int(shap_original.shape[1]),
        "n_missing_expected_raw_cols": int(len(missing_cols)),
        "n_extra_csv_cols_ignored": int(len(extra_cols)),    
        }
    with open(out_dir / "prediction.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 8) Console summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
