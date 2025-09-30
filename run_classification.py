#!/usr/bin/env python3
"""
Classify a single subject from a per-region segmentation CSV and output SHAP values.

Inputs
------
--input_csv:      Path to CSV with columns like ['name','index','GMvalues_unthresholded','GM_values_thresholded'].
--output_folder:  Output directory (created if missing).
--model:          'lgbm' or 'extratrees' (expects ./models/{model}/{thrs|unthrs}/model.joblib).
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
import re
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
import pandas as pd
import joblib
import shap
import config as config

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
    thrs_suffix = "thrs" if thresholded else "unthrs"
    path = Path("./models") / model / thrs_suffix / "model.joblib"
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

ROI_RENAME_MAP_PATH = Path("data/roi_rename_map.csv")


def _load_roi_mapping(path: Path = ROI_RENAME_MAP_PATH) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    expected_cols = {"nospace_name", "space_name"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"ROI rename map at {path} must contain columns {expected_cols}."
        )
    return dict(zip(df["nospace_name"], df["space_name"]))


def _normalize_region_label(label: str, mapping: Dict[str, str]) -> str:
    if pd.isna(label):
        return label
    label_str = str(label)
    if label_str in mapping:
        return mapping[label_str]
    key = label_str.replace(" ", "")
    if key in mapping:
        return mapping[key]
    if key.startswith("x") and key[1:] in mapping:
        return mapping[key[1:]]
    if label_str.startswith("x") and label_str[1:] in mapping:
        return mapping[label_str[1:]]
    return label_str


def _align_to_expected(
    X_source: pd.DataFrame,
    expected_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    X_aligned = pd.DataFrame(index=[0], columns=expected_cols, dtype=float)
    intersect = [c for c in expected_cols if c in X_source.columns]
    if intersect:
        X_aligned.loc[0, intersect] = X_source.loc[0, intersect]
    missing_cols = [c for c in expected_cols if c not in X_source.columns]
    extra_cols = [c for c in X_source.columns if c not in expected_cols]
    return X_aligned, missing_cols, extra_cols


def _prepare_long_format(
    seg_df: pd.DataFrame,
    pipeline,
    *,
    thresholded: bool,
    rename_map: Dict[str, str],
    key: str = "name",
    aggfunc: str = "mean",
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, Optional[str]]:
    if key not in seg_df.columns:
        raise KeyError(f"seg_df missing key column '{key}'. Available: {list(seg_df.columns)}")

    seg_df = seg_df.copy()
    seg_df[key] = seg_df[key].map(lambda v: _normalize_region_label(v, rename_map))

    val_col = _pick_value_column(seg_df, thresholded)
    df = seg_df[[key, val_col]].copy()
    if key == "index":
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
    if expected_cols is None:
        return X_wide.copy(), [], [], X_wide, None

    X_aligned, missing_cols, extra_cols = _align_to_expected(X_wide, expected_cols)
    return X_aligned, missing_cols, extra_cols, X_wide, None


def _prepare_subject_row_format(
    seg_df: pd.DataFrame,
    pipeline,
    *,
    rename_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, Optional[str]]:
    if seg_df.shape[0] != 1:
        raise ValueError(
            "Subject-row CSV must contain exactly one row for inference."
        )

    seg_df = seg_df.copy()
    subject_id = None
    if "PR_number" in seg_df.columns:
        subject_id = str(seg_df.loc[seg_df.index[0], "PR_number"])

    rename_dict = {
        col: _normalize_region_label(col, rename_map) for col in seg_df.columns
    }
    seg_df = seg_df.rename(columns=rename_dict)

    expected_cols = _expected_raw_columns_from_pipeline(pipeline)
    if expected_cols is None:
        raise ValueError(
            "Could not infer expected feature columns from pipeline for subject-row input."
        )

    feature_cols = [c for c in seg_df.columns if c in expected_cols]
    X_wide = seg_df[feature_cols].astype(float)
    X_wide.index = [0]

    X_aligned, missing_cols, extra_cols_from_align = _align_to_expected(X_wide, expected_cols)
    extra_cols = [
        c for c in seg_df.columns
        if c not in expected_cols and c not in {"PR_number"}
    ]
    extra_cols.extend(extra_cols_from_align)
    extra_cols = list(dict.fromkeys(extra_cols))
    return X_aligned, missing_cols, extra_cols, X_wide, subject_id


def prepare_input_for_pipeline(
    seg_df: pd.DataFrame,
    pipeline,
    *,
    thresholded: bool,
    rename_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, Optional[str]]:
    if "PR_number" in seg_df.columns:
        return _prepare_subject_row_format(seg_df, pipeline, rename_map=rename_map)
    return _prepare_long_format(
        seg_df,
        pipeline,
        thresholded=thresholded,
        rename_map=rename_map,
        key="name",
    )


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
    Run X_new through all pipeline steps EXCEPT the final estimator.
    Return:
      X_model  : ndarray the estimator actually sees
      feat_out : np.ndarray of names AFTER all intermediate transforms
    """
    # start with raw
    if isinstance(X_new, pd.DataFrame):
        X_curr = X_new.copy()
        feat_out = np.array(X_curr.columns, dtype=object)
    else:
        X_curr = X_new
        feat_out = np.array([f"col_{i}" for i in range(X_curr.shape[1])], dtype=object)

    for name, step in list(pipeline.steps)[:-1]:
        # ---- transform ----
        if hasattr(step, "transform"):
            X_curr = step.transform(X_curr)
        # normalize to ndarray for reliable shape checks
        if hasattr(X_curr, "toarray"):
            X_nd = X_curr.toarray()
        elif isinstance(X_curr, pd.DataFrame):
            X_nd = X_curr.values
        else:
            X_nd = np.asarray(X_curr)

        # ---- update names ----
        if hasattr(step, "get_feature_names_out"):
            try:
                feat_out = np.asarray(step.get_feature_names_out(feat_out), dtype=object)
            except TypeError:
                feat_out = np.asarray(step.get_feature_names_out(), dtype=object)

        elif hasattr(step, "get_support"):
            # selectors (VarianceThreshold, SelectKBest, SelectFromModel)
            sup = step.get_support(indices=False)
            sup = np.asarray(sup, dtype=bool)
            if sup.shape[0] == feat_out.shape[0]:
                feat_out = feat_out[sup]
            else:
                # dimension changed without a clean mapping
                feat_out = np.array([f"f{i}" for i in range(X_nd.shape[1])], dtype=object)

        elif step.__class__.__name__.lower().startswith("pca"):
            n_comp = getattr(step, "n_components_", None)
            if n_comp is None:
                n_comp = X_nd.shape[1]
            feat_out = np.array([f"pca_{i}" for i in range(int(n_comp))], dtype=object)

        else:
            # if dimensionality changed and we can't infer new names → generic
            if X_nd.shape[1] != feat_out.shape[0]:
                feat_out = np.array([f"f{i}" for i in range(X_nd.shape[1])], dtype=object)

        # carry ndarray forward
        X_curr = X_nd

    return X_curr, np.asarray(feat_out, dtype=object)


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


# -------------------------- SHAP helpers -------------------------

def _select_class_shap(sv, pos_idx: int, n_classes: Optional[int] = None):
    """Normalize SHAP outputs for binary/multiclass models."""

    if isinstance(sv, list):
        return np.asarray(sv[0] if len(sv) == 1 else sv[pos_idx])

    if isinstance(sv, np.ndarray):
        if sv.ndim == 3:
            class_dim = None
            if n_classes is not None:
                candidates = [axis for axis, dim in enumerate(sv.shape) if dim == n_classes]
                if candidates:
                    class_dim = candidates[0]

            if class_dim == 0:
                return sv[pos_idx, :, :]
            if class_dim == 1:
                return sv[:, pos_idx, :]
            if class_dim == 2:
                return sv[:, :, pos_idx]

            k = 0 if sv.shape[0] == 1 else pos_idx
            return sv[k, :, :]

        if sv.ndim == 2:
            return sv

    raise TypeError(
        "Unexpected SHAP return type/shape: type="
        f"{type(sv)}, shape={getattr(sv, 'shape', None)}"
    )


# ----------------------------- HELPER ----------------------------

def _suffix_from_bool(thresholded_bool: bool) -> str:
    return "thrs" if thresholded_bool else "unthrs"


def _subject_from_filename(path: Path) -> Optional[str]:
    match = re.search(r'(PR\d+_[A-Za-z0-9]+)', path.name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

# ----------------------------- Runner ---------------------------

def run_single_prediction(
    *,
    pipeline,
    estimator,
    X_new: pd.DataFrame,
    missing_cols: List[str],
    extra_cols: List[str],
    X_wide: pd.DataFrame,
    source_df: pd.DataFrame,
    pos_idx: int,
    threshold: float,
    classes: List,
    label_mapping: Dict[int, str],
    roi_mapping: Dict[str, str],
    out_dir: Path,
    input_csv: Path,
    model_path: Path,
    model_name: str,
    gm_thresholded: bool,
    threshold_target: str,
    subject_id: Optional[str],
):
    if out_dir.exists():
        for item in out_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_new.to_csv(out_dir / "X_new.csv", index=False)


    proba_arr = pipeline.predict_proba(X_new)
    proba_row = np.asarray(proba_arr)[0]
    proba = float(proba_row[pos_idx])
    label = label_mapping[int(proba >= threshold)]

    X_tr, feat_names = _transform_to_model_input_and_names(pipeline, X_new)
    print(f"[DEBUG] Model input shape: {X_tr.shape} | feature names: {len(feat_names)}")

    try:
        explainer = shap.TreeExplainer(estimator, model_output="log_loss")
        shap_model_output = "log_loss"
    except Exception:
        explainer = shap.TreeExplainer(estimator)
        shap_model_output = "raw"

    sv = explainer.shap_values(X_tr)
    n_classes = len(classes) if classes else None
    sv_class = _select_class_shap(sv, pos_idx, n_classes)
    shap_row = sv_class[:1, :]

    if shap_row.shape[1] != len(feat_names):
        print(
            f"[WARN] Name/shape mismatch: SHAP has {shap_row.shape[1]} cols, names={len(feat_names)}. "
            "Using generic names derived from SHAP matrix."
        )
        feat_names = np.array([f"f{i}" for i in range(shap_row.shape[1])], dtype=object)

    shap_transformed = pd.DataFrame(shap_row, columns=feat_names, index=[0])
    shap_transformed.to_csv(out_dir / "shap_transformed.csv", index=False)

    if any(str(c).startswith(("pca_", "f")) for c in feat_names):
        shap_original = shap_transformed.copy()
        shap_original.index = [0]
    else:
        shap_original = _aggregate_to_original(feat_names, shap_row)
        shap_original.index = [0]
    shap_original.to_csv(out_dir / "shap_original.csv", index=False)

    try:
        exp = explainer.expected_value
        shap_exp = _coerce_expected_value(exp, pos_idx)
    except Exception:
        shap_exp = None

    # --- SHAP sanity check and extra variables ---
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    base_logit = float(shap_exp) if shap_exp is not None else 0.0
    base_proba = float(_sigmoid(base_logit))
    delta_logit = float(shap_transformed.iloc[0].sum())
    pred_logit_from_shap = base_logit + delta_logit
    pred_proba_from_shap = float(_sigmoid(pred_logit_from_shap))
    pred_proba_pipeline = float(proba_row[pos_idx])
    recon_error = abs(pred_proba_from_shap - pred_proba_pipeline)
    shap_space = "raw_logit"

    if {'name', 'index'}.issubset(source_df.columns):
        region_map = (
            source_df[['name', 'index']]
            .drop_duplicates('name')
            .assign(name=lambda d: d['name'].map(lambda v: _normalize_region_label(v, roi_mapping)))
            .set_index('name')['index']
            .to_dict()
        )
    else:
        region_map = {}

    base_vals_aligned = X_wide.reindex(columns=shap_original.columns)

    shap_long = pd.DataFrame({
        "region_name": shap_original.columns,
        "region_index": [region_map.get(c, np.nan) for c in shap_original.columns],
        "shap_value": shap_original.iloc[0].values,
        "feature_value": base_vals_aligned.iloc[0].values,
    })
    shap_long.to_csv(out_dir / "shap_original_long.csv", index=False)

    unmatched_cols = [c for c in shap_original.columns if c not in region_map]
    if region_map and unmatched_cols:
        print("[WARN] The following original feature columns could not be mapped to region indices:")
        print(unmatched_cols)

    shap_cols = {c: f"{c}_shap" for c in shap_original.columns}
    shap_df = shap_original.rename(columns=shap_cols)
    base_cols = list(shap_original.columns)
    base_vals = X_wide.reindex(columns=base_cols)
    merged = pd.concat([base_vals, shap_df], axis=1)
    merged["proba"] = proba
    merged.to_csv(out_dir / "merged_row.csv", index=False)

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_folder": str(out_dir.resolve()),
        "model": model_name,
        "gm_thresholded": bool(gm_thresholded),
        "model_path": str(model_path),
        "subject_id": subject_id,
        "positive_class_index_used": int(pos_idx),
        "proba": proba,
        "predicted_label": label,
        "threshold_used": threshold,
        "threshold_target": threshold_target,
        "shap_model_output": shap_model_output,
        "shap_expected_value_logit": base_logit,
        "shap_expected_value_proba": base_proba,
        "shap_sum_logit": delta_logit,
        "pred_proba_from_shap": pred_proba_from_shap,
        "pred_proba_pipeline": pred_proba_pipeline,
        "prob_reconstruction_error": recon_error,
        "shap_space": shap_space,
        "n_transformed_features": int(shap_transformed.shape[1]),
        "n_original_features": int(shap_original.shape[1]),
        "n_missing_expected_raw_cols": int(len(missing_cols)),
        "n_extra_csv_cols_ignored": int(len(extra_cols)),
        "missing_expected_raw_cols_sample": list(missing_cols)[:10],
        "extra_csv_cols_ignored_sample": list(extra_cols)[:10],

    }
    with open(out_dir / "prediction.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing subject output directories instead of suffixing",
    )
    parser.add_argument(
        "--model", 
        required=False, 
        default="lgbm",
        choices=["lgbm",], 
        help="Model family to load (for now only lgbm works)")
    parser.add_argument(
        "--GM_thrs", 
        required=False, 
        default=False,
        type=str2bool, 
        help="Use thresholded GM values (True/False). Should match the input csv if subject-per-row format.")
    parser.add_argument(
        "--thrs_target",
        required=False,
        default='youden', 
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

    pipeline, model_path = load_pipeline(model=args.model, thresholded=args.GM_thrs)
    est = pipeline[-1]

    seg_df = pd.read_csv(input_csv, sep=None, engine="python")
    roi_mapping = _load_roi_mapping()

    classes = list(getattr(est, "classes_", []))
    pos_idx = 1
    if classes:
        if len(classes) == 2 and 1 in classes:
            pos_idx = classes.index(1)
        elif len(classes) == 2:
            pos_idx = 1

    label_mapping = config.CLASS_LABELS
    results = []

    is_long_format = {'name', 'index'}.issubset(seg_df.columns)

    if is_long_format:
        X_new, missing_cols, extra_cols, X_wide, subject_id = prepare_input_for_pipeline(
            seg_df,
            pipeline,
            thresholded=args.GM_thrs,
            rename_map=roi_mapping,
        )
        file_subject = _subject_from_filename(input_csv)
        subject_id = subject_id or file_subject or "unnamed"

        base_dir = Path(args.output_folder)
        base_dir.mkdir(parents=True, exist_ok=True)
        dir_name = f"{subject_id}_{args.model}_{data_suffix}"
        out_dir = base_dir / dir_name
        if args.overwrite and out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
        elif not args.overwrite and out_dir.exists():
            idx = 0
            candidate = out_dir
            while candidate.exists():
                idx += 1
                candidate = base_dir / f"{dir_name}_{idx}"
            out_dir = candidate
        summary = run_single_prediction(
            pipeline=pipeline,
            estimator=est,
            X_new=X_new,
            missing_cols=missing_cols,
            extra_cols=extra_cols,
            X_wide=X_wide,
            source_df=seg_df,
            pos_idx=pos_idx,
            threshold=THRESHOLD,
            classes=classes,
            label_mapping=label_mapping,
            roi_mapping=roi_mapping,
            out_dir=out_dir,
            input_csv=input_csv,
            model_path=model_path,
            model_name=args.model,
            gm_thresholded=args.GM_thrs,
            threshold_target=args.thrs_target,
            subject_id=subject_id,
        )
        results.append(summary)
    else:
        if seg_df.empty:
            raise ValueError("Subject-row CSV must contain at least one row for inference.")
        base_dir = Path("output_pred")
        base_dir.mkdir(parents=True, exist_ok=True)
        for row_idx, (_, row) in enumerate(seg_df.iterrows()):
            row_df = row.to_frame().T
            X_new, missing_cols, extra_cols, X_wide, subject_id = prepare_input_for_pipeline(
                row_df,
                pipeline,
                thresholded=args.GM_thrs,
                rename_map=roi_mapping,
            )
            label_id = subject_id or f"row{row_idx}"
            dir_name = f"{label_id}_{args.model}_{data_suffix}"
            out_dir = base_dir / dir_name
            if args.overwrite and out_dir.exists():
                import shutil
                shutil.rmtree(out_dir)
            elif not args.overwrite and out_dir.exists():
                out_dir = base_dir / f"{dir_name}_{row_idx}"

            summary = run_single_prediction(
                pipeline=pipeline,
                estimator=est,
                X_new=X_new,
                missing_cols=missing_cols,
                extra_cols=extra_cols,
                X_wide=X_wide,
                source_df=row_df,
                pos_idx=pos_idx,
                threshold=THRESHOLD,
                classes=classes,
                label_mapping=label_mapping,
                roi_mapping=roi_mapping,
                out_dir=out_dir,
                input_csv=input_csv,
                model_path=model_path,
                model_name=args.model,
                gm_thresholded=args.GM_thrs,
                threshold_target=args.thrs_target,
                subject_id=label_id,
            )
            results.append(summary)

    if len(results) > 1:
        print(
            json.dumps(
                {
                    "subjects_processed": len(results),
                    "output_folders": [r["output_folder"] for r in results],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
