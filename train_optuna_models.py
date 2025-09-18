#!/usr/bin/env python3
"""Train AD classification models with Optuna and persist reproducible artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna.exceptions import TrialPruned
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from optuna_search_spaces import SEARCH_SPACES
from dfstruct import DFStruct

TARGET_COL = "CDR"
TARGET_MAPPING = {"CN": 0, "AD": 1}
DATA_SPLIT_SEED = 422
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_SPLITS = 5
METRIC_NAME = "roc_auc"


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_raw: pd.Series
    y_test_raw: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna studies for AD classifiers.")
    parser.add_argument(
        "--dfs-pickle",
        type=Path,
        default=Path("data/dfs.pkl"),
        help="Path to the serialized DFStruct object (default: data/dfs.pkl)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory in which to store trained models and artifacts (default: ./models)",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("optuna_ml_ad.db"),
        help="SQLite file to use for Optuna storage (default: optuna_ml_ad.db)",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default="adclf_tiv_unnorm",
        help="Prefix used when registering Optuna study names.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials per study (default: 100)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=DEFAULT_CV_SPLITS,
        help="Number of folds for StratifiedKFold (default: 5)",
    )
    return parser.parse_args()


def resolve_path(candidate: Path) -> Path:
    if candidate.exists():
        return candidate
    # Try relative to the script directory
    script_dir = Path(__file__).resolve().parent
    alt = (script_dir / candidate).resolve()
    if alt.exists():
        return alt
    # Try one directory up (common when scripts live in repo root but data is outside)
    alt_parent = (script_dir.parent / candidate).resolve()
    if alt_parent.exists():
        return alt_parent
    return candidate


def load_dfs(dfs_path: Path) -> Any:
    dfs_path = resolve_path(dfs_path)
    if not dfs_path.exists():
        raise FileNotFoundError(f"Could not locate DFStruct pickle at {dfs_path}")
    
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("__main__", "DFStruct"):
                return DFStruct
            return super().find_class(module, name)

    with open(dfs_path, "rb") as fh:
        return _CompatUnpickler(fh).load()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers: List[Tuple[str, Any, List[str]]] = []
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_pipe, num_cols))

    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(model_name: str, params: Dict[str, Any], seed: int) -> Any:
    if model_name == "lgbm":
        return LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, **params)
    if model_name == "extratrees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"Unsupported model type: {model_name}")


def hash_dataframe(df: pd.DataFrame, sample_size: int = 1000) -> str:
    sample = df.sample(min(len(df), sample_size), random_state=0).reset_index(drop=True)
    return hashlib.md5(pd.util.hash_pandas_object(sample, index=True).values).hexdigest()


def prepare_dataset(dfs: Any, threshold: bool) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, int]]:
    train_df = dfs.get_for_training(
        threshold=threshold,
        linking="bias",
        tiv_norm=False,
        num_cats=2,
        sigma=4,
        allowed_outliers=5,
        vent_csf=False,
    )
    df = train_df.copy(deep=True)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Column '{TARGET_COL}' not found in dataframe returned by dfs.get_for_training")

    y_raw = df[TARGET_COL]
    mask = y_raw.isin(TARGET_MAPPING)
    dropped = (~mask).sum()
    if dropped:
        drop_summary = df.loc[~mask, TARGET_COL].value_counts(dropna=False).to_dict()
        print(
            f"Dropping {dropped} rows with unsupported labels for target '{TARGET_COL}': {drop_summary}"
        )
    X = df.loc[mask].drop(columns=[TARGET_COL])
    y_filtered = y_raw.loc[mask]
    y_encoded = y_filtered.map(TARGET_MAPPING).astype("int8")
    return X, y_encoded, y_filtered, TARGET_MAPPING


def stratified_split(
    X: pd.DataFrame,
    y_encoded: pd.Series,
    y_raw: pd.Series,
    test_size: float,
    seed: int,
) -> DatasetSplits:
    split = train_test_split(
        X,
        y_encoded,
        y_raw,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = split
    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_train_raw=y_train_raw.reset_index(drop=True),
        y_test_raw=y_test_raw.reset_index(drop=True),
        X_full=X.reset_index(drop=True),
        y_full=y_encoded.reset_index(drop=True),
    )


def determine_cv_splits(y: pd.Series, desired: int) -> int:
    minority = int(y.value_counts().min())
    if minority < 2:
        raise ValueError("Not enough samples in the minority class to perform stratified CV.")
    return max(2, min(desired, minority))


def objective_factory(
    model_name: str,
    search_space_fn,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        fold_scores: List[float] = []
        best_iters: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            prep = clone(preprocessor)
            X_tr_t = prep.fit_transform(X_tr, y_tr)
            X_val_t = prep.transform(X_val)

            estimator = build_estimator(model_name, params, seed)

            if model_name == "lgbm":
                callbacks = [
                    early_stopping(stopping_rounds=100, verbose=False),
                    LightGBMPruningCallback(trial, "auc"),
                ]
                estimator.fit(
                    X_tr_t,
                    y_tr,
                    eval_set=[(X_val_t, y_val)],
                    eval_metric="auc",
                    callbacks=callbacks,
                )
                best_iter = getattr(estimator, "best_iteration_", None)
                if best_iter is not None:
                    best_iters.append(float(best_iter))
            else:
                estimator.fit(X_tr_t, y_tr)

            y_proba = estimator.predict_proba(X_val_t)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            fold_scores.append(score)
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        mean_score = float(np.mean(fold_scores))
        if best_iters:
            trial.set_user_attr("mean_best_iteration", float(np.mean(best_iters)))
        return mean_score

    return objective


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_splits(combo_dir: Path, splits: DatasetSplits) -> Dict[str, str]:
    paths = {
        "X_train": combo_dir / "X_train.csv",
        "X_test": combo_dir / "X_test.csv",
        "y_train": combo_dir / "y_train.csv",
        "y_test": combo_dir / "y_test.csv",
    }
    splits.X_train.to_csv(paths["X_train"], index=False)
    splits.X_test.to_csv(paths["X_test"], index=False)
    splits.y_train_raw.to_frame(name=TARGET_COL).to_csv(paths["y_train"], index=False)
    splits.y_test_raw.to_frame(name=TARGET_COL).to_csv(paths["y_test"], index=False)
    return {name: str(path) for name, path in paths.items()}


def evaluate_pipeline(pipeline: Pipeline, splits: DatasetSplits) -> Dict[str, float]:
    y_proba = pipeline.predict_proba(splits.X_test)[:, 1]
    y_pred = pipeline.predict(splits.X_test)
    return {
        "roc_auc": float(roc_auc_score(splits.y_test, y_proba)),
        "f1": float(f1_score(splits.y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "precision": float(precision_score(splits.y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(splits.y_test, y_pred, zero_division=0)),
    }


def run_training_for_combo(
    model_name: str,
    threshold: bool,
    dfs: Any,
    storage_url: str,
    models_dir: Path,
    study_prefix: str,
    n_trials: int,
    desired_cv_splits: int,
) -> Dict[str, Any]:
    start_time = datetime.utcnow()
    suffix = "thrs" if threshold else "unthrs"
    study_name = f"{study_prefix}_{model_name}_{suffix}"
    combo_dir = ensure_dir(models_dir / model_name / suffix)

    X, y_encoded, y_raw, mapping = prepare_dataset(dfs, threshold)
    splits = stratified_split(X, y_encoded, y_raw, DEFAULT_TEST_SIZE, DATA_SPLIT_SEED)
    save_paths = save_splits(combo_dir, splits)

    preprocessor = build_preprocessor(splits.X_train)
    n_splits = determine_cv_splits(splits.y_train, desired_cv_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DATA_SPLIT_SEED)

    search_space_fn = SEARCH_SPACES[model_name]
    sampler = TPESampler(seed=DATA_SPLIT_SEED)
    pruner = PatientPruner(MedianPruner(n_warmup_steps=1), patience=1)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    objective = objective_factory(
        model_name,
        search_space_fn,
        preprocessor,
        splits.X_train,
        splits.y_train,
        cv,
        DATA_SPLIT_SEED,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_trial.params)
    if model_name == "lgbm":
        mean_best_iter = study.best_trial.user_attrs.get("mean_best_iteration")
        if mean_best_iter is not None:
            best_params["n_estimators"] = int(max(50, round(mean_best_iter)))

    # Train pipeline on training split only (evaluation split remains unseen)
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(splits.X_train)),
            ("model", build_estimator(model_name, best_params, DATA_SPLIT_SEED)),
        ]
    )
    final_pipeline.fit(splits.X_train, splits.y_train)
    metrics = evaluate_pipeline(final_pipeline, splits)

    model_path = combo_dir / "model.joblib"
    joblib.dump(final_pipeline, model_path)

    best_params_path = combo_dir / "best_params.json"
    with open(best_params_path, "w") as fh:
        json.dump(best_params, fh, indent=2)

    trials_df_path = combo_dir / "trials.csv"
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(trials_df_path, index=False)
    except Exception:
        trials_df_path = None

    metadata = {
        "model": model_name,
        "threshold": threshold,
        "study_name": study_name,
        "storage": storage_url,
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "metric": METRIC_NAME,
        "metrics": metrics,
        "n_trials": int(len(study.trials)),
        "cv_splits": n_splits,
        "target_mapping": mapping,
        "dataset_rows": int(len(splits.X_train)),
        "dataset_cols": int(splits.X_train.shape[1]),
        "dataset_hash": {
            "X_train": hash_dataframe(splits.X_train),
            "y_train": hashlib.md5(
                pd.util.hash_pandas_object(splits.y_train, index=True).values
            ).hexdigest(),
            "X_test": hash_dataframe(splits.X_test),
            "y_test": hashlib.md5(
                pd.util.hash_pandas_object(splits.y_test, index=True).values
            ).hexdigest(),
        },
        "artifacts": {
            "model": str(model_path),
            "best_params": str(best_params_path),
            "trials": str(trials_df_path) if trials_df_path else "",
            "splits": save_paths,
        },
    }

    metadata_path = combo_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    log_entry = {
        "model": model_name,
        "threshold": threshold,
        "study_name": study_name,
        "storage": storage_url,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "metric": METRIC_NAME,
        "best_value": float(study.best_value),
        "metrics": metrics,
        "n_trials": int(len(study.trials)),
        "best_params": best_params,
        "artifacts": {
            "model": str(model_path),
            "best_params": str(best_params_path),
            "metadata": str(metadata_path),
            "trials": str(trials_df_path) if trials_df_path else "",
            "splits": save_paths,
        },
    }

    return log_entry


def main() -> None:
    args = parse_args()

    dfs = load_dfs(args.dfs_pickle)
    models_dir = ensure_dir(resolve_path(args.models_dir))
    logs_dir = ensure_dir(models_dir / "logs")
    storage_path = resolve_path(args.storage)
    storage_url = f"sqlite:///{storage_path}"

    combinations = [
        ("lgbm", True),
        ("lgbm", False),
        ("extratrees", True),
        ("extratrees", False),
    ]

    results = []
    for model_name, threshold in combinations:
        print(f"\n=== Running study for model={model_name}, threshold={threshold} ===")
        log_entry = run_training_for_combo(
            model_name=model_name,
            threshold=threshold,
            dfs=dfs,
            storage_url=storage_url,
            models_dir=models_dir,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            desired_cv_splits=args.cv_splits,
        )
        timestamp = datetime.fromisoformat(log_entry["start_time"]).strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"{timestamp}_{model_name}_{'thrs' if threshold else 'unthrs'}.json"
        with open(log_path, "w") as fh:
            json.dump(log_entry, fh, indent=2)
        print(
            f"Completed study '{log_entry['study_name']}' | best {METRIC_NAME}={log_entry['best_value']:.4f}"
        )
        results.append((log_path, log_entry))

    print("\nArtifacts logged:")
    for log_path, entry in results:
        print(f"- {entry['study_name']}: {log_path}")


if __name__ == "__main__":
    main()
