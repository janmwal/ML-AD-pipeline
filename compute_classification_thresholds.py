#!/usr/bin/env python3
"""Recompute decision thresholds for trained classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_MAPPING = {"CN": 0, "AD": 1}
MODELS = ("lgbm", "extratrees")
GM_THRESHOLDS = ("unthrs", "thrs")
STRATEGIES = ("youden", "sensitivity", "f1")
TARGET_SENSITIVITY = 0.90


def load_artifacts(model: str, gm_key: str, base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    combo_dir = base_dir / model / gm_key
    model_path = combo_dir / "model.joblib"
    x_test_path = combo_dir / "X_test.csv"
    y_test_path = combo_dir / "y_test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained pipeline at {model_path}")
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Expected test split CSVs in {combo_dir}; missing one of {x_test_path} / {y_test_path}"
        )

    try:
        pipeline = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        if exc.name == "lightgbm":
            raise ModuleNotFoundError(
                "lightgbm is required to unpickle the saved pipeline. Please install the same "
                "version used during training before running this script."
            ) from exc
        raise
    X_test = pd.read_csv(x_test_path)
    y_test_df = pd.read_csv(y_test_path)

    if "CDR" not in y_test_df.columns:
        raise KeyError(f"y_test.csv in {combo_dir} must contain a 'CDR' column")

    y_true_raw = y_test_df["CDR"].map(CLASS_MAPPING)
    if y_true_raw.isnull().any():
        missing = y_test_df.loc[y_true_raw.isnull(), "CDR"].unique().tolist()
        raise ValueError(f"Encountered unsupported labels {missing} in y_test.csv for {combo_dir}")

    y_true = y_true_raw.astype(int).to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return y_true, y_proba


def evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def compute_candidate_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> List[Dict[str, float]]:
    unique_probs = np.unique(y_proba)
    candidates = np.unique(np.concatenate(([0.0], unique_probs, [1.0])))
    metrics = []
    for threshold in candidates:
        metrics_dict = evaluate_threshold(y_true, y_proba, threshold)
        metrics_dict["threshold"] = float(threshold)
        metrics.append(metrics_dict)
    return metrics


def pick_thresholds(metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    youden = max(
        metrics,
        key=lambda m: (
            m["sensitivity"] + m["specificity"] - 1.0,
            m["specificity"],
            m["f1"],
        ),
    )

    eligible = [m for m in metrics if m["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible:
        sensitivity_target = max(
            eligible,
            key=lambda m: (
                m["specificity"],
                m["threshold"],
            ),
        )
    else:
        sensitivity_target = max(
            metrics,
            key=lambda m: (
                m["sensitivity"],
                m["specificity"],
            ),
        )

    best_f1 = max(
        metrics,
        key=lambda m: (
            m["f1"],
            m["specificity"],
            m["threshold"],
        ),
    )

    return {
        "youden": youden,
        "sensitivity": sensitivity_target,
        "f1": best_f1,
    }


def format_config_block(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> str:
    lines: List[str] = ["PREDICTION_THRESHOLDS = {"]
    for model_index, (model, gm_results) in enumerate(results.items()):
        lines.append(f"    '{model}' : {{")
        gm_keys = list(gm_results.keys())
        for gm_index, gm_key in enumerate(gm_keys):
            entry = gm_results[gm_key]
            auc = entry["auc"]
            thresholds = entry["thresholds"]
            lines.append(f"        '{gm_key}' : {{ # AUC-ROC: {auc:.3f}")
            strategy_keys = list(thresholds.keys())
            for strat_index, strat in enumerate(strategy_keys):
                info = thresholds[strat]
                thr = info["threshold"]
                metrics = info["metrics"]
                suffix = "," if strat_index < len(strategy_keys) - 1 else ""
                lines.append(
                    f"            '{strat}' : {thr:.4f}{suffix} # Sens={metrics['sensitivity']:.3f} "
                    f"Spec={metrics['specificity']:.3f} Prec={metrics['precision']:.3f} F1={metrics['f1']:.3f}"
                )
            closing = "        }" + ("," if gm_index < len(gm_keys) - 1 else "")
            lines.append(closing)
        model_closing = "    }" + ("," if model_index < len(results) - 1 else "")
        lines.append(model_closing)
    lines.append("}\n")
    lines.append('CLASS_LABELS = {0: "CN", 1: "AD"}\n')
    return "\n".join(lines)


def main() -> None:
    models_dir = Path("models")
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for model in MODELS:
        model_entries: Dict[str, Dict[str, Dict[str, float]]] = {}
        for gm_key in GM_THRESHOLDS:
            y_true, y_proba = load_artifacts(model, gm_key, models_dir)
            auc = roc_auc_score(y_true, y_proba)
            metrics = compute_candidate_metrics(y_true, y_proba)
            picked = pick_thresholds(metrics)

            structured: Dict[str, Dict[str, float]] = {}
            for strat, info in picked.items():
                structured[strat] = {
                    "threshold": info["threshold"],
                    "metrics": {
                        "sensitivity": info["sensitivity"],
                        "specificity": info["specificity"],
                        "precision": info["precision"],
                        "f1": info["f1"],
                    },
                }

            model_entries[gm_key] = {
                "auc": auc,
                "thresholds": structured,
            }
        results[model] = model_entries

    config_block = format_config_block(results)
    config_path = Path("config.py")
    config_path.write_text(config_block)
    print("Updated", config_path)


if __name__ == "__main__":
    main()
