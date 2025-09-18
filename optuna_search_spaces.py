"""Search space helpers for Optuna-driven AD classification models."""

from __future__ import annotations

from typing import Any, Dict

import optuna


def lgbm_classifier_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Hyperparameter search space for LightGBM classifiers."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12, 14, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def extratrees_classifier_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Hyperparameter search space for ExtraTrees classifiers."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


SEARCH_SPACES = {
    "lgbm": lgbm_classifier_space,
    "extratrees": extratrees_classifier_space,
}
