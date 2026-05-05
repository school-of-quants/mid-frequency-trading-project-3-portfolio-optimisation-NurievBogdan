"""Model training with PurgedKFold and CPCV diagnostics"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from equity_project.src.utils import PROJECT_PATH, extract_dates, load_config, safe_dataframe_to_parquet, save_dict
from equity_project.src.validation import CombinatorialPurgedCV, PurgedKFold


def instantiate_model(random_seed: int = 42, iterations: int = 500) -> CatBoostClassifier:
    """Create the CatBoost classifier used for stock ranking"""
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.035,
        l2_leaf_reg=8.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=random_seed,
        early_stopping_rounds=60,
        allow_writing_files=False,
        verbose=False,
    )


def load_training_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load, align, and clean training features/labels"""
    X = pd.read_parquet(PROJECT_PATH / "data" / "processed" / "X_train.parquet")
    y = pd.read_parquet(PROJECT_PATH / "data" / "processed" / "y_train.parquet")["target"]

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].replace([np.inf, -np.inf], np.nan)
    y = y.loc[common_index]

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)


    keep_cols = X.isna().mean() < 0.95
    X = X.loc[:, keep_cols]
    X = X.sort_index()
    y = y.loc[X.index]
    return X, y


def multiclass_logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute multiclass log-loss without depending on sklearn"""
    eps = 1e-12
    proba = np.clip(proba, eps, 1.0 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true.astype(int)])))


def evaluate_proba(y_true: pd.Series, proba: np.ndarray) -> Dict[str, float]:
    """Return lightweight classification metrics for probability predictions"""
    pred_class = proba.argmax(axis=1)
    y_arr = y_true.values.astype(int)
    return {
        "accuracy": float((pred_class == y_arr).mean()),
        "logloss": multiclass_logloss(y_arr, proba),
    }


def run_purged_kfold_cv(X: pd.DataFrame, y: pd.Series, cfg: Dict) -> pd.DataFrame:
    """Run PurgedKFold validation and return fold-level model metrics"""
    cv = PurgedKFold(
        n_splits=int(cfg.get("n_splits", 5)),
        purge_days=int(cfg.get("purge_days", 10)),
        embargo_pct=float(cfg.get("embargo_pct", 0.01)),
    )
    rows = []
    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model = instantiate_model(random_seed=int(cfg.get("random_seed", 42)) + fold_id, iterations=350)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        proba = model.predict_proba(X.iloc[val_idx])
        metrics = evaluate_proba(y.iloc[val_idx], proba)
        val_dates = extract_dates(X.iloc[val_idx].index)
        rows.append(
            {
                "fold": fold_id,
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "validation_start": str(val_dates.min().date()),
                "validation_end": str(val_dates.max().date()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_cpcv_oos_predictions(X: pd.DataFrame, y: pd.Series, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate out-of-sample predictions using Combinatorial Purged CV

    The saved prediction matrix is later used for a CPCV portfolio backtest on
    the train/validation period. If a sample appears in several CPCV test paths,
    probabilities are averaged
    """
    cv = CombinatorialPurgedCV(
        n_groups=int(cfg.get("cpcv_n_groups", 6)),
        n_test_groups=int(cfg.get("cpcv_n_test_groups", 2)),
        purge_days=int(cfg.get("purge_days", 10)),
        embargo_pct=float(cfg.get("embargo_pct", 0.01)),
        max_combinations=cfg.get("cpcv_max_combinations", None),
    )

    prediction_frames = []
    metric_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = instantiate_model(random_seed=int(cfg.get("random_seed", 42)) + 100 + fold_id, iterations=300)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])
        pred = pd.DataFrame(proba, index=X.iloc[test_idx].index, columns=[0, 1, 2])
        pred["fold"] = fold_id
        prediction_frames.append(pred)

        metrics = evaluate_proba(y.iloc[test_idx], proba)
        test_dates = extract_dates(X.iloc[test_idx].index)
        metric_rows.append(
            {
                "fold": fold_id,
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                **metrics,
            }
        )

    if not prediction_frames:
        return pd.DataFrame(), pd.DataFrame(metric_rows)

    all_preds = pd.concat(prediction_frames).sort_index()
    averaged_preds = all_preds[[0, 1, 2]].groupby(level=["Date", "Ticker"]).mean()
    metrics = pd.DataFrame(metric_rows)
    return averaged_preds, metrics


def chronological_train_eval_split(X: pd.DataFrame, y: pd.Series, eval_fraction: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the final chronological train/evaluation split for early stopping"""
    dates = pd.DatetimeIndex(sorted(extract_dates(X.index).unique()))
    split_point = int(len(dates) * (1.0 - eval_fraction))
    eval_start = dates[max(0, min(split_point, len(dates) - 1))]
    sample_dates = extract_dates(X.index)
    train_mask = sample_dates < eval_start
    eval_mask = sample_dates >= eval_start
    return X.loc[train_mask], X.loc[eval_mask], y.loc[train_mask], y.loc[eval_mask]


def train() -> None:
    """Train the final model and save validation diagnostics/artifacts"""
    os.makedirs(PROJECT_PATH / "models", exist_ok=True)
    os.makedirs(PROJECT_PATH / "artifacts" / "metrics", exist_ok=True)

    cfg = load_config()
    X, y = load_training_dataset()

    feature_columns = list(X.columns)
    save_dict({"feature_columns": feature_columns}, PROJECT_PATH / "models" / "feature_columns.json")

    purged_metrics = run_purged_kfold_cv(X, y, cfg)
    safe_dataframe_to_parquet(purged_metrics, PROJECT_PATH / "artifacts" / "metrics" / "purged_kfold_metrics.parquet")
    save_dict(
        {
            "purged_kfold_mean_accuracy": float(purged_metrics["accuracy"].mean()) if not purged_metrics.empty else None,
            "purged_kfold_mean_logloss": float(purged_metrics["logloss"].mean()) if not purged_metrics.empty else None,
            "folds": purged_metrics.to_dict(orient="records"),
        },
        PROJECT_PATH / "artifacts" / "metrics" / "purged_kfold_metrics.json",
    )

    if bool(cfg.get("run_cpcv", True)):
        cpcv_preds, cpcv_metrics = run_cpcv_oos_predictions(X, y, cfg)
        if not cpcv_preds.empty:
            safe_dataframe_to_parquet(cpcv_preds, PROJECT_PATH / "data" / "processed" / "cpcv_oos_predictions.parquet")
        safe_dataframe_to_parquet(cpcv_metrics, PROJECT_PATH / "artifacts" / "metrics" / "cpcv_model_metrics.parquet")
        save_dict(
            {
                "cpcv_mean_accuracy": float(cpcv_metrics["accuracy"].mean()) if not cpcv_metrics.empty else None,
                "cpcv_mean_logloss": float(cpcv_metrics["logloss"].mean()) if not cpcv_metrics.empty else None,
                "folds": cpcv_metrics.to_dict(orient="records"),
            },
            PROJECT_PATH / "artifacts" / "metrics" / "cpcv_model_metrics.json",
        )

    X_train, X_eval, y_train, y_eval = chronological_train_eval_split(X, y)
    model = instantiate_model(random_seed=int(cfg.get("random_seed", 42)), iterations=700)
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval))
    joblib.dump(model, PROJECT_PATH / "models" / "model.joblib")


if __name__ == "__main__":
    train()
