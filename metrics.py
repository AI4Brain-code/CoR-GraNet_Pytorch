#!/usr/bin/env python3
"""Evaluation metrics for ASD/TD classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([row[metric] for row in rows if metric in row], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_folds(rows: list[dict]) -> dict[str, float]:
    summary = {}
    for metric in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "mcc", "auc"]:
        if any(metric in row for row in rows):
            mean, std = mean_std(rows, metric)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
    return summary


def weighted_site_summary(site_rows: list[dict], metric: str) -> tuple[float, float]:
    weights = np.asarray([row["n"] for row in site_rows], dtype=float)
    values = np.asarray([row[f"{metric}_mean"] for row in site_rows], dtype=float)
    norm_weights = weights / weights.sum()
    mean = float(np.nansum(values * norm_weights))
    std = float(np.sqrt(np.nansum(weights * (values - mean) ** 2) / np.nansum(weights)))
    return mean, std
