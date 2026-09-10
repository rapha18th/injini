"""DCASE Task 2 scoring: AUC, partial AUC (p=0.1), and the official harmonic-mean score.

pAUC here is the McClish-corrected normalised partial AUC over the FPR range
[0, p], i.e. sklearn's ``roc_auc_score(..., max_fpr=p)``. That matches the DCASE
definition: for a random scorer it is 0.5, not p/2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

P = 0.1


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def partial_auc(y_true: np.ndarray, y_score: np.ndarray, p: float = P) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, max_fpr=p))


def harmonic_mean(values: list[float]) -> float:
    v = np.asarray([x for x in values if x == x and x > 0], dtype=np.float64)
    if len(v) == 0:
        return float("nan")
    return float(len(v) / np.sum(1.0 / v))


def official_score(per_machine: dict[str, dict]) -> float:
    """per_machine[name] = {'auc_source':.., 'auc_target':.., 'pauc':..}."""
    pieces: list[float] = []
    for m in per_machine.values():
        pieces += [m.get("auc_source", np.nan), m.get("auc_target", np.nan), m.get("pauc", np.nan)]
    return harmonic_mean(pieces)
