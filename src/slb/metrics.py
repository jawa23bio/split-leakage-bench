"""Discrimination and calibration metrics for binary predictions."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def discrimination(y, p):
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}


def calibration(y, p, bins=10):
    """Brier score and expected calibration error (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}


def ensemble_uncertainty(y, member_probs):
    """member_probs: (n_models, n_examples). Returns mean prediction, spread, and
    whether spread tracks error (Spearman correlation between spread and |error|;
    positive means the ensemble is less sure where it is more wrong)."""
    from scipy.stats import spearmanr

    P = np.asarray(member_probs, float)
    mean, spread = P.mean(0), P.std(0)
    err = np.abs(mean - np.asarray(y, float))
    rho = spearmanr(spread, err).correlation
    return {"mean": mean, "spread": spread, "spread_vs_error_spearman": float(rho)}
