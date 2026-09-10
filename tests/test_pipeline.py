"""Embedder + anomaly scorer smoke tests (no dataset needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
import features as F  # noqa: E402
from anomaly import MahalanobisScorer, KnnScorer  # noqa: E402
from embed_backends import get_backend  # noqa: E402


def _wave(anom=False, n=6, seed=0):
    rng = np.random.RandomState(seed)
    y = (rng.randn(int(F.SR * n)) * 0.05).astype(np.float32)
    if anom:
        y += 0.05 * np.sin(2 * np.pi * 3200 * np.arange(len(y)) / F.SR).astype(np.float32)
    return y


def test_onnx_matches_torch():
    b_t = get_backend("torch:mn04_as")
    b_o = get_backend("onnx:models/injini_mn04_as_fp32.onnx")
    waves = [_wave(seed=i) for i in range(6)]
    et, eo = b_t.embed(waves), b_o.embed(waves)
    cos = (et * eo).sum(1) / (np.linalg.norm(et, axis=1) * np.linalg.norm(eo, axis=1))
    assert cos.min() > 0.99


def test_scorers_separate_obvious_anomaly():
    b = get_backend("onnx:models/injini_mn04_as_fp32.onnx")
    healthy = b.embed([_wave(seed=i) for i in range(10)])
    test = b.embed([_wave(seed=100 + i) for i in range(4)] + [_wave(anom=True, seed=200 + i) for i in range(4)])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    from sklearn.metrics import roc_auc_score
    for S in (MahalanobisScorer, KnnScorer):
        auc = roc_auc_score(y, S().fit(healthy).score(test))
        assert auc >= 0.9, f"{S.__name__} AUC {auc}"
