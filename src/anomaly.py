"""Distance-based anomaly scoring in embedding space.

This is the whole decision layer. Given the embeddings of a machine's healthy
recordings, score a new clip by how far it sits from them. Two scores, both
standard in DCASE Task 2 systems:

  * ``MahalanobisScorer`` fits a mean and a shrinkage covariance (Ledoit-Wolf)
    on the healthy embeddings and returns the Mahalanobis distance. When source
    and target healthy sets are both given it keeps a covariance for each and
    takes the smaller distance, matching the DCASE "selective Mahalanobis"
    baseline.
  * ``KnnScorer`` returns the mean cosine distance to the k nearest healthy
    embeddings.

Everything here is a few matrix ops on the CPU. It is never exported to ONNX.
On the phone the fitted statistics are what "enrollment" produces and stores.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None


def _fit_gaussian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    xc = x - mu
    if LedoitWolf is not None and len(x) > 2:
        cov = LedoitWolf().fit(xc).covariance_
    else:
        cov = np.cov(xc, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    inv = np.linalg.pinv(cov)
    return mu.astype(np.float64), inv.astype(np.float64)


class MahalanobisScorer:
    def __init__(self) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, source: np.ndarray, target: np.ndarray | None = None) -> "MahalanobisScorer":
        self.stats["source"] = _fit_gaussian(np.asarray(source, dtype=np.float64))
        if target is not None and len(target) >= 2:
            self.stats["target"] = _fit_gaussian(np.asarray(target, dtype=np.float64))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        dists = []
        for mu, inv in self.stats.values():
            xc = x - mu
            dists.append(np.sqrt(np.einsum("ij,jk,ik->i", xc, inv, xc)))
        return np.min(np.stack(dists, axis=0), axis=0)

    def to_dict(self) -> dict:
        return {k: {"mean": mu.tolist(), "inv_cov": inv.tolist()} for k, (mu, inv) in self.stats.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "MahalanobisScorer":
        s = cls()
        s.stats = {k: (np.array(v["mean"]), np.array(v["inv_cov"])) for k, v in d.items()}
        return s


class KnnScorer:
    def __init__(self, k: int = 4) -> None:
        self.k = k
        self.ref: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray | None = None) -> "KnnScorer":
        ref = np.asarray(source, dtype=np.float64)
        if target is not None and len(target):
            ref = np.vstack([ref, np.asarray(target, dtype=np.float64)])
        self.ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.ref is not None
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        sim = x @ self.ref.T
        k = min(self.k, self.ref.shape[0])
        topk = np.partition(sim, -k, axis=1)[:, -k:]
        return 1.0 - topk.mean(axis=1)


SCORERS = {"maha": MahalanobisScorer, "knn": KnnScorer}
