"""Distance-based anomaly scoring in embedding space.

Given the embeddings of a machine's healthy recordings, score a new clip by how
far it sits from them. This is the whole decision layer, and none of it is
exported to ONNX.

Three ingredients, all standard in DCASE Task 2 systems:

  1. Whitening. A raw AudioSet embedding has a few high-variance directions that
     dominate any distance. ``Whitener`` fits PCA-whitening on the pooled
     healthy embeddings of every machine and keeps the top components, so the
     distance is not just measuring loudness or pitch.
  2. A per-machine scorer, ``MahalanobisScorer`` (selective source/target) or
     ``KnnScorer`` (mean cosine distance to the k nearest healthy embeddings).
  3. Per-machine score normalisation. The healthy training clips are scored
     against their own reference (leave-one-out for kNN); a test score is then
     standardised by that healthy-score mean and standard deviation. This
     removes the "some healthy clips are wild outliers" scale problem and makes
     machines comparable before the harmonic mean.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None


class Whitener:
    """PCA-whitening fitted on pooled healthy embeddings."""

    def __init__(self, n_components: int = 128):
        self.n = n_components
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Whitener":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n, Vt.shape[0])
        comps = Vt[:k]
        scale = np.sqrt(len(X)) / (S[:k] + 1e-8)
        self.W_ = (comps * scale[:, None]).T          # (D, k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) @ self.W_
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _fit_gaussian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    xc = x - mu
    if LedoitWolf is not None and len(x) > 2:
        cov = LedoitWolf().fit(xc).covariance_
    else:
        cov = np.cov(xc, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    inv = np.linalg.pinv(cov)
    return mu.astype(np.float64), inv.astype(np.float64)


class _NormMixin:
    """Standardise test scores by the healthy training clips' own scores."""

    def _calibrate(self, train_scores: np.ndarray) -> None:
        self._mu = float(np.mean(train_scores))
        self._sd = float(np.std(train_scores) + 1e-8)

    def _norm(self, s: np.ndarray) -> np.ndarray:
        return (np.asarray(s) - self._mu) / self._sd


class MahalanobisScorer(_NormMixin):
    def __init__(self, normalize: bool = True) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.normalize = normalize
        self._mu, self._sd = 0.0, 1.0

    def fit(self, source: np.ndarray, target: np.ndarray | None = None) -> "MahalanobisScorer":
        source = np.asarray(source, dtype=np.float64)
        self.stats["source"] = _fit_gaussian(source)
        if target is not None and len(target) >= 8:
            self.stats["target"] = _fit_gaussian(np.asarray(target, dtype=np.float64))
        if self.normalize:
            train = source if target is None or len(target) < 2 else np.vstack([source, target])
            self._calibrate(self._raw(train))
        return self

    def _raw(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        dists = []
        for mu, inv in self.stats.values():
            xc = x - mu
            dists.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", xc, inv, xc), 0.0)))
        return np.min(np.stack(dists, axis=0), axis=0)

    def score(self, x: np.ndarray) -> np.ndarray:
        s = self._raw(x)
        return self._norm(s) if self.normalize else s


class KnnScorer(_NormMixin):
    def __init__(self, k: int = 2, normalize: bool = True) -> None:
        self.k = k
        self.normalize = normalize
        self.ref: np.ndarray | None = None
        self._mu, self._sd = 0.0, 1.0

    def fit(self, source: np.ndarray, target: np.ndarray | None = None) -> "KnnScorer":
        ref = np.asarray(source, dtype=np.float64)
        if target is not None and len(target):
            ref = np.vstack([ref, np.asarray(target, dtype=np.float64)])
        self.ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        if self.normalize:
            self._calibrate(self._loo_scores())
        return self

    def _knn(self, x: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        sim = x @ self.ref.T
        if exclude_self:
            np.fill_diagonal(sim, -np.inf)
        k = min(self.k, sim.shape[1] - (1 if exclude_self else 0))
        topk = np.partition(sim, -k, axis=1)[:, -k:]
        return 1.0 - topk.mean(axis=1)

    def _loo_scores(self) -> np.ndarray:
        return self._knn(self.ref, exclude_self=True)

    def score(self, x: np.ndarray) -> np.ndarray:
        s = self._knn(x)
        return self._norm(s) if self.normalize else s


SCORERS = {"maha": MahalanobisScorer, "knn": KnnScorer}
