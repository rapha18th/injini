"""NumPy log-mel vs EfficientAT's AugmentMelSTFT reference."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
import features as F  # noqa: E402


@pytest.mark.parametrize("seconds", [3.0, 6.5, 10.0])
def test_numpy_logmel_matches_reference(seconds):
    rng = np.random.RandomState(7)
    n = int(F.SR * seconds)
    t = np.arange(n) / F.SR
    y = (0.05 * rng.randn(n) + 0.03 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    a = F.logmel(y)
    b = F.reference_logmel(y)
    m = min(a.shape[1], b.shape[1])
    max_abs = np.abs(a[:, :m] - b[:, :m]).max()
    assert max_abs < 1e-3, f"max abs diff {max_abs:.2e}"


def test_fixed_length_shape():
    y = np.zeros(F.SR * 4, dtype=np.float32)
    out = F.logmel(y, fixed=True)
    assert out.shape == (F.N_MELS, F.N_FRAMES)
