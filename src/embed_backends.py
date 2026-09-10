"""Uniform embedding interface over the three backends the eval compares.

  * ``torch:<name>``   frozen EfficientAT MobileNetV3 in PyTorch (mn10_as / mn04_as)
  * ``onnx:<path>``    an exported / quantised embedder run through onnxruntime
  * ``passt``          the transformer teacher, the full-size reference ceiling
                       (needs ``hear21passt``; only used on Kaggle)

All backends take a list of waveforms at ``features.SR`` and return an
(N, D) float32 array of L2-normalised embeddings.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import features as F  # noqa: E402


def _batched_logmels(waves: list[np.ndarray]) -> np.ndarray:
    mels = [F.logmel(w, fixed=True) for w in waves]
    return np.stack(mels, axis=0)[:, None, :, :].astype(np.float32)  # (N,1,128,T)


class TorchBackend:
    def __init__(self, name: str):
        import torch
        from embedder import Embedder

        self.torch = torch
        self.model = Embedder(name)

    def embed(self, waves: list[np.ndarray], batch: int = 16) -> np.ndarray:
        x = _batched_logmels(waves)
        out = []
        for i in range(0, len(x), batch):
            t = self.torch.from_numpy(x[i:i + batch])
            with self.torch.no_grad():
                out.append(self.model(t).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class OnnxBackend:
    def __init__(self, path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.iname = self.sess.get_inputs()[0].name

    def embed(self, waves: list[np.ndarray], batch: int = 16) -> np.ndarray:
        x = _batched_logmels(waves)
        out = []
        for i in range(0, len(x), batch):
            out.append(self.sess.run(None, {self.iname: x[i:i + batch]})[0])
        e = np.concatenate(out, axis=0).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class PasstBackend:
    def __init__(self):
        from hear21passt.base import load_model, get_scene_embeddings

        self.model = load_model(mode="embed_only").eval()
        self._embed = get_scene_embeddings
        import torch

        self.torch = torch

    def embed(self, waves: list[np.ndarray], batch: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(waves), batch):
            chunk = waves[i:i + batch]
            n = max(len(w) for w in chunk)
            arr = np.zeros((len(chunk), n), dtype=np.float32)
            for j, w in enumerate(chunk):
                arr[j, : len(w)] = w
            t = self.torch.from_numpy(arr)
            with self.torch.no_grad():
                out.append(self._embed(t, self.model).cpu().numpy())
        e = np.concatenate(out, axis=0).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def get_backend(spec: str):
    if spec == "passt":
        return PasstBackend()
    if spec.startswith("onnx:"):
        return OnnxBackend(spec.split(":", 1)[1])
    if spec.startswith("torch:"):
        return TorchBackend(spec.split(":", 1)[1])
    raise ValueError(f"unknown backend spec: {spec!r}")
