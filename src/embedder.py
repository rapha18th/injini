"""Frozen EfficientAT MobileNetV3 embedder.

The state-of-the-art DCASE Task 2 recipe scores a clip by its distance, in the
embedding space of a large frozen audio network, from the healthy recordings of
that machine. Injini keeps the recipe and swaps the network for a MobileNetV3
distilled from a transformer teacher on AudioSet (EfficientAT ``mn10_as`` /
``mn04_as``), which is small enough to quantise onto an Arm phone.

This module wraps EfficientAT's ``MN`` so ``forward`` returns only the
L2-normalised embedding (``F.adaptive_avg_pool2d`` of the last feature map), and
exports that subgraph to ONNX. The anomaly score is computed outside the graph
(see ``anomaly.py``); it is a few lines of linear algebra and never needs Arm
acceleration.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(_HERE, os.pardir, "vendor_efficientat")

WIDTH = {"mn10_as": 1.0, "mn04_as": 0.4, "mn05_as": 0.5, "mn01_as": 0.1}
EMBED_DIM = {"mn10_as": 960, "mn04_as": 384, "mn05_as": 480, "mn01_as": 96}


def _load_mn(name: str) -> nn.Module:
    # vendor_efficientat/helpers/utils.py reads metadata/class_labels_indices.csv
    # with a cwd-relative path at import time, so run the import from VENDOR.
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    cwd = os.getcwd()
    try:
        os.chdir(VENDOR)
        from models.mn.model import get_model

        with contextlib.redirect_stdout(io.StringIO()):
            model = get_model(pretrained_name=name, width_mult=WIDTH[name], head_type="mlp")
    finally:
        os.chdir(cwd)
    return model.eval()


class Embedder(nn.Module):
    """(B, 1, 128, T) log-mel  ->  (B, D) L2-normalised embedding."""

    def __init__(self, name: str = "mn10_as", l2: bool = True):
        super().__init__()
        self.name = name
        self.l2 = l2
        self.features = _load_mn(name).features
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        if self.l2:
            x = F.normalize(x, dim=1)
        return x


def export_onnx(name: str, out_path: str, opset: int = 17) -> str:
    from features import N_FRAMES, N_MELS

    model = Embedder(name)
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["logmel"], output_names=["embedding"],
        dynamic_axes={"logmel": {0: "batch", 3: "frames"}, "embedding": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    return out_path


def param_counts(name: str) -> dict:
    m = Embedder(name)
    total = sum(p.numel() for p in m.parameters())
    return {"embedder": name, "embed_dim": EMBED_DIM[name], "params": total}


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, _HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mn10_as", choices=list(WIDTH))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(_HERE, os.pardir, "models", f"injini_{args.name}_fp32.onnx")
    export_onnx(args.name, out)
    size = os.path.getsize(out) / 1e6
    print({**param_counts(args.name), "onnx_mb": round(size, 3), "path": out})
