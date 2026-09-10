"""The DCASE Task 2 autoencoder baseline, reproduced.

Identical in shape to the 2023-2025 official baseline: log-mel with 128 bands,
5 consecutive frames concatenated into each 640-d input vector, a dense
encoder/decoder, trained to minimise reconstruction MSE on normal sound only.
Anomaly score = mean reconstruction MSE over a clip's frames.

This exists so Section 10 can show we reproduce the published baseline before
claiming anything about beating it.

    python src/baseline_ae.py --epochs 100                 # HF mirror (default)
    python src/baseline_ae.py --source dir --root data/dcase2025_dev --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import features as F        # noqa: E402
import metrics as M         # noqa: E402

CTX = 5
IN_DIM = F.N_MELS * CTX


class DenseAE(nn.Module):
    def __init__(self, in_dim: int = IN_DIM):
        super().__init__()
        h = 128
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8), nn.BatchNorm1d(8), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(8, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, in_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def _frames_from_wave(y: np.ndarray) -> np.ndarray:
    mel = F.logmel(y)
    T = mel.shape[1]
    if T < CTX:
        mel = np.pad(mel, ((0, 0), (0, CTX - T)), mode="edge")
        T = CTX
    idx = np.arange(CTX)[None, :] + np.arange(T - CTX + 1)[:, None]
    return mel[:, idx].transpose(1, 0, 2).reshape(T - CTX + 1, -1).astype(np.float32)


def _wave_of(c):
    return c.wave if getattr(c, "wave", None) is not None else F.load_audio(c.path)


def train_machine(train_clips, epochs, device):
    X = np.concatenate([_frames_from_wave(_wave_of(c)) for c in train_clips], axis=0)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X)),
        batch_size=512, shuffle=True, drop_last=True,
    )
    net = DenseAE().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        net.train()
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            lossf(net(xb), xb).backward()
            opt.step()
    return net


@torch.no_grad()
def clip_score(net, c, device):
    net.eval()
    x = torch.from_numpy(_frames_from_wave(_wave_of(c))).to(device)
    return float(((x - net(x)) ** 2).mean().item())


def evaluate(source, root, epochs=100, machines=None, limit=None):
    # Kaggle's current torch build dropped sm_60 (P100), so default to CPU;
    # this net is tiny. Override with INJINI_DEVICE=cuda where the GPU works.
    device = os.environ.get("INJINI_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if source == "hf":
        import dcase_hf as H
        clips, by_machine = H.load(machines=machines, limit=limit), None
        from dcase_hf import by_machine as bm
        groups = bm(clips)
    else:
        import dcase_data as D
        groups = D.by_machine(D.load(root, machines))

    per_machine = {}
    for machine, mclips in groups.items():
        train = [c for c in mclips if c.split == "train"]
        test = [c for c in mclips if c.split == "test"]
        if not train or not test:
            continue
        net = train_machine(train, epochs, device)
        scores = np.array([clip_score(net, c, device) for c in test])
        y = np.array([c.label for c in test])
        dom = np.array([c.domain for c in test])
        per_machine[machine] = {
            "auc_source": M.auc(y[dom == "source"], scores[dom == "source"]),
            "auc_target": M.auc(y[dom == "target"], scores[dom == "target"]),
            "pauc": M.partial_auc(y, scores),
        }
        print(f"  {machine:9s} {per_machine[machine]}")
    aucs = [v["auc_source"] for v in per_machine.values()] + [v["auc_target"] for v in per_machine.values()]
    return {
        "system": "dcase_ae_baseline_mse", "epochs": epochs, "source": source,
        "official_score": M.official_score(per_machine),
        "mean_auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "mean_pauc": float(np.nanmean([v["pauc"] for v in per_machine.values()])) if per_machine else float("nan"),
        "per_machine": per_machine,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf", choices=["hf", "dir"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    res = evaluate(args.source, args.root, args.epochs, args.machines, args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "per_machine"}, indent=2))
    out = os.path.join(_HERE, os.pardir, "models", "eval_dcase_ae_baseline.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)
