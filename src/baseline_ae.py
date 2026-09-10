"""The DCASE Task 2 autoencoder baseline, reproduced.

Identical in shape to the 2023-2025 official baseline: log-mel with 128 bands,
5 consecutive frames concatenated into each 640-d input vector, a dense
encoder/decoder, trained to minimise reconstruction MSE on normal sound only.
Two anomaly scores:

  * "mse"   : mean reconstruction error over a clip's frames
  * "maha"  : Mahalanobis distance between input and reconstruction under a
              covariance fitted on the training residuals (selective, source
              vs target, smaller distance kept)

This exists so Section 10 can show we reproduce the published baseline before
claiming anything about beating it. Runs on the same DCASE dev tree as
eval_dcase.py.

    python src/baseline_ae.py --root data/dcase2025_dev --epochs 100
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
import dcase_data as D      # noqa: E402
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


def _frames(path: str) -> np.ndarray:
    mel = F.logmel(F.load_audio(path))            # (128, T)
    T = mel.shape[1]
    if T < CTX:
        mel = np.pad(mel, ((0, 0), (0, CTX - T)), mode="edge")
        T = CTX
    idx = np.arange(CTX)[None, :] + np.arange(T - CTX + 1)[:, None]
    return mel[:, idx].transpose(1, 0, 2).reshape(T - CTX + 1, -1).astype(np.float32)


def train_machine(train_paths, epochs, device):
    X = np.concatenate([_frames(p) for p in train_paths], axis=0)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
    net = DenseAE().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        net.train()
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = lossf(net(xb), xb)
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def clip_score(net, path, device):
    net.eval()
    x = torch.from_numpy(_frames(path)).to(device)
    r = net(x)
    return float(((x - r) ** 2).mean().item())


def evaluate(root, epochs=100, machines=None, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clips = D.load(root, machines)
    per_machine = {}
    for machine, mclips in D.by_machine(clips).items():
        train = [c for c in mclips if c.split == "train"]
        test = [c for c in mclips if c.split == "test"]
        if limit:
            train, test = train[:limit], test[:limit]
        if not train or not test:
            continue
        net = train_machine([c.path for c in train], epochs, device)
        scores = np.array([clip_score(net, c.path, device) for c in test])
        y = np.array([c.label for c in test])
        dom = np.array([c.domain for c in test])
        per_machine[machine] = {
            "auc_source": M.auc(y[dom == "source"], scores[dom == "source"]),
            "auc_target": M.auc(y[dom == "target"], scores[dom == "target"]),
            "pauc": M.partial_auc(y, scores),
        }
        print(f"  {machine:9s} {per_machine[machine]}")
    return {
        "system": "dcase_ae_baseline_mse",
        "epochs": epochs,
        "official_score": M.official_score(per_machine),
        "per_machine": per_machine,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    res = evaluate(args.root, args.epochs, args.machines, args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "per_machine"}, indent=2))
    out = os.path.join(_HERE, os.pardir, "models", "eval_dcase_ae_baseline.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)
