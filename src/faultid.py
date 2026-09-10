"""Supervised fault-identity head on frozen embeddings.

Secondary mode. Where a labelled corpus covers a fault family, a small MLP on
the same frozen embedding names the likely fault. Trained and evaluated with a
source-disjoint split (see prepare_engine_sounds.py) so the macro-F1 reflects
transfer to unseen recordings, not memorised ones.

    python src/faultid.py --manifest data/engine_sounds_manifest.csv --backend torch:mn10_as
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import features as F                        # noqa: E402
from embed_backends import get_backend      # noqa: E402


def read_manifest(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    classes = sorted({r["class_name"] for r in rows})
    return rows, {c: i for i, c in enumerate(classes)}


def embed_split(backend, rows, split, cap=None):
    sel = [r for r in rows if r["split"] == split]
    if cap:
        sel = sel[:cap]
    waves = [F.load_audio(r["path"]) for r in sel]
    emb = backend.embed(waves)
    y = np.array([int(r["label"]) for r in sel])
    return emb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backend", default="torch:mn10_as")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score, classification_report

    rows, cls_map = read_manifest(args.manifest)
    backend = get_backend(args.backend)

    Xtr, ytr = embed_split(backend, rows, "train", args.cap)
    Xva, yva = embed_split(backend, rows, "val", args.cap)
    n_cls = len(cls_map)

    clf = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, n_cls))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()
    for ep in range(args.epochs):
        clf.train()
        opt.zero_grad()
        loss = lossf(clf(Xt), yt)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(torch.from_numpy(Xva).float()).argmax(1).numpy()
    macro = f1_score(yva, pred, average="macro")
    print(classification_report(yva, pred, target_names=list(cls_map), zero_division=0))

    res = {"backend": args.backend, "macro_f1": float(macro),
           "n_train": int(len(ytr)), "n_val": int(len(yva)), "classes": list(cls_map)}
    out = os.path.join(_HERE, os.pardir, "models",
                       f"faultid_{args.backend.replace(':', '_').replace('/', '_')}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
