"""Evaluate an embedder + distance scorer on the DCASE 2025 Task 2 dev set.

Two passes. First embed every machine's training clips (normal only) and fit a
PCA-whitener on the pooled set. Then per machine: whiten, fit the scorer on the
source and target healthy embeddings, score the test clips, compute source AUC,
target AUC and partial AUC. The official score is the harmonic mean of all
three across all machines.

    python src/eval_dcase.py --root data/dcase2025_dev --backend torch:mn10_as --scorer maha
    python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_int8.onnx --scorer knn
    python src/eval_dcase.py --root data/dcase2025_dev --backend passt --whiten 0

Writes a metrics JSON to models/eval_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import features as F          # noqa: E402
import dcase_data as D        # noqa: E402
import metrics as M           # noqa: E402
from anomaly import SCORERS, Whitener  # noqa: E402
from embed_backends import get_backend  # noqa: E402


def _load(paths: list[str]) -> list[np.ndarray]:
    return [F.load_audio(p) for p in paths]


def evaluate(root: str, backend_spec: str, scorer_name: str, whiten: int = 128,
             machines=None, limit=None) -> dict:
    backend = get_backend(backend_spec)
    clips = D.load(root, machines)
    if not clips:
        raise SystemExit(f"no clips under {root}")
    by_machine = D.by_machine(clips)
    t0 = time.time()

    # pass 1: embed every machine's training clips
    train_emb: dict[str, dict] = {}
    for machine, mclips in by_machine.items():
        train = [c for c in mclips if c.split == "train"]
        if limit:
            train = train[:limit]
        if not train:
            continue
        emb = backend.embed(_load([c.path for c in train]))
        train_emb[machine] = {
            "all": emb,
            "source": emb[[i for i, c in enumerate(train) if c.domain == "source"]],
            "target": emb[[i for i, c in enumerate(train) if c.domain == "target"]],
        }

    whitener = None
    if whiten and whiten > 0:
        stack = np.vstack([v["all"] for v in train_emb.values()])
        whitener = Whitener(whiten).fit(stack)

    def wt(x):
        return whitener.transform(x) if whitener is not None else x

    # pass 2: per machine
    per_machine: dict[str, dict] = {}
    for machine, mclips in by_machine.items():
        if machine not in train_emb:
            continue
        test = [c for c in mclips if c.split == "test"]
        if limit:
            test = test[:limit]
        if not test:
            continue

        src = wt(train_emb[machine]["source"])
        tgt = wt(train_emb[machine]["target"])
        scorer = SCORERS[scorer_name]()
        scorer.fit(src if len(src) else wt(train_emb[machine]["all"]),
                   tgt if len(tgt) >= 2 else None)

        te = wt(backend.embed(_load([c.path for c in test])))
        scores = scorer.score(te)
        y = np.array([c.label for c in test])
        dom = np.array([c.domain for c in test])
        s_m, t_m = dom == "source", dom == "target"
        per_machine[machine] = {
            "auc_source": M.auc(y[s_m], scores[s_m]) if s_m.any() else float("nan"),
            "auc_target": M.auc(y[t_m], scores[t_m]) if t_m.any() else float("nan"),
            "pauc": M.partial_auc(y, scores),
            "n_train": int(len(train_emb[machine]["all"])), "n_test": int(len(test)),
        }
        print(f"  {machine:9s} AUC_src={per_machine[machine]['auc_source']:.4f} "
              f"AUC_tgt={per_machine[machine]['auc_target']:.4f} "
              f"pAUC={per_machine[machine]['pauc']:.4f}")

    aucs = [v["auc_source"] for v in per_machine.values()] + [v["auc_target"] for v in per_machine.values()]
    return {
        "backend": backend_spec,
        "scorer": scorer_name,
        "whiten": whiten,
        "official_score": M.official_score(per_machine),
        "mean_auc": float(np.nanmean(aucs)),
        "mean_pauc": float(np.nanmean([v["pauc"] for v in per_machine.values()])),
        "per_machine": per_machine,
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backend", default="torch:mn10_as")
    ap.add_argument("--scorer", default="knn", choices=list(SCORERS))
    ap.add_argument("--whiten", type=int, default=128, help="PCA-whiten dims; 0 disables")
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = evaluate(args.root, args.backend, args.scorer, args.whiten, args.machines, args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "per_machine"}, indent=2))

    tag = (args.backend.replace(":", "_").replace("/", "_").replace("\\", "_").replace(".onnx", "")
           + f"_{args.scorer}_w{args.whiten}")
    out = args.out or os.path.join(_HERE, os.pardir, "models", f"eval_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)
