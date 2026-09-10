"""Evaluate an embedder + distance scorer on the DCASE 2025 Task 2 dev set.

For each machine: embed the training clips (normal only), fit the scorer on the
source and target healthy embeddings, score every test clip, then compute
source AUC, target AUC and partial AUC. The official score is the harmonic mean
of all three across all machines.

    python src/eval_dcase.py --root data/dcase2025_dev --backend torch:mn10_as --scorer maha
    python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_int8.onnx

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
from anomaly import SCORERS   # noqa: E402
from embed_backends import get_backend  # noqa: E402


def _load(paths: list[str]) -> list[np.ndarray]:
    return [F.load_audio(p) for p in paths]


def evaluate(root: str, backend_spec: str, scorer_name: str, machines=None, limit=None) -> dict:
    backend = get_backend(backend_spec)
    clips = D.load(root, machines)
    if not clips:
        raise SystemExit(f"no clips under {root}")
    per_machine: dict[str, dict] = {}
    t0 = time.time()

    for machine, mclips in D.by_machine(clips).items():
        train = [c for c in mclips if c.split == "train"]
        test = [c for c in mclips if c.split == "test"]
        if limit:
            train, test = train[:limit], test[:limit]
        if not train or not test:
            continue

        tr_emb = backend.embed(_load([c.path for c in train]))
        tr_src = tr_emb[[i for i, c in enumerate(train) if c.domain == "source"]]
        tr_tgt = tr_emb[[i for i, c in enumerate(train) if c.domain == "target"]]

        scorer = SCORERS[scorer_name]()
        scorer.fit(tr_src if len(tr_src) else tr_emb,
                   tr_tgt if len(tr_tgt) >= 2 else None)

        te_emb = backend.embed(_load([c.path for c in test]))
        scores = scorer.score(te_emb)
        y = np.array([c.label for c in test])
        dom = np.array([c.domain for c in test])

        src = dom == "source"
        tgt = dom == "target"
        per_machine[machine] = {
            "auc_source": M.auc(y[src], scores[src]) if src.any() else float("nan"),
            "auc_target": M.auc(y[tgt], scores[tgt]) if tgt.any() else float("nan"),
            "pauc": M.partial_auc(y, scores),
            "n_train": len(train), "n_test": len(test),
        }
        print(f"  {machine:9s} "
              f"AUC_src={per_machine[machine]['auc_source']:.4f} "
              f"AUC_tgt={per_machine[machine]['auc_target']:.4f} "
              f"pAUC={per_machine[machine]['pauc']:.4f}")

    result = {
        "backend": backend_spec,
        "scorer": scorer_name,
        "official_score": M.official_score(per_machine),
        "mean_auc": float(np.nanmean([v["auc_source"] for v in per_machine.values()]
                                     + [v["auc_target"] for v in per_machine.values()])),
        "per_machine": per_machine,
        "seconds": round(time.time() - t0, 1),
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backend", default="torch:mn10_as")
    ap.add_argument("--scorer", default="maha", choices=list(SCORERS))
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap clips per split (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = evaluate(args.root, args.backend, args.scorer, args.machines, args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "per_machine"}, indent=2))

    tag = args.backend.replace(":", "_").replace("/", "_").replace("\\", "_").replace(".onnx", "")
    out = args.out or os.path.join(_HERE, os.pardir, "models", f"eval_{tag}_{args.scorer}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)
