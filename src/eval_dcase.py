"""Evaluate an embedder + distance scorer on the DCASE 2025 Task 2 dev set.

Two passes. First embed every machine's training clips (normal only) and fit a
PCA-whitener on the pooled set. Then per machine: whiten, fit the scorer on the
source and target healthy embeddings, score the test clips, compute source AUC,
target AUC and partial AUC. The official score is the harmonic mean of all
three across all machines.

    python src/eval_dcase.py --backend torch:mn10_as --scorer knn          # HF mirror (default)
    python src/eval_dcase.py --backend onnx:models/injini_mn10_as_int8.onnx --scorer knn
    python src/eval_dcase.py --source dir --root data/dcase2025_dev --backend passt --whiten 0

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
import metrics as M           # noqa: E402
from anomaly import SCORERS, Whitener  # noqa: E402
from embed_backends import get_backend  # noqa: E402


def load_clips(source: str, root: str | None, machines, limit):
    if source == "hf":
        import dcase_hf as H
        return H.load(machines=machines, limit=limit), H.by_machine
    import dcase_data as D
    return D.load(root, machines), D.by_machine


def waves_of(clips) -> list[np.ndarray]:
    out = []
    for c in clips:
        if getattr(c, "wave", None) is not None:
            out.append(c.wave)
        else:
            out.append(F.load_audio(c.path))
    return out


def evaluate(source: str, root: str | None, backend_spec: str, scorer_name: str,
             whiten: int = 128, machines=None, limit=None) -> dict:
    backend = get_backend(backend_spec)
    clips, by_machine_fn = load_clips(source, root, machines, limit)
    if not clips:
        raise SystemExit("no clips loaded")
    by_machine = by_machine_fn(clips)
    t0 = time.time()

    train_emb: dict[str, dict] = {}
    for machine, mclips in by_machine.items():
        train = [c for c in mclips if c.split == "train"]
        if not train:
            continue
        emb = backend.embed(waves_of(train))
        train_emb[machine] = {
            "all": emb,
            "source": emb[[i for i, c in enumerate(train) if c.domain == "source"]],
            "target": emb[[i for i, c in enumerate(train) if c.domain == "target"]],
        }

    whitener = None
    if whiten and whiten > 0 and train_emb:
        whitener = Whitener(whiten).fit(np.vstack([v["all"] for v in train_emb.values()]))

    def wt(x):
        return whitener.transform(x) if whitener is not None else x

    per_machine: dict[str, dict] = {}
    for machine, mclips in by_machine.items():
        if machine not in train_emb:
            continue
        test = [c for c in mclips if c.split == "test"]
        if not test:
            continue

        src = wt(train_emb[machine]["source"])
        tgt = wt(train_emb[machine]["target"])
        scorer = SCORERS[scorer_name]()
        scorer.fit(src if len(src) else wt(train_emb[machine]["all"]),
                   tgt if len(tgt) >= 2 else None)

        te = wt(backend.embed(waves_of(test)))
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
        "backend": backend_spec, "scorer": scorer_name, "whiten": whiten, "source": source,
        "official_score": M.official_score(per_machine),
        "mean_auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "mean_pauc": float(np.nanmean([v["pauc"] for v in per_machine.values()])) if per_machine else float("nan"),
        "per_machine": per_machine,
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf", choices=["hf", "dir"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--backend", default="torch:mn10_as")
    ap.add_argument("--scorer", default="knn", choices=list(SCORERS))
    ap.add_argument("--whiten", type=int, default=128)
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = evaluate(args.source, args.root, args.backend, args.scorer, args.whiten, args.machines, args.limit)
    print(json.dumps({k: v for k, v in res.items() if k != "per_machine"}, indent=2))

    tag = (args.backend.replace(":", "_").replace("/", "_").replace("\\", "_").replace(".onnx", "")
           + f"_{args.scorer}_w{args.whiten}")
    out = args.out or os.path.join(_HERE, os.pardir, "models", f"eval_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)
