"""Emit the Injini training / evaluation notebook (injini_train.ipynb).

Runs entirely on Kaggle from attached inputs, no GitHub needed:

  * dataset  thetraveller/injini-code   (this repo's src/ export/ notebooks/
             vendor_efficientat/ and the cached Kaldi mel matrix)
  * dataset  zeyadzsm/engine-sounds     (supervised fault-ID corpus)
  * internet ON                         (Zenodo DCASE dev set, pip)

Produces every number Section 10 needs and writes artefacts to
/kaggle/working: the ONNX embedders (FP32 + INT8), the metrics JSONs, and a
combined injini_metrics.json.

    python notebooks/build_notebook.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "injini_train.ipynb")

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip()))


def code(s: str) -> None:
    CELLS.append(("code", s.strip()))


md("""
# Injini — train & evaluate on Kaggle

Ships the DCASE Task 2 first-shot recipe compressed for an Arm phone: a frozen
AudioSet-distilled MobileNetV3 embedder plus a Mahalanobis distance score.
This notebook produces the evaluation table and exports the model artefacts.

**Settings:** Internet ON. Accelerator GPU (P100 or T4). Add data:
`thetraveller/injini-code` and `zeyadzsm/engine-sounds`. Then Run All.
""")

md("## 1. Workspace from the attached code dataset")
code("""
import os, sys, shutil, json, glob, time
SRC = "/kaggle/input/injini-code"
WORK = "/kaggle/working/injini"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
os.chdir(WORK)
sys.path.insert(0, os.path.join(WORK, "src"))
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
print("workspace:", WORK)
print(os.listdir(WORK))
""")

md("## 2. Dependencies")
code("""
!pip -q install onnx onnxruntime hear21passt 2>/dev/null
import torch, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
# the Kaldi mel matrix ships in the dataset, so features.py needs no torchaudio
assert os.path.exists("models/mel_kaldi_128x513.npy"), "mel matrix missing from dataset"
""")

md("## 3. DCASE 2025 Task 2 development set (Zenodo 15097779, CC BY-NC-SA 4.0)")
code("""
t0 = time.time()
!python src/fetch_dcase.py --out data/dcase2025_dev
n = len(glob.glob("data/dcase2025_dev/*/*/*.wav"))
print(f"{n} wav files in {time.time()-t0:.0f}s")
assert n > 1000, "DCASE dev set did not download"
""")

md("## 4. Reproduce the DCASE autoencoder baseline")
code("""
!python src/baseline_ae.py --root data/dcase2025_dev --epochs 100
""")

md("""
## 5. Reference ceiling — PaSST transformer embedding + Mahalanobis
The published state of the art in one reproducible form. The phone pipeline is
measured against this.
""")
code("""
!python src/eval_dcase.py --root data/dcase2025_dev --backend passt --scorer maha \
    --out models/eval_passt_maha.json || echo "passt backend failed, continuing"
""")

md("## 6. Frozen EfficientAT mn10_as — FP32, then static INT8")
code("""
!python src/embedder.py --name mn10_as --out models/injini_mn10_as_fp32.onnx
!python src/eval_dcase.py --root data/dcase2025_dev \
    --backend onnx:models/injini_mn10_as_fp32.onnx --scorer maha \
    --out models/eval_mn10_fp32_maha.json
!python export/quantize.py --fp32 models/injini_mn10_as_fp32.onnx \
    --calib-dir data/dcase2025_dev --n-calib 256
!python src/eval_dcase.py --root data/dcase2025_dev \
    --backend onnx:models/injini_mn10_as_int8.onnx --scorer maha \
    --out models/eval_mn10_int8_maha.json
""")

md("## 7. Smaller candidate — mn04_as, FP32 and INT8")
code("""
!python src/embedder.py --name mn04_as --out models/injini_mn04_as_fp32.onnx
!python src/eval_dcase.py --root data/dcase2025_dev \
    --backend onnx:models/injini_mn04_as_fp32.onnx --scorer maha \
    --out models/eval_mn04_fp32_maha.json
!python export/quantize.py --fp32 models/injini_mn04_as_fp32.onnx \
    --calib-dir data/dcase2025_dev --n-calib 256
!python src/eval_dcase.py --root data/dcase2025_dev \
    --backend onnx:models/injini_mn04_as_int8.onnx --scorer maha \
    --out models/eval_mn04_int8_maha.json
""")

md("## 8. kNN scorer cross-check (mn10_as FP32)")
code("""
!python src/eval_dcase.py --root data/dcase2025_dev \
    --backend onnx:models/injini_mn10_as_fp32.onnx --scorer knn \
    --out models/eval_mn10_fp32_knn.json
""")

md("## 9. Supervised fault-ID head (Kaggle engine-sounds, source-disjoint)")
code("""
ES = "/kaggle/input/engine-sounds"
!python src/prepare_engine_sounds.py --root {ES} --out data/engine_sounds_manifest.csv
!python src/faultid.py --manifest data/engine_sounds_manifest.csv \
    --backend onnx:models/injini_mn10_as_fp32.onnx --epochs 80
""")

md("## 10. Collect the table and export artefacts")
code("""
rows = []
for p in sorted(glob.glob("models/eval_*.json")) + sorted(glob.glob("models/faultid_*.json")):
    d = json.load(open(p))
    rows.append({
        "file": os.path.basename(p),
        "backend": d.get("backend") or d.get("system"),
        "scorer": d.get("scorer"),
        "official_score": d.get("official_score"),
        "mean_auc": d.get("mean_auc"),
        "macro_f1": d.get("macro_f1"),
        "per_machine": d.get("per_machine"),
    })
quant = {}
for p in glob.glob("models/*_quant_report.json"):
    quant[os.path.basename(p)] = json.load(open(p))

summary = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "results": rows, "quant": quant}
json.dump(summary, open("/kaggle/working/injini_metrics.json", "w"), indent=2)

for r in rows:
    print(f"{r['file']:34s} official={r['official_score']}  mean_auc={r['mean_auc']}  macro_f1={r['macro_f1']}")
print()
for k, v in quant.items():
    print(k, "->", {kk: v[kk] for kk in ("fp32_mb", "int8_mb", "size_ratio", "approx_macs")})

for f in glob.glob("models/injini_*.onnx") + glob.glob("models/*_quant_report.json") + glob.glob("models/eval_*.json"):
    shutil.copy(f, "/kaggle/working/")
print("\\nartefacts in /kaggle/working/")
print(sorted(os.listdir("/kaggle/working")))
""")


def build() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
