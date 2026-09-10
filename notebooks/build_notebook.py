"""Emit the Injini training / evaluation notebook (injini_train.ipynb).

Run it on Kaggle (GPU optional, CPU is fine for these small models) to produce
every number Section 10 of the working paper needs:

  1. Reproduce the DCASE 2025 Task 2 autoencoder baseline on the dev set.
  2. Evaluate the full-size PaSST transformer embedding + Mahalanobis (the
     reference ceiling).
  3. Evaluate the frozen EfficientAT mn10_as embedder, FP32.
  4. Quantise it to INT8, re-evaluate, measure the AUC gap.
  5. Same for mn04_as (the smaller candidate).
  6. Train and score the supervised fault-ID head on Kaggle engine-sounds with
     a source-disjoint split.
  7. Dump models/*.onnx and a metrics JSON to /kaggle/working.

    python notebooks/build_notebook.py
  then upload notebooks/injini_train.ipynb to a new Kaggle notebook, attach the
  'zeyadzsm/engine-sounds' dataset, enable internet, and Run All.
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
# Injini — train & evaluate

Ships the DCASE Task 2 first-shot recipe compressed for an Arm phone: a frozen
AudioSet-distilled MobileNetV3 embedder plus a Mahalanobis distance score.
This notebook produces the evaluation table.

**Settings:** Internet ON. Add data: `zeyadzsm/engine-sounds`. Accelerator: GPU
optional. Then Run All. Artifacts land in `/kaggle/working`.
""")

md("## 1. Environment")
code("""
!pip -q install onnx onnxruntime "onnxruntime-tools" hear21passt torchaudio 2>/dev/null
import torch, sys, os, json, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
os.makedirs("/kaggle/working/models", exist_ok=True)
""")

md("## 2. Get the code and the EfficientAT weights")
code("""
!git clone -q https://github.com/rapha18th/injini.git || (cd injini && git pull -q)
%cd injini
!git clone -q https://github.com/fschmid56/EfficientAT.git vendor_efficientat || true
!mkdir -p vendor_efficientat/resources
for f in ["mn10_as_mAP_471.pt", "mn04_as_mAP_432.pt"]:
    p = f"vendor_efficientat/resources/{f}"
    if not os.path.exists(p):
        !wget -q -O {p} https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/{f}
!python src/features.py     # writes models/mel_kaldi_128x513.npy
sys.path.insert(0, "src")
""")

md("## 3. DCASE 2025 Task 2 development set")
code("""
!python src/fetch_dcase.py --out data/dcase2025_dev
import glob
print(len(glob.glob("data/dcase2025_dev/*/*/*.wav")), "wav files")
""")

md("## 4. Reproduce the DCASE autoencoder baseline")
code("""
!python src/baseline_ae.py --root data/dcase2025_dev --epochs 100
""")

md("""
## 5. Reference ceiling — PaSST transformer embedding + Mahalanobis
The published state of the art in one reproducible form. This is the number the
phone pipeline is measured against.
""")
code("""
!python src/eval_dcase.py --root data/dcase2025_dev --backend passt --scorer maha \
    --out models/eval_passt_maha.json
""")

md("## 6. Frozen EfficientAT mn10_as — FP32, then INT8")
code("""
!python src/embedder.py --name mn10_as --out models/injini_mn10_as_fp32.onnx
!python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_fp32.onnx \
    --scorer maha --out models/eval_mn10_fp32_maha.json
!python export/quantize.py --fp32 models/injini_mn10_as_fp32.onnx \
    --calib-dir data/dcase2025_dev --n-calib 256
!python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_int8.onnx \
    --scorer maha --out models/eval_mn10_int8_maha.json
""")

md("## 7. Smaller candidate — mn04_as, FP32 and INT8")
code("""
!python src/embedder.py --name mn04_as --out models/injini_mn04_as_fp32.onnx
!python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn04_as_fp32.onnx \
    --scorer maha --out models/eval_mn04_fp32_maha.json
!python export/quantize.py --fp32 models/injini_mn04_as_fp32.onnx \
    --calib-dir data/dcase2025_dev --n-calib 256
!python src/eval_dcase.py --root data/dcase2025_dev --backend onnx:models/injini_mn04_as_int8.onnx \
    --scorer maha --out models/eval_mn04_int8_maha.json
""")

md("## 8. Supervised fault-ID head (Kaggle engine-sounds, source-disjoint)")
code("""
ES = "/kaggle/input/engine-sounds"
!python src/prepare_engine_sounds.py --root {ES} --out data/engine_sounds_manifest.csv
!python src/faultid.py --manifest data/engine_sounds_manifest.csv --backend onnx:models/injini_mn10_as_fp32.onnx
""")

md("## 9. Collect the table")
code("""
import glob, json
rows = []
for p in sorted(glob.glob("models/eval_*.json")) + sorted(glob.glob("models/faultid_*.json")):
    d = json.load(open(p))
    rows.append({"file": os.path.basename(p),
                 "official_score": d.get("official_score"),
                 "mean_auc": d.get("mean_auc"),
                 "macro_f1": d.get("macro_f1")})
summary = {"results": rows}
json.dump(summary, open("/kaggle/working/injini_metrics.json", "w"), indent=2)
for r in rows: print(r)
!cp -r models /kaggle/working/ 2>/dev/null
print("\\nartifacts in /kaggle/working/models and /kaggle/working/injini_metrics.json")
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
