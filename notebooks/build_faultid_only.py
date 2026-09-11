"""Lean fault-ID-only rerun.

The first follow-up run diagnosed the engine-sounds mount (Kaggle put it at
/kaggle/input/datasets, not /kaggle/input/engine-sounds) and got that far, but
produced no fault-ID output and no traceback — almost certainly an OOM kill:
embed_backends.py materialised every clip's log-mel for the whole ~17k-clip
split before batching. That is now fixed to stream in chunks. This notebook
re-runs only the fault-ID head, with per-clip progress printed, so a failure
this time is visible rather than silent.

    python notebooks/build_faultid_only.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "injini_faultid_only.ipynb")

INLINE_FILES = [
    "src/features.py",
    "src/embedder.py",
    "src/embed_backends.py",
    "src/prepare_engine_sounds.py",
    "src/faultid.py",
]

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip()))


def code(s: str) -> None:
    CELLS.append(("code", s.rstrip()))


md("# Injini fault-ID-only rerun — streamed embedding, progress printed")

code("""
import os, sys, glob, subprocess
WORK = "/tmp/injini"
for d in ("src", "models", "data"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, os.path.join(WORK, "src"))
""")

for rel in INLINE_FILES:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")

code("""
!pip -q install onnx onnxruntime 2>/dev/null
if not os.path.isdir("vendor_efficientat"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/fschmid56/EfficientAT.git", "vendor_efficientat"], check=True)
os.makedirs("vendor_efficientat/resources", exist_ok=True)
p = "vendor_efficientat/resources/mn10_as_mAP_471.pt"
if not os.path.exists(p):
    subprocess.run(["wget", "-q", "-O", p,
                    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"], check=True)
import features as F
F.dump_mel_matrix()
""")

md("## Find the real mount, export the embedder, run fault-ID with progress")
code("""
candidates = [d for d in glob.glob("/kaggle/input/*") if len(glob.glob(d + "/**/*.wav", recursive=True)) > 100]
print("candidate roots:", candidates)
assert candidates, "no wav-bearing input directory found"
ES = candidates[0]
!python src/embedder.py --name mn10_as --out models/injini_mn10_as_fp32.onnx
!python src/prepare_engine_sounds.py --root {ES} --out data/engine_sounds_manifest.csv
!python src/faultid.py --manifest data/engine_sounds_manifest.csv \
    --backend onnx:models/injini_mn10_as_fp32.onnx --epochs 80
""")

code("""
import shutil, glob as g
for f in g.glob("models/faultid_*.json"):
    shutil.copy(f, "/kaggle/working/")
print(sorted(os.listdir("/kaggle/working")))
""")


def build() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "id": f"c{i}",
             "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for i, (t, s) in enumerate(CELLS)
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
