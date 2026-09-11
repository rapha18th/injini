"""Lean follow-up notebook: just the two gaps from the first successful run.

  1. Diagnose why /kaggle/input/engine-sounds mounted empty, then run the
     supervised fault-ID head.
  2. A capped, CPU PaSST reference on the DCASE HF mirror, small enough to
     finish in a reasonable time.

Does not repeat the DCASE baseline / mn10 / mn04 evaluation — those numbers
are already banked in models/eval_*.json from the first successful run.

    python notebooks/build_followup.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "injini_followup.ipynb")

INLINE_FILES = [
    "src/features.py",
    "src/metrics.py",
    "src/anomaly.py",
    "src/dcase_hf.py",
    "src/embedder.py",
    "src/embed_backends.py",
    "src/eval_dcase.py",
    "src/prepare_engine_sounds.py",
    "src/faultid.py",
]

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip()))


def code(s: str) -> None:
    CELLS.append(("code", s.rstrip()))


md("""
# Injini follow-up — fault-ID mount + PaSST reference

The main run (injini-train) already produced the anomaly-detection table. This
just chases the two gaps: the engine-sounds dataset mounting empty, and a
capped CPU PaSST reference.
""")

md("## 1. Workspace")
code("""
import os, sys, shutil, json, glob, time
WORK = "/tmp/injini"
OUTDIR = "/kaggle/working"
for d in ("src", "export", "models", "data"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, os.path.join(WORK, "src"))
""")

for rel in INLINE_FILES:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")

md("## 2. Diagnose the input mount")
code("""
print("=== /kaggle/input ===")
!ls -la /kaggle/input
print("\\n=== /kaggle/input/engine-sounds (if it exists) ===")
!ls -la /kaggle/input/engine-sounds 2>&1 | head -20
print("\\n=== recursive wav count under each input dir ===")
for d in glob.glob("/kaggle/input/*"):
    n = len(glob.glob(d + "/**/*.wav", recursive=True))
    print(d, "->", n, "wavs")
""")

md("## 3. Dependencies and EfficientAT weights")
code("""
!pip -q install onnx onnxruntime "datasets>=2.19" 2>/dev/null
!pip -q install hear21passt 2>/dev/null || echo "hear21passt install failed"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not os.path.isdir("vendor_efficientat"):
    import subprocess
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/fschmid56/EfficientAT.git", "vendor_efficientat"], check=True)
os.makedirs("vendor_efficientat/resources", exist_ok=True)
for f in ["mn10_as_mAP_471.pt"]:
    p = f"vendor_efficientat/resources/{f}"
    if not os.path.exists(p):
        import subprocess
        subprocess.run(["wget", "-q", "-O", p,
                        f"https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/{f}"], check=True)
import features as F
F.dump_mel_matrix()
""")

md("## 4. Fault-ID head, using whichever input path actually has the wavs")
code("""
candidates = [d for d in glob.glob("/kaggle/input/*") if len(glob.glob(d + "/**/*.wav", recursive=True)) > 100]
print("candidate roots:", candidates)
if candidates:
    ES = candidates[0]
    !python src/embedder.py --name mn10_as --out models/injini_mn10_as_fp32.onnx
    !python src/prepare_engine_sounds.py --root {ES} --out data/engine_sounds_manifest.csv
    !python src/faultid.py --manifest data/engine_sounds_manifest.csv \
        --backend onnx:models/injini_mn10_as_fp32.onnx --epochs 80
else:
    print("no candidate input directory had wav files; engine-sounds dataset likely not attached to this kernel")
""")

md("""
## 5. CPU PaSST reference
No GPU this run (see cell 3), so this is a transformer forward pass on CPU
over the full five-machine set. Slower than the GPU path but not capped, so
the number is the real one, not a subsample.
""")
code("""
!python src/eval_dcase.py --backend passt --scorer knn --whiten 128 \
    --out models/eval_passt_knn_w128.json || echo "passt failed, see traceback above"
""")

md("## 6. Export")
code("""
for f in glob.glob("models/*.json") + glob.glob("models/*.onnx"):
    shutil.copy(f, OUTDIR)
print(sorted(os.listdir(OUTDIR)))
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
