"""Emit a fully self-contained Injini training / evaluation notebook.

No private dataset, no GitHub repo of our own. The notebook writes every
pipeline file from inline cells, clones EfficientAT (public) for the model
code, wgets its weights (public GitHub release), downloads the DCASE 2025 dev
set from Zenodo, and reads Kaggle's public `zeyadzsm/engine-sounds`.

Outputs to /kaggle/working: the ONNX embedders (FP32 + INT8), per-eval metrics
JSONs, quant reports, and a combined injini_metrics.json.

    python notebooks/build_notebook.py

Kaggle settings: Internet ON, GPU (P100/T4), add data `zeyadzsm/engine-sounds`.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "injini_train.ipynb")

INLINE_FILES = [
    "src/features.py",
    "src/metrics.py",
    "src/anomaly.py",
    "src/dcase_data.py",
    "src/embedder.py",
    "src/embed_backends.py",
    "src/eval_dcase.py",
    "src/baseline_ae.py",
    "src/fetch_dcase.py",
    "src/prepare_engine_sounds.py",
    "src/faultid.py",
    "src/dcase_hf.py",
    "export/quantize.py",
]

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip()))


def code(s: str) -> None:
    CELLS.append(("code", s.rstrip()))


md("""
# Injini — train & evaluate

DCASE Task 2 first-shot recipe compressed for an Arm phone: a frozen
AudioSet-distilled MobileNetV3 embedder plus a Mahalanobis distance score.
This notebook is self-contained. It writes the pipeline, pulls EfficientAT and
the DCASE dev set, runs the full evaluation, and exports the model artefacts.

**Settings:** Internet ON. GPU (P100 or T4). Add data: `zeyadzsm/engine-sounds`.
Then Run All.
""")

md("## 1. Workspace and pipeline files")
code("""
import os, sys, shutil, json, glob, time, subprocess
# workspace OUTSIDE /kaggle/working so the 2 GB DCASE download is not saved as
# kernel output; only the small artefacts are copied there at the end.
WORK = "/tmp/injini"
OUTDIR = "/kaggle/working"
for d in ("src", "export", "models", "data"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, os.path.join(WORK, "src"))
print("workspace", WORK)
""")

for rel in INLINE_FILES:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")

md("## 2. Dependencies, EfficientAT code and weights")
code("""
!pip -q install onnx onnxruntime "datasets>=2.19" 2>/dev/null
!pip -q install hear21passt 2>/dev/null || echo "hear21passt install failed; PaSST reference row will be skipped"
try:
    import torchaudio  # noqa: F401
except Exception:
    !pip -q install torchaudio 2>/dev/null
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

if not os.path.isdir("vendor_efficientat"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/fschmid56/EfficientAT.git", "vendor_efficientat"], check=True)
os.makedirs("vendor_efficientat/resources", exist_ok=True)
for f in ["mn10_as_mAP_471.pt", "mn04_as_mAP_432.pt"]:
    p = f"vendor_efficientat/resources/{f}"
    if not os.path.exists(p):
        subprocess.run(["wget", "-q", "-O", p,
                        f"https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/{f}"], check=True)
print(os.listdir("vendor_efficientat/resources"))

# cache the Kaldi mel matrix (needs torchaudio, present on Kaggle)
import features as F
F.dump_mel_matrix()
print("mel matrix", os.path.exists("models/mel_kaldi_128x513.npy"))
""")

md("""
## 3. DCASE 2025 Task 2 development set
Read from the HuggingFace mirror `HTill/dcase2025_task2_dev` (Zenodo, the
official host, has been returning 504s). Also dump 256 healthy training clips
to a folder for the INT8 calibration pass.
""")
code("""
t0 = time.time()
import dcase_hf as H, soundfile as sf
_clips = H.load()
print(f"{len(_clips)} clips in {time.time()-t0:.0f}s")
assert len(_clips) > 1000, "DCASE mirror did not load"
os.makedirs("data/calib", exist_ok=True)
cal = [c for c in _clips if c.split == "train"][:256]
for i, c in enumerate(cal):
    sf.write(f"data/calib/{i:03d}.wav", c.wave, F.SR)
print("calibration wavs:", len(cal))
del _clips
""")

md("## 4. Reproduce the DCASE autoencoder baseline")
code("!python src/baseline_ae.py --epochs 100")

md("""
## 5. Reference ceiling — PaSST transformer embedding
The published state of the art in one reproducible form: a frozen transformer
embedding with whitening plus a kNN distance.
""")
code("""
!python src/eval_dcase.py --backend passt --scorer knn --whiten 128 \
    --out models/eval_passt_knn_w128.json || echo "passt backend failed, continuing"
""")

md("""
## 6. Frozen EfficientAT mn10_as — the shipped pipeline
Whitening + kNN + per-machine score normalisation, FP32 then static INT8. A
Mahalanobis variant and a no-whitening variant for the ablation.
""")
code("""
!python src/embedder.py --name mn10_as --out models/injini_mn10_as_fp32.onnx
FP=onnx:models/injini_mn10_as_fp32.onnx
!python src/eval_dcase.py --backend $FP --scorer knn  --whiten 128 --out models/eval_mn10_fp32_knn_w128.json
!python src/eval_dcase.py --backend $FP --scorer maha --whiten 128 --out models/eval_mn10_fp32_maha_w128.json
!python src/eval_dcase.py --backend $FP --scorer knn  --whiten 0   --out models/eval_mn10_fp32_knn_w0.json
!python export/quantize.py --fp32 models/injini_mn10_as_fp32.onnx --calib-dir data/calib --n-calib 256
!python src/eval_dcase.py --backend onnx:models/injini_mn10_as_int8.onnx --scorer knn --whiten 128 --out models/eval_mn10_int8_knn_w128.json
""")

md("## 7. Smaller candidate — mn04_as, FP32 and INT8")
code("""
!python src/embedder.py --name mn04_as --out models/injini_mn04_as_fp32.onnx
!python src/eval_dcase.py --backend onnx:models/injini_mn04_as_fp32.onnx --scorer knn --whiten 128 --out models/eval_mn04_fp32_knn_w128.json
!python export/quantize.py --fp32 models/injini_mn04_as_fp32.onnx --calib-dir data/calib --n-calib 256
!python src/eval_dcase.py --backend onnx:models/injini_mn04_as_int8.onnx --scorer knn --whiten 128 --out models/eval_mn04_int8_knn_w128.json
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
    rows.append({"file": os.path.basename(p),
                 "backend": d.get("backend") or d.get("system"),
                 "scorer": d.get("scorer"),
                 "official_score": d.get("official_score"),
                 "mean_auc": d.get("mean_auc"),
                 "macro_f1": d.get("macro_f1"),
                 "mean_pauc": d.get("mean_pauc"),
                 "whiten": d.get("whiten"),
                 "per_machine": d.get("per_machine")})
quant = {os.path.basename(p): json.load(open(p)) for p in glob.glob("models/*_quant_report.json")}
summary = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "results": rows, "quant": quant}
json.dump(summary, open(os.path.join(OUTDIR, "injini_metrics.json"), "w"), indent=2)

for r in rows:
    print(f"{r['file']:34s} official={r['official_score']}  mean_auc={r['mean_auc']}  "
          f"mean_pauc={r.get('mean_pauc')}  macro_f1={r['macro_f1']}")
print()
for k, v in quant.items():
    print(k, {kk: v.get(kk) for kk in ("fp32_mb", "int8_mb", "size_ratio", "approx_macs")})

for f in (glob.glob("models/injini_*.onnx") + glob.glob("models/*_quant_report.json")
          + glob.glob("models/eval_*.json") + glob.glob("models/faultid_*.json")):
    shutil.copy(f, OUTDIR)
print("\\nworking:", sorted(os.listdir(OUTDIR)))
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
