# Injini

Offline acoustic condition monitoring for engines and diesel generators, on a mid-range Android phone. Sibling of [SiloSense](https://github.com/rapha18th/SiloSense).

Injini ships the DCASE Task 2 first-shot recipe, compressed for an Arm phone: a **frozen AudioSet-distilled MobileNetV3 embedder** (EfficientAT), quantised to INT8, plus a **distance score** against the healthy recordings of one specific machine. The neural work is the embedding. The decision is a few lines of linear algebra on the CPU.

See `Injini.docx` in the Sensing Work set for the full argument, the history, the Arm rationale, and the living evaluation table.

## The recipe

| Stage | What | Where |
|---|---|---|
| Features | 32 kHz, pre-emphasis, 128 Kaldi mel bands, EfficientAT's exact spec, pure NumPy | `src/features.py`, ported to `AudioFeatures.kt` |
| Embedder | Frozen EfficientAT `mn10_as` (960-d) or `mn04_as` (384-d), feature extractor only | `src/embedder.py` → ONNX |
| Quantise | Static QDQ INT8, per-channel, MinMax calibration on real mels | `export/quantize.py` |
| Score | Per-machine enrollment → Mahalanobis (full-cov in eval, diagonal + kNN on device) | `src/anomaly.py`, `AnomalyScorer.kt` |
| Fault ID | Optional supervised head on frozen embeddings, source-disjoint split | `src/faultid.py` |

## Evaluation

Every claim is anchored to a published DCASE figure.

```bash
pip install -r requirements.txt

# EfficientAT code + weights
git clone https://github.com/fschmid56/EfficientAT.git vendor_efficientat
mkdir -p vendor_efficientat/resources
wget -P vendor_efficientat/resources \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt
python src/features.py                       # cache the Kaldi mel matrix

# DCASE 2025 Task 2 development set (CC BY-NC-SA 4.0, ~2.3 GB)
python src/fetch_dcase.py --out data/dcase2025_dev

# baseline, reference ceiling, then the phone pipeline
python src/baseline_ae.py  --root data/dcase2025_dev --epochs 100
python src/eval_dcase.py   --root data/dcase2025_dev --backend passt --scorer maha
python src/embedder.py     --name mn10_as --out models/injini_mn10_as_fp32.onnx
python src/eval_dcase.py   --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_fp32.onnx
python export/quantize.py  --fp32 models/injini_mn10_as_fp32.onnx --calib-dir data/dcase2025_dev --n-calib 256
python src/eval_dcase.py   --root data/dcase2025_dev --backend onnx:models/injini_mn10_as_int8.onnx
```

`notebooks/injini_train.ipynb` (built by `notebooks/build_notebook.py`) runs the whole sequence on Kaggle.

## Android

```bash
cp models/injini_mn10_as_int8.onnx models/injini_mn10_as_fp32.onnx android/app/src/main/assets/
cd android && ./gradlew assembleDebug
```

Two flows: **Enrol** records six healthy ten-second clips of a machine and stores its fingerprint; **Check** records one clip and returns a three-tier verdict. The "Full model results" dialog shows the matched FP32-vs-INT8 embedder benchmark and the execution-provider trace, same methodology as SiloSense.

Feature parity (Kotlin vs the Python reference):

```bash
cd android && ./gradlew testDebugUnitTest --tests "com.injini.app.AudioFeaturesParityTest"
```

## Local artefacts measured so far

| Embedder | Params (extractor) | ONNX FP32 | ONNX INT8 |
|---|---|---|---|
| `mn10_as` | 2.97 M | 11.93 MB | 3.55 MB |
| `mn04_as` | 0.52 M | 2.14 MB | 0.93 MB |

AUC numbers on DCASE come from the Kaggle run and land in `Injini.docx` Section 10.

## Licences

Code MIT. EfficientAT weights: MIT (fschmid56/EfficientAT). DCASE 2025 dev set: CC BY-NC-SA 4.0 (not redistributed here). Kaggle `zeyadzsm/engine-sounds`: Apache-2.0 declared. `malakragaie/car-diagnostics-dataset`: licence unstated, used for local evaluation only.
