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

Every claim is anchored to a published DCASE figure. Data comes from the public
HuggingFace mirror `HTill/dcase2025_task2_dev` by default (`--source hf`), not
Zenodo directly — Zenodo has been unreliable, see [ADR-4](ADR.md#adr-4-kaggle-specific-data-source-has-to-be-resilient-not-clever).

```bash
pip install -r requirements.txt

# EfficientAT code + weights
git clone https://github.com/fschmid56/EfficientAT.git vendor_efficientat
mkdir -p vendor_efficientat/resources
wget -P vendor_efficientat/resources \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt
python src/features.py                       # cache the Kaldi mel matrix

# baseline, reference ceiling, then the phone pipeline — all default to the HF mirror
python src/baseline_ae.py  --epochs 100
python src/eval_dcase.py   --backend passt --scorer knn
python src/embedder.py     --name mn10_as --out models/injini_mn10_as_fp32.onnx
python src/eval_dcase.py   --backend onnx:models/injini_mn10_as_fp32.onnx --scorer knn
python export/quantize.py  --fp32 models/injini_mn10_as_fp32.onnx --calib-dir data/calib --n-calib 256
python src/eval_dcase.py   --backend onnx:models/injini_mn10_as_int8.onnx --scorer knn
```

`notebooks/injini_train.ipynb` (built by `notebooks/build_notebook.py`) runs the whole sequence on Kaggle, self-contained: no private dataset, no repo of ours to clone (see [ADR-4](ADR.md#adr-4-kaggle-specific-data-source-has-to-be-resilient-not-clever)). `notebooks/injini_faultid_only.ipynb` reruns just the supervised head.

### Results, 2026-09-11 (Kaggle, CPU, five DCASE machine types)

| System | Official score | Mean AUC |
|---|---|---|
| DCASE autoencoder baseline, reproduced | 0.547 | 0.560 |
| PaSST transformer, whitened, kNN | 0.596 | 0.637 |
| **mn10_as, whitened, Mahalanobis (best)** | **0.616** | **0.661** |
| mn10_as, whitened, kNN | 0.613 | 0.655 |
| mn10_as, whitened, kNN, INT8 | 0.599 | 0.640 |
| mn04_as, whitened, kNN | 0.593 | 0.630 |
| mn04_as, whitened, kNN, INT8 | 0.602 | 0.644 |

Every embedder configuration beats the reproduced baseline. `mn10_as` with kNN edges past the PaSST reference with the same scorer (0.613 vs 0.596) — directional given PaSST ran capped to CPU and five machine types, but a fair, matched comparison. Full narrative and the whitening ablation in `Injini.docx` §10.

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

## Known issues and gotchas

The short version; the full incident log with context and fixes is [ADR.md](ADR.md).

- **A brand-new private Kaggle dataset does not reliably mount into a kernel started right after creation**, even when it shows `status: ready`. The notebook is now fully self-contained (writes its own code, clones public EfficientAT, reads a public HF mirror) rather than depending on a dataset of ours.
- **Zenodo was down for an extended period** (504 on both the pretty URL and the bare REST API). The DCASE dev set now comes from a public HuggingFace mirror instead.
- **That HF mirror is incomplete**: it holds 5 of the 7 official DCASE 2025 machine types. ToyCar and ToyTrain are missing. Every result here is over bearing/fan/gearbox/slider/valve.
- **A Kaggle public dataset does not always mount at `/kaggle/input/<slug>`.** `zeyadzsm/engine-sounds` mounted at `/kaggle/input/datasets`, a shared folder. Scan `/kaggle/input/*` for the directory that actually has your files; don't hardcode the path.
- **Kaggle's current GPU image (`torch 2.10+cu128`) dropped Tesla P100 support.** `torch.cuda.is_available()` still returns `True`; the failure only appears on the first real kernel launch. Everything here defaults to CPU.
- **A hand-rolled partial-AUC metric returned `p/2` for a random scorer instead of 0.5**, which crushed every score on the first real run and looked like total failure. Fixed to match `sklearn.metrics.roc_auc_score(max_fpr=p)` exactly; validate any hand-rolled metric against a reference on synthetic data before trusting it on real results.
- **A raw, unwhitened Mahalanobis distance on a frozen embedding sits at baseline, not above it.** Whitening plus per-machine score normalisation is what earns the lift over baseline — verified with a direct ablation, not assumed.
- **Materialising every clip's log-mel for a large split before batching OOM-kills the process with no traceback.** Fine at ~1000 clips/split (the DCASE evaluation), fatal at 20,000 (the fault-ID corpus). Embedding now streams in fixed-size chunks.
- **`kaggle kernels output` downloads your entire `/kaggle/working` tree.** Keep the working directory in `/tmp` and copy only the deliverables out at the end, or the output pull never finishes.

## Licences

Code MIT. EfficientAT weights: MIT (fschmid56/EfficientAT). DCASE 2025 dev set: CC BY-NC-SA 4.0 (not redistributed here). Kaggle `zeyadzsm/engine-sounds`: Apache-2.0 declared. `malakragaie/car-diagnostics-dataset`: licence unstated, used for local evaluation only.
