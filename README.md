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

Supervised fault-ID head (secondary mode): **0.362 macro F1** across 12 classes, source-disjoint split, mn10_as embeddings — well below the ~98-99% clean-data ceiling, which is expected on a source-disjoint split of a messy, unlicensed-provenance public corpus (see [ADR-10](ADR.md#adr-10-a-shared-kaggle-input-mount-can-silently-contaminate-a-recursive-glob) for a contamination bug this caught before it reached the paper).

Every embedder configuration beats the reproduced baseline. `mn10_as` with kNN edges past the PaSST reference with the same scorer (0.613 vs 0.596) — directional given PaSST ran capped to CPU and five machine types, but a fair, matched comparison. Full narrative and the whitening ablation in `Injini.docx` §10.

## Android

```bash
cp models/injini_mn10_as_int8.onnx models/injini_mn10_as_fp32.onnx android/app/src/main/assets/
cd android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

A named-machine fleet, not one hardcoded scorer. The home screen carries a dropdown ("Select or add a machine ⌄") — tap it for every known machine plus a trailing "+ Add new machine", no detour required. Picking one shows an inline card right there on the home screen: engine type, category, enrolled state, check count, training clip count. A separate, always-visible **Machines** button opens the full fleet screen (`MachineListActivity`) for management — add, delete, review any machine's queue in one place. "+ Add machine" (on either screen, same form — `MachineForms.kt`) collects a name, engine type (Petrol/Diesel/Not sure) and category (Vehicle/Generator/Pump/Other), because a rod knock on a diesel generator and the same fault on a petrol kombi are not one training example. `MachineRegistry.kt` (one JSON file, same shape as TapSense's vessel registry) persists it.

Two flows, both against whichever machine is selected, and both feed the training corpus with no extra button:

- **Enrol** records six healthy ten-second clips, builds that machine's fingerprint, and writes every one of those clips straight into the corpus under the healthy label — a clip recorded during enrolment is healthy by definition, so there is nothing left to ask.
- **Check** records one clip and returns a three-tier verdict, then queues that same clip with the model's tier as a hint, not a label. What Check cannot know is the real answer, which usually does not exist for hours or days, until a mechanic has actually looked. A visible **Add verdict** button appears — on the home screen and on that machine's row in the fleet list — the moment a machine has a recording waiting; tapping it (`VerdictFlow.kt`, shared by both screens) lists each queued recording, and picking one opens the label dialog: healthy confirmed, or faulty against the fault-ID head's own class list (`FaultLabel.kt`), free text for a fault the taxonomy has no name for yet. Confirming moves the file out of the pending queue into its real corpus folder (`LabeledClipStore.kt`) with a manifest row shaped like `prepare_engine_sounds.py` already expects. This is the field data-collection tool the working paper's Section 09 once imagined as separate software, built into the same two flows instead of a third one.

Every recording — Enrol, Check, either screen — drives a genuinely live waveform and a counting-down status line (`WaveformView.kt`): a continuous travelling ripple plus a breathing REC dot run the whole ten seconds regardless of how quiet the room is, so it never looks like a frozen screen. The "Full model results" dialog shows the matched FP32-vs-INT8 embedder benchmark, the execution-provider trace, and a field benchmark of the on-device model itself (see below). Custom adaptive icon: a five-bar amber waveform on the app's own instrument-panel dark ground, not a placeholder.

**The app is a field benchmark, not just a data collector.** Every Check clip's on-device tier is compared against the verdict it eventually gets: a false positive (flagged, turned out healthy) or a false negative (read healthy, turned out faulty) is surfaced immediately and written into `manifest.csv`'s `benchmark_outcome` column. A false negative also flags that machine **needs re-enrollment** — its healthy fingerprint may include a bad sample — cleared automatically the next time it's enrolled. Fleet-wide totals sit in "Full model results".

**Dataset export and remote sync**, both from the Machines screen: a live counter ("Dataset: N healthy · M faulty · T total", tap for the per-fault-type breakdown) answers "have we collected enough yet" without guessing. **Export dataset (.zip)** zips the whole corpus (confirmed clips and the still-pending queue) and hands it straight to Android's share sheet — WhatsApp, Gmail, Drive, or any file manager via the public Downloads folder, no extra steps either way (`DatasetExporter.kt`). **Sync to Injini Cloud** (`HfSyncActivity.kt`, `HfSync.kt`) is one button, one sentence, no technical terms anywhere in it. Behind that button: the same zip goes through **Injini Relay** (`hf_space/`, a small backend deployed as a Hugging Face Space), never straight to Hugging Face, and the relay is the only thing that ever holds a real Hugging Face write token. The relay's URL and a separate, low-privilege upload key are bundled into the app itself — pasting a key into every phone this gets distributed to isn't workable in the field, and that key is designed to be safe to distribute this way: if it leaks, the most it allows is opening a pull request against the dataset, never writing to it directly, since the relay always stages uploads as a PR for a human to review. Purely user-triggered; the app is offline-only otherwise. The relay also carries a minimal admin dashboard (`/dashboard`, same login as its manual-upload page) showing where the data is coming from and what it looks like once reviewed. See [ADR-18](ADR.md#adr-18-dataset-export-a-live-label-counter-and-hugging-face-as-the-sync-target), [ADR-19](ADR.md#adr-19-adr-18s-direct-to-hf-sync-was-corrected--a-phone-cant-safely-hold-a-write-token) and [ADR-20](ADR.md#adr-20-the-relay-went-live-the-upload-key-was-deliberately-bundled-and-a-real-404-turned-out-to-be-leftover-test-data) for the full reasoning. `hf_space/README.md` has the deploy steps.

Feature parity (Kotlin vs the Python reference):

```bash
cd android && ./gradlew testDebugUnitTest --tests "com.injini.app.AudioFeaturesParityTest"
```

### On-device, 2026-09-11 (Samsung SM-M075F, not the originally planned M16)

| Measurement | INT8 | FP32 |
|---|---|---|
| Steady-state embed | 44.1 ms | 87.5 ms |
| Load time | 241.7 ms | 87.1 ms |
| Executed per node | CPUExecutionProvider: all 3670 nodes | same |

XNNPACK registered but ran nothing on this device — the opposite of SiloSense's Galaxy M16, where it did all the real work. Process memory 178.2 MB PSS, thermal status LIGHT. See [ADR-12](ADR.md#adr-12-a-verdict-shown-and-immediately-overwritten-by-the-idle-reset) and [ADR-13](ADR.md#adr-13-the-results-link-sat-underneath-the-system-navigation-bar) for two real bugs the first on-device run caught, and the note on unvalidated 6-clip on-device calibration in [Open items](ADR.md#open-items).

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
- **`android:path` is not the vector-drawable path attribute** — every icon in the app rendered as a totally blank shape (background chips showed, foreground content did not) until this was caught. The real attribute is `android:pathData`; `android:path` compiles without error (it's a legitimate but unrelated framework attribute) so nothing fails at build time, it just silently draws nothing. See [ADR-17](ADR.md#adr-17-field-benchmark-tracking-minimalist-icons-and-a-real-vector-drawable-bug).

## Licences

Code MIT. EfficientAT weights: MIT (fschmid56/EfficientAT). DCASE 2025 dev set: CC BY-NC-SA 4.0 (not redistributed here). Kaggle `zeyadzsm/engine-sounds`: Apache-2.0 declared. `malakragaie/car-diagnostics-dataset`: licence unstated, used for local evaluation only.
