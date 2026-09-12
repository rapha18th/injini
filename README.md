# Injini

Offline acoustic condition monitoring for engines and diesel generators. Built for a mid-range Android phone in Zimbabwe and South Africa. Sibling of [SiloSense](https://github.com/rapha18th/SiloSense).

A phone can listen to a machine the way a mechanic does. Injini learns the sound of one machine when it runs well. It flags the sound when something changes. No cloud connection required. No specialist hardware.

Injini ships the DCASE Task 2 first-shot recipe, compressed for an Arm phone. A frozen AudioSet-distilled MobileNetV3 embedder (EfficientAT), quantised to INT8. A distance score against the healthy recordings of one specific machine. The neural work is the embedding. The decision is a few lines of linear algebra on the CPU.

See `Injini.docx` in the Sensing Work set for the full argument, the history, the Arm rationale, and the living evaluation table.

## Architecture

Data to deployment, end to end.

![Injini architecture, data to deployment](diagrams/architecture.svg)

## The recipe

| Stage | What | Where |
|---|---|---|
| Features | 32 kHz, pre-emphasis, 128 Kaldi mel bands, EfficientAT's exact spec, pure NumPy | `src/features.py`, ported to `AudioFeatures.kt` |
| Embedder | Frozen EfficientAT `mn10_as` (960-d) or `mn04_as` (384-d), feature extractor only | `src/embedder.py` → ONNX |
| Quantise | Static QDQ INT8, per-channel, MinMax calibration on real mels | `export/quantize.py` |
| Score | Per-machine enrollment → Mahalanobis (full-cov in eval, diagonal + kNN on device) | `src/anomaly.py`, `AnomalyScorer.kt` |
| Fault ID | Optional supervised head on frozen embeddings, source-disjoint split | `src/faultid.py` |

## Evaluation

Every claim is anchored to a published DCASE figure. Data comes from the public HuggingFace mirror `HTill/dcase2025_task2_dev` by default (`--source hf`). Zenodo has been unreliable. See [ADR-4](ADR.md#adr-4-kaggle-specific-data-source-has-to-be-resilient-not-clever).

```bash
pip install -r requirements.txt

# EfficientAT code + weights
git clone https://github.com/fschmid56/EfficientAT.git vendor_efficientat
mkdir -p vendor_efficientat/resources
wget -P vendor_efficientat/resources \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt \
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt
python src/features.py                       # cache the Kaldi mel matrix

# baseline, reference ceiling, then the phone pipeline: all default to the HF mirror
python src/baseline_ae.py  --epochs 100
python src/eval_dcase.py   --backend passt --scorer knn
python src/embedder.py     --name mn10_as --out models/injini_mn10_as_fp32.onnx
python src/eval_dcase.py   --backend onnx:models/injini_mn10_as_fp32.onnx --scorer knn
python export/quantize.py  --fp32 models/injini_mn10_as_fp32.onnx --calib-dir data/calib --n-calib 256
python src/eval_dcase.py   --backend onnx:models/injini_mn10_as_int8.onnx --scorer knn
```

`notebooks/injini_train.ipynb` runs the whole sequence on Kaggle. It is self-contained. No private dataset. No repo of ours to clone. See [ADR-4](ADR.md#adr-4-kaggle-specific-data-source-has-to-be-resilient-not-clever). `notebooks/injini_faultid_only.ipynb` reruns just the supervised head.

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

Supervised fault-ID head (secondary mode): **0.362 macro F1** across 12 classes, source-disjoint split, mn10_as embeddings. The clean-data ceiling sits around 98 to 99%. A source-disjoint split of a messy, unlicensed-provenance public corpus earns a lower number honestly. See [ADR-10](ADR.md#adr-10-a-shared-kaggle-input-mount-can-silently-contaminate-a-recursive-glob) for a contamination bug this caught before it reached the paper.

Every embedder configuration beats the reproduced baseline. `mn10_as` with kNN edges past the PaSST reference with the same scorer, 0.613 versus 0.596. PaSST ran capped to CPU and five machine types, so treat this as directional. The comparison itself is fair and matched. Full narrative and the whitening ablation in `Injini.docx` Section 10.

## Android

```bash
cp models/injini_mn10_as_int8.onnx models/injini_mn10_as_fp32.onnx android/app/src/main/assets/
cd android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Every machine gets its own fingerprint. The home screen carries a dropdown, "Select or add a machine." Tap it for every known machine, plus "+ Add new machine." Picking one shows an inline card right there: engine type, category, enrolled state, check count, training clip count. A separate, always-visible **Machines** button opens the full fleet screen for management: add, delete, review any machine's queue in one place. "+ Add machine" collects a name, engine type (Petrol, Diesel, Not sure) and category (Vehicle, Generator, Pump, Other). A rod knock on a diesel generator and the same fault on a petrol kombi are different training examples. `MachineRegistry.kt` persists it in one JSON file, the same shape as TapSense's vessel registry.

Two flows run against whichever machine is selected. Both feed the training corpus automatically.

- **Enrol** records six healthy ten-second clips and builds that machine's fingerprint. Every one of those clips writes straight into the corpus under the healthy label. A clip recorded during enrolment is healthy by definition.
- **Check** records one clip and returns a three-tier verdict, then queues that same clip with the model's tier as a hint. The real answer usually arrives hours or days later, once a mechanic has actually looked. A visible **Add verdict** button appears, on the home screen and on that machine's row in the fleet list, the moment a machine has a recording waiting. Tapping it lists each queued recording. Picking one opens the label dialog: healthy confirmed, or faulty against the fault-ID head's own class list, free text for a fault the taxonomy has no name for yet. Confirming moves the file out of the pending queue into its real corpus folder, with a manifest row shaped the way `prepare_engine_sounds.py` already expects. This is the field data-collection tool the working paper's Section 09 once imagined as separate software. It lives inside these same two flows instead.

Every recording, Enrol or Check, drives a genuinely live waveform and a counting-down status line. A continuous travelling ripple and a breathing REC dot run the whole ten seconds, even in a silent room. The screen never looks frozen. The "Full model results" dialog shows the matched FP32-vs-INT8 embedder benchmark, the execution-provider trace, and a field benchmark of the on-device model itself. The custom adaptive icon is a five-bar amber waveform on the app's own instrument-panel dark ground.

Injini benchmarks itself in the field. Every Check clip's on-device tier gets compared against the verdict it eventually receives. A false positive means the model flagged something that turned out healthy. A false negative means the model read something as healthy that turned out faulty. Both get surfaced immediately and written into `manifest.csv`'s `benchmark_outcome` column. A false negative also flags that machine as needing re-enrollment, since its healthy fingerprint may include a bad sample. The flag clears automatically the next time the machine is enrolled. Fleet-wide totals sit in "Full model results."

Dataset export and remote sync both live on the Machines screen. A live counter reads "Dataset: N healthy, M faulty, T total," tap for the per-fault-type breakdown. It answers whether enough has been collected, without guessing. **Export dataset (.zip)** zips the whole corpus, confirmed clips and the still-pending queue together, and hands it straight to Android's share sheet. WhatsApp, Gmail, Drive, or any file manager through the public Downloads folder all work the same way. **Sync to Injini Cloud** is one button and one sentence. No technical terms appear anywhere in it.

Behind that button, the zip travels through **Injini Relay**, a small backend deployed as a Hugging Face Space. It never goes straight to Hugging Face. The relay is the only thing that holds a real Hugging Face write token. The relay's URL and a separate, low-privilege upload key are bundled into the app itself. Pasting a key into every phone this gets distributed to is not workable in the field, so the key is designed to be safe to distribute this way. If it leaks, the most it allows is opening a pull request against the dataset. The relay always stages uploads as a pull request for a human to review, so a direct write stays out of reach either way. Every sync is one deliberate tap. The app stays offline otherwise. The relay also carries a minimal admin dashboard, at `/dashboard`, behind the same login as its manual-upload page. It shows where the data is coming from and what it looks like once reviewed. See [ADR-18](ADR.md#adr-18-dataset-export-a-live-label-counter-and-hugging-face-as-the-sync-target), [ADR-19](ADR.md#adr-19-adr-18s-direct-to-hf-sync-was-corrected--a-phone-cant-safely-hold-a-write-token) and [ADR-20](ADR.md#adr-20-the-relay-went-live-the-upload-key-was-deliberately-bundled-and-a-real-404-turned-out-to-be-leftover-test-data) for the full reasoning. `hf_space/README.md` has the deploy steps.

Feature parity (Kotlin vs the Python reference):

```bash
cd android && ./gradlew testDebugUnitTest --tests "com.injini.app.AudioFeaturesParityTest"
```

### On-device, 2026-09-11 (Samsung SM-M075F, in place of the M16 originally planned)

| Measurement | INT8 | FP32 |
|---|---|---|
| Steady-state embed | 44.1 ms | 87.5 ms |
| Load time | 241.7 ms | 87.1 ms |
| Executed per node | CPUExecutionProvider: all 3670 nodes | same |

XNNPACK registered on this device but ran nothing. SiloSense's Galaxy M16 ran the same workload entirely through XNNPACK. Process memory sits at 178.2 MB PSS, thermal status LIGHT. See [ADR-12](ADR.md#adr-12-a-verdict-shown-and-immediately-overwritten-by-the-idle-reset) and [ADR-13](ADR.md#adr-13-the-results-link-sat-underneath-the-system-navigation-bar) for two real bugs the first on-device run caught, and the note on unvalidated 6-clip on-device calibration in [Open items](ADR.md#open-items).

## Local artefacts measured so far

| Embedder | Params (extractor) | ONNX FP32 | ONNX INT8 |
|---|---|---|---|
| `mn10_as` | 2.97 M | 11.93 MB | 3.55 MB |
| `mn04_as` | 0.52 M | 2.14 MB | 0.93 MB |

AUC numbers on DCASE come from the Kaggle run and land in `Injini.docx` Section 10.

## Known issues and gotchas

The short version follows. The full incident log with context and fixes lives in [ADR.md](ADR.md).

- **A brand-new private Kaggle dataset does not reliably mount into a kernel started right after creation**, even when it shows `status: ready`. The notebook is now fully self-contained. It writes its own code, clones public EfficientAT, and reads a public HF mirror, rather than depending on a dataset of ours.
- **Zenodo was down for an extended period**, a 504 on both the pretty URL and the bare REST API. The DCASE dev set now comes from a public HuggingFace mirror instead.
- **That HF mirror is incomplete**. It holds 5 of the 7 official DCASE 2025 machine types. ToyCar and ToyTrain are missing. Every result here covers bearing, fan, gearbox, slider, and valve.
- **A Kaggle public dataset does not always mount at `/kaggle/input/<slug>`.** `zeyadzsm/engine-sounds` mounted at `/kaggle/input/datasets`, a shared folder. Scan `/kaggle/input/*` for the directory that actually holds your files. Never hardcode the path.
- **Kaggle's current GPU image (`torch 2.10+cu128`) dropped Tesla P100 support.** `torch.cuda.is_available()` still returns `True`. The failure only appears on the first real kernel launch. Everything here defaults to CPU.
- **A hand-rolled partial-AUC metric returned `p/2` for a random scorer instead of 0.5.** It crushed every score on the first real run and looked like total failure. The fix matches `sklearn.metrics.roc_auc_score(max_fpr=p)` exactly. Validate any hand-rolled metric against a reference on synthetic data before trusting it on real results.
- **A raw, unwhitened Mahalanobis distance on a frozen embedding sits at baseline level.** Whitening plus per-machine score normalisation earns the lift over baseline. A direct ablation confirmed this. It was verified, not assumed.
- **Materialising every clip's log-mel for a large split before batching kills the process with an out-of-memory error and no traceback.** It works fine at around 1000 clips per split, the DCASE evaluation size. It fails at 20,000, the fault-ID corpus size. Embedding now streams in fixed-size chunks.
- **`kaggle kernels output` downloads your entire `/kaggle/working` tree.** Keep the working directory in `/tmp` and copy only the deliverables out at the end. Otherwise the output pull never finishes.
- **`android:path` is not the vector-drawable path attribute.** Every icon in the app rendered as a blank shape until this was caught. Background chips showed. Foreground content did not. The real attribute is `android:pathData`. `android:path` compiles without error, since it is a legitimate but unrelated framework attribute. It simply draws nothing. See [ADR-17](ADR.md#adr-17-field-benchmark-tracking-minimalist-icons-and-a-real-vector-drawable-bug).
- **Standing up a Hugging Face Space does not create the dataset repo it pushes into.** The relay went live and 404'd on its first real upload, since `rairo/injini-field-data` did not exist yet. The two repos need creating independently.
- **A stale value in `SharedPreferences` can look exactly like a network bug.** A placeholder URL typed in during an earlier manual test silently overrode a newly bundled default, and the resulting 404 survived forcing http/1.1, adding a retry, and testing from three different IPs before a request logger revealed the actual URL being called. See [ADR-20](ADR.md#adr-20-the-relay-went-live-the-upload-key-was-deliberately-bundled-and-a-real-404-turned-out-to-be-leftover-test-data).

## Licences

Code is MIT. EfficientAT weights are MIT (fschmid56/EfficientAT). The DCASE 2025 dev set is CC BY-NC-SA 4.0 and is not redistributed here. Kaggle `zeyadzsm/engine-sounds` declares Apache-2.0. `malakragaie/car-diagnostics-dataset` has no stated licence and is used for local evaluation only.
