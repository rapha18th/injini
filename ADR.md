# Injini — architecture decisions and incident log

Canonical log of the decisions made building this pipeline, and every mistake
and gotcha hit along the way. Each entry is dated. Consequences and gotchas are
kept even when later superseded, because the reason not to do something again
is the whole value of the entry.

## ADR-1: Ship the DCASE first-shot recipe, not a novel method

**Context.** The first pass at this project (2026-09-10) proposed a homegrown
"order-domain normalisation" method as the headline technical bet, plus a
local audio-recording programme to build a training corpus.

**Decision.** The user rejected both and redirected: find the field's actual
baseline and 2026 state of the art, and ship that, compressed for a phone. No
data-capture programme.

**Consequences.** The product became: a frozen AudioSet-distilled MobileNetV3
embedder (EfficientAT `mn10_as` / `mn04_as`) plus a distance score in embedding
space, which is what DCASE 2025 Task 2's top-ranked systems do. Training data
is entirely public (DCASE dev set, Kaggle engine-sounds). See the working paper
`Injini.docx` Section 04 for the full field survey this is based on.

**Gotcha.** A frozen pretrained embedding plus a *raw, unwhitened* Mahalanobis
distance is not automatically state of the art just because the shape matches
what papers describe. The first real evaluation (below) landed at baseline
level, not above it. The field's papers gloss over the score-normalisation and
whitening work that separates a working system from a merely shaped-right one.

## ADR-2: Whitening and per-machine score normalisation are not optional

**Context.** First Kaggle evaluation (kernel `injini-train` v1, 2026-09-10):
mn10_as embedding, raw Mahalanobis distance, no whitening. Mean AUC landed at
0.61, essentially at the DCASE baseline's own level, not above it.

**Decision.** Added `Whitener` (PCA-whitening fitted on the pooled healthy
embeddings across all machine types) and per-machine score normalisation
(standardise a test score against the healthy training clips' own leave-one-out
scores) as mandatory pipeline stages, not optional flags.

**Consequences.** Directly ablated: mn10_as + kNN, no whitening = 0.585
official score; same model and scorer, whitened = 0.613. Whitening earns its
place with a controlled comparison, not an assumption. See `src/anomaly.py`
(`Whitener`, `_NormMixin`) and `Injini.docx` Section 10.

## ADR-3: The DCASE partial-AUC metric must match sklearn's `max_fpr` exactly

**Context.** The first hand-rolled `metrics.partial_auc` computed the mean
fraction of positives ranked above the top-`p` negatives, without the
McClish normalisation DCASE's own definition uses. For a random classifier
this returns roughly `p/2` (e.g. 0.05 for p=0.1) instead of 0.5.

**Mistake.** This was not caught before the first Kaggle run. Every
`official_score` came back catastrophically low (0.16-0.21) and looked like
the whole embedding approach had failed, when the AUC numbers underneath were
actually reasonable (mean AUC ~0.61).

**Fix.** Replaced with `sklearn.metrics.roc_auc_score(y, s, max_fpr=p)`,
verified against sklearn on synthetic separable / random / weak-separation
cases before trusting it on real data again.

**Lesson.** Never trust a hand-rolled metric implementation against real data
first. Validate it on synthetic ground truth (a random scorer must score 0.5,
a perfect one must score 1.0) before it touches anything that matters.

## ADR-4: Kaggle-specific data source has to be resilient, not clever

**Context, in order of failure:**

1. **A brand-new private Kaggle dataset does not reliably mount into a kernel
   started immediately after creation.** `thetraveller/injini-code` showed
   `status: ready` and listed all its files via the API, and the kernel's own
   pulled metadata confirmed it was in `dataset_sources` — but two consecutive
   kernel runs (v1, v2), minutes apart, both failed with
   `FileNotFoundError: /kaggle/input/injini-code`. Re-pushing did not help.
2. **Zenodo (the official DCASE host) was down.** `zenodo.org/records/15097779`
   and even the bare REST API (`zenodo.org/api/records/...`) returned HTTP 504
   for an extended period, killing both a local download attempt and a full
   Kaggle kernel run (v4) that had otherwise gotten past every earlier problem.

**Decision.** Made the notebook fully self-contained: every pipeline `.py`
file is written from an inline `%%writefile` cell rather than pulled from a
dataset or a git repo of ours. Model code and weights come from the public
`fschmid56/EfficientAT` GitHub repo (cloned + wget'd fresh each run — a public
repo mounts reliably where a fresh private dataset did not). The DCASE dev set
is read from a public HuggingFace mirror (`HTill/dcase2025_task2_dev`, via the
`datasets` library) instead of Zenodo directly.

**Consequences, and a new gotcha found this way.** The HF mirror is not a
complete copy of the official release: it holds exactly 6000 clips, which is
5 machine types x 1200 clips, not the official 7. ToyCar and ToyTrain are
absent from this mirror entirely. Every result in this project's evaluation
table is over the five real machine types (bearing, fan, gearbox, slider,
valve) — arguably the more relevant set for engines and gensets anyway, but
stated here so nobody assumes DCASE-official coverage. Never assume a
community mirror of a benchmark is complete; check row and class counts
against the paper's own numbers.

## ADR-5: Kaggle's public-dataset input path is not `/kaggle/input/<slug>`

**Context.** `zeyadzsm/engine-sounds` was correctly attached to every kernel
version via `dataset_sources`. `prepare_engine_sounds.py --root
/kaggle/input/engine-sounds` reported "0 files, 0 source keys" every time —
not an error, just silently nothing, because that path did not exist.

**Diagnosis.** A follow-up run's first cell listed `/kaggle/input/` directly:
the dataset had mounted at `/kaggle/input/datasets`, a shared folder, not a
per-slug one. This looks like a newer Kaggle mounting scheme that groups
multiple attached datasets together rather than the older one-folder-per-slug
layout most examples online assume.

**Fix.** Never hardcode an input path. Scan `glob.glob("/kaggle/input/*")` for
whichever directory actually contains more than a handful of the expected
file type, and use that. See the diagnostic cell in
`notebooks/build_faultid_only.py`.

## ADR-6: Kaggle's default torch build has dropped Tesla P100 (sm_60) support

**Context.** Kernels default to a P100 when GPU is requested. The pinned base
image ships `torch 2.10.0+cu128`. Both `baseline_ae.py`'s AE training and the
`hear21passt` PaSST wrapper crashed identically:

```
torch.AcceleratorError: CUDA error: no kernel image is available for
execution on the device
```

**The trap.** `torch.cuda.is_available()` returns `True` and `.to("cuda")`
succeeds without complaint. The failure only surfaces on the first real kernel
launch, deep inside a `forward()` call, which makes it look like a bug in the
model rather than a hardware/build mismatch.

**Fix.** Default every training and inference path in this project to CPU
(`baseline_ae.py`'s `INJINI_DEVICE` env var, `embed_backends.PasstBackend`
probing `torch.zeros(1, device="cuda")` before trusting `is_available()`).
The models here are small enough that CPU is a runtime cost, not a blocker —
the full five-machine PaSST evaluation took roughly three hours on CPU.
Do not assume a Kaggle GPU kernel is usable without a real forward-pass smoke
test on that exact base image first.

## ADR-7: Workspace must live outside `/kaggle/working`

**Context.** The first three notebook versions used `WORK =
"/kaggle/working/injini"` and downloaded the ~2.3 GB DCASE set into it.
`kaggle kernels output` downloads the **entire** `/kaggle/working` tree, no
selective fetch. The first output pull attempt ran for over ten minutes and
was killed without finishing; a second attempt, run to completion, pulled
back thousands of individual `.wav` files before it ever reached the actual
result JSONs.

**Fix.** `WORK = "/tmp/injini"`. Only the small deliverables (ONNX files, eval
JSONs, quant reports) are explicitly copied to `/kaggle/working` in the final
cell. The next output pull, of the exact same shape of results, took seconds.

## ADR-8: Streaming embedding, not batch-materialised

**Context.** `embed_backends.TorchBackend`/`OnnxBackend` originally built the
full array of log-mel spectrograms for an entire input list before batching it
for inference: `np.stack([logmel(w) for w in waves])`. This was invisible on
the DCASE evaluation (splits of ~1000-1200 clips) but on the 20,738-clip
engine-sounds fault-ID manifest it tried to hold roughly 9 GB of float32
spectrograms in memory before running a single inference.

**Failure mode, and why it was hard to diagnose.** The Linux OOM killer
SIGKILLs a process with no Python traceback. A `!python script.py` line inside
a Jupyter cell does not raise on a non-zero exit code, so the cell reports
success and moves on. The result: a script that silently produced nothing, no
error anywhere in the log, in a notebook that still reported COMPLETE. Losing
an afternoon to "why is there no output" is the direct cost of this bug.

**Fix.** Both backends now embed in fixed-size chunks (`CHUNK = 512` in
`embed_backends.py`), so peak memory is bounded regardless of corpus size.
`faultid.py` also gained explicit per-split progress prints and a top-level
`try/except` with a full traceback, so a future failure of any kind is visible
in the log rather than silent.

**Lesson.** Any evaluation or training script written against a small dataset
and later pointed at a much larger one needs its memory profile re-checked,
not just its correctness. "Works on `data/dcase2025_dev`" (≤1200 clips/split)
said nothing about behaviour on 20,000 clips.

## ADR-9: Commit the shipped ONNX artefacts, not just the code that makes them

**Decision.** Following the SiloSense precedent, the exported `.onnx`
embedders and every `eval_*.json` / `*_quant_report.json` are committed
directly to the repo (`models/`), not left as CI/notebook output only.
Genuinely intermediate files (`*_fp32_preproc.onnx`, quantize.py's
shape-inference pre-pass) stay gitignored.

**Why.** A training run's numbers are useless without a git-addressable
artefact they belong to. The alternative — regenerate on demand — means the
paper's reported numbers can silently drift from what's actually in the app's
`assets/` folder.

## ADR-10: A shared Kaggle input mount can silently contaminate a recursive glob

**Context.** With ADR-5's fix (scan `/kaggle/input/*` for whichever directory
holds wav files) in place, the fault-ID rerun completed and produced a number:
macro F1 = 0.427 across 25 classes. Reading the class list caught the problem
immediately: `Birds`, `Cats`, `Dogs`, `Door`, `Footsteps`, `Rain`, `Sirens`,
`Thunder`, `Traffic`, `Wind`, `Silence`, and a bare `Data_Fixed` sat alongside
the real fault classes.

**Root cause.** `/kaggle/input/datasets` (the shared mount from ADR-5) held
more than one attached dataset's files, and `prepare_engine_sounds.collect()`
globbed `root/**/*.wav` unconditionally, then fell back to `parts[-2]` as the
class name whenever a path didn't contain a `Data_Fixed` segment. Every wav
from whatever unrelated environmental-sound dataset also shared that mount
point got silently absorbed as a plausible-looking fault class. Separately, a
doubly-nested `Data_Fixed/Data_Fixed/...` path (malformed upstream, or an
artefact of the shared mount) produced a class literally named `Data_Fixed`.

**Fix.** `collect()` now requires the literal `Data_Fixed` path segment to
reject anything that isn't structurally this dataset, and explicitly rejects
a path where the segment right after `Data_Fixed` is itself `Data_Fixed`
(no class folder). Both rejection counts are printed. Verified locally against
a synthetic tree combining a valid engine-sounds layout, a contaminating file
from an unrelated dataset, and a malformed nested path — the fix keeps
exactly the two legitimate files and reports the two it dropped.

**Lesson.** A recursive glob rooted at a Kaggle input path is not safe to
trust just because it found "enough" files of the right extension — a shared
mount point means "enough wav files under this directory" does not mean "only
this dataset's wav files under this directory". Validate structure, not just
file count. And read the class list of any classifier trained on scraped
labels before trusting its score; the wrong number here would have looked
completely plausible (0.427 macro F1 across 25 classes reads fine at a glance)
while actually measuring how easily an AudioSet embedder tells birdsong from
piston slap, not fault identification.

## ADR-11: A plain `<Button>` under Material Components ignores `android:background`

**Context.** First on-device build (2026-09-11, Samsung SM-M075F), first real
screenshot. The Check button rendered as designed (flat dark surface, amber
stroke and text), but the Enrol button came back a solid Material-filled
amber block with white text — visually inconsistent with every other surface
in the app.

**Root cause.** The Material Components theme (`Theme.MaterialComponents.*`,
required for `androidx.appcompat`'s theming) auto-promotes a plain XML
`<Button>` tag to `com.google.android.material.button.MaterialButton` at
inflation time. `MaterialButton`'s default style paints its own
`MaterialShapeDrawable` background tinted by `colorPrimary` and ignores a
plain `android:background="@drawable/..."` attribute entirely. The Check
button happened to look closer to intended only because it started disabled
(not yet enrolled), and Material's disabled-state tinting muted the fill
enough to look coincidentally close to the flat design.

**Fix.** Both buttons declared explicitly as
`androidx.appcompat.widget.AppCompatButton` in the layout XML instead of the
bare `<Button>` tag, which is not auto-promoted and honours `android:background`
as written. Kotlin code needed no change: `AppCompatButton extends Button`, so
the existing `lateinit var checkButton: Button` field type still resolves via
`findViewById`. Also added an explicit disabled-state alpha (1.0 / 0.4) in
`MainActivity.setIdle()`, since the flat background drawable carries no
built-in disabled visual on its own.

**Lesson.** Under a Material Components theme, `<Button>` is not "just a
button" — it is silently a MaterialButton with its own opinionated background
handling. Any custom flat/outlined button style needs either
`AppCompatButton` explicitly, or a `style=` attribute pointing at one of
Material's own outlined/text button styles configured with the right
attributes (`app:strokeColor`, `app:backgroundTint`, etc.) — a plain
`android:background` on `<Button>` is silently a no-op.

## ADR-12: a verdict shown and immediately overwritten by the idle reset

**Context.** Second on-device screenshot: `runCheck()`'s success path called
`status(word, ...)` to show the three-tier verdict, then called `setIdle()`
in the same `runOnUiThread` block. `setIdle()` unconditionally calls its own
`status(...)` to show either "NOT ENROLLED" or "READY: Enrolled...", which
ran immediately after and overwrote the verdict before a human eye could ever
see it. The screenshot simply showed "READY" with the default enrolled
message, no sign anything had gone wrong, because nothing crashed — the app
was doing exactly what the code said, the code just said the wrong thing.

**Fix.** Split `setIdle()`'s button-enabling logic out into `refreshButtons()`
(touches only `isEnabled`/`alpha`, never the status text), and call that from
`runCheck()`'s success path instead of `setIdle()`. `setIdle()` itself still
calls `refreshButtons()` plus sets the default message, used everywhere a
default message is actually correct (startup, after Enrol, on error paths).

**Lesson.** A function whose name describes an app-level *state* ("idle") is
a poor place to also hide UI *side effects* that not every caller wants.
Every call site needs to be re-read once such a function's meaning depends on
what else it happens to be doing under the hood.

## ADR-13: the results link sat underneath the system navigation bar

**Context.** Third on-device interaction: tapping "Full model results" did
nothing, twice, with no crash and no log line. A `uiautomator dump` gave the
exact bounds: `resultsLink` at `[265,1514][455,1547]`, `navigationBarBackground`
at `[0,1510][720,1600]` — the link sat entirely inside the nav bar's own
window, which silently owns any touch in that region on an edge-to-edge
(targetSdk 35) window. The link wasn't broken; it was drawn under a pane of
glass that intercepted every tap before it reached the app.

**First attempted fix, also wrong.** `android:fitsSystemWindows="true"` on
the root layout does reserve the inset space and stops the overlap, but it
*replaces* the view's own `android:padding` with the inset-derived padding
rather than adding to it — the fix traded a hidden button for text sitting
flush against the left and right screen edges, losing the 28dp margin used
everywhere else in the layout.

**Real fix.** A manual `ViewCompat.setOnApplyWindowInsetsListener` on the
root view (`MainActivity.applySystemBarInsetsAsExtraPadding()`), which
captures the layout's own padding once and adds the system bar insets on top
of it rather than replacing it.

**Lesson.** "It isn't clickable" and "it's genuinely broken" look identical
from a screenshot alone on a device with on-screen navigation. `uiautomator
dump` (bounds of every view, including the system nav bar) found this in
under a minute where guessing tap coordinates from screenshots had already
failed twice. And a one-line insets fix is not automatically an equivalent
fix — `fitsSystemWindows` and a manual insets listener solve the same overlap
problem through different mechanisms with different side effects.

## ADR-14: fleet capability and real recording feedback, in the app itself

**Context.** Two gaps, raised directly: a ten-second recording showed a
status line that never changed until the clip ended, reading as a stuck
screen; and the app only ever knew one hardcoded "machine", with no way to
name a fleet, and no way to collect the labelled data the fault-ID head
(Section 09's secondary mode) actually needs to improve past 0.362 macro F1.
A follow-up correction: any data collection had to point at real supervised
retraining, not just sit as loose recordings.

**Decision.**

- `WaveformView.kt`: a live level meter in the same five-bar shape as the app
  icon, fed by `AudioCapture.record()`'s new per-chunk `onProgress` callback
  (RMS level plus elapsed fraction). Every recording, Enrol, Check or a
  labelled clip, now drives the waveform and a live "N s remaining" line
  through one shared `recordWithFeedback()` path, so none of them can regress
  independently back to a frozen screen.
- `MachineRegistry.kt`, the same JSON-registry shape as TapSense's
  `VesselRegistry`: named machines, each with its own persisted
  `AnomalyScorer`, check count and last verdict, and a per-label count of
  training clips collected. Enrol and Check now operate on whichever machine
  is selected, not a single global scorer in SharedPreferences.
- `FaultLabel.kt` + `LabeledClipStore.kt`: "Add training clip" records one
  clip and saves it as a real WAV plus a manifest row shaped exactly like
  `prepare_engine_sounds.py`'s existing columns (`path, label, class_name,
  source_key, split, ...`), with the machine id as `source_key` — the same
  unit that script already splits on. The label set is the corpus's own
  fault classes (including its "Engine kanocking" typo, preserved on purpose
  so a clip joins the existing class instead of forking a near-duplicate
  one), plus a free-text "Other" path so a fault this taxonomy has no name
  for yet gets kept, not discarded. The point of this shape is that `adb
  pull` of one directory is the entire step between a phone in the field and
  a retrain, not a separate export tool to design later.

**Verified on-device**, Samsung SM-M075F: a full 6-clip Enrol cycle (waveform
correctly resets and re-shows between clips, no hang, no crash), a Check, and
an "Add training clip" run end to end — machine created, fault selected,
mechanic's verdict typed, clip recorded with a live waveform and countdown,
saved. The written WAV opens cleanly in Python's stdlib `wave` module (mono,
16-bit, 16 kHz, exactly 10.0 s) and the manifest row carries the real device
model, capture source (`UNPROCESSED`), and the machine id as source key.

## ADR-15: ADR-14 was a redundant button, not a design. Corrected.

**Context.** ADR-14 shipped a third button, "Add training clip", as a
manually-triggered recording separate from Enrol and Check. Direct
correction: every Enrol clip is already a healthy recording of a real
machine, so writing it to the corpus is not an extra decision, it is what
Enrol already means. Every Check clip is a real recording too; what it is
missing is not a button; it is a verdict, which does not exist yet at record
time and normally will not exist for hours or days, until a mechanic has
actually opened the machine up. A button that asks "healthy or faulty?" at
the moment of Check is asking a question nobody present can honestly answer.
A further correction: any of this had to point at real supervised
retraining, not just a habit of collecting files.

**Decision.**

- Enrol now writes every one of its clips straight into the training corpus
  under the healthy label, no separate save step, no dialog. This removed
  the third button entirely — `AddClipButton`, `FaultLabel`'s trigger from
  `MainActivity`, and the "Add training clip" flow are gone from the main
  screen. `LabeledClipStore.save()` is now called only from inside
  `runEnrol()`'s loop.
- Check now saves its clip too, but into a new pending queue
  (`LabeledClipStore.savePending`/`pending.csv`), tagged only with the
  on-device model's tier as a hint, never a final label. `MainActivity`'s
  machine row grows a live "N awaiting a verdict" flag the moment a Check
  produces one.
- `MachineListActivity` (new) replaces the old one-line "type a name" dialog
  entirely: a real fleet screen, "+ Add machine" opens a form collecting
  engine type (Petrol/Diesel/Not sure) and machine category
  (Vehicle/Generator/Pump/Other) alongside the name, because a rod knock on a
  diesel generator and a rod knock on a petrol kombi are not the same
  training example and a future model needs to know which is which. Tapping
  a machine with pending recordings opens that machine's queue directly;
  picking one clip opens the same label dialog from ADR-14, now firing
  `LabeledClipStore.confirmPending()`, which moves the file from `_pending/`
  into its final label folder and appends the confirmed manifest row.

**Verified on-device**, Samsung SM-M075F, from a clean install: added a
Diesel/Generator machine ("Genset_Clinic") through the new form, ran a full
6-clip Enrol (all six landed in `manifest.csv` under `Normal` with
`engine_type=Diesel, category=Generator`, no extra tap), ran a Check (landed
in `pending.csv`, the model's tier as `provisional_tier`, machine row showed
"1 awaiting a verdict" immediately), then labelled it from the machine list
as "Alternator Bearing Noise" with a typed mechanic's verdict — confirmed the
WAV physically moved out of `_pending/` into `Alternator Bearing Noise/`, the
pending row was removed, and the confirmed manifest row carried the real
verdict text and the machine's engine type and category.

## Results summary (for context; full table and narrative in Injini.docx §10)

Five DCASE 2025 machine types (bearing, fan, gearbox, slider, valve), CPU,
2026-09-11.

| System | Official score | Mean AUC |
|---|---|---|
| DCASE autoencoder baseline, reproduced | 0.547 | 0.560 |
| PaSST transformer, whitened, kNN | 0.596 | 0.637 |
| mn10_as, whitened, kNN | 0.613 | 0.655 |
| mn10_as, whitened, Mahalanobis | 0.616 | 0.661 |
| mn10_as, whitened, kNN, INT8 | 0.599 | 0.640 |
| mn04_as, whitened, kNN | 0.593 | 0.630 |
| mn04_as, whitened, kNN, INT8 | 0.602 | 0.644 |

Supervised fault-ID head (secondary mode, not the table above's metric):
0.362 macro F1, 12 classes, source-disjoint split, mn10_as FP32 embeddings.

Every embedder configuration beats the reproduced baseline. mn10_as with kNN
edges past the PaSST transformer reference with the same scorer (0.613 vs
0.596) — directional, not definitive, since PaSST ran capped to CPU and to
five machine types, but a fair, matched comparison.

## Open items

- **Fault-ID head**: resolved. 0.362 macro F1 over the real 12 fault classes,
  8550 train / 1888 val, 332 source recordings, source-disjoint split
  (`models/faultid_mn10_as_fp32.json`). Well below the ~98-99% ceiling clean,
  single-vehicle, randomly-split data gets — expected, given a source-disjoint
  split on a messy, unlicensed-provenance public corpus. Not a target to beat,
  a number to report honestly.
- **On-device measurement**: resolved, on a Samsung SM-M075F (not the M16
  originally planned — whatever phone was connected). mn10_as embedder,
  INT8 44.05 ms steady-state (min 36.24), FP32 87.54 ms (min 79.54). Load
  time INT8 241.7 ms vs FP32 87.1 ms, the same QDQ-compile-on-first-use cost
  SiloSense saw. Executed-per-node trace: `CPUExecutionProvider=3670` — on
  this device XNNPACK registered but ran nothing, everything actually
  executed on plain CPU, unlike SiloSense's Galaxy M16 where XNNPACK did the
  real work. Process memory 178,224 KB PSS, thermal status LIGHT.
- **MAC counting**: `export/quantize.py`'s `onnx_macs()` is a rough Conv/Gemm
  estimate from static shapes where present, not a full profiler count.
- **On-device calibration is unvalidated with a 6-clip enrollment.** A real
  Check against a freshly-enrolled machine, minutes later in the same
  environment, scored "GET IT LOOKED AT" (0.895) rather than healthy. The
  on-device diagonal Mahalanobis (`AnomalyScorer.kt`) estimates a per-dimension
  variance from only 6 healthy clips across a 960-d embedding; any dimension
  with near-zero variance in that tiny sample produces a huge inverse-variance
  weight, so an ordinary embedding wobble in one dimension can dominate the
  score. This is a real calibration gap, not confirmed as a bug: the DCASE
  evaluation's full-covariance Mahalanobis (Section 10's numbers) was run on
  hundreds of training clips, not 6, and was never meant to validate this
  diagonal on-device approximation. Needs either more enrollment clips, a
  regularised/shrunk variance estimate, or a validated bounds recalibration
  before the three-tier verdict can be trusted the way Section 10's DCASE
  numbers can.
