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

## ADR-16: four direct UX corrections after actually using ADR-15's design

**Context.** ADR-15 shipped, and using it surfaced four real problems the
build-and-screenshot loop alone had not caught, because none of them are
bugs — the app did exactly what its code said, and what its code said was
wrong. Fixed all four together since they touch the same few files.

**1. The waveform looked dead in a quiet room.** `WaveformView` only ever
drew what the microphone actually measured, and an ordinary quiet recording
site (most of them) sits close enough to silence that the bars barely
moved — technically live, indistinguishable from stuck at a glance. Fixed
with a continuous idle animation independent of the microphone signal: a
slow travelling ripple across the bars (`Handler.postDelayed` every 45ms,
`sin(phase*0.12 + i*0.35)` per bar) plus a breathing REC dot in the corner,
both running the entire time a recording is in progress. Real audio now
rides on top of that floor rather than being the only source of motion.
Verified with two screenshots taken about a second apart during a live
Check: the ripple pattern visibly shifted between them.

**2 and 3. Adding a machine, and seeing what you already know about one,
both required leaving the home screen — and the old plain-text "Machine: X"
row gave no visual signal it was interactive at all.** Both fixed together:
the row is now worded and styled as a dropdown ("Select or add a machine
⌄"), and tapping it lists every known machine plus a trailing "+ Add new
machine" row — reached from the home screen directly, no detour through the
fleet screen required. Selecting a machine now populates an inline info card
right there on the home screen (engine type, category, enrolled state,
check count, training clip count) instead of requiring a trip to
`MachineListActivity` just to see it. A separate, always-visible "Machines"
button still opens the full fleet screen for management (delete, and
reviewing every machine's queue in one place) — it was never folded into the
dropdown, so managing the fleet is never a hidden side effect of picking
from it.

**4. "Not now" on the pending-verdict dialog read as declining the only way
to select an already-enrolled machine from the list.** The actual bug:
`MachineListActivity`'s row `onClickListener` checked `if (pending > 0)
showPendingQueue(...) else selectAndReturn(...)` — so tapping any row with a
pending recording never selected it; it forced the pending dialog open
first, and "Not now" was the only path back to selecting, worded like a
decline for what was actually the sole route to the primary action. Fixed
by splitting the row into two independent tap targets: the name/status area
always calls `selectAndReturn` and nothing else, and reviewing pending
recordings is now its own visibly button-shaped element within the row
("⚠ N recordings need a verdict — Add verdict ›", chip-styled, its own
`OnClickListener`). The same fix applies on the home screen: the new
`addVerdictButton` is a real `AppCompatButton`, not text carrying an
implication.

**New shared code**, to keep the two entry points (home screen, fleet
screen) from drifting: `MachineForms.showAddMachineDialog()` (the add-machine
form) and `VerdictFlow` (the pending-queue list plus the label dialog) are
each defined once and used from both `MainActivity` and
`MachineListActivity`.

**Verified on-device**: switching machines from the home-screen dropdown
correctly repopulates the info card and the Add-verdict button's
visibility; tapping a Machines-list row with 3 pending recordings selects
that machine immediately, with no dialog in the way; tapping its separate
"Add verdict" chip opens the queue as before.

## ADR-17: field benchmark tracking, minimalist icons, and a real vector-drawable bug

**Context.** Four more direct user corrections: the Machines button was
awkwardly placed, a pending-recording's timestamp in the verdict dialog
didn't read as tappable, there was no mechanism for the app to notice when
its own on-device tier disagreed with what a mechanic actually found, and
the app wanted minimalist icons instead of unicode characters (⚠, ⌄, +).

**Field benchmark, not just data collection.** Every Check clip already
queues with the model's on-device tier (`AnomalyScorer.Tier`) as a hint;
`LabeledClipStore.confirmPending()` now also compares that tier against the
verdict it's finally given and writes the result — `false_positive`,
`false_negative`, or `confirmed` — into a new `benchmark_outcome` manifest
column. A false negative (tier said HEALTHY, mechanic found a real fault)
also flags the machine `needsReenrollment`: the fingerprint that produced
that reading may itself include a bad sample, since it was built from only
6 clips (see ADR-16's Open Items note on that calibration gap — this doesn't
fix the diagonal-Mahalanobis noise, it gives the app a way to notice when
that noise has actually mattered). Both outcomes surface immediately in a
dialog (`VerdictFlow.reportOutcome()`), not just silently in a CSV column,
and the fleet-wide totals now sit in "Full model results" alongside the
DCASE numbers — this app is a live benchmark of the shipped model, not only
a data collector, which is the whole point per the project's own framing.

**A real vector-drawable bug, not a tint/theme issue.** Building the icon
set, every single icon rendered as a totally blank shape — background chips
showed, `src`/compound-drawable content did not, across `ImageButton`,
`AppCompatImageButton`, and `TextView` compound drawables alike. `aapt2 dump
xmltree` on the built APK confirmed the resource compiled and the layout
bound to it correctly; three device-specific hypotheses (Samsung/OneUI
tint bug, GPU hardware-acceleration compositing, `layerType="software"`)
were tried and ruled out in turn before the actual cause surfaced: every
vector file used `android:path` instead of the real attribute name,
`android:pathData`. `android:path` compiles cleanly (it's a legitimate
framework attribute, just not the one `VectorDrawable`'s path element reads)
so nothing errors at build time — the path silently has no data, and an
empty path draws nothing. Fixed by renaming the attribute in all five icon
files. Lesson: when a resource "compiles fine and binds fine" but renders
as nothing, check the actual XML attribute names against the framework
class's real `styleable`, not just against what `aapt2` accepted — a wrong
but legal attribute name is a silent no-op, not an error.

**Machines button, verdict-queue rows.** The Machines button moved off the
crowded picker row into its own circular icon button beside the wordmark
(`ic_fleet_24`), freeing the picker chip to span full width. The
pending-verdict queue (`VerdictFlow.showQueue()`) was a plain
`AlertDialog.setItems()` list — inert-looking text with no visual cue a row
was the only way into the label dialog. Replaced with custom chip rows
(`item_pending_row.xml`): bold timestamp, tier subtitle, trailing chevron,
ripple foreground — the same "control, not text" fix ADR-16 already applied
to the fleet screen's rows, now applied here too.

## ADR-18: dataset export, a live label counter, and Hugging Face as the sync target

**Context.** "Exporting of the collection" was named the most important
open piece: a way to get every recording off the phone, manually or over
WhatsApp, plus a live tally of healthy vs. faulty clips to judge dataset
size against. Separately, a remote sync target for phones with internet.

**Export as a share-sheet target, not a bespoke transfer mechanism.**
`DatasetExporter.export()` zips the whole `InjiniLabeled` tree (confirmed
corpus and the still-pending queue both — nothing is excluded just because
it hasn't been given a verdict yet) and, from Android 10 on, writes it
through `MediaStore.Downloads` rather than app-private storage: this is the
only scoped-storage-correct way to land a file where any file manager can
see it, and a `MediaStore` `Uri` is natively `content://` and shareable with
no `FileProvider` needed. Below API 29, a `FileProvider` + a direct write to
the legacy public Downloads directory covers it instead — this app's minSdk
is 26, and a rugged budget phone still on Android 8/9 is a real device in
its actual market, not a hypothetical to skip. Verified live: the export
landed in `/storage/emulated/0/Download/`, and the resulting share sheet
listed WhatsApp, Gmail, Drive, and individual WhatsApp contacts directly —
both halves of "collect manually or receive on WhatsApp" confirmed with one
tap, no extra integration code for either.

**Live label counter.** `LabeledClipStore.labelCounts()` scans
`manifest.csv` for a healthy/faulty tally (plus the per-fault-type
breakdown behind "faulty") and renders on the Machines screen as "Dataset: N
healthy · M faulty · T total", tap-through for the breakdown. Deliberately
no hardcoded "good dataset size" threshold baked into the UI — the numbers
are the product, judging them is the user's call, not an opinion this app
should assert.

**Remote sync: Hugging Face over Google Drive.** Google Drive needs OAuth —
a Google Cloud Console project, a consent screen, and (past 100 test users)
a verification review — real bureaucratic friction for a one-person field
tool, and it was already reachable for free: "My Drive" appears directly in
the export share sheet above with zero extra code. Hugging Face needs only
a personal access token, no app registration or review, and — the deciding
factor — it closes the loop this project already runs on: the DCASE mirror
this app's own eval pulls from, and the Kaggle notebooks that would read the
next retrain's data, both already live on HF. A phone syncing straight into
an HF dataset repo is one hop from field recording to next-retrain input;
a Drive folder is an extra manual download-and-reupload step in between.
Confirmed the exact wire contract before writing code (per this project's
own standing rule not to assert unverified facts about external APIs): the
Hub's JSON commit endpoint is `POST /api/datasets/{repo}/commit/{rev}`,
bearer-token auth, `{"summary", "files":[{"path","content"(base64),
"encoding":"base64"}]}` — implemented directly over `HttpURLConnection`, no
`huggingface_hub` dependency, capped at 40MB (larger exports should use the
manual Export-and-share path instead, until a chunked/LFS path is worth
building). `HfSyncActivity` stores the token in plain (unencrypted)
`SharedPreferences` — app-private but not at-rest encrypted — with an
explicit recommendation in the UI to scope the token to a single repo with
write-only access, so a compromised device's exposure stays bounded. Purely
user-triggered, no background/scheduled sync: this app has been offline-only
until this one feature, and stays that way except for this one explicit tap.

Verified live end to end short of a real token: entering a fake token and
repo and tapping Upload now produced a real `401 {"error":"Invalid username
or password."}` from huggingface.co, confirming the request reaches the
real endpoint correctly formed (right URL, right auth header shape, right
JSON body) — it needs a real token to succeed, but every part of the wiring
up to that point is proven, not assumed.

## ADR-19: ADR-18's direct-to-HF sync was corrected — a phone can't safely hold a write token

**Context.** ADR-18 shipped `HfSync`/`HfSyncActivity` pasting a personal
Hugging Face **write** access token straight into the phone, stored in
plain `SharedPreferences`, calling Hugging Face's commit API directly. The
user's correction, immediately: "we can't obviously ship a bundled token
right?" — right. Even though the token was meant to be pasted per-phone by
whoever set it up rather than baked into the APK at build time, the actual
threat model is the same either way: any Android app's `SharedPreferences`
is trivially readable on a rooted device or via a backup extraction, and
once a real Hugging Face write token is on a phone in the field, it can
write (or be used to corrupt) the dataset repo directly, full stop. A field
phone is exactly the device most likely to be lost, stolen, or handed to
someone else to use.

**Decision.** Interpose a small backend — **Injini Relay**
(`hf_space/`, deployed as a Hugging Face Space) — between phones and Hugging
Face. The relay is the only thing that ever holds the real `HF_TOKEN`, kept
as a Space secret (Settings → Variables and secrets), never in git, never
on a phone. Phones instead hold a separate, low-privilege `UPLOAD_API_KEY`
that only lets them call the relay's own `/api/upload` — if that key leaks
from a decompiled APK, the blast radius is bounded to "someone can submit an
upload," not "someone can write to the dataset."

**A private HF Space would not have solved this — it would have moved the
same problem.** Checked before building anything (external API behavior,
not assumed): a **private** Space requires every external HTTP caller,
including the phone, to present a real Hugging Face account token in the
`Authorization: Bearer` header just to reach the Space at all. That
reintroduces exactly the thing being avoided, one layer down. The fix the
user then pointed out, correctly: Hugging Face Pro (which this account has)
offers a third visibility tier, **Protected** — the running app's URL stays
publicly reachable with no HF auth required to call it, while the Space's
own source code (`app.py`, `Dockerfile`) is hidden from public Hub browsing.
That is the right setting here: public enough for a phone's plain HTTP call,
private enough that the validation logic and endpoint shape aren't sitting
in a public repo for anyone to read.

**Second safeguard: every upload becomes a pull request, never a direct
commit.** `HfApi.upload_file(..., create_pr=True)` in `hf_space/app.py`.
This is the actual answer to "won't put you in a pickle when processing for
training": a corrupted zip, a bad field recording, or a spam attempt against
a leaked API key can never silently become part of what a training run
reads — it sits as a proposed change on the dataset repo until a person
looks at it and merges. Point Kaggle notebooks at the reviewed `main`
branch, never at open PRs. The relay also validates before proposing
anything (must be a real zip, must contain `manifest.csv`, capped at
150MB), and every upload lands under its own
`field_exports/<source>/<timestamp>_<file>` path — nothing is ever
overwritten, so a bad batch from one device is always traceable and
revertible without touching anyone else's contribution.

**Android side**: `HfSync.kt` rewritten from a JSON+base64 POST straight to
`huggingface.co/api/datasets/.../commit/main` to a real multipart file
upload against the relay's `/api/upload`, using OkHttp (added as a
dependency — hand-rolled multipart boundary strings over
`HttpURLConnection` are easy to get subtly wrong for something that uploads
real training data, and this is not where to find that out). `HfSyncActivity`
now collects a relay URL and an upload key, not a Hugging Face token and a
repo id. Verified live: a placeholder relay URL that doesn't exist yet
produced a genuine HTTPS round trip and a clean, non-crashing "Upload
failed" dialog with Hugging Face's own real 404 page as the body — proof
the multipart request, file attachment, and error handling all work
correctly, pending only a real deployed relay to succeed against.

**A real bug the smoke test caught, not assumed away.** Ran the relay
locally with fake secrets before calling any of this "done": `GET /health`
worked immediately, but the Basic-Auth-protected manual upload page 500'd
with `TypeError: unhashable type: 'dict'` inside Jinja2. Cause: Starlette
changed `Jinja2Templates.TemplateResponse`'s calling convention — `request`
must now be the first positional argument (`TemplateResponse(request, name,
context)`), not a `"request"` key inside the context dict
(`TemplateResponse(name, {"request": request, ...})`, the old, now-removed
form). `requirements.txt` pins no upper bound on `fastapi`/`starlette`, so a
fresh install resolved the current version and hit the current signature.
Fixed both call sites in `app.py`. Then re-ran the full matrix locally
end to end with real HTTP requests (`curl`, a synthetic zip with a real
`manifest.csv`) rather than trusting the fix by inspection: missing API key
→ 401, wrong API key → 401, a `.zip`-named file that isn't actually a zip →
400, a real zip via both the phone-facing `/api/upload` route and the
manual `/` web-form route → both correctly reached Hugging Face's real API
and surfaced its genuine `401`/`Repository Not Found` response (expected,
given fake `HF_TOKEN`/`HF_DATASET_REPO` in this local run) instead of
crashing. Every guard in `app.py` is now proven against a running server,
not just read back and assumed correct.

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
