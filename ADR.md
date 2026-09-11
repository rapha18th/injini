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

Every embedder configuration beats the reproduced baseline. mn10_as with kNN
edges past the PaSST transformer reference with the same scorer (0.613 vs
0.596) — directional, not definitive, since PaSST ran capped to CPU and to
five machine types, but a fair, matched comparison.

## Open items

- **Fault-ID head**: the first clean number, 0.427 macro F1, turned out to be
  over a contaminated 25-class problem (ADR-10). The fix is in and verified
  against a synthetic tree; a rerun (`injini-faultid` kernel v2) was in flight
  as of 2026-09-11.
- **On-device measurement**: latency, execution-provider trace, memory,
  thermal, battery all need a real Galaxy M16 (or equivalent), not Kaggle.
- **MAC counting**: `export/quantize.py`'s `onnx_macs()` is a rough Conv/Gemm
  estimate from static shapes where present, not a full profiler count.
