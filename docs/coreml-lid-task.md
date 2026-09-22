# The language-ID specialist on Apple silicon

A task document, published here rather than in `.plans/` (which is not committed)
so it is available on any machine that clones this repository.

**Goal:** Convert the trained stage-2 specialists to Core ML, prove the converted models give the
same answers as the ONNX ones, and publish a table of accuracy against size, latency and peak
memory measured on this machine.

**Why:** The repository already shows that a **0.8 MB** head scores 0.881 group macro-F1 where the
**450 MB** fine-tune scores 0.913 — a 560× size difference for 3.2 points. What it does not show is
what either costs to *run* on a device. That number is the difference between "trained a language
ID model" and "shipped one inside a device budget", and it can only be measured on Apple hardware.

**Run this on the Mac.** Everything here needs Apple silicon; none of it can be done on the Windows
machine the models were trained on.

---

## Before you start

Read `AGENTS.md` first — it has the commands, the commit rules and the setup steps.

You need the trained specialist files copied from the training machine, because they are not in
the repository:

| file | size | what it is |
|---|---:|---|
| `models/lid-specialist-e5-head.onnx` | 0.8 MB | a classifier over e5 embeddings — **start here** |
| `models/lid-specialist-e5-head.json` | tiny | labels, prefix, embedding dim |
| `models/lid-specialist-e5-finetune.onnx` | 450 MB | a fine-tuned e5 encoder plus head |
| `models/lid-specialist-e5-finetune.json` | tiny | its sidecar |

Install the converter into the project environment:

```bash
python -m uv add --dev coremltools
```

## What the models actually take and return

Read `backend/pipeline/specialist.py` before converting anything. In short:

**The head** (`OnnxHeadScorer`) is a small classifier over sentence embeddings.

- input name `embedding`, float32, shape `(batch, 384)`
- output name `logits`, shape `(batch, 12)`
- the 12 labels are in the sidecar JSON, in order
- the caller applies softmax afterwards; the model returns raw logits
- the embeddings come from multilingual-e5-small with the prefix `"query: "` already applied

**The fine-tune** (`OnnxClassifierScorer`) is the whole encoder plus a head, fed token ids from the
e5 tokenizer with a fixed `max_length` and `pad_id`. It is much larger and harder to convert.

**Do the head first.** It is 0.8 MB, converts in seconds, and is the model that plausibly belongs on
a device. Only attempt the fine-tune once the head is done end to end.

---

## Task 1: Convert the head and prove it gives the same answers

A conversion that silently changes its outputs is worse than no conversion, because it looks like
success. This repository already takes that seriously: the sidecar records
`onnx_max_abs_diff: 0.0001220703125` from when the model was exported to ONNX. Do the same thing
for Core ML.

- [ ] **Step 1.** Write `scripts/coreml_export.py`. It loads the ONNX head, converts it with
      `coremltools`, and writes `models/lid-specialist-e5-head.mlpackage`.

- [ ] **Step 2.** In the same script, feed **the same input** to the ONNX model and to the Core ML
      model and compare the outputs. Use at least 256 random float32 vectors of shape `(384,)`,
      and also real embeddings if you can produce them. Record the **maximum absolute difference**
      across all of them.

- [ ] **Step 3.** Write that number into the sidecar JSON as `coreml_max_abs_diff`, beside the
      existing `onnx_max_abs_diff`.

- [ ] **Step 4.** Decide what counts as acceptable and say so in the commit message. The ONNX
      export accepted 1.2e-4. If Core ML's difference is far larger, **stop and report it** rather
      than continuing — a converted model that disagrees with the one whose accuracy was measured
      no longer has a measured accuracy.

- [ ] **Step 5.** Commit. Suggested message: `Export the language-ID head to Core ML and record its
      numerical agreement`.

## Task 2: Measure latency and peak memory

This is the point of the whole plan. Be careful, because a sloppy benchmark is worse than none.

- [ ] **Step 1.** Write `scripts/coreml_bench.py` measuring, for one input at a time (batch size 1,
      which is what a real caller does):
      - **warm-up runs that are discarded** — the first call includes model load and compilation
      - **median and 95th-percentile latency** over at least 200 runs, not the mean, because a few
        slow outliers will otherwise hide the typical case
      - **peak resident memory** during the run
      - the same three figures for the ONNX model through `onnxruntime`, so there is a baseline

- [ ] **Step 2.** Run each of the Core ML compute units separately — CPU only, CPU and GPU, and
      CPU and Neural Engine — and record all three. Which one the Neural Engine actually accepts is
      itself a finding worth writing down.

- [ ] **Step 3.** Record the machine: chip, core counts, memory, macOS version, `coremltools`
      version. Latency figures without the machine they came from say nothing, and this repository
      already states that about its evaluation numbers.

- [ ] **Step 4.** Write the results to `results/coreml-bench-<date>.json`. Do not type numbers into
      prose by hand — every other table in this repository is generated from a results file, and
      `eval/readme_table.py` and `training/results_table.py` show the pattern.

- [ ] **Step 5.** Commit. Suggested message: `Measure the Core ML head's latency and memory on
      Apple silicon`.

## Task 3: Publish the frontier

- [ ] **Step 1.** Extend `training/results_table.py`, or add a small generator beside it, to emit a
      table joining what is already known to what you just measured:

      | model | group macro-F1 | size | median latency | p95 | peak memory | compute unit |

      The head and the fine-tune are both rows if both converted; the head alone is still worth
      publishing.

- [ ] **Step 2.** Add it to `training/RESULTS.md` under a new heading, generated, not hand-typed.

- [ ] **Step 3.** Add a short section to `README.md` near the language-identification section
      pointing at it. Two or three sentences. State the trade-off plainly: what the small model
      costs in accuracy and what it saves in size, latency and memory.

- [ ] **Step 4.** If the fine-tune did not convert, **say so in the README** and say why. A failed
      conversion honestly recorded is more useful than silence, and "this architecture did not
      convert cleanly" is a real finding.

- [ ] **Step 5.** Commit. Suggested message: `Publish the accuracy, size and latency frontier for
      the language-ID specialists`.

## Task 4: Attempt the fine-tune, only after the head is finished

- [ ] The 450 MB fine-tune is a transformer encoder and may not convert cleanly. Give it a bounded
      attempt. If it fails, record the error and move on — do not spend the session on it.
- [ ] If it converts, run it through Tasks 1 and 2 unchanged: parity first, then the benchmark.

---

## What not to do

- **Do not retrain anything.** The models are trained and their accuracy is already measured. This
  plan is about deployment cost, not accuracy.
- **Do not change the accuracy numbers** in `training/RESULTS.md`. They come from a recorded
  evaluation and are not yours to edit here.
- **Do not push.** Commit locally and let the owner decide.
- **Do not report a latency figure you have not measured**, and do not report a mean where the
  median and p95 were asked for. This repository's worth is that its numbers are true.
