# 0001: Ship GlotLID compressed

Status: accepted, 2026-09-15.

## Context

GlotLID covers about two thousand languages, which is what makes the app answer someone who writes
in a language it does not fully support. The published model is 1.69 GB and loads its matrices into
memory, which is too much to ask of a laptop that is also running a local answer model.

## Decision

Product-quantize the model with fastText's own quantization, without vocabulary pruning, and ship
the 225 MB result as the default detector. `scripts/compress_glotlid.py` does it, and the full model
stays available for comparison.

## Consequences

- The app's language-ID model fits next to an answer model on an ordinary machine.
- Compression changes accuracy per language. The measured change is in
  `results/langid-eval-2026-09-15.json` and in the tables generated from it; it is not assumed to be
  zero.
- Quantization needs the full fastText build, which has no macOS arm64 wheel for Python 3.12, so the
  compression step runs on Windows, Linux or in the container and the file is copied to a Mac.
