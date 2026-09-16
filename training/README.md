# Language-ID training

These scripts build a language-ID dataset, train three kinds of specialist for the app's confusion groups (Malay/Indonesian, Simplified/Traditional Chinese, romanized Hindi/Urdu/English), and evaluate them on data they never saw. The design is in section 16 of [the design document](../docs/design/2026-09-15-multilingual-rag.md).

- Data sources, licences and counts: [data/DATACARD.md](data/DATACARD.md)
- Results, generated from `results/`: [RESULTS.md](RESULTS.md)

## Models

| Model | Script | Hardware |
|---|---|---|
| fastText with character n-grams | `training/fasttext_train.py` | CPU, in an isolated environment with `fasttext-wheel` |
| Small MLP on frozen multilingual-e5-small embeddings | `training/embed_cache.py`, `training/embed_head.py` | GPU; CPU works, slower |
| multilingual-e5-small fine-tuned end to end | `training/finetune_e5.py` | GPU |

## Setup

```bash
uv sync --group train
PYTHONUTF8=1 uv run --group train python -m training.env_report --require-cuda
uv run python scripts/download_models.py --glotlid --e5-torch
```

On Windows and Linux, PyTorch comes from the CUDA 13.0 wheel index (`[tool.uv.index]` in `pyproject.toml`); `--require-cuda` fails on a CPU-only build. On macOS uv installs the PyPI build: pass `--device mps` or `--device cpu` to the embedding, head and fine-tune scripts.

The app itself never imports PyTorch. Exported models run with fasttext-predict or onnxruntime.

## Order of steps

```bash
PYTHONUTF8=1 uv run --group train python -m training.data.build
PYTHONUTF8=1 uv run --no-project --python 3.12 --with fasttext-wheel==0.9.2 --with numpy==1.26.4 python -m training.fasttext_train --sweep
PYTHONUTF8=1 uv run --group train python -m training.embed_cache
PYTHONUTF8=1 uv run --group train python -m training.embed_head --sweep
PYTHONUTF8=1 uv run --group train python -m training.finetune_e5
uv run python -m training.select
# export each family's winner (run ids are in results/lid-selection-*.json), then:
PYTHONUTF8=1 uv run --group train python -m eval.langid_eval
PYTHONUTF8=1 uv run --group train python -m eval.flores_benchmark
uv run python -m training.results_table
```

Every run writes `runs/<run_id>/config.json` (settings, GPU, driver, PyTorch and CUDA versions, commit), `metrics.jsonl` (one line per epoch) and `final.json` (per-label F1 and confusion matrices for the confusion groups, wall-clock time). `runs/`, `models/` and `data/cache/` are git-ignored; `training/select.py` copies the chosen runs' text records to `results/lid-runs/`.

The GPU scripts first check for a running `llama-server` (DaybreakOS latency measurements share this GPU) and wait, checking every 5 minutes for up to 2 hours.

To use a trained specialist in the app, set `LANGID_SPECIALIST` in `.env` to the exported file name in `models/`.

## Limits

- The fastText sweep trains 6 of the 36 configurations listed in the design, and the fine-tune stops after 2 epochs or 170 minutes, whichever comes first (the design allows 4 epochs). Both caps keep the full run to one night on an 8 GB laptop GPU.
- Models are chosen on `val_select`. For labels whose validation split is not source-disjoint (see the data card), those scores are likely higher than on text from new sources.
- FLORES-200 is professionally translated Wikimedia text. Short, informal chat questions are harder, which is why the evaluation also reports 1-3 word crops and code-mixed lines.
- The published GlotLID and OpenLID figures come from the papers' model versions and label mappings, so the comparison in RESULTS.md is a consistency check, not a reproduction.
- Runs are not bit-for-bit reproducible: fastText trains with several threads and GPU kernels are not deterministic.
