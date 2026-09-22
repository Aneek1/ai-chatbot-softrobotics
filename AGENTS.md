# Working in this repository

A multilingual retrieval-augmented chatbot about soft robotics. Python/FastAPI backend, React
frontend, and a two-stage language identifier whose specialist model was trained here.

Keep this file short — it is loaded into context every session.

If you were pointed here to do the Core ML work, the task is in
[docs/coreml-lid-task.md](docs/coreml-lid-task.md).

## Commands

Backend, from the repo root. `uv` is often not on PATH; `python -m uv` works either way.

```bash
python -m uv run pytest -q          # 544 tests
python -m uv run ruff check .
python -m uv run uvicorn backend.app.main:app --port 8000
```

Frontend, from `frontend/`:

```bash
npm test          # 129 tests, Vitest
npm run typecheck
npm run lint
npm run build
npm run check:bundle   # fails if an external URL ends up in the built bundle
npm run dev
```

On Windows only, set `PYTHONUTF8=1` for any Python command that prints Devanagari, Chinese,
Japanese, Korean or Tamil, or the console raises `UnicodeEncodeError`. macOS and Linux do not
need it.

## Rules

- **Commits:** plain sentence messages describing what changed and why. **No `Co-Authored-By`
  and no trailers of any kind.** One logical change per commit.
- **Never push without being asked.** `main` is public and CI runs on it.
- Some clones are **sparse and blob-less**. Never run `git sparse-checkout disable` — it would
  download a 400 MB virtualenv.
- Files under `results/` and `training/data/DATACARD.md` often show as modified with no real
  diff. That is line-ending noise. Leave them alone and stage only your own files.

## Invariants — do not break these

- **`frontend/src/lib/api.ts` is the only module that may touch the network.** Every component
  and hook is tested by mocking that one module. If `fetch` appears anywhere else in `src/`,
  that testability is gone.
- **`frontend/src/lib/` holds plain functions with no React in them**, so they can be tested
  without rendering anything.
- **Generated or retrieved content must keep its provenance.** The Sources tab shows a licence
  per document because the corpus is redistributable only on those terms. Never present
  anything as a source that is not one.
- **Private mode must never silently fall back to a non-private path.** See
  `docs/decisions/0004-app-wide-private-mode.md`. If something cannot be done privately,
  refuse and say so rather than doing it anyway.
- **The README states measured limits with numbers.** Do not soften them, and do not add a
  claim the evidence does not support. If a measurement changes, re-run it and update the
  number rather than editing the prose.

## Traps that have already cost time

- A test asserting a mock call count passed only because Vitest 3's `restoreMocks` cleared call
  history as a side effect. `clearMocks: true` is now set explicitly. Do not rely on undeclared
  defaults.
- Tests that slice a C or Python function by searching for `static void NAME(` can match a
  forward declaration instead of the definition, and then assert against a comment.
- An answer once degenerated into one repeated word. The model sets `repeat_penalty 1` in its own
  Modelfile, so the backend now sends explicit decoding options. This was never reproduced on
  demand, so it is mitigated, not fixed.

## Setting up on a fresh machine

`models/`, `data/index/` and `data/knowledge-base.jsonl` are **not committed**.

```bash
uv sync
uv run python scripts/download_models.py          # multilingual-e5-small
ollama pull qwen3:8b                              # the default answer model
uv run python -m ingest.build_kb                  # fetches arXiv + Wikipedia
uv run python -m ingest.load_jsonl data/knowledge-base.jsonl
```

Two things `download_models.py` cannot give you:

- **The trained LID specialist** (`models/lid-specialist-*.onnx` and its `.json`) was trained in
  this repository and is not published. Copy it from a machine that has it. Without it, stage two
  falls back to a script rule and `GET /api/health` reports `specialist: false`.
- **`models/glotlid-q.ftz`** (214 MB) is produced by compressing the 1.69 GB GlotLID model. The
  compression step needs `fasttext-wheel`, which may have no prebuilt wheel for Apple silicon and
  would then need Xcode command line tools. Copying the compressed file avoids that entirely.

Check `GET /api/health` before trusting any language result.
