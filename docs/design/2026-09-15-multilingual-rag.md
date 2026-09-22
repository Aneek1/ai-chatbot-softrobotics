# Multilingual soft-robotics assistant: design

Date: 2026-09-15
Status: approved design, amended the same day after dependency tests (training added, detector and platform decisions updated). Implementation not started.

## 1. Why

The current repo is a Tkinter chatbot that matches spaCy keywords against a 9-row CSV and sends the question to Gemini. It has four copies of the same script, a committed 400 MB virtual environment and unsourced dataset values.

The rebuild turns it into a working multilingual retrieval assistant with its own trained language-ID model. Its purpose is to show end-to-end work on text language identification: data curation, model design and training, evaluation broken down by language, and a deployable app. The subject stays soft-robotics fabrication.

## 2. Scope

In scope:
- Web app: FastAPI backend, React + TypeScript frontend.
- Language detection for 2,000+ languages, with 8 languages fully supported and evaluated.
- Training our own language-ID models for the hard cases (section 16).
- Retrieval over a licence-tracked knowledge base, with cited answers.
- Two answer models behind one interface: Gemini and Ollama.
- Private mode with an in-app egress allowlist.
- Evaluation scripts whose output feeds the README.
- Runs on Windows, macOS (Apple Silicon and Intel) and Linux, including WSL.
- Repo cleanup, tests, CI, Docker.

Out of scope:
- User accounts and authentication (single-user app).
- Training translation models, or fine-tuning embeddings for retrieval.
- Translating the interface text (it stays English; answers follow the user's language).
- Renaming the repo, voice input, mobile apps.

## 3. Decisions

| Topic | Decision |
|---|---|
| Stack | FastAPI (Python 3.12, managed with uv) + React/TypeScript/Vite/Tailwind |
| Platforms | Windows, macOS, Linux incl. WSL; CI runs on all three |
| Language ID | Two stages: compressed GlotLID for broad coverage, then a specialist for known confusion groups (section 5) |
| Detector runtime | `fasttext-predict` (prediction-only fastText; wheels for all three platforms) |
| Default detector file | GlotLID product-quantized to 225 MB (from 1.69 GB) |
| Simplified vs Traditional Chinese | OpenCC-based rule until the trained specialist replaces it |
| Supported languages | English, Chinese (Simplified and Traditional), Indonesian, Malay, Japanese, Korean, Tamil, Hindi (Devanagari and romanized) |
| Trained models | fastText sweep, classifier on frozen e5 embeddings, e5 fine-tune (section 16) |
| Embeddings | `intfloat/multilingual-e5-small` (384 dims) via fastembed (ONNX), loaded from a local folder |
| Vector store | Qdrant in local (embedded) mode |
| Answer models | Gemini (`google-genai` SDK), Ollama (local) |
| Streaming | FastAPI's built-in server-sent events (`fastapi.sse`) |
| Web search | Google Custom Search in normal mode, DuckDuckGo in private mode, snippets only |
| Private mode | App-wide switch; Ollama only; DuckDuckGo only; nothing written to history |
| Layout | Workspace: chat list, chat, inspector with Language / Sources / Privacy tabs |
| Hosted demo | Normal mode only (no Ollama on free hosting) |

## 4. Architecture

```
frontend (React)  ──HTTP + server-sent events──>  backend (FastAPI)
                                                    │
   question → normalize → detect language → retrieve (Qdrant) ─┐
                                        → web search (snippets) ┼→ answer model → citation check → stream
                                                                ┘
   egress guard: wraps all outbound sockets for the whole process
   history: SQLite (skipped in private mode)
```

### 4.1 Layout

```
backend/
  app/main.py            FastAPI app, startup checks, route registration
  app/config.py          settings from environment (pydantic-settings)
  app/api/chat.py        POST /api/chat (SSE stream)
  app/api/mode.py        GET/PUT /api/mode
  app/api/egress.py      GET /api/egress (SSE log stream)
  app/api/chats.py       chat history list/get/delete
  app/api/health.py      GET /api/health
  pipeline/normalize.py  NFKC, invisible-character removal, newline removal, script detection
  pipeline/langid.py     LanguageDetector interface, two-stage detector
  pipeline/han_script.py Simplified vs Traditional rule (OpenCC)
  pipeline/retrieve.py   embedding + Qdrant search
  pipeline/chunk.py      script-aware chunking
  pipeline/citations.py  citation validation
  providers/base.py      AnswerModel interface
  providers/gemini.py
  providers/ollama.py
  search/base.py         WebSearch interface
  search/google.py
  search/duckduckgo.py
  privacy/egress.py      socket-level allowlist + connection log
  privacy/mode.py        app-wide mode state and switching
  store/history.py       SQLite chat history
ingest/                  knowledge-base build scripts
training/                language-ID dataset building, training, sweeps, export (section 16)
eval/                    language-ID and RAG evaluation
scripts/download_models.py
scripts/compress_glotlid.py
```

### 4.2 API

- `POST /api/chat` with `{chat_id?, message, language_override?}` returns a server-sent event stream. Events, in order:
  - `language`: `{candidates: [{code, name, script, probability}], chosen, uncertain, stage}` (top 3 candidates; `stage` is `general`, `specialist` or `rule`)
  - `sources`: `[{id, title, snippet, language, source, url, licence}]`
  - `token`: `{text}` (repeated)
  - `citations`: `{valid: [n...], removed: [n...]}`
  - `notice`: `{kind, message}` (e.g. web search unavailable), any time
  - `done` or `error`: `{code, message}`
- `GET /api/mode` returns `{private, proxy_configured, ollama_reachable}`. `PUT /api/mode` with `{private: bool}` switches mode.
- `GET /api/egress` streams connection log entries: `{time, host, port, verdict, component}`.
- `GET /api/chats`, `GET /api/chats/{id}`, `DELETE /api/chats/{id}`.
- `GET /api/health` reports model files present, Ollama reachable, Gemini configured, search available, and index document counts per language.

## 5. Language identification

- **Normalization before detection:** Unicode NFKC, remove zero-width and other invisible characters, replace newlines with spaces (fastText rejects them), collapse whitespace. Script detection uses Unicode character properties, and the dominant script is recorded.
- **Stage 1 (general):** compressed GlotLID returns the top 5 labels (ISO 639-3 plus script, e.g. `ind_Latn`, `hin_Latn`, `cmn_Hani`).
- **Stage 2 (specialist):** if the stage-1 top label is in a confusion group, the specialist chooses within that group. Labels it returns from outside the group are ignored; if none remain, the result falls back as if there were no specialist:
  - Malay/Indonesian: `ind_Latn`, `zsm_Latn`
  - Han script: `cmn_Hani`, `yue_Hani`, `zho_Hans`, `zho_Hant`
  - Romanized Hindi: `hin_Latn`, `urd_Latn`, `eng_Latn`
- **Until the trained specialist exists** (or when it has no answer inside the group): Han-script text goes through the OpenCC rule, reported as stage `rule` (text unchanged by Traditional-to-Simplified conversion but changed by Simplified-to-Traditional is Simplified, the reverse is Traditional, otherwise ambiguous); the other groups keep the stage-1 result.
- **Mapping:** a table converts labels to display names and flags the 10 supported languages.
- **Uncertainty:** a result is `uncertain` if the top probability is below `LANGID_MIN_CONFIDENCE` (default 0.3), two different scripts each make up more than 30% of letters, or the Han rule is ambiguous. The default threshold is re-tuned from the evaluation's threshold sweep.
- **Answer language:**
  1. The user's manual override, if set.
  2. Otherwise the detected language, if not uncertain.
  3. Otherwise the language of the most recent confident message in the chat.
  4. Otherwise English.
- **Swappable:** `LanguageDetector.detect(text) -> Detection` is the only interface the rest of the app uses.

## 6. Knowledge base

### 6.1 Document record

`id, text, language, script, source (csv|arxiv|wikipedia), title, url, licence, retrieved_at, revision`

### 6.2 Sources

- **Curated methods (`data/methods.csv`):** about 50 fabrication methods. Each row cites a DOI or URL. A value with no source is dropped. Rows are drafted from real papers and every row is checked by a person before commit. The current 9 rows are not carried over unless their values can be sourced.
- **arXiv:** titles and abstracts from the arXiv API for soft-robotics queries. Licence: CC0 (arXiv metadata).
- **Wikipedia:** start from English soft-robotics articles, follow the links to the same articles in other languages, and store the revision ID. Licence: CC BY-SA 4.0.

### 6.3 Processing

- **Chunking:** Chinese and Japanese split on 。！？; other languages split on sentence boundaries. Target 300 tokens, 50-token overlap, measured with the embedding model's tokenizer.
- **Deduplication:** exact duplicates removed by hash of the normalized text.
- **Embeddings:** e5 prefixes (`query: ` for questions, `passage: ` for chunks). Scores are only used for ranking: an unrelated English sentence still scored 0.73 cosine against a related query in testing, so no fixed score threshold is used.
- **Retrieval:** top 6 chunks.
- **`DATASHEET.md`:** generated by `ingest/datasheet.py`. Counts per language, source and licence, including supported languages with zero documents.

## 7. Answer generation

- **Interface:** `AnswerModel.stream(messages) -> iterator[str]`.
- **Gemini:** enabled only when both `GOOGLE_API_KEY` and `GEMINI_MODEL` are set. There is no default model name; health reports "Gemini not configured" otherwise.
- **Ollama:** model from `OLLAMA_MODEL` (default `qwen3:8b`) at `OLLAMA_URL` (default `http://localhost:11434`). Requests set `think: false` (with thinking on, the model can spend its whole token budget on hidden reasoning and return an empty answer). `OLLAMA_TIMEOUT` defaults to 120 s because a cold model load took 45 s in testing; the health check warms the model.
- **Prompt:** answer in the chosen language, use only the numbered sources and web snippets, cite as `[n]`, and say so when the sources don't cover the question.
- **Citation check:** after generation, any `[n]` that doesn't match a provided source is removed and reported in the `citations` event.
- **No sources:** if retrieval and web search both return nothing, the answer is labelled "No sources found".
- **Normal mode:** the answer model is user-selectable (Gemini or Ollama). Private mode forces Ollama.

## 8. Private mode

### 8.1 Rule

A private request that cannot be served privately fails. It never falls back to a non-private path.

### 8.2 Switching

- App-wide, held in `privacy/mode.py`.
- Turning private mode on waits for in-flight answers to finish, then applies.
- If Ollama is unreachable, the switch still happens, but chat requests return `error` with code `ollama_unavailable`. Gemini is never used.

### 8.3 Egress guard

- Installed at process start. It wraps `socket.getaddrinfo` and `socket.socket.connect`, so every library (Gemini SDK, search library, HTTP clients) passes through it.
- Allowlists:
  - Always: `localhost`, `127.0.0.1`, `::1`.
  - Normal: `generativelanguage.googleapis.com`, `www.googleapis.com`.
  - Private: `html.duckduckgo.com` (the only host the pinned `ddgs` DuckDuckGo engine uses). If `PRIVATE_PROXY` is set, only the proxy host.
- A blocked lookup raises before any DNS query is sent.
- Every attempt is logged (`time, host, port, verdict, component`) into a 500-entry ring buffer streamed by `/api/egress`.

### 8.4 Other ways data could leak

- **Search:** snippets and links only; result pages are never fetched. The search library is pinned to its DuckDuckGo engine.
- **Proxy:** `PRIVATE_PROXY=socks5h://host:port` supported (Tor works); DNS resolution goes through the proxy.
- **Models:** downloaded only by `scripts/download_models.py` or the Docker build, into a flat `models/` folder (Hugging Face `local_dir`, no cache symlinks). At runtime `HF_HUB_OFFLINE=1` is always set, and models load from local paths. Loading e5 this way was verified with networking blocked.
- **Frontend:** bundles all fonts and scripts; no CDNs, no analytics.
- **History and logs:** private chats live in memory only and are cleared on switching back or restart. Question text is left out of server logs in private mode.

### 8.5 Stated limits (README)

- An in-app guard, not an operating-system firewall; it doesn't cover the browser or other programs.
- Without a proxy, DuckDuckGo sees the query and IP address.
- The README shows how to run the container with network restrictions for a stronger guarantee.

## 9. Frontend

- **Layout:** left, chat list and new chat; centre, the chat; right, an inspector with Language, Sources and Privacy tabs showing details for the selected message. Below 900 px width the inspector becomes a bottom sheet.
- **Language tab:** top 3 candidates with probability bars, script, which stage decided, an uncertain state, and an override menu.
- **Sources tab:** cited sources with language, source, licence and link.
- **Privacy tab:** mode switch, proxy status, live connection log.
- **Private mode:** dark header with a lock badge; the model selector is fixed to Ollama.
- **Themes:** light and dark.
- **Visual style:** restrained and specific to this app, not a default component-library or gradient look.

## 10. Error handling

| Situation | Behaviour |
|---|---|
| Uncertain language | `uncertain: true`, top 2 shown, answer-language rule from §5 |
| Retrieval returns nothing | Continue with web snippets; if also none, label "No sources found" |
| Web search fails | `notice` event; answer continues without web results |
| Answer model error or timeout | `error` event with `provider_error` or `provider_timeout`; UI shows retry |
| Ollama unreachable in private mode | `error` with `ollama_unavailable`; no fallback |
| Blocked connection | Logged; the calling component gets an `EgressBlocked` exception mapped to a `notice` or `error` |
| Gemini key or model not set | Gemini hidden from the model selector; health reports it |
| Missing Google search key | Normal-mode web search off, with a visible notice |
| Missing model files | Startup fails with a message naming `scripts/download_models.py` |

## 11. Evaluation

Results go to `results/<name>-<date>.json` with the date, hardware, platform, commit hash and package versions. `eval/readme_table.py` generates the README tables from these files.

### 11.1 Language ID (`eval/langid_eval.py`)

- **Data:** FLORES+ devtest for the 8 languages (requires a Hugging Face login and accepting the dataset terms); the Dakshina test split for romanized Hindi; Wikipedia language labels from the knowledge base.
- **Slices:** length (1-3 words, one sentence, paragraph), confusion groups (ind/zsm, Simplified/Traditional, hin_Latn/urd_Latn/eng_Latn), code-mixed text.
- **Metrics:** per-language precision, recall and F1; macro-F1; confusion matrix; p50/p95 latency; model file size and RAM.
- **Detectors compared:** full GlotLID, compressed GlotLID, fastText `lid.176`, and the trained models from section 16, alone and as the stage-2 specialist.
- **Compression cost:** accuracy change per language between full and compressed GlotLID.
- **Threshold sweep:** accuracy and coverage at confidence thresholds 0.3-0.9, used to set `LANGID_MIN_CONFIDENCE`.

### 11.2 RAG (`eval/rag_eval.py`)

- **Questions:** `eval/questions.jsonl`, 20 per supported language (160). Each records the expected source ids and `verified_by_native_speaker: true|false`. English is written by hand; other languages are machine-translated, then checked where a native speaker is available.
- **Metrics, per language and per answer model:** retrieval hit@5, answer-language accuracy (detector run on the answer), citation validity.

### 11.3 Error analysis

`docs/error-analysis.md`: real misclassified or mis-answered examples, each with a short explanation.

## 12. Testing and CI

- **Backend (pytest):**
  - normalize, chunk, citation validation, Han script rule
  - language mapping, confusion-group routing and the uncertainty rule
  - egress guard: blocked hosts, allowed hosts, DNS not sent for blocked names, proxy-only mode
  - private mode: no Gemini, no fallback, no history writes
  - search library pinned to DuckDuckGo
  - API with a fake answer model
  - training utilities on tiny fixtures (split logic, run logging, metrics)
- **Tests needing real models or GPUs** are marked `slow` and excluded from CI.
- **Frontend:** Vitest for components; one Playwright test that asks a question and sees a language and sources.
- **Build check:** a script fails if the built frontend contains external URLs.
- **GitHub Actions:** ruff, eslint, backend tests on ubuntu, windows and macos runners, frontend tests, build, external-URL check.
- **Scripts** set `PYTHONUTF8=1`; Windows consoles otherwise crash printing Tamil or Chinese.

## 13. Repo cleanup and documentation

- **Remove:** `main.py`, `gui_chatbot.py`, `oop_gui_chatbot.py`, `newcodev1.py`, `setup.sh`, the old `requirements.txt`, spaCy.
- **Venv:** untrack `spacy_env/`. Purging it from history needs `git filter-repo` and a force push, done only with explicit approval, after telling collaborator Kavya9878.
- **Dependency pins:** `uv.lock` (backend), `package-lock.json` (frontend).
- **Line endings:** `.gitattributes` normalizes text files to LF.
- **Licences:** code under MIT; data licences in `DATASHEET.md`; model licences (GlotLID Apache-2.0, e5 MIT) in `MODELS.md`.
- **`README.md`:** what it does, a screenshot or GIF of the real app, running it in three commands on each platform, generated results tables, limitations, Windows notes (long paths).
- **`docs/decisions/`:** short records for GlotLID compression, the two-stage detector, snippets-only search, app-wide private mode, e5-small over larger embedding models.
- **Docker:** one image with the API and built frontend; `docker-compose.yml` adds Ollama. Multi-architecture (amd64 and arm64).

## 14. Writing standards

- No emoji headings, marketing adjectives, badge walls or comments that restate the code.
- Every number in documentation comes from a committed results file.
- Small commits, each doing one thing.
- Limitations stated plainly.

## 15. Risks

- **Google Custom Search availability:** Google has closed the Custom Search JSON API to new customers and announced it is being wound down. Availability is checked at implementation; if it's unusable, normal mode also uses DuckDuckGo through the same `WebSearch` interface.
- **fastText builds:** the official `fastText` package has no wheels. The app uses `fasttext-predict`. Compressing GlotLID needs the full build (`fasttext-wheel`, which has no macOS arm64 wheel for Python 3.12), so `scripts/compress_glotlid.py` runs on Windows, Linux or in Docker, and macOS users download the compressed file.
- **Windows file system:** long paths are off by default and Hugging Face cache symlinks need Developer Mode. Models go into a short flat `models/` folder; the README explains long-path settings.
- **Training data licences:** OpenLID lists its licence as "other" (it varies by source). The repo publishes scripts, never the data; trained model licence notes cite the sources.
- **Synthetic training data:** OpenCC conversions don't read like natively written Traditional Chinese, and short-text crops aren't real queries. Synthetic rows are tagged, and results are reported with and without them.
- **Evaluation access:** FLORES+ is gated; without a Hugging Face login the language-ID evaluation falls back to Dakshina and Wikipedia labels, and the results file records which sets were used.
- **Thin non-English content:** Wikipedia coverage in Tamil and Malay may be close to zero. That is reported in the datasheet and evaluated as cross-language retrieval, not hidden.
- **Machine-translated evaluation questions** may flatter or penalise some languages; the verified flag lets results be split by verification status.
- **Small local models** may ignore the answer-language instruction; answer-language accuracy is measured per model so this shows up in results.

## 16. Training our own language-ID model

### 16.1 Goal

A specialist that beats compressed GlotLID on the confusion groups and short text, while staying small and fast, measured on data it never saw.

### 16.2 Labels

`eng_Latn, ind_Latn, zsm_Latn, zho_Hans, zho_Hant, jpn_Jpan, kor_Hang, tam_Taml, hin_Deva, hin_Latn, urd_Latn`, plus `other`, trained on a sample of the remaining OpenLID languages so the model can decline.

### 16.3 Data (`training/data/`)

- **OpenLID** (`laurievb/open-lid-dataset`, 118M lines, 201 labels): up to 100,000 lines per label, streamed per shard rather than downloading all 16.6 GB.
- **Dakshina** (CC BY-SA 4.0): romanized Hindi sentences for `hin_Latn`.
- **Augmentation, tagged `synthetic`:**
  - OpenCC conversions to give matched Simplified/Traditional pairs.
  - Short crops of 1-3 words.
  - Code-mixed lines built by joining sentences from two languages.
- **Cleaning:** the same normalization as the app (section 5), exact deduplication, lines under 2 characters dropped.
- **Splits:** train/validation/test are separated by OpenLID's `dataset_source` field, so sources seen in training never appear in test. A deduplication pass removes any test line that also appears in training.
- **Held out entirely:** FLORES+ devtest and the Dakshina test split are never used for training or model selection.

### 16.4 Models

1. **fastText supervised** (`training/fasttext_train.py`): character n-grams. Sweep over epochs {5, 10, 25}, learning rate {0.1, 0.5, 1.0}, dimension {16, 64}, n-gram range {1-4, 2-5}. Runs on CPU. Needs the full fastText build, so it runs on Windows, Linux, WSL or in Docker.
2. **Classifier on frozen e5 embeddings** (`training/embed_head.py`): embeddings computed once and cached; a 1-hidden-layer network in PyTorch. Sweep over hidden size {128, 512}, learning rate {1e-3, 3e-4}, dropout {0.1, 0.3}; up to 30 epochs with early stopping when validation macro-F1 hasn't improved for 3 epochs.
3. **e5-small fine-tune** (`training/finetune_e5.py`): encoder plus classification head trained end to end for up to 4 epochs, sequence length 128, mixed precision, early stopping on validation macro-F1.

GPU training uses PyTorch CUDA 12.8+ builds (the RTX 5060 has compute capability 12.0). On macOS, models 2 and 3 run on MPS or CPU, slower; nothing in CI trains.

### 16.5 Run tracking

- Each run writes `runs/<run_id>/config.json`, `metrics.jsonl` (one line per epoch: train loss, validation loss, validation macro-F1, validation F1 per confusion group), and `final.json`.
- `training/sweep.py` runs a grid and writes `runs/sweep-<name>.csv`.
- `training/plot_runs.py` draws training curves from `metrics.jsonl`.
- `runs/` is git-ignored; the chosen runs' summaries are copied into `results/`.

### 16.6 Selection and export

- The model is chosen on validation macro-F1 over the confusion groups; the test split and held-out sets are evaluated once, after selection.
- `training/export.py` saves the fastText winner as a quantized `.ftz`, and the neural winners as ONNX, so the app keeps using `fasttext-predict` and onnxruntime with no PyTorch at runtime.
- The exported model becomes the stage-2 specialist in `pipeline/langid.py`.
