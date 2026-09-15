# Multilingual soft-robotics assistant: design

Date: 2026-09-15
Status: approved design, implementation not started

## 1. Why

The current repo is a Tkinter chatbot that matches spaCy keywords against a 9-row CSV and sends the question to Gemini. It has four copies of the same script, a committed 400 MB virtual environment and unsourced dataset values.

The rebuild turns it into a working multilingual retrieval assistant. Its purpose is to show end-to-end work on text language identification: data curation, model choice, evaluation broken down by language, and a deployable app. The subject stays soft-robotics fabrication.

## 2. Scope

In scope:
- Web app: FastAPI backend, React + TypeScript frontend.
- Language detection for 100+ languages, with 8 languages fully supported and evaluated.
- Retrieval over a licence-tracked knowledge base, with cited answers.
- Two answer models behind one interface: Gemini and Ollama.
- Private mode with an in-app egress allowlist.
- Evaluation scripts whose output feeds the README.
- Repo cleanup, tests, CI, Docker.

Out of scope:
- User accounts and authentication (single-user app).
- Training a language-ID model. That is a separate repo; this app's detector interface is where such a model would plug in later.
- Translating the interface text (it stays English; answers follow the user's language).
- Renaming the repo, voice input, mobile apps.

## 3. Decisions

| Topic | Decision |
|---|---|
| Stack | FastAPI (Python 3.11) + React/TypeScript/Vite/Tailwind |
| Language ID | GlotLID (fastText) behind a `LanguageDetector` interface |
| Supported languages | English, Chinese (Simplified and Traditional), Indonesian, Malay, Japanese, Korean, Tamil, Hindi (Devanagari and romanized) |
| Embeddings | `intfloat/multilingual-e5-small` (384 dims, CPU) |
| Vector store | Qdrant in local (embedded) mode |
| Answer models | Gemini (`google-genai` SDK), Ollama (local) |
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

### 4.1 Backend layout

```
backend/
  app/main.py            FastAPI app, startup checks, route registration
  app/config.py          settings from environment (pydantic-settings)
  app/api/chat.py        POST /api/chat (SSE stream)
  app/api/mode.py        GET/PUT /api/mode
  app/api/egress.py      GET /api/egress (SSE log stream)
  app/api/chats.py       chat history list/get/delete
  app/api/health.py      GET /api/health
  pipeline/normalize.py  NFKC, invisible-character removal, script detection
  pipeline/langid.py     LanguageDetector interface + GlotLID implementation
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
eval/                    language-ID and RAG evaluation
scripts/download_models.py
```

### 4.2 API

- `POST /api/chat` with `{chat_id?, message, language_override?}` returns a server-sent event stream. Events, in order:
  - `language`: `{candidates: [{code, name, script, probability}], chosen, uncertain}` (top 3 candidates)
  - `sources`: `[{id, title, snippet, language, source, url, licence}]`
  - `token`: `{text}` (repeated)
  - `citations`: `{valid: [n...], removed: [n...]}`
  - `notice`: `{kind, message}` (e.g. web search unavailable), any time
  - `done` or `error`: `{code, message}`
- `GET /api/mode` returns `{private, proxy_configured, ollama_reachable}`. `PUT /api/mode` with `{private: bool}` switches mode.
- `GET /api/egress` streams connection log entries: `{time, host, port, verdict, component}`.
- `GET /api/chats`, `GET /api/chats/{id}`, `DELETE /api/chats/{id}`.
- `GET /api/health` reports model files present, Ollama reachable, Gemini key present, search available, and index document counts per language.

## 5. Language identification

- **Normalization before detection:** Unicode NFKC, remove zero-width and other invisible characters, collapse whitespace. Script detection uses Unicode character properties, and the dominant script is recorded.
- **Detector:** GlotLID returns labels as ISO 639-3 plus script (e.g. `ind_Latn`, `hin_Latn`). A mapping table converts labels to display names and flags the 8 supported languages. Exact label strings are checked against the model file during implementation.
- **Uncertainty:** a result is `uncertain` if the top probability is below `LANGID_MIN_CONFIDENCE` (default 0.60) or two different scripts each make up more than 30% of letters. The default threshold is re-tuned from the evaluation's threshold sweep.
- **Answer language:**
  1. The user's manual override, if set.
  2. Otherwise the detected language, if not uncertain.
  3. Otherwise the language of the most recent confident message in the chat.
  4. Otherwise English.
- **Swappable:** `LanguageDetector.detect(text) -> list[Candidate]` is the only interface the rest of the app uses.

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
- **Embeddings:** e5 prefixes (`query: ` for questions, `passage: ` for chunks).
- **Retrieval:** top 6 chunks.
- **`DATASHEET.md`:** generated by `ingest/datasheet.py`. Counts per language, source and licence, including supported languages with zero documents.

## 7. Answer generation

- **Interface:** `AnswerModel.stream(messages) -> iterator[str]`.
- **Gemini:** model from `GEMINI_MODEL`; the default is a current Flash model, confirmed against the API's model list during implementation.
- **Ollama:** model from `OLLAMA_MODEL`, default `qwen3:8b`, at `OLLAMA_URL` (default `http://localhost:11434`).
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
  - Private: DuckDuckGo hosts used by the pinned `ddgs` version (listed in code, confirmed during implementation). If `PRIVATE_PROXY` is set, only the proxy host.
- A blocked lookup raises before any DNS query is sent.
- Every attempt is logged (`time, host, port, verdict, component`) into a 500-entry ring buffer streamed by `/api/egress`.

### 8.4 Other ways data could leak

- **Search:** snippets and links only; result pages are never fetched. The search library is pinned to the DuckDuckGo backend.
- **Proxy:** `PRIVATE_PROXY=socks5h://host:port` supported (Tor works); DNS resolution goes through the proxy.
- **Models:** downloaded only by `scripts/download_models.py` or the Docker build. At runtime `HF_HUB_OFFLINE=1` is always set.
- **Frontend:** bundles all fonts and scripts; no CDNs, no analytics.
- **History and logs:** private chats live in memory only and are cleared on switching back or restart. Question text is left out of server logs in private mode.

### 8.5 Stated limits (README)

- An in-app guard, not an operating-system firewall; it doesn't cover the browser or other programs.
- Without a proxy, DuckDuckGo sees the query and IP address.
- The README shows how to run the container with network restrictions for a stronger guarantee.

## 9. Frontend

- **Layout:** left, chat list and new chat; centre, the chat; right, an inspector with Language, Sources and Privacy tabs showing details for the selected message. Below 900 px width the inspector becomes a bottom sheet.
- **Language tab:** top 3 candidates with probability bars, script, an uncertain state, and an override menu.
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
| Missing Gemini key at startup | Gemini hidden from the model selector; health reports it |
| Missing Google search key | Normal-mode web search off, with a visible notice |
| Missing model files | Startup fails with a message naming `scripts/download_models.py` |

## 11. Evaluation

Results go to `results/<name>-<date>.json` with the date, hardware, commit hash and package versions. `eval/readme_table.py` generates the README tables from these files.

### 11.1 Language ID (`eval/langid_eval.py`)

- **Data:** FLORES-200 devtest for the 8 languages (including `zho_Hant`, `zsm_Latn`); romanized Hindi from the Dakshina dataset.
- **Slices:** length (1-3 words, one sentence, paragraph), lookalike pairs (ind/zsm, zho_Hans/zho_Hant, hin_Latn/eng_Latn), a code-mixed set built from real mixed-language text in the knowledge base.
- **Metrics:** per-language precision, recall and F1; macro-F1; confusion matrix; p50/p95 latency; model size.
- **Baselines:** GlotLID vs. fastText `lid.176` vs. lingua (marked "not supported" where a language isn't covered).
- **Threshold sweep:** accuracy and coverage at confidence thresholds 0.3-0.9, used to set `LANGID_MIN_CONFIDENCE`.

### 11.2 RAG (`eval/rag_eval.py`)

- **Questions:** `eval/questions.jsonl`, 20 per supported language (160). Each records the expected source ids and `verified_by_native_speaker: true|false`. English is written by hand; other languages are machine-translated, then checked where a native speaker is available.
- **Metrics, per language and per answer model:** retrieval hit@5, answer-language accuracy (detector run on the answer), citation validity.

### 11.3 Error analysis

`docs/error-analysis.md`: real misclassified or mis-answered examples, each with a short explanation.

## 12. Testing and CI

- **Backend (pytest):**
  - normalize, chunk, citation validation
  - language mapping and the uncertainty rule
  - egress guard: blocked hosts, allowed hosts, DNS not sent for blocked names, proxy-only mode
  - private mode: no Gemini, no fallback, no history writes
  - search library pinned to DuckDuckGo
  - API with a fake answer model
- **Tests needing real models** are marked `slow` and excluded from CI.
- **Frontend:** Vitest for components; one Playwright test that asks a question and sees a language and sources.
- **Build check:** a script fails if the built frontend contains external URLs.
- **GitHub Actions:** ruff, eslint, backend tests, frontend tests, build, external-URL check.

## 13. Repo cleanup and documentation

- **Remove:** `main.py`, `gui_chatbot.py`, `oop_gui_chatbot.py`, `newcodev1.py`, `setup.sh`, the old `requirements.txt`, spaCy.
- **Venv:** untrack `spacy_env/`. Purging it from history needs `git filter-repo` and a force push, done only with explicit approval.
- **Dependency pins:** `uv.lock` (backend), `package-lock.json` (frontend).
- **Licences:** code under MIT; data licences in `DATASHEET.md`.
- **`README.md`:** what it does, a screenshot or GIF of the real app, running it in three commands, generated results tables, limitations.
- **`docs/decisions/`:** short records for GlotLID, snippets-only search, app-wide private mode, e5-small over larger embedding models.
- **Docker:** one image with the API and built frontend; `docker-compose.yml` adds Ollama.

## 14. Writing standards

- No emoji headings, marketing adjectives, badge walls or comments that restate the code.
- Every number in documentation comes from a committed results file.
- Small commits, each doing one thing.
- Limitations stated plainly.

## 15. Risks

- **Google Custom Search availability:** Google has closed the Custom Search JSON API to new customers and announced it is being wound down. Availability is checked at implementation; if it's unusable, normal mode also uses DuckDuckGo through the same `WebSearch` interface.
- **fastText on Windows / newer Python:** wheels are unreliable, so development and Docker use Python 3.11 via uv.
- **Thin non-English content:** Wikipedia coverage in Tamil and Malay may be close to zero. That is reported in the datasheet and evaluated as cross-language retrieval, not hidden.
- **Machine-translated evaluation questions** may flatter or penalise some languages; the verified flag lets results be split by verification status.
- **Small local models** may ignore the answer-language instruction; answer-language accuracy is measured per model so this shows up in results.
