# Soft-robotics assistant

A retrieval assistant for soft-robotics fabrication that works in the language you write in. It
identifies the language of a question, retrieves passages from a licence-tracked knowledge base,
answers with a local or a hosted model, and cites the passages it used. Language identification is
the part of the project with the most work in it: a compressed GlotLID model covers two thousand
languages, and a specialist trained in this repository decides the cases GlotLID confuses.

**Measured, not asserted.** GlotLID is product-quantized from **1.69 GB to 225 MB** at no accuracy
cost — **0.957 macro-F1 on FLORES-200** over 195 languages and 206,448 sentences, against the
published model's 0.917. A specialist trained in this repository then decides the pairs GlotLID
confuses: Malay against Indonesian, Simplified against Traditional Chinese, and romanized Hindi
against Urdu and English. Its **0.8 MB** variant reaches 0.881 group macro-F1 where the 470 MB one
reaches 0.913. How that was built, and where it is weakest, is in
[docs/language-identification.md](docs/language-identification.md).

Retrieval quality, answer-language accuracy and citation faithfulness are measured over 200
questions in ten languages, and the limits are stated with numbers — including the two supported
languages that have no documents at all.

The design document is [docs/design/2026-09-15-multilingual-rag.md](docs/design/2026-09-15-multilingual-rag.md);
the decisions behind it are in [docs/decisions/](docs/decisions/).

## What it does

- Detects the language of a question in two stages, reports the top candidates with their
  probabilities, and says which stage decided.
- Fully supports English, Chinese (Simplified and Traditional), Indonesian, Malay, Japanese, Korean,
  Tamil and Hindi, in Devanagari and romanized.
- Retrieves from a local index of arXiv abstracts and Wikipedia lead sections, each document
  carrying its source, link, licence and revision.
- Answers with Ollama on your own machine, or with Gemini when you configure a key, and removes any
  citation that does not point at a source it was given.
- Has a private mode: local model only, DuckDuckGo only, nothing written to disk, with an in-app
  guard that refuses any other outbound connection.

## Interface

The interface is a React workspace in `frontend/`: the chat list on the left, the chat in the
middle, and an inspector on the right with three tabs, navigated with a roving tabindex and the
arrow keys, per the WAI-ARIA tabs pattern.

- **Language** shows the top candidates the detector returned with their probabilities, each one's
  raw label (`ind_Latn`, `zho_Hant`) alongside its name, and which stage decided. The answer
  language can be overridden from the same tab.
- **Sources** lists the documents the answer was built from, each with its language, where it came
  from, its licence and a link to the original, and marks the ones the answer cited.
- **Privacy** holds the private-mode switch, says whether a proxy is configured, and streams the
  connection log from `GET /api/egress`.

Both themes, light and dark, are remembered in `localStorage` and seeded from
`prefers-color-scheme` on first load. The frontend has its own suite of 129 tests, covering these
components and the state around them.

What it does not do: chat deletion is a single click with no confirmation and no undo; the
connection log tells you when it could not be loaded rather than rendering an empty list as if
nothing had happened, but that also means the log you see can be incomplete; it has no accounts and
no server-side session, so anyone who can open the page can use the backend; it never translates its
own labels, which stay in English while answers follow the question's language; and it shows only
what the backend reports, so an empty Sources tab means retrieval and web search both returned
nothing, not that the answer was invented. The suite above runs against a mocked API; nothing here
has yet been exercised against a running backend end to end.

Run it in development with the backend on port 8000 and the dev server proxying to it:

```bash
PYTHONUTF8=1 uv run uvicorn backend.app.main:app --port 8000
cd frontend && npm ci && npm run dev
```

The dev server prints a `http://localhost:5173` address. For a production build, `npm run build`
writes `frontend/dist/`, which the API serves at `/` when it is present, and `npm run check:bundle`
fails if anything in that build points at a host other than this origin: the app bundles its own
scripts and styles, uses the system font stack, and loads nothing from a CDN.

## Running it

You need Python 3.12, [uv](https://docs.astral.sh/uv/), Node 24 (only for the frontend) and, for
answers, [Ollama](https://ollama.com/).

```bash
uv sync
uv run python scripts/download_models.py --glotlid
uv run uvicorn backend.app.main:app --port 8000
```

The first command installs the locked dependencies, the second downloads multilingual-e5-small and
the 1.69 GB GlotLID model into `models/`, and the third starts the API on
`http://127.0.0.1:8000`. `GET /api/health` reports which model files it found.

GlotLID is compressed to 225 MB before the app uses it, in an environment of its own, because the
full fastText build and the prediction-only one share a module name:

```bash
uv run --no-project --python 3.12 --with fasttext-wheel==0.9.2 --with numpy==1.26.4 \
    python scripts/compress_glotlid.py models/glotlid/model.bin models/glotlid-q.ftz
```

Then build the knowledge base and index it:

```bash
PYTHONUTF8=1 uv run python -m ingest.build_kb
PYTHONUTF8=1 uv run python -m ingest.load_jsonl data/knowledge-base.jsonl
PYTHONUTF8=1 uv run python -m ingest.datasheet
```

The first writes `data/knowledge-base.jsonl` (not committed), the second indexes it into
`data/index/`, and the third regenerates [DATASHEET.md](DATASHEET.md). These three scripts reach the
network themselves, outside the backend process, so the egress guard does not apply to them: they
connect to `export.arxiv.org` and the `*.wikipedia.org` API hosts, and nothing else.

Ask a question:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Apa itu pencetakan silikon?", "model": "ollama"}'
```

The answer arrives as server-sent events: the detected language, the sources, the answer tokens, the
citation check, then `done`.

### Windows

Set `PYTHONUTF8=1` before any command that prints Tamil, Chinese or Devanagari, or the console
raises a `UnicodeEncodeError`:

```powershell
$env:PYTHONUTF8 = "1"
```

Turn on long paths (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` = 1, or
Group Policy) before downloading models. Model files are written into a flat `models/` folder with
no cache symlinks, which is why Developer Mode is not needed.

### macOS

Everything runs on Apple Silicon and Intel except the compression step: `fasttext-wheel` has no
macOS arm64 wheel for Python 3.12. Compress `glotlid-q.ftz` on Windows, on Linux, or in the
container, and copy the 225 MB file into `models/`. The app itself uses `fasttext-predict`, which
does have a macOS arm64 wheel.

### Linux and WSL

No extra steps. Under WSL, keep the repository inside the Linux file system if you can: model files
load noticeably faster than across `/mnt/c`.

### The frontend

```bash
cd frontend
npm ci
npm run dev
```

The dev server runs on port 5173 and proxies `/api` to the backend, so the browser stays on one
origin. `npm run build` writes `frontend/dist`, which the API serves at `/` when it is present.

## Private mode

Private mode is an app-wide switch (`PUT /api/mode`, or `PRIVATE_MODE=true` at startup). While it
is on:

- Answers come from Ollama on your own machine. Gemini is never called, and a request that cannot be
  answered privately fails instead of falling back.
- Web search goes to DuckDuckGo, and only the search page itself is fetched: the assistant reads
  titles, links and snippets, and never opens a result page.
- Chats are kept in memory. Nothing is written to `data/history.db`, and the private chats are
  cleared when you switch back or restart.
- Question text is kept out of the server log.

An egress guard runs inside the backend process. It wraps name resolution and socket connections and
checks both against a list of allowed hosts: loopback and your Ollama host at all times,
`generativelanguage.googleapis.com` and `www.googleapis.com` in normal mode, and
`html.duckduckgo.com` (or your proxy) in private mode. A blocked name is refused before any DNS
query is sent, and every attempt is logged. `GET /api/egress` streams that log.

Set `PRIVATE_PROXY=socks5h://host:port` to send private searches through a SOCKS5 proxy, such as
Tor. With `socks5h` the proxy resolves the host name, so the search host is never looked up from
your machine, and the proxy becomes the only outside host the guard allows.

### What private mode does not do

- The guard covers this application only. It is not a firewall: your browser and every other program
  on the machine are unaffected.
- Without a proxy, DuckDuckGo still sees your query and your IP address.
- The Ollama host in `OLLAMA_URL` is allowed in both modes. Point it at another machine and your
  private questions go to that machine.
- The guard works by wrapping Python's socket functions, so it covers libraries that use them. A
  library with its own network stack in C or Rust would not be covered, which is why the search
  client here is plain `httpx`.
- Stronger isolation has to come from outside the app. The Docker section below shows how.

## Docker

One image holds the API and the built frontend. Models and the index stay outside it:

```bash
docker build -t softrobotics-assistant .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/models:ro" -v softrobotics-data:/data \
  softrobotics-assistant
```

`docker compose up` starts the same image with an Ollama container next to it and points
`OLLAMA_URL` at it.

For a stronger version of the private-mode guarantee, give the container a network that cannot leave
the host:

```bash
docker network create --internal softrobotics-private
docker run --rm --network softrobotics-private -p 8000:8000 \
  -e PRIVATE_MODE=true -e WEB_SEARCH=off \
  -v "$PWD/models:/models:ro" -v softrobotics-data:/data \
  softrobotics-assistant
```

On that network the app can reach an Ollama container and nothing else, so the guarantee no longer
depends on the app's own guard. Web search is off, because DuckDuckGo is unreachable.

The image is built for `linux/amd64` and `linux/arm64` in CI, each on a runner of that architecture.
Only the amd64 image has been run by hand.

## Language identification

Stage one is GlotLID, product-quantized from 1.69 GB to 225 MB, returning the top five labels over
about two thousand languages. Stage two is a specialist, trained in this repository, that decides
inside the groups GlotLID confuses: Malay against Indonesian, Simplified against Traditional
Chinese, and romanized Hindi against Urdu and English. Until a specialist file is configured,
Han-script text falls back to an OpenCC conversion rule, reported as stage `rule`.

The training data, the sweeps and the selection are in [training/](training/): how the data was
built in [training/data/DATACARD.md](training/data/DATACARD.md), what came out in
[training/RESULTS.md](training/RESULTS.md).

## Results

These tables are generated by `eval/readme_table.py` from the files in `results/`. Nothing here is
typed by hand: to change a number, re-run the evaluation and the generator.

```bash
PYTHONUTF8=1 uv run --group train python -m eval.langid_eval
PYTHONUTF8=1 OLLAMA_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M uv run python -m eval.rag_eval
PYTHONUTF8=1 uv run python -m eval.readme_table
```

`training/RESULTS.md` has the language-ID work in full, including the FLORES-200 comparison with the
published GlotLID and OpenLID figures, and `docs/error-analysis.md` shows real mistakes.

<!-- results:start (generated by eval/readme_table.py; do not edit by hand) -->

### Language identification

Source: `results/langid-eval-2026-09-21.json`, detector `two-stage+specialist-e5-finetune`, 696.2 MB of model files, 2.48 ms per question at the median and 77.74 ms at the 95th percentile. F1 per language:

| Language | Label | FLORES sentence | Taiwan vocabulary | Dakshina sentence |
|---|---|---:|---:|---:|
| English | `eng_Latn` | 0.998 | n/a | n/a |
| Indonesian | `ind_Latn` | 0.931 | n/a | n/a |
| Malay | `zsm_Latn` | 0.934 | n/a | n/a |
| Chinese (Simplified) | `zho_Hans` | 0.989 | 0.989 | n/a |
| Chinese (Traditional) | `zho_Hant` | 0.951 | 0.996 | n/a |
| Japanese | `jpn_Jpan` | 0.997 | n/a | n/a |
| Korean | `kor_Hang` | 1.000 | n/a | n/a |
| Tamil | `tam_Taml` | 1.000 | n/a | n/a |
| Hindi | `hin_Deva` | 0.981 | n/a | n/a |
| Hindi (romanized) | `hin_Latn` | n/a | n/a | 0.912 |

Taiwan vocabulary is FLORES Simplified text converted with OpenCC `s2twp`, not text written in Taiwan. FLORES-200 has no romanized Hindi or Urdu, so those are scored on the Dakshina test files.

Where the confusion groups go wrong:

malay_indonesian:

| Gold, predicted right | `ind_Latn` | `zsm_Latn` | (outside group) |
|---|---:|---:|---:|
| `ind_Latn` | 942 | 70 | 0 |
| `zsm_Latn` | 64 | 948 | 0 |

han:

| Gold, predicted right | `zho_Hans` | `zho_Hant` | (outside group) |
|---|---:|---:|---:|
| `zho_Hans` | 989 | 0 | 23 |
| `zho_Hant` | 0 | 1006 | 6 |

romanized_hindi:

| Gold, predicted right | `eng_Latn` | `hin_Latn` | `urd_Latn` | (outside group) |
|---|---:|---:|---:|---:|
| `eng_Latn` | 1011 | 1 | 0 | 0 |
| `hin_Latn` | 0 | 0 | 0 | 0 |
| `urd_Latn` | 0 | 0 | 0 | 0 |

Accuracy and coverage at each confidence threshold, which is how `LANGID_MIN_CONFIDENCE` was set:

| Confidence threshold | Coverage | Accuracy |
|---:|---:|---:|
| 0.3 | 0.994 | 0.961 |
| 0.4 | 0.989 | 0.964 |
| 0.5 | 0.982 | 0.967 |
| 0.6 | 0.967 | 0.973 |
| 0.7 | 0.953 | 0.978 |
| 0.8 | 0.935 | 0.982 |
| 0.9 | 0.911 | 0.986 |

### Retrieval and answers

Source: `results/rag-eval-2026-09-21.json`. Answers from `qwen3:30b-a3b-instruct-2507-q4_K_M` over the local index, web search off. Retrieval counts a hit when a retrieved chunk comes from the question's topic article in any language.

| Language | Label | Questions | hit@5 | Answer in the right language | Citations all valid | Cited chunk supports the sentence |
|---|---|---:|---:|---:|---:|---:|
| English | `eng_Latn` | 20 | 1.000 | 1.000 | 1.000 | 0.932 |
| Hindi | `hin_Deva` | 20 | 0.900 | 0.700 | 1.000 | 0.822 |
| Hindi (romanized) | `hin_Latn` | 20 | 0.850 | 0.450 | 1.000 | 0.899 |
| Indonesian | `ind_Latn` | 20 | 0.950 | 0.850 | 1.000 | 0.844 |
| Japanese | `jpn_Jpan` | 20 | 0.950 | 0.950 | 1.000 | 0.910 |
| Korean | `kor_Hang` | 20 | 0.950 | 1.000 | 1.000 | 0.913 |
| Tamil | `tam_Taml` | 20 | 0.950 | 1.000 | 1.000 | 0.889 |
| Chinese (Simplified) | `zho_Hans` | 20 | 0.750 | 1.000 | 1.000 | 0.844 |
| Chinese (Traditional) | `zho_Hant` | 20 | 0.800 | 0.850 | 1.000 | 0.849 |
| Malay | `zsm_Latn` | 20 | 0.900 | 1.000 | 1.000 | 0.859 |
| All languages | | 200 | 0.900 | 0.880 | 1.000 | 0.875 |

"Citations all valid" only means every [n] points at a source that was provided. The last column is different: a model read each of the 616 cited sentences next to the chunk it cites and judged 539 supported, 77 unsupported and 0 unclear.
That judge was `qwen3:30b-a3b-instruct-2507-q4_K_M`, and it agreed with the person on 0.750 of 20 hand-labelled pairs (the judge called 0.500 of them supported, the person 0.550).

- The knowledge base holds arXiv metadata and Wikipedia lead sections only: data/methods.csv ships with its header row, so no curated fabrication method is indexed. Retrieval numbers describe that corpus, not the fabrication literature.
- A question is a hit when a retrieved chunk comes from its topic's Wikipedia article in any language, so hits can be cross-language. DATASHEET.md records no documents at all in Traditional Chinese or romanized Hindi.
- Questions other than the English ones were machine-translated and no question or answer was checked by a native speaker.
- citations_all_valid means every [n] points at a source that was provided. Whether that source supports the sentence is the faithfulness section, judged by a model; eval/judge_agreement.py reports how often that judge agreed with a hand-labelled sample.
- Web search is off during this run, so nothing outside the index can answer a question.
- The index is checked before and during the run: a retrieved chunk whose source is a test fixture stops the run, so no number here was produced with test data in the corpus.

<!-- results:end -->

## Repository layout

| Path | What is in it |
|---|---|
| `backend/` | FastAPI app: pipeline, providers, search, privacy, storage |
| `frontend/` | React and Vite project |
| `ingest/` | Knowledge-base build, indexing, datasheet |
| `training/` | Language-ID data building, training, sweeps, export |
| `eval/` | Language-ID evaluation and the generated tables |
| `scripts/` | Model download and GlotLID compression |
| `tests/` | The test suite |
| `results/` | Committed result files; every number in the documentation comes from one |

## Limits

- `data/methods.csv`, the hand-curated part of the knowledge base, ships with its header row only:
  none of the prototype's nine rows could be traced to a source anyone can open. The knowledge base
  is arXiv abstracts and Wikipedia lead sections until someone checks rows into it, following
  [docs/knowledge-base-curation.md](docs/knowledge-base-curation.md).
- Two supported languages have no documents at all, Chinese (Traditional) and romanized Hindi, so
  questions in them are answered from documents in another language. The counts are in
  [DATASHEET.md](DATASHEET.md).
- Retrieval quality, answer-language accuracy and citation faithfulness are now measured, over 200
  questions in ten languages: see Results. Two caveats travel with those numbers. Answering in the
  question's language is weakest for romanized Hindi (0.450) and Hindi (0.700), so a reader asking in
  those often gets an answer in another language. And "the cited chunk supports the sentence" (0.875)
  was judged by a model that agreed with a hand-labelled sample only 0.750 of the time, so treat it
  as an indicator rather than a measurement.
- The frontend has not been checked against a running backend; see [Interface](#interface). Its
  test suite runs against a mocked API only.
- Google's Custom Search JSON API is being wound down; `WEB_SEARCH=duckduckgo` uses the same
  interface if your key stops working, and `WEB_SEARCH=off` turns web search off.
- The evaluation ran on one machine, recorded in each results file. Latency figures are from that
  machine and say nothing about yours.

## Licences

The code is MIT, in [LICENSE](LICENSE).

- Model files and their licences: [MODELS.md](MODELS.md). GlotLID is Apache-2.0, multilingual-e5-small
  is MIT; neither is committed here.
- Knowledge-base documents and their licences: [DATASHEET.md](DATASHEET.md). arXiv metadata is CC0,
  Wikipedia text is CC BY-SA 4.0.
- Training data: [training/data/DATACARD.md](training/data/DATACARD.md). The data is never
  redistributed, only the scripts that sample it.
