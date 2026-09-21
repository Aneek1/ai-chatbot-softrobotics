# Soft-robotics assistant

A retrieval assistant for soft-robotics fabrication that works in the language you write in. It
identifies the language of a question, retrieves passages from a licence-tracked knowledge base,
answers with a local or a hosted model, and cites the passages it used. Language identification is
the part of the project with the most work in it: a compressed GlotLID model covers two thousand
languages, and a specialist trained in this repository decides the cases GlotLID confuses.

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

The HTTP API is the working interface. The React frontend is set up, themed and built by the
container, but its workspace screens are not written yet, so what it serves today is a placeholder
page.

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

<!-- results:start -->
Macro-F1 by slice, from `results/langid-eval-2026-09-21.json`, measured on 2026-09-21.
Items: flores200-devtest 43,749, dakshina-test 24,899, internal-test 20,555, knowledge-base 86.

| Detector | FLORES sentence | FLORES 1-3 words | code-mixed | Dakshina sentence | internal test | p50 ms | Model MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| glotlid-full | 0.974 | 0.756 | 0.582 | 0.938 | 0.777 | 0.35 | 1687 |
| glotlid-compressed | 0.975 | 0.756 | 0.592 | 0.935 | 0.781 | 0.78 | 225 |
| lid.176 | 0.817 | 0.750 | 0.359 | 0.001 | 0.678 | 0.06 | 1 |
| specialist-fasttext | 0.963 | 0.843 | 0.719 | 0.848 | 0.838 | 0.08 | 2 |
| two-stage+specialist-fasttext | 0.977 | 0.774 | 0.599 | 0.839 | 0.766 | 0.89 | 227 |
| specialist-e5-head | 0.859 | 0.842 | 0.374 | 0.814 | 0.816 | 100.67 | 471 |
| two-stage+specialist-e5-head | 0.972 | 0.766 | 0.564 | 0.823 | 0.758 | 2.90 | 696 |
| specialist-e5-finetune | 0.964 | 0.859 | 0.692 | 0.934 | 0.850 | 77.37 | 472 |
| two-stage+specialist-e5-finetune | 0.978 | 0.774 | 0.600 | 0.910 | 0.765 | 2.48 | 696 |

Per-language precision and recall, the confusion groups, the threshold sweep and the comparison with the published GlotLID and OpenLID figures are in [training/RESULTS.md](training/RESULTS.md), generated from the same files.
<!-- results:end -->

Written by `eval/readme_table.py`; re-run it after an evaluation instead of editing the table.

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
- Retrieval quality, answer-language accuracy and whether a cited chunk really supports the sentence
  citing it are not measured yet. The results below are language identification only.
- The frontend workspace is not built. The API is the interface.
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
