# Knowledge base datasheet

Written by `ingest/datasheet.py` on 2026-09-16 from
`data/knowledge-base.jsonl`, the file `ingest/build_kb.py` builds. Every number here is
counted from that file.

Documents: 86

## Documents by language

| Language | Label | Documents |
|---|---|---|
| English | eng_Latn | 60 |
| Indonesian | ind_Latn | 4 |
| Malay | zsm_Latn | 3 |
| Chinese (Simplified) | zho_Hans | 4 |
| Chinese (Traditional) | zho_Hant | 0 |
| Japanese | jpn_Jpan | 5 |
| Korean | kor_Hang | 5 |
| Tamil | tam_Taml | 1 |
| Hindi | hin_Deva | 4 |
| Hindi (romanized) | hin_Latn | 0 |

## Documents by source

| Source | Documents |
|---|---|
| arxiv | 54 |
| wikipedia | 32 |

## Licences

| Licence | Documents |
|---|---|
| CC BY-SA 4.0 | 32 |
| arXiv metadata, CC0 1.0 | 54 |

Each document records its `source`, `url`, `licence` and `retrieved_at`, and, where the
source has one, its `revision`: the arXiv version or the Wikipedia revision id.

## Curated rows

Rows in `data/methods.csv` not checked against a source: 0.
Those rows produce no documents. They are never retrieved, never cited and never sent to an
answer model. A row becomes a document only when it carries a source title, an http(s) link
to that source, a licence, and the name and date of the person who checked it.
`docs/knowledge-base-curation.md` has the rules.

## Known gaps

- No documents at all in: Chinese (Traditional) (zho_Hant), Hindi (romanized) (hin_Latn). Questions in those languages are answered
  from sources in other languages, or not at all.
- Wikipedia documents are lead sections only, not whole articles, and only for the languages
  that have an article linked from the English one.
- arXiv documents are titles and abstracts in English. Full texts are not fetched.
- Chinese comes from the Simplified variant of one Wikipedia, so Traditional Chinese
  (`zho_Hant`) has no documents of its own.
- Nothing here is translated, and no text is generated. What a source said is what is stored.

## How it was collected

`ingest/build_kb.py` reads `data/methods.csv`, queries the arXiv API and the Wikipedia API,
drops exact duplicates by hash of the normalized text, and writes the JSONL. It runs outside
the backend process, so the egress guard does not apply to it; it connects to
`export.arxiv.org` and the `*.wikipedia.org` API hosts.
