# Models and their licences

No model file is committed to this repository. `scripts/download_models.py` downloads them into a
flat `models/` folder and writes `models/MANIFEST.json` with the repository, the revision and a
SHA-256 for every file, so what is on disk can be checked against what was downloaded.

| Model | Source | Licence | Used for |
|---|---|---|---|
| GlotLID (`models/glotlid-q.ftz`) | `cis-lmu/glotlid` on Hugging Face, product-quantized locally by `scripts/compress_glotlid.py` | Apache-2.0 | Stage-1 language identification |
| multilingual-e5-small (`models/multilingual-e5-small`) | `intfloat/multilingual-e5-small` on Hugging Face | MIT | Chunk and query embeddings |
| Stage-2 specialist (`models/`, set by `LANGID_SPECIALIST`) | trained in this repository, exported by `training/export.py` | MIT, same as this repository's code | Deciding inside the confusion groups |

At runtime `HF_HUB_OFFLINE=1` is set and models load from these local paths, so nothing is
downloaded while the app is running.

## Data behind the trained specialist

The specialist is trained on data this repository does not redistribute. `training/data/DATACARD.md`
records what was used and how it was sampled; `training/RESULTS.md` records what came out.

- OpenLID (`laurievb/open-lid-dataset`) lists its licence as "other": it varies by source corpus.
  The repository publishes the scripts that sample it, never the data.
- Dakshina (romanized Hindi) is CC BY-SA 4.0.
- FLORES+ devtest is CC BY-SA 4.0 and is held out entirely: never used for training or selection.

## Knowledge-base data

Document licences are in `DATASHEET.md`, generated from the built knowledge base by
`ingest/datasheet.py`.
