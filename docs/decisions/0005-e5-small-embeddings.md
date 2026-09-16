# 0005: multilingual-e5-small for embeddings

Status: accepted, 2026-09-15.

## Context

The knowledge base is small and multilingual, and a question in Tamil often has to reach a document
in English. The embedding model runs on the same machine as the answer model, and both compete for
memory.

## Decision

Use `intfloat/multilingual-e5-small`, 384 dimensions, through fastembed's ONNX runtime, loaded from
a local folder with `HF_HUB_OFFLINE=1` set. Queries carry the `query: ` prefix and chunks the
`passage: ` prefix, as the model expects. Larger multilingual encoders are not used, and the
embedding model is not fine-tuned.

## Consequences

- No PyTorch at runtime: the app depends on onnxruntime, which has wheels for every platform the
  project supports, including arm64.
- Retrieval scores are used for ranking only. An unrelated English sentence scored 0.73 cosine
  against a related query in testing, so no fixed score threshold is applied anywhere.
- Retrieval quality is bounded by a small model on a small corpus. Measuring it is the work the
  design calls the RAG evaluation, which is not done yet.
