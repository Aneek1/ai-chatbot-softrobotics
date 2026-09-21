"""RAG evaluation (spec section 11.2).

Usage:
    PYTHONUTF8=1 OLLAMA_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M uv run python -m eval.rag_eval
    PYTHONUTF8=1 OLLAMA_MODEL=... uv run python -m eval.rag_eval --per-language 2 --no-judge

Each question is retrieved for and answered exactly as the app would do it, with web search off, so
the numbers describe the local knowledge base alone. That knowledge base is arXiv metadata and
Wikipedia lead sections: data/methods.csv ships with its header row only, so no curated fabrication
method is indexed, and DATASHEET.md records zero documents in Traditional Chinese and zero in
romanized Hindi.
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from backend.pipeline.citations import check_citations
from backend.pipeline.langid import LanguageDetector
from backend.pipeline.prompt import build_messages
from backend.pipeline.retrieve import Retriever
from backend.providers.base import AnswerModel
from eval.faithfulness import ModelJudge, faithfulness, judge_all, pairs_for
from eval.questions import QUESTIONS_PATH, Question, read_questions
from eval.rag_metrics import cited_doc_ids, hit_at_k, summarize
from eval.topics import load_topics, missing_from
from training.env_report import environment

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
NOTES = (
    "The knowledge base holds arXiv metadata and Wikipedia lead sections only: data/methods.csv "
    "ships with its header row, so no curated fabrication method is indexed. Retrieval numbers "
    "describe that corpus, not the fabrication literature.",
    "A question is a hit when a retrieved chunk comes from its topic's Wikipedia article in any "
    "language, so hits can be cross-language. DATASHEET.md records no documents at all in "
    "Traditional Chinese or romanized Hindi.",
    "Questions other than the English ones were machine-translated and no question or answer was "
    "checked by a native speaker.",
    "citations_all_valid means every [n] points at a source that was provided. Whether that source "
    "supports the sentence is the faithfulness section, judged by a model; "
    "eval/judge_agreement.py reports how often that judge agreed with a hand-labelled sample.",
    "Web search is off during this run, so nothing outside the index can answer a question.",
)


def answer_question(
    question: Question,
    detector: LanguageDetector,
    retriever: Retriever,
    model: AnswerModel,
    topics: dict[str, list[str]],
    top_k: int,
) -> tuple[dict, list, str, list]:
    started = time.perf_counter()
    hits = retriever.search(question.text, top_k)
    doc_ids = [hit.chunk.doc_id for hit in hits]
    answer = "".join(model.stream(build_messages(question.text, question.language, hits)))
    check = check_citations(answer, source_count=len(hits))
    detection = detector.detect(check.text)
    expected = topics[question.topic]
    record = {
        "id": question.id,
        "language": question.language,
        "topic": question.topic,
        "hit_at_5": hit_at_k(doc_ids, expected, 5),
        "hit_at_6": hit_at_k(doc_ids, expected, 6),
        "retrieved_doc_ids": doc_ids,
        "answer_language_detected": detection.chosen,
        "answer_language_uncertain": detection.uncertain,
        "answer_language_correct": detection.chosen == question.language,
        "has_citation": bool(check.valid or check.removed),
        # Without a citation there is nothing to be valid or invalid, so this stays unmeasured.
        "citations_all_valid": (not check.removed) if (check.valid or check.removed) else None,
        "citations_valid": list(check.valid),
        "citations_removed": list(check.removed),
        "cited_doc_ids": cited_doc_ids(check.valid, doc_ids),
        "answer_chars": len(check.text),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return record, hits, check.text, pairs_for(question, check.text, hits)


def evaluate(
    questions: Sequence[Question],
    detector: LanguageDetector,
    retriever: Retriever,
    model: AnswerModel,
    topics: dict[str, list[str]],
    top_k: int,
    judge=None,
    log: Callable[[str], None] = print,
) -> dict:
    records, answers, pairs = [], [], []
    for number, question in enumerate(questions, start=1):
        record, _, answer, question_pairs = answer_question(
            question, detector, retriever, model, topics, top_k
        )
        records.append(record)
        answers.append({"id": question.id, "language": question.language, "answer": answer})
        pairs += question_pairs
        log(f"[{number}/{len(questions)}] {question.id} {record['seconds']} s")
    judged = []
    if judge is not None:
        log(f"Judging {len(pairs)} cited sentences with {judge.name}")
        judged = judge_all(pairs, judge)
    by_language: dict[str, list[dict]] = {}
    for record in judged:
        by_language.setdefault(record["language"], []).append(record)
    return {
        "questions": len(questions),
        "top_k": top_k,
        "judge": judge.name if judge is not None else None,
        "per_language": summarize(records),
        "per_topic": summarize(records, key="topic"),
        "overall": summarize([{**r, "all": "all"} for r in records], key="all")["all"],
        "faithfulness": faithfulness(judged) if judged else None,
        "faithfulness_by_language": {
            language: faithfulness(rows) for language, rows in sorted(by_language.items())
        },
        "judged": judged,
        "records": records,
        "answers": answers,
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from backend.app.config import Settings
    from backend.pipeline.embeddings import E5Embedder
    from backend.pipeline.langid import FastTextScorer, TwoStageDetector
    from backend.pipeline.retrieve import ChunkIndex
    from backend.pipeline.specialist import load_specialist
    from backend.providers.ollama import OllamaModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", type=int, default=0, help="limit questions per language")
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness pass")
    args = parser.parse_args()

    settings = Settings()
    questions = read_questions(QUESTIONS_PATH)
    if args.per_language:
        seen: dict[str, int] = {}
        kept = []
        for question in questions:
            seen[question.language] = seen.get(question.language, 0) + 1
            if seen[question.language] <= args.per_language:
                kept.append(question)
        questions = kept
    topics = load_topics()

    embedder = E5Embedder(settings.models_dir / settings.e5_dir)
    specialist = (
        load_specialist(settings.models_dir, settings.langid_specialist, embedder)
        if settings.langid_specialist
        else None
    )
    detector = TwoStageDetector(
        general=FastTextScorer(settings.models_dir / settings.glotlid_file),
        min_confidence=settings.langid_min_confidence,
        specialist=specialist,
    )
    index = ChunkIndex.open(settings.index_dir, embedder)
    model = OllamaModel(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
    judge = None if args.no_judge else ModelJudge(model, settings.ollama_model)
    try:
        results = evaluate(questions, detector, index, model, topics, settings.retrieval_top_k, judge)
        results = {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": environment(),
            "answer_model": settings.ollama_model,
            "detector": settings.glotlid_file
            + (f" + {settings.langid_specialist}" if specialist else " + OpenCC rule"),
            "questions_file": "eval/questions.jsonl",
            "topics_file": "eval/topics.json",
            "documents_by_language": index.count_by_language(),
            "topic_documents_missing_from_index": missing_from(topics, index.doc_ids()),
            "notes": list(NOTES),
            **results,
        }
    finally:
        index.close()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"rag-eval-{datetime.now(UTC):%Y-%m-%d}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = results["overall"]
    print(
        f"hit@5 {overall['hit_at_5']}, answer language {overall['answer_language_correct']}, "
        f"citations all valid {overall['citations_all_valid']}"
    )
    if results["faithfulness"]:
        print(f"cited chunk supports the sentence: {results['faithfulness']['supported_rate']}")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
