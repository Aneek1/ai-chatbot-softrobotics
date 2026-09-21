import pytest

from eval.readme_table import END, START, confusion_tables, language_table, rag_tables, render, replace_block

EVALUATION = {
    "date": "2026-09-16T10:00:00+00:00",
    "detectors": {
        "two-stage+specialist-e5-finetune": {
            "slices": {
                "flores200-devtest/sentence": {
                    "per_label": {
                        "eng_Latn": {"f1": 0.9985, "support": 1012},
                        "zho_Hant": {"f1": 0.8997, "support": 1012},
                        "hin_Latn": {"f1": 0.0, "support": 0},
                    },
                    "confusions": {
                        "han": {
                            "zho_Hans": {"zho_Hans": 988, "zho_Hant": 0, "(outside group)": 24},
                            "zho_Hant": {"zho_Hans": 0, "zho_Hant": 897, "(outside group)": 115},
                        }
                    },
                },
                "flores200-devtest/taiwan-vocabulary": {
                    "per_label": {"zho_Hant": {"f1": 0.95, "support": 1000}},
                    "confusions": {},
                },
            },
            "latency": {"p50_ms": 0.77, "p95_ms": 1.13},
            "model_bytes": 695_000_000,
            "threshold_sweep": [
                {"threshold": 0.3, "coverage": 0.9948, "accuracy": 0.964},
                {"threshold": 0.9, "coverage": 0.8866, "accuracy": 0.9871},
            ],
        }
    },
}
RAG = {
    "date": "2026-09-16T20:00:00+00:00",
    "answer_model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "judge": "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "questions": 200,
    "per_language": {
        "eng_Latn": {
            "questions": 20,
            "hit_at_5": 0.85,
            "hit_at_6": 0.9,
            "answer_language_correct": 1.0,
            "has_citation": 0.95,
            "citations_all_valid": 0.9,
        }
    },
    "overall": {
        "questions": 200,
        "hit_at_5": 0.6,
        "hit_at_6": 0.65,
        "answer_language_correct": 0.8,
        "has_citation": 0.9,
        "citations_all_valid": 0.85,
    },
    "faithfulness": {"pairs": 300, "supported": 180, "unsupported": 90, "unclear": 30, "supported_rate": 0.6},
    "faithfulness_by_language": {"eng_Latn": {"pairs": 30, "supported_rate": 0.7}},
    "judge_agreement": {
        "labelled": 30,
        "agreement": 0.8,
        "judge_supported_rate": 0.6,
        "human_supported_rate": 0.5,
    },
    "notes": ["data/methods.csv ships with its header row."],
}


def entry():
    return EVALUATION["detectors"]["two-stage+specialist-e5-finetune"]


def test_language_table_reports_a_label_without_support_as_not_available():
    lines = language_table(entry())
    assert "| English | `eng_Latn` | 0.999 | n/a | n/a |" in lines
    assert "| Chinese (Traditional) | `zho_Hant` | 0.900 | 0.950 | n/a |" in lines
    assert "| Hindi (romanized) | `hin_Latn` | n/a | n/a | n/a |" in lines


def test_confusion_tables_print_the_counts():
    lines = confusion_tables(entry())
    assert "| `zho_Hant` | 0 | 897 | 115 |" in lines


def test_rag_tables_show_faithfulness_next_to_the_agreement():
    text = "\n".join(rag_tables(RAG, "results/rag-eval-2026-09-21.json"))
    assert "| English | `eng_Latn` | 20 | 0.850 | 1.000 | 0.900 | 0.700 |" in text
    assert "| All languages | | 200 | 0.600 | 0.800 | 0.850 | 0.600 |" in text
    assert "agreed with the person on 0.800 of 30" in text
    assert "data/methods.csv ships with its header row." in text


def test_render_names_its_sources_and_the_detector():
    text = render(
        (EVALUATION, "results/langid-eval-2026-09-16.json"),
        (RAG, "results/rag-eval-2026-09-21.json"),
        detector="two-stage+specialist-e5-finetune",
    )
    assert "two-stage+specialist-e5-finetune" in text
    assert "results/langid-eval-2026-09-16.json" in text
    assert "0.77" in text and "695.0" in text


def test_replace_block_keeps_the_text_around_the_markers():
    readme = f"# Title\n\nbefore\n\n{START}\n\nold\n\n{END}\n\nafter\n"
    updated = replace_block(readme, "new table")
    assert "before" in updated and "after" in updated and "old" not in updated and "new table" in updated
    with pytest.raises(ValueError, match="markers"):
        replace_block("# Title\n", "new table")
