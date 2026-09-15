from eval.flores_benchmark import NOTES, PUBLISHED
from training.results_table import comparison_sentence, latest, render

SELECTION = {
    "selection_metric": "validation macro-F1 over confusion-group labels (val_select)",
    "overall": "e5-head-03-20260915-230000",
    "ranking": [
        {
            "run_id": "e5-head-03-20260915-230000",
            "family": "e5_head",
            "val_group_macro_f1": 0.9123,
            "val_macro_f1": 0.95,
            "val_group_f1": {"malay_indonesian": 0.8, "han": 0.97, "romanized_hindi": 0.96},
            "model_bytes": 2_500_000,
            "wall_clock_seconds": 240.0,
        }
    ],
}


def benchmark(full_f1: float) -> dict:
    group = {"malay_indonesian": 0.7, "han": 0.9, "romanized_hindi": 0.99}
    return {
        "benchmark": {"sentences": 206448},
        "published": list(PUBLISHED),
        "notes": list(NOTES),
        "native_space": {"glotlid-full": {"languages": 195, "macro_f1": full_f1, "macro_fpr": 0.0041}},
        "specialist_space": {
            "glotlid-compressed": {
                "macro_f1": 0.93,
                "group_macro_f1": 0.86,
                "group_f1": group,
                "per_label": {"other": {"f1": 0.99}},
                "sentences_per_second": 51234.5,
            }
        },
        "contamination": {
            "flores200-dev": {"train": 0, "test": 0},
            "flores200-devtest": {"train": 0, "test": 0},
        },
    }


EVALUATION = {
    "detectors": {
        "glotlid-compressed": {
            "slices": {
                "flores200-devtest/sentence": {"macro_f1": 0.931},
                "dakshina-test/sentence": {"macro_f1": 0.412},
            },
            "latency": {"p50_ms": 0.08, "p95_ms": 0.21},
            "model_bytes": 225_000_000,
        }
    }
}


def test_render_builds_every_section_from_the_files():
    text = render(
        (SELECTION, "results/lid-selection-2026-09-16.json"),
        (EVALUATION, "results/langid-eval-2026-09-16.json"),
        (benchmark(0.915), "results/lid-flores200-devtest-2026-09-16.json"),
    )
    assert text.startswith("# Language-ID results")
    assert (
        "| `e5-head-03-20260915-230000` | e5_head | 0.912 | 0.950 | 0.800 | 0.970 | 0.960 | 2.5 | 4 |" in text
    )
    assert "| glotlid-full | this repository | 195 | 0.915 | 0.0041 |" in text
    assert (
        "| GlotLID-M | Kargaran et al. 2023 (arXiv:2310.16248), Table 2, FLORES-200 all "
        "| 196 | 0.917 | 0.0042 |" in text
    )
    assert "| glotlid-compressed | 0.930 | 0.860 | 0.700 | 0.900 | 0.990 | 0.990 | 51234 |" in text
    assert "| glotlid-compressed | 0.931 | n/a | n/a | n/a | 0.412 | n/a | 0.08 | 0.21 | 225.0 |" in text
    assert "0 dataset rows match a held-out line" in text
    assert "not a reproduction" in text


def test_comparison_never_claims_more_than_the_numbers_show():
    assert "within 0.01 of the 0.917" in comparison_sentence(benchmark(0.912))
    assert "0.050 below the 0.917" in comparison_sentence(benchmark(0.867))
    assert "0.020 above" in comparison_sentence(benchmark(0.937))


def test_missing_inputs_leave_sections_out():
    text = render(None, None, None)
    assert "## Validation" not in text and "## FLORES-200" not in text


def test_latest_picks_the_newest_dated_file(tmp_path):
    for name in ("langid-eval-2026-09-15.json", "langid-eval-2026-09-16.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert latest("langid-eval-*.json", tmp_path).name == "langid-eval-2026-09-16.json"
    assert latest("missing-*.json", tmp_path) is None
