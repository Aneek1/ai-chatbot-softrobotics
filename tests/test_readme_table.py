import pytest

from eval.readme_table import END, START, render, replace_block

EVALUATION = {
    "date": "2026-09-15T13:10:41+00:00",
    "items_per_set": {"flores200-devtest": 41749, "dakshina-test": 24899, "internal-test": 20555},
    "detectors": {
        "glotlid-compressed": {
            "slices": {
                "flores200-devtest/sentence": {"macro_f1": 0.9741},
                "flores200-devtest/1-3 words": {"macro_f1": 0.7619},
                "flores200-devtest/code-mixed": {"macro_f1": 0.5502},
                "dakshina-test/sentence": {"macro_f1": 0.9381},
                "internal-test/all": {"macro_f1": 0.7772},
            },
            "latency": {"p50_ms": 0.637, "p95_ms": 0.886},
            "model_bytes": 225_000_000,
        },
        "two-stage+specialist-e5-finetune": {
            "slices": {"flores200-devtest/sentence": {"macro_f1": 0.9810}},
            "latency": {"p50_ms": 3.42, "p95_ms": 5.10},
            "model_bytes": 470_700_000,
        },
    },
}


def test_every_detector_gets_a_row_with_its_scores():
    block = render(EVALUATION, "results/langid-eval-2026-09-15.json")
    assert "| glotlid-compressed | 0.974 | 0.762 | 0.550 | 0.938 | 0.777 | 0.64 | 225 |" in block
    assert "| two-stage+specialist-e5-finetune |" in block


def test_a_slice_a_detector_was_not_measured_on_prints_n_a():
    block = render(EVALUATION, "results/langid-eval-2026-09-15.json")
    row = next(line for line in block.splitlines() if line.startswith("| two-stage"))
    assert row.count("n/a") == 4


def test_the_block_names_the_file_and_the_number_of_items():
    block = render(EVALUATION, "results/langid-eval-2026-09-15.json")
    assert "`results/langid-eval-2026-09-15.json`" in block
    assert "2026-09-15" in block
    assert "flores200-devtest 41,749" in block


def test_only_what_is_between_the_markers_is_replaced():
    readme = f"# Title\n\nbefore\n\n{START}\nold\n{END}\n\nafter\n"
    updated = replace_block(readme, "new table\n")
    assert updated.startswith("# Title\n\nbefore\n\n")
    assert updated.endswith(f"{END}\n\nafter\n")
    assert "old" not in updated
    assert "new table" in updated


def test_writing_the_block_twice_changes_nothing():
    readme = f"{START}\n{END}\n"
    once = replace_block(readme, "new table\n")
    assert replace_block(once, "new table\n") == once


def test_a_readme_without_markers_is_refused():
    with pytest.raises(ValueError, match="results:start"):
        replace_block("# Title\n", "new table\n")
