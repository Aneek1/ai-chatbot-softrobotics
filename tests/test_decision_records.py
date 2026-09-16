from pathlib import Path

DECISIONS = Path(__file__).resolve().parents[1] / "docs" / "decisions"
RECORDS = {
    "0001-compressed-glotlid.md",
    "0002-two-stage-detection.md",
    "0003-snippets-only-search.md",
    "0004-app-wide-private-mode.md",
    "0005-e5-small-embeddings.md",
}
SECTIONS = ("## Context", "## Decision", "## Consequences")
MARKETING = ("seamless", "cutting-edge", "state-of-the-art", "blazing", "revolutionary", "powerful")


def test_every_decision_the_design_names_has_a_record():
    assert {path.name for path in DECISIONS.glob("[0-9]*.md")} == RECORDS


def test_each_record_states_its_status_context_decision_and_consequences():
    for name in sorted(RECORDS):
        text = (DECISIONS / name).read_text(encoding="utf-8")
        assert text.startswith("# "), name
        assert "Status:" in text, name
        for section in SECTIONS:
            assert section in text, f"{name} has no {section}"


def test_records_are_plain_text_without_marketing_words():
    for name in sorted(RECORDS):
        text = (DECISIONS / name).read_text(encoding="utf-8")
        assert all(ord(char) < 0x2190 for char in text), f"{name} has a symbol outside plain text"
        for word in MARKETING:
            assert word not in text.lower(), f"{name} uses {word}"


def test_the_index_links_every_record():
    index = (DECISIONS / "README.md").read_text(encoding="utf-8")
    for name in sorted(RECORDS):
        assert f"({name})" in index, name
