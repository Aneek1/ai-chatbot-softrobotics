from training.data.clean import Cleaner, clean_rows, held_out_keys, text_key
from training.data.rows import Row


def row(text: str, label: str = "eng_Latn") -> Row:
    return Row(text, label, "openlid", "lti")


def test_text_is_normalized_like_the_app():
    kept, _ = clean_rows([row("  soft\nrobot​ arm ")])
    assert kept[0].text == "soft robot arm"


def test_short_lines_are_dropped():
    kept, stats = clean_rows([row("a"), row(" \n"), row("ok")])
    assert [r.text for r in kept] == ["ok"]
    assert stats.too_short["eng_Latn"] == 2


def test_exact_duplicates_after_normalization_are_dropped():
    kept, stats = clean_rows([row("Soft robot"), row("Soft  robot\n"), row("Soft robot", "ind_Latn")])
    assert len(kept) == 1
    assert stats.duplicates == {"eng_Latn": 1, "ind_Latn": 1}
    assert stats.label_conflicts == {"ind_Latn": 1}


def test_evaluation_sentences_are_removed():
    held_out = held_out_keys(["The FLORES sentence.\n"])
    kept, stats = clean_rows([row("The FLORES  sentence."), row("A training sentence.")], held_out)
    assert [r.text for r in kept] == ["A training sentence."]
    assert stats.contaminated == {"eng_Latn": 1}


def test_cleaner_shares_state_across_calls():
    cleaner = Cleaner()
    assert cleaner.accept(row("same line")) is not None
    assert cleaner.accept(row("same line", "zsm_Latn")) is None
    assert cleaner.stats.as_dict()["kept"] == {"eng_Latn": 1}


def test_text_key_is_stable():
    assert text_key("abc") == text_key("abc")
    assert len(text_key("abc")) == 16
