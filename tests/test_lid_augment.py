import random

from training.data.augment import code_mixed, crop, opencc_pairs, short_crops
from training.data.rows import Row


def test_opencc_pairs_convert_in_both_directions():
    rows = [
        Row("如何制作软气动执行器", "zho_Hans", "openlid", "lti"),
        Row("如何製作軟氣動執行器", "zho_Hant", "openlid", "lti"),
    ]
    pairs = opencc_pairs(rows, random.Random(0), per_label=5)
    assert Row("如何製作軟氣動執行器", "zho_Hant", "openlid", "lti", "opencc") in pairs
    assert Row("如何制作软气动执行器", "zho_Hans", "openlid", "lti", "opencc") in pairs


def test_opencc_skips_lines_that_do_not_change():
    rows = [Row("中文", "zho_Hans", "openlid", "lti"), Row("soft robot", "eng_Latn", "openlid", "lti")]
    assert opencc_pairs(rows, random.Random(0), per_label=5) == []


def test_opencc_respects_the_per_label_cap():
    rows = [Row(f"软体{i}", "zho_Hans", "openlid", "lti") for i in range(10)]
    assert len(opencc_pairs(rows, random.Random(0), per_label=3)) == 3


def test_word_crop_has_one_to_three_words_from_the_line():
    rng = random.Random(3)
    text = "soft robots are made from silicone rubber"
    for _ in range(50):
        piece = crop(text, rng)
        assert 1 <= len(piece.split()) <= 3
        assert piece in text


def test_unspaced_text_is_cropped_by_characters():
    rng = random.Random(3)
    text = "ソフト空気圧アクチュエータの作り方"
    for _ in range(50):
        piece = crop(text, rng)
        assert 2 <= len(piece) <= 8
        assert piece in text


def test_short_crops_are_tagged_and_sized():
    rows = [Row(f"line number {i} with words", "eng_Latn", "openlid", "lti") for i in range(20)]
    crops = short_crops(rows, random.Random(0), fraction=0.25)
    assert len(crops) == 5
    assert {r.synthetic for r in crops} == {"crop"}


def test_code_mixed_label_is_the_longer_part():
    rows = [
        Row("hello", "eng_Latn", "openlid", "a"),
        Row("Bagaimana cara membuat aktuator pneumatik lunak", "ind_Latn", "openlid", "b"),
        Row("ignored line in another language", "other", "openlid", "c"),
    ]
    mixed = code_mixed(rows, random.Random(0), count=4)
    assert len(mixed) == 4
    for row in mixed:
        assert row.label == "ind_Latn"
        assert row.synthetic == "code_mixed"
        assert "hello" in row.text and "Bagaimana" in row.text


def test_code_mixed_needs_english_and_another_language():
    assert code_mixed([Row("hello", "eng_Latn", "openlid", "a")], random.Random(0), count=3) == []
