from backend.pipeline.normalize import normalize, script_profile


def test_collapses_whitespace_and_newlines():
    assert normalize("  Hello\n\tworld  ") == "Hello world"


def test_removes_zero_width_space_and_bom():
    assert normalize("\ufeffsili\u200bcone") == "silicone"


def test_keeps_zero_width_non_joiner():
    # ZWNJ changes spelling in Persian and some Indic text, so it must survive.
    text = "می\u200cخواهم"
    assert normalize(text) == text


def test_nfkc_folds_full_width_characters():
    assert normalize("ＡＢＣ１２３") == "ABC123"


def test_latin_text_profile():
    profile = script_profile("How do you make a soft actuator?")
    assert profile.dominant == "Latn"
    assert profile.mixed is False


def test_japanese_counts_kana_and_kanji_together():
    profile = script_profile("ソフト空気圧アクチュエータの作り方は？")
    assert profile.dominant == "Jpan"
    assert profile.mixed is False


def test_chinese_and_english_mix_is_flagged():
    profile = script_profile("软体机器人 soft robot")
    assert profile.mixed is True


def test_no_letters_has_no_dominant_script():
    profile = script_profile("123 !!")
    assert profile.dominant is None
    assert profile.shares == {}
