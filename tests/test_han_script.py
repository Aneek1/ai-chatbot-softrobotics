from backend.pipeline.han_script import han_variant


def test_simplified_text():
    assert han_variant("如何制作软气动执行器？") == "zho_Hans"


def test_traditional_text():
    assert han_variant("如何製作軟氣動執行器？") == "zho_Hant"


def test_characters_shared_by_both_are_ambiguous():
    assert han_variant("中文") == "ambiguous"


def test_text_mixing_both_variants_is_ambiguous():
    assert han_variant("软體") == "ambiguous"


def test_empty_text_is_ambiguous():
    assert han_variant("") == "ambiguous"


def test_non_han_text_is_ambiguous():
    # The detector only calls this for Han-script labels; stray non-Han input must not look Chinese.
    assert han_variant("silicone") == "ambiguous"
