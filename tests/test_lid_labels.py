from training.labels import (
    GROUP_LABELS,
    GROUPS,
    LABELS,
    OPENLID_LABELS,
    OTHER,
    label_index,
    training_label,
)


def test_label_set_matches_the_spec():
    assert LABELS == (
        "eng_Latn",
        "ind_Latn",
        "zsm_Latn",
        "zho_Hans",
        "zho_Hant",
        "jpn_Jpan",
        "kor_Hang",
        "tam_Taml",
        "hin_Deva",
        "hin_Latn",
        "urd_Latn",
        "other",
    )


def test_groups_only_hold_trainable_labels():
    assert GROUPS == {
        "malay_indonesian": ("ind_Latn", "zsm_Latn"),
        "han": ("zho_Hans", "zho_Hant"),
        "romanized_hindi": ("eng_Latn", "hin_Latn", "urd_Latn"),
    }
    assert GROUP_LABELS == (
        "eng_Latn",
        "ind_Latn",
        "zsm_Latn",
        "zho_Hans",
        "zho_Hant",
        "hin_Latn",
        "urd_Latn",
    )


def test_openlid_languages_outside_the_set_become_other():
    assert training_label("zsm_Latn") == "zsm_Latn"
    assert training_label("urd_Arab") == OTHER
    assert training_label("fra_Latn") == OTHER


def test_openlid_labels_are_a_subset_without_romanized_labels():
    assert OPENLID_LABELS < set(LABELS)
    assert "hin_Latn" not in OPENLID_LABELS and "urd_Latn" not in OPENLID_LABELS


def test_label_index():
    assert label_index("eng_Latn") == 0
    assert label_index(OTHER) == 11
