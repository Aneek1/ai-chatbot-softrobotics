from backend.pipeline.languages import SUPPORTED, display_name, group_of, script_of_label


def test_eight_languages_as_ten_labels():
    assert len(SUPPORTED) == 10
    assert "hin_Latn" in SUPPORTED and "zho_Hant" in SUPPORTED


def test_confusion_groups():
    assert group_of("zsm_Latn") == "malay_indonesian"
    assert group_of("cmn_Hani") == "han"
    assert group_of("urd_Latn") == "romanized_hindi"
    assert group_of("jpn_Jpan") is None


def test_display_names():
    assert display_name("zsm_Latn") == "Malay"
    assert display_name("nld_Latn") == "nld_Latn"
    assert display_name("und") == "Undetermined"


def test_script_of_label():
    assert script_of_label("hin_Latn") == "Latn"
    assert script_of_label("und") == ""
