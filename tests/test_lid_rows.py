import pytest

from training.data.rows import Row, read_jsonl, write_jsonl


def test_round_trip_keeps_non_latin_text(tmp_path):
    rows = [
        Row("如何制作软气动执行器", "zho_Hans", "openlid", "lti"),
        Row("aap kaise hain", "hin_Latn", "dakshina", "hi.romanized.rejoined.dev.roman.txt", None),
        Row("silikon", "ind_Latn", "openlid", "leipzig", "crop"),
    ]
    path = tmp_path / "nested" / "train.jsonl"
    assert write_jsonl(rows, path) == 3
    assert list(read_jsonl(path)) == rows
    assert "如何" in path.read_text(encoding="utf-8")


def test_missing_field_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "a", "label": "eng_Latn"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl line 1: missing source, dataset_source, synthetic"):
        list(read_jsonl(path))


def test_unknown_synthetic_kind_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '\n{"text": "a", "label": "eng_Latn", "source": "openlid", "dataset_source": "x", '
        '"synthetic": "paraphrase"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2: unknown synthetic kind 'paraphrase'"):
        list(read_jsonl(path))
