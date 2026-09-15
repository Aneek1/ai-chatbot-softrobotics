import pytest

from backend.app.config import Settings
from backend.pipeline.chunk import load_token_counter
from backend.pipeline.langid import FastTextScorer, TwoStageDetector

pytestmark = pytest.mark.slow
settings = Settings(_env_file=None)
GLOTLID = settings.models_dir / settings.glotlid_file
E5 = settings.models_dir / settings.e5_dir


@pytest.fixture(scope="module")
def detector():
    if not GLOTLID.exists():
        pytest.skip(f"{GLOTLID} not found")
    return TwoStageDetector(general=FastTextScorer(GLOTLID), min_confidence=0.6)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How do you make a soft pneumatic actuator?", "eng_Latn"),
        ("Bagaimana cara membuat aktuator pneumatik lunak?", "ind_Latn"),
        ("如何制作软气动执行器？", "zho_Hans"),
        ("如何製作軟氣動執行器？", "zho_Hant"),
        ("ソフト空気圧アクチュエータの作り方は？", "jpn_Jpan"),
        ("소프트 공압 액추에이터는 어떻게 만드나요?", "kor_Hang"),
        ("மென்மையான காற்றழுத்த இயக்கியை எப்படி உருவாக்குவது?", "tam_Taml"),
        ("सॉफ्ट न्यूमेटिक एक्चुएटर कैसे बनाते हैं?", "hin_Deva"),
    ],
)
def test_compressed_glotlid_on_clear_sentences(detector, text, expected):
    assert detector.detect(text).chosen == expected


def test_e5_embeds_across_languages():
    if not (E5 / "onnx" / "model.onnx").exists():
        pytest.skip(f"{E5} not found")
    from backend.pipeline.embeddings import E5Embedder

    embedder = E5Embedder(E5)
    query = embedder.embed_queries(["how do you make a soft pneumatic actuator?"])[0]
    related, unrelated = embedder.embed_passages(
        [
            "Aktuator pneumatik lunak dibuat dengan menuang silikon ke dalam cetakan.",
            "The stock market closed higher today after strong bank earnings.",
        ]
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(query, related) > dot(query, unrelated)


def test_token_counter_counts_subwords():
    if not (E5 / "tokenizer.json").exists():
        pytest.skip("tokenizer not found")
    count = load_token_counter(E5 / "tokenizer.json")
    assert count("silicone") >= 1
    assert count("") == 0
