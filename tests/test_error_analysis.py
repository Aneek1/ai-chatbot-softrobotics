from eval.error_analysis import citation_section, explain, language_section, render

EVALUATION = {
    "detectors": {
        "two-stage+specialist-e5-finetune": {
            "mistakes": [
                {
                    "set": "flores200-devtest",
                    "slice": "sentence",
                    "gold": "zsm_Latn",
                    "predicted": "ind_Latn",
                    "probability": 0.62,
                    "text": "Beliau berkata pasukan itu sedang menyiasat kejadian tersebut.",
                },
                {
                    "set": "flores200-devtest",
                    "slice": "1-3 words",
                    "gold": "zho_Hant",
                    "predicted": "zho_Hans",
                    "probability": 0.51,
                    "text": "軟體",
                },
            ]
        }
    }
}
RAG = {
    "judge": "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "judged": [
        {
            "question_id": "q07-eng_Latn",
            "language": "eng_Latn",
            "citation": 1,
            "chunk_id": "wikipedia:en:3342099:0",
            "sentence": "Silicone rubber withstands 300 C [1].",
            "chunk_text": "Silicone rubber is a polymer containing silicon.",
            "verdict": "unsupported",
        },
        {
            "question_id": "q05-eng_Latn",
            "language": "eng_Latn",
            "citation": 1,
            "chunk_id": "wikipedia:en:3342099:0",
            "sentence": "It contains silicon [1].",
            "chunk_text": "Silicone rubber is a polymer containing silicon.",
            "verdict": "supported",
        },
    ],
}


def test_explanations_name_the_group_and_the_slice():
    assert "Malay" in explain(EVALUATION["detectors"]["two-stage+specialist-e5-finetune"]["mistakes"][0])
    assert "one to three words" in explain(
        EVALUATION["detectors"]["two-stage+specialist-e5-finetune"]["mistakes"][1]
    )


def test_language_section_lists_the_examples():
    text = "\n".join(language_section(EVALUATION, "two-stage+specialist-e5-finetune", "results/x.json"))
    assert "`zsm_Latn`" in text and "`ind_Latn`" in text and "0.62" in text
    assert "Beliau berkata" in text


def test_citation_section_shows_only_the_unsupported_pairs():
    text = "\n".join(citation_section(RAG, "results/y.json"))
    assert "Silicone rubber withstands 300 C [1]." in text
    assert "It contains silicon [1]." not in text
    assert "1 of 2" in text


def test_render_without_inputs_says_so():
    text = render(None, None)
    assert text.startswith("# Error analysis")
    assert "No results file" in text
