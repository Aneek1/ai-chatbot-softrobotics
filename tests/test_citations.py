from backend.pipeline.citations import check_citations


def test_keeps_valid_and_removes_invalid_citations():
    result = check_citations("Silicone is cast [1]. It cures [9].", source_count=2)
    assert result.text == "Silicone is cast [1]. It cures."
    assert result.valid == (1,)
    assert result.removed == (9,)


def test_repeated_citations_are_listed_once():
    result = check_citations("A [2]. B [2]. C [0].", source_count=2)
    assert result.valid == (2,)
    assert result.removed == (0,)


def test_no_sources_means_every_citation_is_removed():
    result = check_citations("Claim [1].", source_count=0)
    assert result.text == "Claim."
    assert result.removed == (1,)
