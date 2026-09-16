"""Taiwan-vocabulary Chinese for the Han confusion group (spec section 11.1).

FLORES-200's zho_Hant side is Traditional characters with mainland vocabulary. Text written in
Taiwan differs in words as well as characters, which is the harder case for a detector that decides
Simplified against Traditional. OpenCC's s2twp conversion adds that vocabulary, so each pair says
the same thing in both variants. The conversions are synthetic and are labelled as such wherever
they are reported.
"""

from collections.abc import Sequence
from functools import lru_cache

import opencc

CONFIG = "s2twp"


@lru_cache(maxsize=1)
def _converter() -> opencc.OpenCC:
    return opencc.OpenCC(CONFIG)


def to_taiwan(text: str) -> str:
    return _converter().convert(text)


def taiwan_pairs(simplified: Sequence[str]) -> list[tuple[str, str]]:
    """(Simplified line, Taiwan conversion) for the lines the conversion changes.

    A line the conversion leaves alone carries no evidence either way, so it is dropped rather than
    counted as Traditional.
    """
    pairs = [(text, to_taiwan(text)) for text in simplified]
    return [(source, converted) for source, converted in pairs if converted != source]
