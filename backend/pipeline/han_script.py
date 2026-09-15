from functools import lru_cache
from typing import Literal

import opencc

HanVariant = Literal["zho_Hans", "zho_Hant", "ambiguous"]


@lru_cache(maxsize=1)
def _converters() -> tuple[opencc.OpenCC, opencc.OpenCC]:
    return opencc.OpenCC("s2t"), opencc.OpenCC("t2s")


def han_variant(text: str) -> HanVariant:
    """Simplified text survives t2s unchanged but not s2t; Traditional is the reverse."""
    to_traditional, to_simplified = _converters()
    unchanged_by_s2t = to_traditional.convert(text) == text
    unchanged_by_t2s = to_simplified.convert(text) == text
    if unchanged_by_t2s and not unchanged_by_s2t:
        return "zho_Hans"
    if unchanged_by_s2t and not unchanged_by_t2s:
        return "zho_Hant"
    return "ambiguous"
