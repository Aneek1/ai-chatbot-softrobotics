import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# Zero-width space, word joiner, BOM and soft hyphen carry no meaning.
# ZWJ and ZWNJ are kept: they change spelling in Persian and several Indic scripts.
_INVISIBLE = re.compile("[\u200b\u2060\ufeff\u00ad]")
_WHITESPACE = re.compile(r"\s+")

_SCRIPT_BY_NAME_PREFIX = {
    "LATIN": "Latn",
    "CJK": "Hani",
    "HIRAGANA": "Hira",
    "KATAKANA": "Kana",
    "KATAKANA-HIRAGANA": "Kana",
    "HANGUL": "Hang",
    "TAMIL": "Taml",
    "DEVANAGARI": "Deva",
    "ARABIC": "Arab",
    "CYRILLIC": "Cyrl",
    "GREEK": "Grek",
    "THAI": "Thai",
}
MIXED_SHARE = 0.30


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class ScriptProfile:
    dominant: str | None
    shares: dict[str, float] = field(default_factory=dict)

    @property
    def mixed(self) -> bool:
        return sum(1 for share in self.shares.values() if share > MIXED_SHARE) >= 2


def _script_of(char: str) -> str:
    prefix = unicodedata.name(char, "").split(" ")[0]
    return _SCRIPT_BY_NAME_PREFIX.get(prefix, "Other")


def script_profile(text: str) -> ScriptProfile:
    counts = Counter(_script_of(char) for char in text if char.isalpha())
    # Japanese mixes kana with kanji; treat all three as one script when kana is present.
    if counts["Hira"] or counts["Kana"]:
        counts["Jpan"] = counts.pop("Hira", 0) + counts.pop("Kana", 0) + counts.pop("Hani", 0)
    counts = Counter({script: n for script, n in counts.items() if n})
    total = sum(counts.values())
    if total == 0:
        return ScriptProfile(dominant=None)
    shares = {script: n / total for script, n in counts.items()}
    return ScriptProfile(dominant=counts.most_common(1)[0][0], shares=shares)
