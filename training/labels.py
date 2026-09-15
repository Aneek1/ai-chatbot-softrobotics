from backend.pipeline.languages import CONFUSION_GROUPS

OTHER = "other"

# Spec section 16.2: the specialist's label set. "other" lets it decline.
LABELS: tuple[str, ...] = (
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
    OTHER,
)

# Labels sampled from OpenLID under their own name; every other OpenLID language becomes "other".
OPENLID_LABELS: frozenset[str] = frozenset(
    {
        "eng_Latn",
        "ind_Latn",
        "zsm_Latn",
        "zho_Hans",
        "zho_Hant",
        "jpn_Jpan",
        "kor_Hang",
        "tam_Taml",
        "hin_Deva",
    }
)

# Dakshina language folder -> label of its romanized text.
DAKSHINA_LABELS: dict[str, str] = {"hi": "hin_Latn", "ur": "urd_Latn"}

# The app's confusion groups restricted to labels the specialist can output
# (GlotLID's cmn_Hani and yue_Hani are general-stage labels only).
GROUPS: dict[str, tuple[str, ...]] = {
    name: tuple(label for label in LABELS if label in members) for name, members in CONFUSION_GROUPS.items()
}
GROUP_LABELS: tuple[str, ...] = tuple(label for label in LABELS if any(label in g for g in GROUPS.values()))


def training_label(openlid_language: str) -> str:
    return openlid_language if openlid_language in OPENLID_LABELS else OTHER


def label_index(label: str) -> int:
    return LABELS.index(label)
