SUPPORTED: dict[str, str] = {
    "eng_Latn": "English",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "tam_Taml": "Tamil",
    "hin_Deva": "Hindi",
    "hin_Latn": "Hindi (romanized)",
}

CONFUSION_GROUPS: dict[str, frozenset[str]] = {
    "malay_indonesian": frozenset({"ind_Latn", "zsm_Latn"}),
    "han": frozenset({"cmn_Hani", "yue_Hani", "zho_Hans", "zho_Hant"}),
    "romanized_hindi": frozenset({"hin_Latn", "urd_Latn", "eng_Latn"}),
}


def group_of(label: str) -> str | None:
    for name, members in CONFUSION_GROUPS.items():
        if label in members:
            return name
    return None


def display_name(label: str) -> str:
    if label == "und":
        return "Undetermined"
    return SUPPORTED.get(label, label)


def script_of_label(label: str) -> str:
    return label.split("_", 1)[1] if "_" in label else ""
