"""Tiny in-memory archives shaped like FLORES-200 and Dakshina, for tests only."""

import hashlib
import io
import tarfile
from pathlib import Path


def tar_bytes(files: dict[str, str], compression: str = "") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=f"w:{compression}" if compression else "w") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FLORES_FILES = {
    "./flores200_dataset/README": "test fixture",
    "./flores200_dataset/dev/eng_Latn.dev": "Dev sentence one.\nDev sentence two.\n",
    "./flores200_dataset/dev/zsm_Latn.dev": "Ayat dev satu.\nAyat dev dua.\n",
    "./flores200_dataset/devtest/eng_Latn.devtest": "Devtest sentence one.\nDevtest sentence two.\n",
    "./flores200_dataset/devtest/zsm_Latn.devtest": "Ayat ujian satu.\nAyat ujian dua.\n",
    "./flores200_dataset/devtest/zho_Hant.devtest": "如何製作軟氣動執行器\n",
    "./flores200_dataset/devtest/fra_Latn.devtest": "Une phrase de test.\n",
}


def write_flores_archive(path: Path) -> Path:
    path.write_bytes(tar_bytes(FLORES_FILES, "gz"))
    return path


def dakshina_files() -> dict[str, str]:
    files = {}
    for lang, sentence, word in (("hi", "aap kaise hain", "kitaab"), ("ur", "aap kaisay hain", "kitab")):
        root = f"dakshina_dataset_v1.0/{lang}"
        files[f"{root}/romanized/{lang}.romanized.rejoined.dev.roman.txt"] = (
            f"{sentence} dev\n\n{sentence} dev two\n"
        )
        files[f"{root}/romanized/{lang}.romanized.rejoined.test.roman.txt"] = f"{sentence} test\n"
        files[f"{root}/romanized/{lang}.romanized.rejoined.dev.native.txt"] = "native script, not needed\n"
        files[f"{root}/lexicons/{lang}.translit.sampled.train.tsv"] = f"n1\t{word}\t3\nn2\t{word}on\t1\n"
        files[f"{root}/lexicons/{lang}.translit.sampled.dev.tsv"] = f"n3\t{word}dev\t2\n"
        files[f"{root}/lexicons/{lang}.translit.sampled.test.tsv"] = f"n4\t{word}test\t2\n"
    files["dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"] = "n\tboi\t1\n"
    return files
