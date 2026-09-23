"""Record what the ONNX language-ID head answers, so another runtime can be checked against it.

The accuracy in training/RESULTS.md was measured with this ONNX model. A conversion to some other
runtime — Core ML, for instance — only carries that accuracy over if it gives the same answers, and
`coremltools` cannot run predictions anywhere but macOS. So the reference is generated here, from
the model that was actually measured, and committed. The other runtime then checks itself against a
fixed artifact rather than against whatever happens to be installed beside it.

Writes results/lid-head-reference.json:

    embeddings  the exact float32 input vectors, so the comparison feeds identical input
    logits      what the ONNX head returned for each
    predicted   the argmax label for each, which is what a user would actually see

Run:  python -m uv run python scripts/lid_head_reference.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
# The other scripts here are standalone; this one reuses the application's own embedder so the
# reference is produced by exactly the code path the measured accuracy came from, rather than a
# reimplementation that might normalise differently.
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.pipeline.embeddings import E5Embedder  # noqa: E402

OUT = REPO_ROOT / "results" / "lid-head-reference.json"

# Two sentences per supported language, plus the confusable pairs the specialist exists to separate.
# Short and long, because identification from a few words is a different problem from a sentence.
SENTENCES: list[tuple[str, str]] = [
    ("eng_Latn", "How do you cast a silicone pneumatic actuator?"),
    ("eng_Latn", "soft robot gripper"),
    ("ind_Latn", "Bagaimana cara membuat aktuator pneumatik lunak?"),
    ("ind_Latn", "pencetakan silikon"),
    ("zsm_Latn", "Bagaimanakah cara membuat penggerak pneumatik lembut?"),
    ("zsm_Latn", "pengacuan silikon"),
    ("zho_Hans", "如何制作软气动执行器？"),
    ("zho_Hans", "硅胶浇注"),
    ("zho_Hant", "如何製作軟氣動執行器？"),
    ("zho_Hant", "矽膠澆鑄"),
    ("jpn_Jpan", "柔らかい空気圧アクチュエータの作り方を教えてください。"),
    ("jpn_Jpan", "シリコーン成形"),
    ("kor_Hang", "부드러운 공압 액추에이터는 어떻게 만드나요?"),
    ("kor_Hang", "실리콘 주조"),
    ("tam_Taml", "மென்மையான காற்றழுத்த இயக்கியை எப்படி உருவாக்குவது?"),
    ("tam_Taml", "சிலிக்கான் வார்ப்பு"),
    ("hin_Deva", "नरम वायवीय एक्चुएटर कैसे बनाएं?"),
    ("hin_Deva", "सिलिकॉन ढलाई"),
    ("hin_Latn", "narm vayaviya actuator kaise banaye?"),
    ("hin_Latn", "silicone dhalai kaise karte hain"),
    ("urd_Latn", "narm hawai actuator kaise banate hain?"),
    ("urd_Latn", "silicone dhalai ka tareeqa"),
]


def main() -> None:
    settings = Settings()
    meta = json.loads(
        (settings.models_dir / "lid-specialist-e5-head.json").read_text(encoding="utf-8")
    )
    labels: list[str] = meta["labels"]

    embedder = E5Embedder(settings.models_dir / settings.e5_dir)
    session = ort.InferenceSession(
        str(settings.models_dir / "lid-specialist-e5-head.onnx"),
        providers=["CPUExecutionProvider"],
    )

    texts = [text for _, text in SENTENCES]
    embeddings = np.asarray(embedder.embed_queries(texts), dtype=np.float32)
    # Round BEFORE running the model, not after. The file stores rounded vectors and whoever
    # checks against it feeds those exact rounded vectors back in, so the logits recorded here
    # have to be the ones those rounded inputs produce. Rounding afterwards would bake a
    # difference into the reference and charge it to the runtime being tested.
    embeddings = np.round(embeddings, 7).astype(np.float32)
    logits = session.run(["logits"], {"embedding": embeddings})[0]
    predicted = [labels[int(row.argmax())] for row in logits]

    agree = sum(p == expected for p, (expected, _) in zip(predicted, SENTENCES, strict=True))
    print(f"{agree}/{len(SENTENCES)} predictions match the language the sentence was written in")
    for (expected, text), got in zip(SENTENCES, predicted, strict=True):
        if got != expected:
            print(f"  differs: expected {expected}, got {got} — {text[:48]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "model": "lid-specialist-e5-head.onnx",
                "run_id": meta.get("run_id"),
                "labels": labels,
                "embedding_dim": meta["embedding_dim"],
                "prefix": meta["prefix"],
                "generated_on": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "onnxruntime": ort.__version__,
                },
                "cases": [
                    {
                        "text": text,
                        "written_in": expected,
                        "predicted": got,
                        "embedding": [float(x) for x in vector],
                        "logits": [round(float(x), 7) for x in row],
                    }
                    for (expected, text), got, vector, row in zip(
                        SENTENCES, predicted, embeddings, logits, strict=True
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
