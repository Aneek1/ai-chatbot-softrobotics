"""Convert the language-ID head to Core ML and check it answers the same as the ONNX model.

**Runs on macOS only.** `coremltools` can convert elsewhere but cannot run a prediction off a Mac,
and a conversion nobody has run is not a conversion anybody should trust.

The comparison is against `results/lid-head-reference.json`, generated on the training machine from
the ONNX model whose accuracy is recorded in training/RESULTS.md. That file holds the exact input
vectors and the logits they produced, so this script feeds identical input rather than embedding
text again — an embedder that normalised differently would otherwise look like a conversion error.

Two things are checked, and they are not the same thing:

    numerical   how far the logits moved. Small drift is expected from a different runtime.
    decisions   whether the predicted label changed. This is what a user would notice, and a
                conversion that keeps the numbers close while flipping a label has not preserved
                the behaviour that was measured.

Run:  python -m uv run python scripts/coreml_export.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REFERENCE = REPO_ROOT / "results" / "lid-head-reference.json"
ONNX_MODEL = REPO_ROOT / "models" / "lid-specialist-e5-head.onnx"
SIDECAR = REPO_ROOT / "models" / "lid-specialist-e5-head.json"
OUT_MODEL = REPO_ROOT / "models" / "lid-specialist-e5-head.mlpackage"

# What the ONNX export already accepted when this model was produced, recorded in the sidecar as
# onnx_max_abs_diff. Holding Core ML to the same bar rather than inventing a looser one.
TOLERANCE = 1.5e-4


def main() -> int:
    if platform.system() != "Darwin":
        print("This needs macOS: coremltools cannot run a prediction anywhere else.")
        return 2
    if not REFERENCE.exists():
        print(f"Missing {REFERENCE.relative_to(REPO_ROOT)}. Generate it on a machine with the")
        print("models by running: python -m uv run python scripts/lid_head_reference.py")
        return 2

    import coremltools as ct

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    labels: list[str] = reference["labels"]
    dim: int = reference["embedding_dim"]
    cases = reference["cases"]

    print(f"reference: {len(cases)} cases, {len(labels)} labels, {dim}-dim input")
    print(f"generated on: {reference['generated_on']['platform']}")

    # Convert. The head takes one named input, "embedding", shaped (batch, 384), and returns
    # "logits" shaped (batch, 12). A fixed batch of 1 is what the application actually asks for
    # and lets Core ML specialise; change this only if you intend to batch.
    print("\nconverting...")
    model = ct.convert(
        str(ONNX_MODEL),
        inputs=[ct.TensorType(name="embedding", shape=(1, dim), dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    model.save(str(OUT_MODEL))
    size_kb = sum(f.stat().st_size for f in OUT_MODEL.rglob("*") if f.is_file()) / 1024
    print(f"wrote {OUT_MODEL.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")

    # Compare against the reference, feeding the exact vectors it recorded.
    print("\nchecking against the ONNX reference...")
    worst = 0.0
    flipped: list[tuple[str, str, str]] = []
    for case in cases:
        vector = np.asarray([case["embedding"]], dtype=np.float32)
        got = model.predict({"embedding": vector})["logits"]
        got = np.asarray(got, dtype=np.float32).reshape(-1)
        expected = np.asarray(case["logits"], dtype=np.float32)

        worst = max(worst, float(np.max(np.abs(got - expected))))
        label = labels[int(got.argmax())]
        if label != case["predicted"]:
            flipped.append((case["text"], case["predicted"], label))

    print(f"largest logit difference: {worst:.3e}   (ONNX export accepted {TOLERANCE:.1e})")
    if flipped:
        print(f"\n{len(flipped)} prediction(s) CHANGED — the conversion did not preserve behaviour:")
        for text, was, now in flipped:
            print(f"  {was} -> {now}   {text[:52]}")
    else:
        print(f"all {len(cases)} predictions unchanged")

    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    sidecar["coreml_max_abs_diff"] = worst
    sidecar["coreml_predictions_changed"] = len(flipped)
    SIDECAR.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"recorded both figures in {SIDECAR.relative_to(REPO_ROOT)}")

    if flipped:
        print("\nSTOP. A model that answers differently no longer has the accuracy that was")
        print("measured for it. Do not benchmark this or ship it until you know why.")
        return 1
    if worst > TOLERANCE:
        print(f"\nDrift of {worst:.3e} is larger than the ONNX export accepted. No prediction")
        print("changed, so it may be harmless — but say so deliberately rather than by omission.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
