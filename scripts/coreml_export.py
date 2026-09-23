"""Convert the language-ID head to Core ML and check it answers the same as the ONNX model.

**Runs on macOS only.** `coremltools` can convert elsewhere but cannot run a prediction off a Mac,
and a conversion nobody has run is not a conversion anybody should trust.

`coremltools` no longer converts ONNX — the ONNX route was deprecated in 5.x and removed in 6, and
the accepted sources are now TensorFlow, PyTorch and milinternal. The head was trained in PyTorch
and exported to ONNX by `training/export.py`, so this rebuilds the PyTorch module and converts that.
No checkpoint survives in `results/lid-runs/`, so the weights are read back out of the ONNX file,
whose initializers are named after the state-dict keys they came from (`net.0.weight` and so on).

Reconstruction is the risky step, and it is exactly what the reference catches. The comparison is
against `results/lid-head-reference.json`, generated on the training machine from the ONNX model
whose accuracy is recorded in training/RESULTS.md. It holds the exact input vectors and the logits
they produced, so this feeds identical input rather than embedding text again — an embedder that
normalised differently would otherwise look like a conversion error, and a mis-wired weight would
look like nothing at all.

Two things are checked, and they are not the same thing:

    numerical   how far the logits moved. Small drift is expected from a different runtime.
    decisions   whether the predicted label changed. This is what a user would notice, and a
                conversion that keeps the numbers close while flipping a label has not preserved
                the behaviour that was measured.

Run:  python -m uv run --group train python scripts/coreml_export.py
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

# The architecture the selected run was trained with, from
# results/lid-runs/e5-head-04-20260915-113244/config.json. Dropout is inert once the module is in
# eval mode; it is passed only so the layer indices line up with the state-dict keys.
HIDDEN = 512
DROPOUT = 0.1

# What the ONNX export accepted when this model was produced, recorded in the sidecar as
# onnx_max_abs_diff. Core ML is held to the same bar rather than a looser one invented for it.
TOLERANCE = 1.5e-4


def rebuild_from_onnx(dim: int, n_labels: int):
    """Load the trained weights out of the ONNX file into the module they were trained in."""
    import onnx
    import torch
    from onnx import numpy_helper

    from training.embed_head import MLPHead

    graph = onnx.load(str(ONNX_MODEL)).graph
    state = {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in graph.initializer
    }
    model = MLPHead(dim=dim, hidden=HIDDEN, n_labels=n_labels, dropout=DROPOUT)
    # strict=True: a renamed or missing key raises here rather than silently leaving a layer at
    # its random initialisation, which would convert cleanly and answer nonsense.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> int:
    if platform.system() != "Darwin":
        print("This needs macOS: coremltools cannot run a prediction anywhere else.")
        return 2
    if not REFERENCE.exists():
        print(f"Missing {REFERENCE.relative_to(REPO_ROOT)}. Generate it on a machine with the")
        print("models by running: python -m uv run python scripts/lid_head_reference.py")
        return 2

    import coremltools as ct
    import torch

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    labels: list[str] = reference["labels"]
    dim: int = reference["embedding_dim"]
    cases = reference["cases"]

    print(f"reference: {len(cases)} cases, {len(labels)} labels, {dim}-dim input")
    print(f"generated on: {reference['generated_on']['platform']}")

    torch_model = rebuild_from_onnx(dim, len(labels))
    print(f"rebuilt MLPHead(dim={dim}, hidden={HIDDEN}, n_labels={len(labels)}) from ONNX weights")

    print("\nconverting...")
    example = torch.zeros(1, dim, dtype=torch.float32)
    traced = torch.jit.trace(torch_model, example)
    model = ct.convert(
        traced,
        source="pytorch",
        inputs=[ct.TensorType(name="embedding", shape=(1, dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    model.save(str(OUT_MODEL))
    size_kb = sum(f.stat().st_size for f in OUT_MODEL.rglob("*") if f.is_file()) / 1024
    print(f"wrote {OUT_MODEL.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")

    print("\nchecking against the ONNX reference...")
    worst = 0.0
    flipped: list[tuple[str, str, str]] = []
    for case in cases:
        vector = np.asarray([case["embedding"]], dtype=np.float32)
        out = model.predict({"embedding": vector})
        got = np.asarray(next(iter(out.values())), dtype=np.float32).reshape(-1)
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
