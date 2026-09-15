from scripts.download_models import E5_FILES, E5_TORCH_FILES, merge_manifest

E5 = {"repo": "intfloat/multilingual-e5-small", "revision": "r1", "files": {"onnx/model.onnx": {"bytes": 1}}}
GLOTLID = {"repo": "cis-lmu/glotlid", "revision": "g1", "files": {"model.bin": {"bytes": 2}}}


def test_torch_weights_are_a_separate_opt_in():
    assert E5_TORCH_FILES == ["model.safetensors"]
    assert "model.safetensors" not in E5_FILES


def test_earlier_models_are_kept():
    existing = {"downloaded_at": "old", "models": [E5, GLOTLID]}
    torch_entry = {**E5, "files": {"model.safetensors": {"bytes": 3}}}
    manifest = merge_manifest(existing, [torch_entry], "new")
    assert manifest["downloaded_at"] == "new"
    assert [m["repo"] for m in manifest["models"]] == ["intfloat/multilingual-e5-small", "cis-lmu/glotlid"]
    assert set(manifest["models"][0]["files"]) == {"onnx/model.onnx", "model.safetensors"}


def test_new_revision_replaces_old_files():
    manifest = merge_manifest(
        {"models": [E5]}, [{**E5, "revision": "r2", "files": {"config.json": {"bytes": 4}}}], "t"
    )
    assert manifest["models"][0]["revision"] == "r2"
    assert set(manifest["models"][0]["files"]) == {"config.json"}


def test_first_download_without_manifest():
    assert merge_manifest(None, [E5], "t") == {"downloaded_at": "t", "models": [E5]}
