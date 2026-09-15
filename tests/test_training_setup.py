import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_training_dependencies_stay_out_of_the_app_install():
    train = PYPROJECT["dependency-groups"]["train"]
    assert "torch==2.14.0" in train
    assert any(dep.startswith("transformers") for dep in train)
    assert not any(
        dep.startswith(("torch", "transformers", "onnx>", "onnxscript"))
        for dep in PYPROJECT["project"]["dependencies"]
    )


def test_torch_comes_from_the_cuda_index_on_windows_and_linux():
    assert PYPROJECT["tool"]["uv"]["sources"]["torch"] == [
        {
            "index": "pytorch-cu130",
            "group": "train",
            "marker": "sys_platform == 'win32' or sys_platform == 'linux'",
        }
    ]
    indexes = {index["name"]: index for index in PYPROJECT["tool"]["uv"]["index"]}
    assert indexes["pytorch-cu130"] == {
        "name": "pytorch-cu130",
        "url": "https://download.pytorch.org/whl/cu130",
        "explicit": True,
    }


def test_downloads_models_and_runs_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in ("models/", "runs/", "data/cache/"):
        assert entry in ignored
