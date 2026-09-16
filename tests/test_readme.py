import json
from pathlib import Path

from eval.readme_table import END, START, render
from training.results_table import latest

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_the_results_block_is_what_the_generator_produces_now():
    path = latest("langid-eval-*.json")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    body = README.split(START, 1)[1].split(END, 1)[0]
    assert body.strip() == render(evaluation, f"results/{path.name}").strip()


def test_the_readme_covers_the_three_platforms_the_privacy_limits_and_the_licences():
    for heading in (
        "## Running it",
        "### Windows",
        "### macOS",
        "### Linux and WSL",
        "## Private mode",
        "### What private mode does not do",
        "## Docker",
        "## Results",
        "## Limits",
        "## Licences",
    ):
        assert heading in README, heading
