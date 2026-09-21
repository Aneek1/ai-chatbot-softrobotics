from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


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
