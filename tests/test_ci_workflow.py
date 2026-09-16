from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(TEXT)
JOBS = WORKFLOW["jobs"]
# YAML 1.1 reads a bare `on:` key as the boolean true, which is how GitHub spells it.
TRIGGERS = WORKFLOW.get("on", WORKFLOW.get(True))


def run_commands(job: dict) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def test_the_backend_suite_runs_on_all_three_platforms():
    matrix = JOBS["backend"]["strategy"]["matrix"]["os"]
    assert matrix == ["ubuntu-latest", "windows-latest", "macos-latest"]
    assert JOBS["backend"]["strategy"]["fail-fast"] is False


def test_the_backend_job_pins_python_and_installs_the_locked_dependencies():
    uses = {step["uses"]: step.get("with", {}) for step in JOBS["backend"]["steps"] if "uses" in step}
    setup = next(config for name, config in uses.items() if name.startswith("astral-sh/setup-uv"))
    assert setup["python-version"] == "3.12"
    assert setup["version"] == "0.12.6"
    assert "uv sync --frozen" in run_commands(JOBS["backend"])


def test_ruff_and_pytest_run_on_every_platform():
    commands = run_commands(JOBS["backend"])
    assert "uv run ruff check ." in commands
    assert "uv run pytest -q" in commands


def test_ci_needs_no_model_file_and_no_secret():
    assert "secrets." not in TEXT
    assert "download_models" not in TEXT
    assert "HF_TOKEN" not in TEXT


def test_the_frontend_job_lints_type_checks_tests_builds_and_checks_the_bundle():
    commands = run_commands(JOBS["frontend"])
    assert commands == [
        "npm ci",
        "npm run lint",
        "npm run typecheck",
        "npm test",
        "npm run build",
        "npm run check:bundle",
    ]


def test_the_workflow_runs_on_a_push_and_on_a_pull_request():
    assert "push" in TRIGGERS
    assert "pull_request" in TRIGGERS
