"""Tests for scripts/run_fill.py, the documented fill entry point.

It exists because `python -m shiftprofile.fill` fails on the Kaggle kernel with
ModuleNotFoundError even with PYTHONPATH set, while invoking a script by path
works. That makes this wrapper the path a real run actually takes, so its
argument handling is worth pinning: a default silently overriding an explicit
flag would send a run at the wrong config or the wrong budget and still look
like it worked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_fill.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_fill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists():
    assert SCRIPT.exists(), "the documented entry point must be in the repo"


def test_delegates_to_the_real_fill_main(runner):
    """The wrapper must not reimplement the CLI, only reach it."""
    from shiftprofile.fill import main as fill_main

    assert runner.main is fill_main


def test_repo_root_is_derived_from_the_script_location(runner):
    """Not hardcoded to /kaggle/working, so a local run works unchanged."""
    assert runner.REPO_ROOT == REPO_ROOT


def test_defaults_are_injected_when_absent(runner):
    argv = runner.build_argv([])
    for flag, _ in runner.DEFAULTS:
        assert flag in argv, f"{flag} default was not applied"


def test_default_config_points_at_a_config_that_exists(runner):
    argv = runner.build_argv([])
    config = Path(argv[argv.index("--config") + 1])
    assert config.exists(), f"default config {config} is missing"


def test_explicit_flags_win_over_defaults(runner):
    """An explicit value must not be shadowed or duplicated.

    argparse takes the last occurrence, so appending a default after a
    user-supplied flag would silently override it. That would be invisible: the
    run would proceed against the wrong budget or config and report success.
    """
    for flag, default in runner.DEFAULTS:
        argv = runner.build_argv([flag, "EXPLICIT"])
        assert argv.count(flag) == 1, f"{flag} appears twice; argparse takes the last"
        assert argv[argv.index(flag) + 1] == "EXPLICIT"
        assert default not in argv or default == "EXPLICIT"


def test_missing_cache_read_is_omitted_rather_than_passed(runner, monkeypatch):
    """A first session has no cache to resume from, and that is not an error.

    Passing a path that does not exist would make a typo in the Dataset slug
    indistinguishable from a legitimate first run.
    """
    monkeypatch.setattr(runner.Path, "exists", lambda self: False)
    assert "--cache-read" not in runner.build_argv([])


def test_present_cache_read_is_passed(runner, monkeypatch):
    monkeypatch.setattr(runner.Path, "exists", lambda self: True)
    argv = runner.build_argv([])
    assert argv[argv.index("--cache-read") + 1] == runner.OPTIONAL_READ_ROOT


def test_explicit_cache_read_is_respected(runner, monkeypatch):
    monkeypatch.setattr(runner.Path, "exists", lambda self: True)
    argv = runner.build_argv(["--cache-read", "/somewhere/else"])
    assert argv.count("--cache-read") == 1
    assert argv[argv.index("--cache-read") + 1] == "/somewhere/else"
