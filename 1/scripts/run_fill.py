"""Run the fill by file path, with Kaggle-shaped defaults.

Why this exists rather than `python -m shiftprofile.fill`: on the Kaggle kernel,
invoking a script by path imports shiftprofile successfully -- scripts/setup_check.py
does it -- while `python -m shiftprofile.fill` reports

    ModuleNotFoundError: No module named 'shiftprofile'

even with PYTHONPATH pointing at the repo root. `pip install -e` registers against
one interpreter and the one a notebook `!` line resolves to need not be the same,
and `-m` is the invocation that depends on getting that right. Rather than keep
guessing at an interpreter's search path, this wrapper puts the repo root on
sys.path itself, from its own location, and calls the same main().

It also fills in the Kaggle paths, so the whole run is one short line with no shell
continuations to get mangled:

    !python /kaggle/working/shiftprofile/scripts/run_fill.py

Any flag passed explicitly wins over the default, so the same script drives a local
CPU run:

    python scripts/run_fill.py --config configs/pilot.yaml --cache-write ./cache \\
        --data-root ./data --corrupt-root ./data --budget-minutes 10
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiftprofile.data import (  # noqa: E402
    discover_cifar10c_root,
    resolve_cifar10_root,
)
from shiftprofile.fill import main  # noqa: E402  (needs sys.path set first)

CONVENTIONAL_CORRUPT_ROOT = "/kaggle/input/cifar-10-c"

# Defaults applied only when the flag is absent from the command line.
DEFAULTS: list[tuple[str, str]] = [
    ("--config", str(REPO_ROOT / "configs" / "pilot.yaml")),
    ("--budget-minutes", "600"),
    ("--cache-write", "/kaggle/working/cache"),
]

# Added only if it exists. A read root that is not there is not an error -- the
# first session has no cache to resume from -- but passing a phantom path would
# make a real typo indistinguishable from a first run.
OPTIONAL_READ_ROOT = "/kaggle/input/shiftprofile-cache"


def resolve_corrupt_root() -> str:
    """Where CIFAR-10-C is, without the user having to know the Dataset slug.

    Prefers the conventional path, then searches the attached Datasets for one
    that actually holds the arrays. Falls back to the conventional path so the
    error message names something recognisable rather than nothing at all.
    """
    # Returned verbatim rather than round-tripped through Path, which would
    # rewrite the separators on Windows and make a local run's output differ
    # from the path that was actually configured.
    if Path(CONVENTIONAL_CORRUPT_ROOT).exists():
        return CONVENTIONAL_CORRUPT_ROOT

    found = discover_cifar10c_root()
    if found is not None:
        return str(found)

    return CONVENTIONAL_CORRUPT_ROOT


def build_argv(argv: list[str]) -> list[str]:
    """Fill in defaults for flags the caller did not pass."""
    out = list(argv)
    for flag, value in DEFAULTS:
        if flag not in out:
            out += [flag, value]
    if "--corrupt-root" not in out:
        out += ["--corrupt-root", resolve_corrupt_root()]
    if "--data-root" not in out:
        out += ["--data-root", resolve_cifar10_root()]
    if "--cache-read" not in out and Path(OPTIONAL_READ_ROOT).exists():
        out += ["--cache-read", OPTIONAL_READ_ROOT]
    return out


if __name__ == "__main__":
    resolved = build_argv(sys.argv[1:])
    print(f"repo        {REPO_ROOT}")
    if "--cache-read" not in resolved:
        print(f"cache read  (none: {OPTIONAL_READ_ROOT} is not attached)")
    sys.exit(main(resolved))
