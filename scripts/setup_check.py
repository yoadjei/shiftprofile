"""Session-zero check: does this machine have everything the fill run needs?

Run this instead of stepping through notebook 00. Same checks, no notebook to
import — which matters because importing a .ipynb is the one step in the runbook
with a silent failure mode: paste the file's JSON into a cell rather than
importing it as a notebook and the kernel tries to execute `null` as Python.

On Kaggle, paste this into one cell of a blank notebook:

    !git clone --depth 1 https://github.com/yoadjei/shiftprofile.git /kaggle/working/shiftprofile
    !pip install -q -e /kaggle/working/shiftprofile
    !python /kaggle/working/shiftprofile/scripts/setup_check.py

Every check that fails prints what to do about it, and the script exits non-zero
so a failure cannot be mistaken for a pass scrolled off the top.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

FAILURES: list[str] = []

# sha256 of fixed_eval_indices(1000) as little-endian int64, first 16 hex digits.
# Pinned here and in tests/test_data_cifar.py: if the index stream ever changes,
# both fail loudly rather than quietly scoring a different set of images.
EXPECTED_INDEX_CHECKSUM = "60c5b821cd131d37"


def _ok(msg: str) -> None:
    print(f"  OK    {msg}")


def _fail(check: str, msg: str, fix: str) -> None:
    print(f"  FAIL  {msg}")
    print(f"        fix: {fix}")
    FAILURES.append(check)


def check_gpu() -> None:
    print("\n[1/6] GPU")
    try:
        import torch
    except ImportError:
        _fail("gpu", "torch is not installed", "the pip install step did not run")
        return

    print(f"  torch {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        _ok(f"{torch.cuda.get_device_name(0)}, {props.total_memory / 1e9:.1f} GB")
    else:
        _fail(
            "gpu",
            "CUDA is not available",
            "Kaggle right-hand panel -> Accelerator -> GPU T4 x2, then restart the session",
        )


def check_clean_data(data_root: Path):
    print("\n[2/6] CIFAR-10 clean test set")
    from shiftprofile.data import load_cifar10_test

    data_root.mkdir(parents=True, exist_ok=True)
    try:
        images, labels = load_cifar10_test(data_root)
    except Exception as exc:
        _fail(
            "clean",
            f"could not load CIFAR-10: {type(exc).__name__}: {exc}",
            "needs Internet: On, which needs phone verification",
        )
        return None

    _ok(f"{images.shape}, labels {labels.shape}, dtype {images.dtype}")
    return images, labels


def _looks_like_cifar10c(directory: Path) -> bool:
    """Could this directory be the CIFAR-10-C Dataset?

    Looser than the library's `holds_cifar10c`, deliberately. That one requires
    labels.npy, because without it the data is unusable. This one also accepts a
    directory of corruption arrays with the labels missing, so the diagnostic can
    point at the Dataset the user clearly meant and name what is absent from it,
    rather than reporting that nothing was found.
    """
    from shiftprofile.data import holds_cifar10c

    try:
        if holds_cifar10c(directory):
            return True
        return any(directory.glob("*.npy")) or any(directory.glob("*/*.npy"))
    except OSError:
        return False


def _diagnose_missing_mount(base: Path = Path("/kaggle/input")) -> str:
    """Say what IS attached, not just what is not.

    The expected path is derived from the Dataset slug, so a Dataset named
    anything other than `cifar-10-c` mounts somewhere else and the check fails
    with a path the user never chose. Reporting only "does not exist" leaves
    them guessing between "I forgot to attach it" and "I named it differently",
    which are very different fixes. Listing the mounts distinguishes the two
    immediately.
    """
    if not base.exists():
        return (
            f"no {base} on this machine, so nothing is mounted. Pass "
            f"--corrupt-root pointing at a local CIFAR-10-C directory."
        )

    try:
        mounts = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError as exc:
        return f"could not read {base}: {exc}"

    if not mounts:
        return (
            "nothing is attached to this notebook at all. Add Input -> your "
            "cifar-10-c Dataset, then re-run."
        )

    candidates = [p for p in mounts if _looks_like_cifar10c(p)]
    listing = ", ".join(p.name for p in mounts)

    if candidates:
        best = candidates[0]
        return (
            f"attached Datasets are: {listing}. {best.name} contains .npy files, "
            f"so it is probably the one -- re-run with "
            f"--corrupt-root {best}"
        )

    return (
        f"attached Datasets are: {listing}. None of them contains .npy files, so "
        f"the CIFAR-10-C Dataset is not attached to this notebook yet. Add Input "
        f"-> select it, then re-run."
    )


def check_corrupt_data(corrupt_root: Path):
    print(f"\n[3/6] CIFAR-10-C under {corrupt_root}")
    from shiftprofile.data.cifar import resolve_cifar10c_dir

    if not corrupt_root.exists():
        _fail(
            "corrupt",
            f"{corrupt_root} does not exist",
            _diagnose_missing_mount(),
        )
        return None

    try:
        resolved = resolve_cifar10c_dir(corrupt_root)
    except FileNotFoundError as exc:
        _fail("corrupt", str(exc), "re-upload the Dataset including labels.npy")
        return None

    layout = "nested in CIFAR-10-C/" if resolved != corrupt_root else "flat"
    found = sorted(p.stem for p in resolved.glob("*.npy") if p.stem != "labels")
    _ok(f"{layout}; {len(found)} corruptions: {', '.join(found) if found else 'NONE'}")

    for needed, label in (
        ({"gaussian_noise", "defocus_blur", "fog"}, "pilot"),
        ({"gaussian_noise", "shot_noise", "defocus_blur", "fog", "jpeg_compression"}, "version A"),
    ):
        missing = sorted(needed - set(found))
        if missing:
            _fail(
                f"corrupt-{label}",
                f"{label} needs {', '.join(missing)}, which are absent",
                "add those .npy files to the cifar-10-c Dataset and re-publish",
            )
        else:
            _ok(f"{label} corruptions all present")

    return resolved


def check_alignment(clean, corrupt_dir: Path) -> None:
    """The check worth running before any GPU time is spent.

    If clean row i and corrupted row i are different images, every paired
    comparison in the study silently compares unrelated things. It is cheap to
    verify and impossible to notice later.
    """
    print("\n[4/6] Clean/corrupted row alignment")
    if clean is None or corrupt_dir is None:
        print("  SKIP  needs both datasets above")
        return

    import numpy as np

    clean_images, clean_labels = clean
    try:
        corrupt = np.load(corrupt_dir / "gaussian_noise.npy", mmap_mode="r")
        corrupt_labels = np.load(corrupt_dir / "labels.npy")
    except FileNotFoundError as exc:
        _fail("alignment", f"missing file: {exc}", "re-upload the Dataset")
        return

    sev1 = np.asarray(corrupt[0:10000])
    sev1_labels = corrupt_labels[0:10000]

    if not np.array_equal(clean_labels, sev1_labels):
        _fail(
            "alignment",
            "clean and corrupted labels DO NOT match - rows are misaligned",
            "the CIFAR-10-C upload is wrong or truncated; do not run the pilot until this passes",
        )
        return

    if np.array_equal(clean_images, sev1):
        _fail(
            "alignment",
            "clean and corrupted images are identical - the corruption is not applied",
            "the .npy is not real CIFAR-10-C data",
        )
        return

    _ok("labels match and images differ, for all 10000 rows")


def check_indices() -> None:
    """Verify the evaluation set is the same one every other machine will use.

    The digest is taken over a canonical byte layout, little-endian int64, not
    over whatever the platform's default integer happens to be. NumPy's default
    int is 32-bit on Windows and 64-bit on Linux, so hashing the raw buffer made
    identical indices digest differently on a laptop and on a Kaggle runner --
    a check meant to catch a real mismatch reporting a fake one.
    """
    print("\n[5/6] Evaluation indices")
    import numpy as np

    from shiftprofile.data import fixed_eval_indices

    idx = fixed_eval_indices(1000)
    canonical = np.asarray(idx, dtype="<i8").tobytes()
    checksum = hashlib.sha256(canonical).hexdigest()[:16]

    print(f"  n=1000, range [{idx.min()}, {idx.max()}], first five {idx[:5].tolist()}")
    if checksum == EXPECTED_INDEX_CHECKSUM:
        _ok(f"checksum {checksum} matches the committed value")
    else:
        _fail(
            "indices",
            f"checksum {checksum} != expected {EXPECTED_INDEX_CHECKSUM}",
            "the evaluation set differs from every other run - stop and report this, "
            "because results computed here would not be comparable",
        )


def check_cache(cache_dir: Path) -> None:
    print("\n[6/6] Cache directory")
    from shiftprofile.cache import ArtifactCache

    cache_dir.mkdir(parents=True, exist_ok=True)
    ArtifactCache(write_root=cache_dir)
    _ok(f"{cache_dir} ready")
    print("        After notebook 01, Save Version to persist this as a Dataset.")
    print("        Skip that and the session's work is gone.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument(
        "--corrupt-root",
        type=Path,
        default=None,
        help="CIFAR-10-C root. Discovered from the attached Datasets if omitted.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/cache"))
    args = parser.parse_args(argv)

    print("=" * 66)
    print("shiftprofile session check")
    print("=" * 66)

    check_gpu()
    clean = check_clean_data(args.data_root)

    # Resolve the corruption root after the package is importable, so discovery
    # can use the library's own notion of what a CIFAR-10-C directory looks like.
    corrupt_root = args.corrupt_root
    if corrupt_root is None:
        from shiftprofile.data import discover_cifar10c_root

        conventional = Path("/kaggle/input/cifar-10-c")
        discovered = None if conventional.exists() else discover_cifar10c_root()
        corrupt_root = discovered or conventional
        if discovered is not None:
            print(f"\n  note: using discovered CIFAR-10-C at {discovered}")

    corrupt_dir = check_corrupt_data(corrupt_root)
    check_alignment(clean, corrupt_dir)
    check_indices()
    check_cache(args.cache_dir)

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"NOT READY - {len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        print("Fix these before running notebook 01. Each failure prints its fix above.")
        print("=" * 66)
        return 1

    print("READY - every check passed. Proceed to notebook 01.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
