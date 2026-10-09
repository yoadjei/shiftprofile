"""Tests for the mechanism report's two derived quantities.

Both are second copies of a fact that lives somewhere else -- the roll offsets
belong to `explain.roll_attribution`, and the mask belongs to
`curves._get_removal_mask` -- so both are pinned against their originals rather
than against hand-computed expectations. Every cache-key defect in this project
was a second copy of one fact drifting from the first.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mechanism_report import (  # noqa: E402
    MASK_FRACTION,
    mask_geometry,
    roll_displacements,
)

from shiftprofile.curves import _get_removal_mask  # noqa: E402
from shiftprofile.explain import roll_attribution  # noqa: E402


class TestRollDisplacements:
    """The displacement the report bins by must be the one that was applied."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_it_matches_what_roll_attribution_actually_did(self, seed):
        """Measured from the roll itself, not from a reimplementation of it.

        One hot pixel at the origin, so wherever it lands IS the shift that was
        applied, and the test compares the recovered distance against the
        observed one. If `roll_attribution` changed its draw -- a different RNG
        call, a different offset range, a shared offset instead of a per-image
        one -- this fails, which is the point: TEST 1 of the report bins images
        by this number, and a wrong number would silently reorder the bins and
        flatten the very trend the test exists to detect.
        """
        n = 64
        attribution = np.zeros((n, 32, 32), dtype=np.float32)
        attribution[:, 0, 0] = 1.0

        rolled = roll_attribution(attribution, seed=seed)
        landed = np.array([np.unravel_index(np.argmax(r), r.shape) for r in rolled])
        ys, xs = landed[:, 0], landed[:, 1]
        observed = np.hypot(np.minimum(ys, 32 - ys), np.minimum(xs, 32 - xs))

        np.testing.assert_allclose(
            roll_displacements(n, seed=seed), observed,
            err_msg="the recovered displacement is not the one that was applied",
        )

    def test_the_distance_wraps_rather_than_running_off_the_edge(self):
        """`np.roll` wraps, so a shift of 30 on a 32-wide axis is a displacement
        of 2. Treating it as 30 would put the images that moved LEAST in the
        bin for the images that moved MOST, which inverts TEST 1's trend."""
        assert roll_displacements(2000, seed=0).max() <= np.hypot(16, 16)

    def test_no_image_is_left_where_it_started(self):
        """Offsets are drawn from [1, h*w), so the control is never silently the
        attribution itself for some fraction of the images."""
        assert roll_displacements(2000, seed=0).min() > 0

    def test_it_is_reproducible(self):
        a = roll_displacements(500, seed=3)
        b = roll_displacements(500, seed=3)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, roll_displacements(500, seed=4))


class TestMaskGeometry:
    """The geometry measured must be the geometry the curve was computed from."""

    def test_it_measures_the_same_mask_the_curves_use(self):
        """Pinned against `_get_removal_mask`, which is what actually selected
        the pixels. Measuring a different mask -- ranking by signed value rather
        than magnitude, say, or a different k -- would describe a mask no curve
        was ever computed from, and TEST 2 would be relating faithfulness to the
        geometry of something that did not happen."""
        rng = np.random.RandomState(0)
        attribution = rng.randn(8, 32, 32).astype(np.float32)

        centre, spread = mask_geometry(attribution, fraction=MASK_FRACTION)

        mask = _get_removal_mask(
            torch.from_numpy(attribution), MASK_FRACTION
        ).numpy()
        for i in range(len(attribution)):
            ys, xs = np.nonzero(mask[i])
            cy, cx = ys.mean(), xs.mean()
            assert centre[i] == pytest.approx(np.hypot(cy - 15.5, cx - 15.5), abs=1e-4)
            assert spread[i] == pytest.approx(
                np.hypot(ys - cy, xs - cx).mean(), abs=1e-4
            )

    def test_a_centred_blob_reads_as_centred_and_an_offset_one_does_not(self):
        # 8x8 = 64 pixels, comfortably more than the 52 the mask takes. A
        # smaller blob would leave the mask to pick its remainder from pixels
        # tied at zero, and the centroid would then describe the tie-break
        # rather than the blob.
        attribution = np.zeros((2, 32, 32), dtype=np.float32)
        attribution[0, 12:20, 12:20] = 1.0  # middle of the frame
        attribution[1, 0:8, 0:8] = 1.0      # corner

        centre, _ = mask_geometry(attribution)

        assert centre[0] < 1.0, "a blob over the frame centre must read as centred"
        assert centre[1] > 15.0, "a blob in the corner must read as far off centre"

    def test_a_contiguous_mask_has_lower_spread_than_a_scattered_one(self):
        """`spread` is the axis the random_lowres ladder varies, so it has to
        order a blob below scattered pixels or the ladder's own rungs would not
        come out in order."""
        k = int(np.ceil(MASK_FRACTION * 32 * 32))
        blob = np.zeros((1, 32, 32), dtype=np.float32)
        flat = blob.reshape(1, -1)
        flat[0, :k] = 1.0  # k consecutive pixels: two contiguous rows

        scattered = np.zeros((1, 32, 32), dtype=np.float32)
        rng = np.random.RandomState(0)
        scattered.reshape(1, -1)[0, rng.choice(1024, k, replace=False)] = 1.0

        _, blob_spread = mask_geometry(blob)
        _, scattered_spread = mask_geometry(scattered)

        assert blob_spread[0] < scattered_spread[0]

    def test_magnitude_ranking_so_strong_negative_evidence_is_masked(self):
        """Ranking is by |attribution|, matching `_get_removal_mask`. A signed
        ranking would exclude the most negative pixels, and for Integrated
        Gradients those are evidence AGAINST the class -- a different mask, and
        one no curve in the cache was built from."""
        attribution = np.zeros((1, 32, 32), dtype=np.float32)
        attribution[0, 0:2, :] = -5.0   # strongest magnitude, negative
        attribution[0, 30:32, :] = 1.0  # weaker, positive

        centre, _ = mask_geometry(attribution, fraction=2 * 32 / 1024)

        # The top rows win on magnitude, so the centroid sits near row 0.5.
        assert centre[0] == pytest.approx(np.hypot(0.5 - 15.5, 15.5 - 15.5), abs=0.5)
