import numpy as np
import pytest

from eval.metrics import reprojection_error


def test_exact_match():
    result = reprojection_error([[10, 20], [30, 40]], [[10, 20], [30, 40]])
    assert result.count == 2
    assert result.excluded_count == 0
    assert result.mean_px == result.rms_px == result.max_px == 0


def test_known_euclidean_distances():
    result = reprojection_error([[0, 0], [0, 0]], [[3, 4], [0, 0]])
    assert result.mean_px == 2.5
    assert result.rms_px == pytest.approx(np.sqrt(12.5))
    assert result.max_px == 5


def test_multicamera_mask_excludes_missing_points():
    observed = np.zeros((2, 3, 4, 2))
    projected = np.full_like(observed, 2)
    mask = np.ones((2, 3, 4), dtype=bool)
    mask[1, 2] = False
    observed[1, 2] = np.nan
    projected[1, 2] = np.inf
    result = reprojection_error(observed, projected, mask=mask)
    assert result.count == 20
    assert result.excluded_count == 4
    assert result.rms_px == pytest.approx(np.sqrt(8))


@pytest.mark.parametrize("observed,projected,mask", [
    ([[0, 0]], [[0, 0], [1, 1]], None),
    ([0, 0], [0, 0], None),
    ([[0, 0, 0]], [[0, 0, 0]], None),
    (np.empty((0, 2)), np.empty((0, 2)), None),
    ([[0, 0]], [[0, 0]], [False]),
    ([[0, 0]], [[0, 0]], [1]),
    ([[0, 0]], [[0, 0]], True),
    ([[0, 0]], [[0, 0]], [[True]]),
    ([[np.nan, 0]], [[0, 0]], None),
    ([[0, 0]], [[np.inf, 0]], None),
    ([[1e308, 0]], [[-1e308, 0]], None),
])
def test_invalid_inputs_fail_explicitly(observed, projected, mask):
    with pytest.raises(ValueError):
        reprojection_error(observed, projected, mask=mask)


def test_large_finite_distances_do_not_overflow_rms():
    result = reprojection_error([[0, 0]], [[1e200, 0]])
    assert result.rms_px == pytest.approx(1e200)
