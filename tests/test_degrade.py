import numpy as np
from m3.data.degrade import corrupt, resolution, CORRUPTIONS


def _imgs(): return np.random.default_rng(0).integers(0, 255, (20, 48, 48), dtype=np.uint8)


def test_shape_identity_determinism():
    x = _imgs()
    for c in CORRUPTIONS:
        y = corrupt(x, c, 3); assert y.shape == x.shape and y.dtype == np.uint8
        assert np.array_equal(y, corrupt(x, c, 3)); assert np.array_equal(corrupt(x, c, 0), x)
    assert np.array_equal(resolution(x, 48), x)


def test_monotone():
    x = _imgs()
    for c in ["gauss_noise", "motion_blur", "defocus_blur"]:
        d = [np.abs(corrupt(x, c, s).astype(float) - x).mean() for s in range(1, 6)]
        assert all(a <= b + 1e-9 for a, b in zip(d, d[1:])), c
