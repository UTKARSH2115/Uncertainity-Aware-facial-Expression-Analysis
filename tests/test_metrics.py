import numpy as np
from scipy.special import softmax
from m3.calib.metrics import ece, metrics
from m3.calib.temperature import fit_T
from m3.uncertainty.selective import aurc, oracle_aurc


def test_ece_calibrated():
    rng = np.random.default_rng(0); q = softmax(rng.normal(size=(100000, 5)) * 2, axis=1)
    y = np.array([rng.choice(5, p=r) for r in q[:20000]]); q = q[:20000]
    assert ece(q.max(1), (q.argmax(1) == y).astype(float))[0] < 0.02


def test_ts_recovers_T():
    rng = np.random.default_rng(1); z = rng.normal(size=(40000, 6)) * 3; q = softmax(z / 2, axis=1)
    cum = q.cumsum(1); y = (rng.random((len(q), 1)) > cum).sum(1)
    assert abs(fit_T(z, y) - 2) / 2 < 0.05


def test_softnll_equals_ce():
    rng = np.random.default_rng(2); q = softmax(rng.normal(size=(500, 8)), axis=1); y = rng.integers(0, 8, 500); p = np.eye(8)[y]
    m = metrics(q, p, y, ranking=False); assert abs(m["soft_nll"] - m["nll"]) < 1e-9


def test_aurc():
    rng = np.random.default_rng(3); n = 5000; err = rng.random(n) < 0.3
    assert abs(aurc(1 - err + 0.0, err) - oracle_aurc(err)) < 5e-3
    assert abs(aurc(rng.random(n), err) - err.mean()) < 0.02
