import numpy as np
from m3.quantification import methods as Q


def _sim(pi, M, n, rng):
    y = rng.choice(8, n, p=pi); yh = np.array([rng.choice(8, p=M[:, k]) for k in y]); q = np.eye(8)[yh]; return q, y


def test_acc_recovers():
    rng = np.random.default_rng(0); M = np.eye(8) * 0.7 + 0.3 / 8; M /= M.sum(0)
    pi = rng.dirichlet(np.ones(8)); q, _ = _sim(pi, M, 10000, rng)
    assert np.abs(Q.acc(q, M) - pi).mean() < 0.02


def test_emq_prior_shift():
    rng = np.random.default_rng(1); pi_tr = np.array([.5, .5]); pi_te = np.array([.8, .2]); n = 20000
    y = rng.choice(2, n, p=pi_te); x = rng.normal(y * 2.0 - 1, 1.0)
    lik = lambda c: np.exp(-(x - (c * 2.0 - 1)) ** 2 / 2); post = np.stack([lik(0) * pi_tr[0], lik(1) * pi_tr[1]], 1); post /= post.sum(1, keepdims=True)
    import m3.quantification.methods as m; m.K = 2
    try: est = m.emq(post, pi_tr)
    finally: m.K = 8
    assert abs(est[0] - 0.8) < 0.03
