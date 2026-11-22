import numpy as np
from m3.conformal.cp import calibrate, predict_sets, raps_params


def test_coverage_exchangeable():
    rng = np.random.default_rng(0); N = 6000
    q = rng.dirichlet(np.ones(8) * 0.4, N); y = (rng.random((N, 1)) > q.cumsum(1)).sum(1).clip(0, 7)
    for method in ["LAC", "APS", "RAPS"]:
        covs = []
        for t in range(60):
            idx = rng.permutation(N); c, te = idx[:1000], idx[1000:]
            u = rng.random(N); kw = dict(lam=0.01, kreg=2) if method == "RAPS" else {}
            thr = calibrate(q[c], y[c], method, 0.1, u[c], **kw)
            C = predict_sets(q[te], thr, method, u[te], **kw); covs.append(C[np.arange(len(te)), y[te]].mean())
        assert abs(np.mean(covs) - 0.9) < 0.01, method
