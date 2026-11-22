"""Split conformal prediction: LAC, APS (randomised), RAPS; Mondrian (group-conditional) thresholds; evaluation."""
import numpy as np
from scipy.stats import spearmanr


def all_scores(q, method, u=None, lam=0.0, kreg=0):
    """Nonconformity score for every candidate class, [N,K]."""
    if method == "LAC": return 1 - q
    o = np.argsort(-q, axis=1); qs = np.take_along_axis(q, o, 1); cum_before = np.cumsum(qs, 1) - qs
    if u is None: u = np.full(len(q), 0.5)
    s_sorted = cum_before + u[:, None] * qs
    if method == "RAPS":
        rank = np.arange(1, q.shape[1] + 1)[None, :]; s_sorted = s_sorted + lam * np.maximum(0, rank - kreg)
    S = np.empty_like(s_sorted); np.put_along_axis(S, o, s_sorted, 1); return S


def qhat(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def calibrate(q, labels, method, alpha, u=None, **kw):
    S = all_scores(q, method, u, **kw); return qhat(S[np.arange(len(q)), labels], alpha)


def predict_sets(q, thr, method, u=None, **kw):
    return all_scores(q, method, u, **kw) <= np.asarray(thr).reshape(-1, 1) if np.ndim(thr) else all_scores(q, method, u, **kw) <= thr


def raps_params(qA, yA, alpha, lambdas=(0.001, 0.01, 0.1, 0.2, 0.5), seed=0):
    """k_reg: smallest k with top-k coverage on valA >= 1-alpha; lambda minimising mean set size on valA."""
    rank = (np.argsort(-qA, 1) == yA[:, None]).argmax(1) + 1
    kreg = next(k for k in range(1, 9) if (rank <= k).mean() >= 1 - alpha)
    u = np.random.default_rng(seed).random(len(qA)); best = None
    for lam in lambdas:
        t = calibrate(qA, yA, "RAPS", alpha, u, lam=lam, kreg=kreg)
        sz = predict_sets(qA, t, "RAPS", u, lam=lam, kreg=kreg).sum(1).mean()
        if best is None or sz < best[0]: best = (sz, lam)
    return kreg, best[1]


def evaluate_sets(C, p, y, H=None):
    N = len(y); size = C.sum(1)
    out = dict(cov_maj=C[np.arange(N), y].mean(), cov_annot=(C * p).sum(1).mean(), size_mean=size.mean(),
               frac_singleton=(size == 1).mean(), frac_empty=(size == 0).mean())
    cc = [C[y == k, k].mean() for k in range(8) if (y == k).sum() >= 20]; out["worst_class_cov"] = min(cc)
    if H is not None and size.std() > 0: out["rho_size_entropy"] = spearmanr(size, H)[0]
    return out


def sample_votes(n, rng):
    """One annotator vote per image sampled from raw counts n [N,8]."""
    c = n.astype(float).cumsum(1); r = rng.random(len(n)) * c[:, -1]
    return (r[:, None] >= c).sum(1)
