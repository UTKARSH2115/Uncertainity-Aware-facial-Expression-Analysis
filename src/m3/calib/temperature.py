import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def _nll(logT, z, tgt, soft):
    lp = log_softmax(z / np.exp(logT), axis=1)
    return -(tgt * lp).sum(1).mean() if soft else -lp[np.arange(len(z)), tgt].mean()


def fit_T(z, target, soft=False):
    """Bounded scalar search over log T in [-3,3]; target = labels (hard) or distributions (soft)."""
    r = minimize_scalar(_nll, bounds=(-3, 3), args=(z, target, soft), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def apply_T(z, T):
    return softmax(np.asarray(z, dtype=np.float64) / T, axis=1)
