"""Quantifiers (NumPy). All return a prevalence vector summing to 1."""
import numpy as np
from scipy.optimize import minimize

K = 8


def cc(q): return np.bincount(q.argmax(1), minlength=K) / len(q)
def pcc(q): return q.mean(0)


def conf_matrix_hard(q, y):
    M = np.zeros((K, K)); yh = q.argmax(1)
    for k in range(K):
        m = y == k
        M[:, k] = np.bincount(yh[m], minlength=K) / max(m.sum(), 1) if m.any() else 0
    return M


def conf_matrix_soft(q, y):
    M = np.zeros((K, K))
    for k in range(K):
        m = y == k; M[:, k] = q[m].mean(0) if m.any() else 0
    return M


def solve_simplex(M, v):
    def f(x): return ((M @ x - v) ** 2).sum()
    x0 = np.full(K, 1 / K)
    r = minimize(f, x0, jac=lambda x: 2 * M.T @ (M @ x - v), bounds=[(0, 1)] * K, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
    x = r.x if r.success else np.clip(np.linalg.lstsq(M, v, rcond=None)[0], 0, None)
    x = np.clip(x, 0, None); return x / x.sum()


def acc(q, M): return solve_simplex(M, cc(q))
def pacc(q, M): return solve_simplex(M, pcc(q))


def emq(q, prior_tr, tol=1e-6, maxit=1000):
    pi = prior_tr.copy()
    for _ in range(maxit):
        w = q * (pi / prior_tr); w /= w.sum(1, keepdims=True); new = w.mean(0)
        if np.abs(new - pi).max() < tol: pi = new; break
        pi = new
    return pi


def abst_cc(q, sets):
    single = sets.sum(1) == 1
    if single.sum() == 0: return np.full(K, 1 / K)
    return np.bincount(q[single].argmax(1), minlength=K) / single.sum()
