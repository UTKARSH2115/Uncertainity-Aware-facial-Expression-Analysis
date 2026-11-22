import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

EPS = 1e-12


def entropy(p):
    p = np.clip(p, EPS, 1); return -(p * np.log(p)).sum(1)


def ece(conf, acc, bins=15, adaptive=False):
    N = len(conf)
    if adaptive:
        o = np.argsort(conf); parts = np.array_split(o, bins)
    else:
        e = np.linspace(0, 1, bins + 1); parts = [np.where((conf > e[i]) & (conf <= e[i + 1]))[0] for i in range(bins)]
    out, mx = 0.0, 0.0
    for ix in parts:
        if len(ix) == 0: continue
        g = abs(acc[ix].mean() - conf[ix].mean()); out += len(ix) / N * g; mx = max(mx, g)
    return out, mx


def classwise_ece(q, y, bins=15):
    return float(np.mean([ece(q[:, k], (y == k).astype(float), bins)[0] for k in range(q.shape[1])]))


def amb_groups(p):
    m = p.max(1); return m <= 0.5, m >= 0.8


def metrics(q, p, y, ranking=True):
    """q: model probs [N,8]; p: human soft labels; y: majority labels."""
    q = np.clip(q, EPS, 1); q = q / q.sum(1, keepdims=True); yh = q.argmax(1); c = q.max(1); N = len(y)
    acc = (yh == y).astype(float); sacc = p[np.arange(N), yh]
    Hq, Hp = entropy(q), entropy(p)
    out = dict(acc=acc.mean(), macro_f1=f1_score(y, yh, average="macro", labels=list(range(8)), zero_division=0),
               bal_acc=balanced_accuracy_score(y, yh), soft_acc=sacc.mean(),
               nll=-np.log(q[np.arange(N), y]).mean(), soft_nll=-(p * np.log(q)).sum(1).mean())
    out["kl"] = out["soft_nll"] - Hp.mean()
    out["brier"] = ((q - np.eye(8)[y]) ** 2).sum(1).mean()
    out["ece"], out["mce"] = ece(c, acc); out["aece"] = ece(c, acc, adaptive=True)[0]
    out["cw_ece"] = classwise_ece(q, y); out["soft_ece"] = ece(c, sacc)[0]
    if ranking:
        out["rho_entropy"] = spearmanr(Hq, Hp)[0]
        a, cl = amb_groups(p); k = a | cl
        out["auroc_amb"] = roc_auc_score(a[k], Hq[k]) if a[k].any() and cl[k].any() else np.nan
    return out


def boot_ci(fn, N, B=500, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); v = np.array([fn(rng.integers(0, N, N)) for _ in range(B)])
    return np.quantile(v, [alpha / 2, 1 - alpha / 2])
