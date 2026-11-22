import numpy as np


def risk_coverage(conf, risk):
    o = np.argsort(-conf, kind="stable"); r = np.asarray(risk, dtype=float)[o]
    return np.cumsum(r) / np.arange(1, len(r) + 1)


def aurc(conf, risk):
    return float(risk_coverage(conf, risk).mean())


def oracle_aurc(risk):
    r = float(np.mean(risk)); return r + (1 - r) * np.log(1 - r) if r < 1 else r


def sel_acc(conf, correct, cov):
    o = np.argsort(-conf, kind="stable"); k = int(round(cov * len(conf))); return float(np.asarray(correct)[o][:k].mean())
