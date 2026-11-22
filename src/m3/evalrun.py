"""Per-run evaluation: Phases 4-7 (calibration, selective, conformal, resolution sweep, Mondrian, corruptions).
Writes tidy CSVs to results/runs/<id>/eval_*.csv"""
import argparse, time
import numpy as np, pandas as pd
from scipy.special import softmax
from m3.utils.common import ROOT, load_npz
from m3.infer import predict as P
from m3.infer.shifts import shifted_logits, tta_logits, mcd_logits, RES
from m3.calib.metrics import metrics, entropy, amb_groups
from m3.calib.temperature import fit_T, apply_T
from m3.uncertainty.selective import aurc, oracle_aurc, sel_acc
from m3.conformal import cp
from m3.geometry import camera as G
from m3.data.degrade import CORRUPTIONS

ALPHAS = [0.05, 0.10, 0.20]
METHODS = ["LAC", "APS", "RAPS"]


class RunData:
    def __init__(self, rid, D=None):
        self.rid = rid; self.arch, self.reg, self.seed = P.parse_id(rid); self.D = D or load_npz(); self.idx = P.split_idx(self.D)
        d = self.D; ix = self.idx
        self.y = {k: d["y_maj"][v].astype(int) for k, v in ix.items()}; self.p = {k: d["p"][v] for k, v in ix.items()}; self.n = {k: d["n"][v] for k, v in ix.items()}
        self.z = {k: np.load(ROOT / f"results/runs/{rid}/logits/{k}.npy") for k in ix}
        self.zeff = {k: P.eff_logits(self.reg, v) for k, v in self.z.items()}
        self.T_maj = fit_T(self.zeff["valA"], self.y["valA"]); self.T_soft = fit_T(self.zeff["valA"], self.p["valA"], soft=True)

    def q(self, z, T=None): return apply_T(P.eff_logits(self.reg, z), self.T_maj if T is None else T)
    def meta(self): return dict(arch=self.arch, regime=self.reg, seed=self.seed)


def cp_setup(R, qA, qB=None, zB=None):
    """Return per-alpha RAPS params (selected on valA)."""
    return {a: cp.raps_params(qA, R.y["valA"], a) for a in ALPHAS}


def cp_thresholds(R, qA, qB, labelsB, rp, u):
    """thr[(method,alpha)] calibrated on valB with given labels."""
    th = {}
    for m in METHODS:
        for a in ALPHAS:
            kw = dict(kreg=rp[a][0], lam=rp[a][1]) if m == "RAPS" else {}
            th[(m, a)] = (cp.calibrate(qB, labelsB, m, a, u, **kw), kw)
    return th


def cp_sets(q, thr_kw, m, u):
    thr, kw = thr_kw; return cp.predict_sets(q, thr, m, u, **kw)


def clean_eval(R):
    rows, sel, cprow = [], [], []
    qT = {"raw": P.probs_from_logits(R.reg, R.z["test"]), "ts_maj": R.q(R.z["test"]), "ts_soft": R.q(R.z["test"], R.T_soft)}
    y, p = R.y["test"], R.p["test"]
    for cal, q in qT.items(): rows.append(dict(**R.meta(), calibrator=cal, T=dict(raw=1, ts_maj=R.T_maj, ts_soft=R.T_soft)[cal], **metrics(q, p, y)))
    # --- selective prediction
    q = qT["ts_maj"]; ests = {"MSP": q.max(1), "ENT": -entropy(q), "MARGIN": np.sort(q, 1)[:, -1] - np.sort(q, 1)[:, -2]}
    cost = {"MSP": 1, "ENT": 1, "MARGIN": 1}; Hp = entropy(p) / np.log(8); amb, clr = amb_groups(p)
    probs_for = {k: q for k in ests}
    if R.arch in ("tinycnn", "mnv3s") and R.reg in ("maj", "soft", "single"):
        qt = np.mean([R.q(z) for z in tta_logits(R.rid, D=R.D, idx=R.idx)], 0); ests["TTA8"] = qt.max(1); probs_for["TTA8"] = qt; cost["TTA8"] = 8
        qm = np.mean([R.q(z) for z in mcd_logits(R.rid, D=R.D, idx=R.idx)], 0); ests["MCD20"] = qm.max(1); probs_for["MCD20"] = qm; cost["MCD20"] = 20
    for e, conf in ests.items():
        qe = probs_for[e]; yh = qe.argmax(1); err = (yh != y).astype(float); serr = 1 - p[np.arange(len(y)), yh]
        k = int(0.8 * len(y)); o = np.argsort(-conf, kind="stable"); acc_set, abs_set = o[:k], o[k:]
        r = err.mean(); row = dict(**R.meta(), estimator=e, cost_x=cost[e], aurc=aurc(conf, err), eaurc=aurc(conf, err) - oracle_aurc(err), soft_aurc=aurc(conf, serr))
        from scipy.stats import spearmanr
        row["rho"] = spearmanr(-conf, Hp)[0]
        for c in (50, 70, 80, 90): row[f"selacc@{c}"] = sel_acc(conf, 1 - err, c / 100)
        row.update(H_abst=Hp[abs_set].mean(), H_acc=Hp[acc_set].mean(), frac_amb_abst=amb[abs_set].mean(), frac_amb_acc=amb[acc_set].mean(), cov=0.8); sel.append(row)
    # --- conformal (clean)
    qA, qB = R.q(R.z["valA"]), R.q(R.z["valB"]); rp = cp_setup(R, qA); rng = np.random.default_rng(0)
    uB, uT = rng.random(len(qB)), rng.random(len(q)); Hn = entropy(p)
    th = cp_thresholds(R, qA, qB, R.y["valB"], rp, uB)
    SB = {m: cp.all_scores(qB, m, uB, **({"kreg": rp[0.1][0], "lam": rp[0.1][1]} if m == "RAPS" else {})) for m in METHODS}
    for m in METHODS:
        for a in ALPHAS:
            C = cp_sets(q, th[(m, a)], m, uT); cprow.append(dict(**R.meta(), method=m, cal_labels="maj", alpha=a, **cp.evaluate_sets(C, p, y, Hn), n_votes_rho=np.nan))
            # CAL-VOTE: sampled single annotator votes, R=100 resamples
            kw = dict(kreg=rp[a][0], lam=rp[a][1]) if m == "RAPS" else {}
            Sb = cp.all_scores(qB, m, uB, **kw); St = cp.all_scores(q, m, uT, **kw); ev = []
            for _ in range(100):
                lab = cp.sample_votes(R.n["valB"], rng); t = cp.qhat(Sb[np.arange(len(lab)), lab], a); ev.append(cp.evaluate_sets(St <= t, p, y))
            ev = pd.DataFrame(ev); row = dict(**R.meta(), method=m, cal_labels="vote", alpha=a, **ev.mean().to_dict(), **{c + "_sd": ev[c].std() for c in ["cov_maj", "cov_annot", "size_mean"]})
            cprow.append(row)
    # mondrian per class (P1-ish, cheap): per-class thresholds LAC
    return pd.DataFrame(rows), pd.DataFrame(sel), pd.DataFrame(cprow), rp, th, uB, uT


def sweep_eval(R, rp, th, uT):
    rows = []; y, p = R.y["test"], R.p["test"]
    for r in [48] + RES:
        zt = R.z["test"] if r == 48 else shifted_logits(R.rid, f"res{r}", "test", D=R.D, idx=R.idx)
        q = R.q(zt); m = metrics(q, p, y); conf = q.max(1); err = (q.argmax(1) != y).astype(float)
        row = dict(**R.meta(), r=r, acc=m["acc"], soft_acc=m["soft_acc"], soft_nll=m["soft_nll"], ece=m["ece"], rho=m["rho_entropy"], aurc=aurc(conf, err))
        for mname, key in (("LAC", "lac"), ("APS", "aps")):
            C = cp_sets(q, th[(mname, 0.1)], mname, uT); e = cp.evaluate_sets(C, p, y); row[f"cov_{key}"] = e["cov_maj"]; row[f"size_{key}"] = e["size_mean"]; row[f"covann_{key}"] = e["cov_annot"]
        rows.append(row)
    return pd.DataFrame(rows)


LEVELS = [48, 40, 32, 24, 20, 16, 12, 8]
BIN_REPS = {"B1": (48, 40), "B2": (32, 24), "B3": (20, 16), "B4": (12, 8)}


def _z(R, r, split): return R.z[split] if r == 48 else shifted_logits(R.rid, f"res{r}", split, D=R.D, idx=R.idx)


def mondrian_eval(R, rp, th, uT, alpha=0.1, methods=("LAC", "APS")):
    rng = np.random.default_rng(7); rows = []; yB, yA = R.y["valB"], R.y["valA"]; y, p = R.y["test"], R.p["test"]
    # per-bin calibration set: each valB/valA image appears once per bin, with resolution picked from the bin's pair
    thr_b, T_b = {}, {}
    for b, reps in BIN_REPS.items():
        pick = rng.integers(0, 2, len(yB)); zB = np.stack([_z(R, reps[0], "valB"), _z(R, reps[1], "valB")])[pick, np.arange(len(yB))]
        pickA = rng.integers(0, 2, len(yA)); zA = np.stack([_z(R, reps[0], "valA"), _z(R, reps[1], "valA")])[pickA, np.arange(len(yA))]
        Tb = fit_T(P.eff_logits(R.reg, zA), yA); T_b[b] = Tb; qB = P.eff_logits(R.reg, zB); qB = apply_T(qB, Tb); u = rng.random(len(yB))
        qAb = apply_T(P.eff_logits(R.reg, zA), Tb)
        for m in methods:
            kw = dict(kreg=rp[alpha][0], lam=rp[alpha][1]) if m == "RAPS" else {}
            thr_b[(b, m)] = (cp.calibrate(qB, yB, m, alpha, u, **kw), kw)
    for mode in ("B", "C", "D"):
        s = G.seats(mode); rs = s[s.visible].r.values; rs = rs[rs >= 1]
        lv = np.array([min(LEVELS, key=lambda l: abs(l - r)) for r in rng.choice(rs, len(y))])  # r drawn from seat distribution
        zt = np.zeros_like(R.z["test"])
        for l in set(lv.tolist()): zt[lv == l] = _z(R, l, "test")[lv == l]
        bins = np.array([G.bin_of(l) for l in lv])
        qg = R.q(zt); qm = np.zeros_like(qg)
        for b in BIN_REPS:
            k = bins == b
            if k.any(): qm[k] = apply_T(P.eff_logits(R.reg, zt[k]), T_b[b])
        for m in methods:
            Cg = cp_sets(qg, th[(m, alpha)], m, uT)
            Cm = np.zeros_like(Cg)
            for b in BIN_REPS:
                k = bins == b
                if k.any(): Cm[k] = cp.predict_sets(qm[k], thr_b[(b, m)][0], m, uT[k], **thr_b[(b, m)][1])
            for b in list(BIN_REPS) + ["all"]:
                k = np.ones(len(y), bool) if b == "all" else bins == b
                if k.sum() < 10: continue
                rows.append(dict(**R.meta(), mode=mode, method=m, bin=b, n=int(k.sum()), cov_global=Cg[k, y[k]].mean() if False else Cg[np.where(k)[0], y[k]].mean(),
                                 cov_mondrian=Cm[np.where(k)[0], y[k]].mean(), size_global=Cg[k].sum(1).mean(), size_mondrian=Cm[k].sum(1).mean(),
                                 ece_global=metrics(qg[k], p[k], y[k], ranking=False)["ece"], ece_binT=metrics(qm[k], p[k], y[k], ranking=False)["ece"]))
    return pd.DataFrame(rows)


def corruption_eval(R, th, uT):
    rows = []; y, p = R.y["test"], R.p["test"]
    for c in CORRUPTIONS:
        for s in range(1, 6):
            q = R.q(shifted_logits(R.rid, f"{c}_s{s}", "test", D=R.D, idx=R.idx)); m = metrics(q, p, y, ranking=False)
            C = cp_sets(q, th[("LAC", 0.1)], "LAC", uT)
            rows.append(dict(**R.meta(), corruption=c, severity=s, acc=m["acc"], ece=m["ece"], soft_nll=m["soft_nll"], aurc=aurc(q.max(1), (q.argmax(1) != y).astype(float)), cov=C[np.arange(len(y)), y].mean()))
    return pd.DataFrame(rows)


def eval_run(rid, corr=False, D=None):
    t = time.time(); R = RunData(rid, D); out = ROOT / f"results/runs/{rid}"
    clean, sel, cpr, rp, th, uB, uT = clean_eval(R)
    clean.to_csv(out / "eval_clean.csv", index=False); sel.to_csv(out / "eval_selective.csv", index=False); cpr.to_csv(out / "eval_conformal.csv", index=False)
    sweep_eval(R, rp, th, uT).to_csv(out / "eval_sweep.csv", index=False)
    mondrian_eval(R, rp, th, uT).to_csv(out / "eval_mondrian.csv", index=False)
    if corr: corruption_eval(R, th, uT).to_csv(out / "eval_corr.csv", index=False)
    print(f"{rid} evaluated in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="+"); ap.add_argument("--corr", action="store_true"); a = ap.parse_args()
    for r in a.run: eval_run(r, a.corr)
