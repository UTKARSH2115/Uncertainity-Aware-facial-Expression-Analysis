"""Phase 9: class-level quantification protocols (APP/NPP/CPP) and virtual classrooms with seat-dependent degradation."""
import argparse
import numpy as np, pandas as pd
from m3.utils.common import ROOT, NEG
from m3.infer import predict as P
from m3.infer.shifts import shifted_logits
from m3.calib.temperature import fit_T, apply_T
from m3.quantification import methods as Q
from m3.conformal import cp
from m3.geometry import camera as G
from m3.evalrun import RunData, LEVELS, BIN_REPS, cp_setup

HAPPY = 1


def nas(pi): return np.asarray(pi)[..., NEG].sum(-1)
def wbi(pi): return np.asarray(pi)[..., HAPPY] - nas(pi)


def prov_fp32(R):
    return lambda split, l: R.z[split] if l == 48 else shifted_logits(R.rid, f"res{l}", split, D=R.D, idx=R.idx)


def prov_int8(R):
    def f(split, l):
        d = ROOT / f"results/runs/{R.rid}/logits_int8/{'clean' if l == 48 else f'res{l}'}/{split}.npy"; return np.load(d)
    return f


def errs(est, true, n):
    eps = 1 / (2 * n); a = (est + 0); out = {}
    out["ae"] = np.abs(a - true).mean(-1)
    sm = lambda v: (v + eps) / (1 + eps * v.shape[-1])
    out["rae"] = (np.abs(a - true) / (true + eps)).mean(-1)
    e, t = sm(a), sm(true); out["kld"] = (t * np.log(t / e)).sum(-1)
    out["nas_err"] = np.abs(nas(est) - nas(true)); out["wbi_err"] = np.abs(wbi(est) - wbi(true)); return out


def quantify_all(q, M_h, M_s, prior, sets):
    return {"CC": Q.cc(q), "PCC": Q.pcc(q), "ACC": Q.acc(q, M_h), "PACC": Q.pacc(q, M_s), "EMQ": Q.emq(q, prior), "ABST-CC": Q.abst_cc(q, sets)}


def protocols(R, samples=1000, n_list=(10, 20, 40, 60, 100), seed=0, prov=None):
    prov = prov or prov_fp32(R); zt = prov("test", 48); zA = prov("valA", 48)
    T = fit_T(P.eff_logits(R.reg, zA), R.y["valA"]); qT = apply_T(P.eff_logits(R.reg, zt), T); qA = apply_T(P.eff_logits(R.reg, zA), T)
    M_h, M_s = Q.conf_matrix_hard(qA, R.y["valA"]), Q.conf_matrix_soft(qA, R.y["valA"])
    ytr = R.D["y_maj"][(R.D["split"] == "train") & R.D["main"]]; prior = np.bincount(ytr, minlength=8) / len(ytr)
    rp = cp.raps_params(qA, R.y["valA"], 0.1); uB = np.random.default_rng(1).random(len(R.y["valB"])); qB = apply_T(P.eff_logits(R.reg, prov("valB", 48)), T)
    thr = cp.calibrate(qB, R.y["valB"], "LAC", 0.1); sets_all = cp.predict_sets(qT, thr, "LAC")
    y = R.y["test"]; by = [np.where(y == k)[0] for k in range(8)]; pi_test = np.bincount(y, minlength=8) / len(y)
    rng = np.random.default_rng(seed); rows = []; per_sample = []
    for proto in ("APP", "NPP", "CPP"):
        for n in n_list:
            acc = {}
            for s in range(samples):
                pi = rng.dirichlet(np.ones(8)) if proto == "APP" else pi_test if proto == "NPP" else rng.dirichlet(20 * pi_test)
                if proto == "NPP":
                    idx = rng.integers(0, len(y), n)
                else:
                    cnt = rng.multinomial(n, pi); idx = np.concatenate([rng.choice(by[k], c) if c else np.array([], int) for k, c in enumerate(cnt)]).astype(int)
                true = np.bincount(y[idx], minlength=8) / n; est = quantify_all(qT[idx], M_h, M_s, prior, sets_all[idx])
                for m, e in est.items():
                    acc.setdefault(m, []).append(errs(e, true, n))
                    if proto == "APP" and n == 40: per_sample.append(dict(method=m, sample=s, ae=acc[m][-1]["ae"]))
            for m, lst in acc.items():
                df = pd.DataFrame(lst); rows.append(dict(**R.meta(), protocol=proto, n=n, method=m, **{k: df[k].mean() for k in df}, **{k + "_sd": df[k].std() for k in df}))
    return pd.DataFrame(rows), pd.DataFrame(per_sample)


def bin_calib(R, prov, reps, seed):
    rng = np.random.default_rng(seed); yA = R.y["valA"]; pick = rng.integers(0, 2, len(yA))
    zA = np.stack([prov("valA", reps[0]), prov("valA", reps[1])])[pick, np.arange(len(yA))]
    ze = P.eff_logits(R.reg, zA); T = fit_T(ze, yA); q = apply_T(ze, T)
    return T, Q.conf_matrix_hard(q, yA), Q.conf_matrix_soft(q, yA)


def virtual_classrooms(R, prov, label, n_rooms=1000, seed=0):
    ytr = R.D["y_maj"][(R.D["split"] == "train") & R.D["main"]]; prior = np.bincount(ytr, minlength=8) / len(ytr)
    y = R.y["test"]; by = [np.where(y == k)[0] for k in range(8)]; pi_test = np.bincount(y, minlength=8) / len(y)
    T0 = fit_T(P.eff_logits(R.reg, prov("valA", 48)), R.y["valA"]); qA0 = apply_T(P.eff_logits(R.reg, prov("valA", 48)), T0)
    M0h, M0s = Q.conf_matrix_hard(qA0, R.y["valA"]), Q.conf_matrix_soft(qA0, R.y["valA"])
    bins = {b: bin_calib(R, prov, reps, 11) for b, reps in BIN_REPS.items()}
    Z = {l: prov("test", l) for l in LEVELS}; rng = np.random.default_rng(seed); rows = []
    for mode in ("A", "B", "C", "D"):
        s = G.seats(mode); v = s[s.visible]; nv = len(v)
        lv = np.array([min(LEVELS, key=lambda l: abs(l - r)) for r in v.r.values]); bn = np.array([G.bin_of(l) for l in lv])
        acc = {}
        for _ in range(n_rooms):
            pi = rng.dirichlet(20 * pi_test); cnt = rng.multinomial(nv, pi); cls = rng.permutation(np.repeat(np.arange(8), cnt))
            idx = np.array([rng.choice(by[k]) for k in cls]); true = np.bincount(y[idx], minlength=8) / nv
            zc = np.stack([Z[l][i] for l, i in zip(lv, idx)]); ze = P.eff_logits(R.reg, zc)
            q0 = apply_T(ze, T0); est = {"CC": Q.cc(q0), "PCC": Q.pcc(q0), "ACC (clean M)": Q.acc(q0, M0h), "PACC (clean M)": Q.pacc(q0, M0s), "EMQ": Q.emq(q0, prior)}
            pa, aa = np.zeros(8), np.zeros(8)
            for b, (Tb, Mh, Ms) in bins.items():
                k = bn == b
                if k.any():
                    qb = apply_T(ze[k], Tb); w = k.sum() / nv; pa += w * Q.pacc(qb, Ms); aa += w * Q.acc(qb, Mh)
            est["ACC (seat-aware)"] = aa; est["PACC (seat-aware)"] = pa
            for m, e in est.items(): acc.setdefault(m, []).append(errs(e, true, nv))
        for m, lst in acc.items():
            df = pd.DataFrame(lst); rows.append(dict(**R.meta(), precision=label, mode=mode, visible=nv, method=m, calibration="seat-aware" if "seat" in m else "clean",
                                                      **{k: df[k].mean() for k in ["ae", "nas_err", "wbi_err"]}, **{k + "_sd": df[k].std() for k in ["ae", "nas_err", "wbi_err"]}))
    return pd.DataFrame(rows)


def alerts(R, samples=1000, n_list=(10, 20, 40, 60, 100), seed=3):
    """Alert iff NAS_hat > 0.30. False-alert: true NAS <= 0.20; miss: true NAS >= 0.40 (CPP, with rejection sampling on pi)."""
    prov = prov_fp32(R); T = fit_T(P.eff_logits(R.reg, prov("valA", 48)), R.y["valA"]); qA = apply_T(P.eff_logits(R.reg, prov("valA", 48)), T)
    qT = apply_T(P.eff_logits(R.reg, prov("test", 48)), T); M_h, M_s = Q.conf_matrix_hard(qA, R.y["valA"]), Q.conf_matrix_soft(qA, R.y["valA"])
    y = R.y["test"]; by = [np.where(y == k)[0] for k in range(8)]; pi_test = np.bincount(y, minlength=8) / len(y); rng = np.random.default_rng(seed); rows = []
    for n in n_list:
        res = {m: dict(fa=[], miss=[]) for m in ("CC", "PCC", "ACC", "PACC")}
        for _ in range(samples):
            pi = rng.dirichlet(np.ones(8) * 1.0 if False else 20 * pi_test); # classroom prior
            # push negative mass to cover both regimes: mix pi with a negative-heavy prior
            if rng.random() < 0.5: pi = 0.5 * pi + 0.5 * rng.dirichlet(np.ones(8));
            cnt = rng.multinomial(n, pi); idx = np.concatenate([rng.choice(by[k], c) if c else np.array([], int) for k, c in enumerate(cnt)]).astype(int)
            true = nas(np.bincount(y[idx], minlength=8) / n); q = qT[idx]
            est = {"CC": Q.cc(q), "PCC": Q.pcc(q), "ACC": Q.acc(q, M_h), "PACC": Q.pacc(q, M_s)}
            for m, e in est.items():
                al = nas(e) > 0.30
                if true <= 0.20: res[m]["fa"].append(al)
                if true >= 0.40: res[m]["miss"].append(not al)
        for m, d in res.items():
            rows.append(dict(**R.meta(), n=n, method=m, false_alert_rate=np.mean(d["fa"]) if d["fa"] else np.nan, n_fa=len(d["fa"]), miss_rate=np.mean(d["miss"]) if d["miss"] else np.nan, n_miss=len(d["miss"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="+"); ap.add_argument("--samples", type=int, default=1000); ap.add_argument("--rooms", type=int, default=1000)
    ap.add_argument("--int8", action="store_true"); a = ap.parse_args()
    for rid in a.run:
        R = RunData(rid); o = ROOT / f"results/runs/{rid}"
        if not a.int8:
            t, ps = protocols(R, a.samples); t.to_csv(o / "quant_protocols.csv", index=False); ps.to_csv(o / "quant_app40_samples.csv", index=False)
            alerts(R, a.samples).to_csv(o / "quant_alerts.csv", index=False)
            virtual_classrooms(R, prov_fp32(R), "fp32", a.rooms).to_csv(o / "quant_classroom_fp32.csv", index=False)
        else:
            virtual_classrooms(R, prov_int8(R), "int8", a.rooms).to_csv(o / "quant_classroom_int8.csv", index=False)
        print("quant done", rid, flush=True)
