"""INT8 pipeline per run + FP32-vs-INT8 reliability (core table for N1). CLI: python -m m3.edge.compare --run ... [--select-method]"""
import argparse, json
import numpy as np, pandas as pd
from m3.utils.common import ROOT, load_npz, jdump
from m3.infer import predict as P
from m3.infer.shifts import shifted_logits
from m3.edge import onnx_tools as O
from m3.calib.metrics import metrics, entropy
from m3.calib.temperature import fit_T, apply_T
from m3.uncertainty.selective import aurc
from m3.conformal import cp
from m3.evalrun import RunData, LEVELS

R_EVAL = [48, 24, 16, 12]
METHOD_FILE = ROOT / "results/int8_method.json"


def int8_method(): return json.load(open(METHOD_FILE))["method"] if METHOD_FILE.exists() else "Percentile"


def make_int8_logits(rid, method=None, D=None):
    """INT8 logits for every split x resolution level (needed by Phase 9)."""
    D = D or load_npz(); idx = P.split_idx(D); arch, _, _ = P.parse_id(rid); size = P.run_size(rid)
    out = O.quantize(rid, method or int8_method()); s = O.session(out, 4)
    from m3.data.degrade import resolution
    for l in LEVELS:
        for k, ii in idx.items():
            f = ROOT / f"results/runs/{rid}/logits_int8/{'clean' if l == 48 else f'res{l}'}/{k}.npy"
            if f.exists(): continue
            f.parent.mkdir(parents=True, exist_ok=True); np.save(f, O.ort_logits(s, resolution(D["img"][ii], l), arch, size))


def select_method(rids):
    D = load_npz(); idx = P.split_idx(D); res = {}
    for m in O.CAL:
        v = []
        for rid in rids:
            arch, reg, _ = P.parse_id(rid); size = P.run_size(rid); out = O.quantize(rid, m, tag=f"int8_{m}"); s = O.session(out, 4)
            z = O.ort_logits(s, D["img"][idx["valA"]], arch, size); q = P.probs_from_logits(reg, z); p = D["p"][idx["valA"]]
            v.append(-(p * np.log(np.clip(q, 1e-12, 1))).sum(1).mean())
        res[m] = float(np.mean(v)); print("method", m, res[m], flush=True)
    best = min(res, key=res.get); jdump(dict(method=best, valA_soft_nll=res, runs=rids), METHOD_FILE); return best


def _fit(R, zA, yA): return fit_T(P.eff_logits(R.reg, zA), yA)


def compare(rid, B=200):
    R = RunData(rid); rows = []; flipf = []; y, p = R.y["test"], R.p["test"]; Hp = entropy(p) / np.log(8)
    rp = cp.raps_params(R.q(R.z["valA"]), R.y["valA"], 0.1)
    for r in R_EVAL:
        f = lambda s: (R.z[s] if r == 48 else shifted_logits(rid, f"res{r}", s, D=R.D, idx=R.idx))
        i8 = lambda s: np.load(ROOT / f"results/runs/{rid}/logits_int8/{'clean' if r == 48 else f'res{r}'}/{s}.npy")
        z32 = {s: f(s) for s in ("valA", "valB", "test")}; z8 = {s: i8(s) for s in ("valA", "valB", "test")}
        T32 = R.T_maj; T8 = _fit(R, z8["valA"], R.y["valA"])
        q32 = apply_T(P.eff_logits(R.reg, z32["test"]), T32); q8_old = apply_T(P.eff_logits(R.reg, z8["test"]), T32); q8_new = apply_T(P.eff_logits(R.reg, z8["test"]), T8)
        flips = q32.argmax(1) != q8_old.argmax(1)
        for dec_by, v in (("conf", q32.max(1)), ("humanH", Hp)):
            dec = np.minimum((pd.Series(v).rank(pct=True).values * 10).astype(int), 9)
            for d in range(10): flipf.append(dict(**R.meta(), r=r, by=dec_by, decile=d, flip_rate=flips[dec == d].mean()))
        rng = np.random.default_rng(0); uA, uT = np.random.default_rng(0).random(len(R.y["valB"])), np.random.default_rng(1).random(len(y))
        def thr(zB, T, m, qA): return cp.calibrate(apply_T(P.eff_logits(R.reg, zB), T), R.y["valB"], m, 0.1, uA, **({"kreg": rp[0], "lam": rp[1]} if m == "RAPS" else {}))
        def metric_set(q, thrs):
            m = metrics(q, p, y); d = dict(acc=m["acc"], soft_nll=m["soft_nll"], ece=m["ece"], rho=m["rho_entropy"], aurc=aurc(q.max(1), (q.argmax(1) != y).astype(float)))
            for mm in ("LAC", "APS"):
                C = cp.predict_sets(q, thrs[mm], mm, uT); e = cp.evaluate_sets(C, p, y); d[f"cov_{mm}"] = e["cov_maj"]; d[f"size_{mm}"] = e["size_mean"]
            return d
        th32 = {m: thr(z32["valB"], T32, m, None) for m in ("LAC", "APS")}; th8_old = th32; th8_new = {m: thr(z8["valB"], T8, m, None) for m in ("LAC", "APS")}
        base = metric_set(q32, th32); variants = {"int8_reuse": metric_set(q8_old, th8_old), "int8_refit": metric_set(q8_new, th8_new)}
        # paired bootstrap CIs for delta (accuracy, ECE, soft-NLL, coverage LAC)
        idxs = [rng.integers(0, len(y), len(y)) for _ in range(B)]
        def boot(qa, qb, thr_a, thr_b):
            out = {k: [] for k in ("acc", "ece", "soft_nll", "cov_LAC")}
            for ii in idxs:
                ma, mb = metrics(qa[ii], p[ii], y[ii], ranking=False), metrics(qb[ii], p[ii], y[ii], ranking=False)
                Ca = cp.predict_sets(qa[ii], thr_a["LAC"], "LAC", uT[ii]); Cb = cp.predict_sets(qb[ii], thr_b["LAC"], "LAC", uT[ii]); n = np.arange(len(ii))
                out["acc"].append(mb["acc"] - ma["acc"]); out["ece"].append(mb["ece"] - ma["ece"]); out["soft_nll"].append(mb["soft_nll"] - ma["soft_nll"])
                out["cov_LAC"].append(Cb[n, y[ii]].mean() - Ca[n, y[ii]].mean())
            return {k: np.quantile(v, [0.025, 0.975]) for k, v in out.items()}
        cis = {"int8_reuse": boot(q32, q8_old, th32, th8_old), "int8_refit": boot(q32, q8_new, th32, th8_new)}
        for var, d in variants.items():
            for k, v in d.items():
                ci = cis[var].get(k, (np.nan, np.nan))
                rows.append(dict(**R.meta(), r=r, variant=var, metric=k, fp32=base[k], int8=v, delta=v - base[k], ci_lo=ci[0], ci_hi=ci[1]))
            rows.append(dict(**R.meta(), r=r, variant=var, metric="flip_rate", fp32=0.0, int8=flips.mean(), delta=flips.mean(), ci_lo=np.nan, ci_hi=np.nan))
            rows.append(dict(**R.meta(), r=r, variant=var, metric="T", fp32=T32, int8=T8 if var == "int8_refit" else T32, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan))
    pd.DataFrame(rows).to_csv(ROOT / f"results/runs/{rid}/eval_int8.csv", index=False); pd.DataFrame(flipf).to_csv(ROOT / f"results/runs/{rid}/eval_flips.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="+"); ap.add_argument("--select-method", action="store_true"); ap.add_argument("--skip-compare", action="store_true"); a = ap.parse_args()
    if a.select_method: print("selected", select_method(a.run))
    for rid in a.run:
        make_int8_logits(rid)
        if not a.skip_compare: compare(rid)
        print("int8 done", rid, flush=True)
