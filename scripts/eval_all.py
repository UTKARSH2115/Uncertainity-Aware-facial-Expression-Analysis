"""Phases 4-9 on every finished run. --watch keeps polling until all expected runs are evaluated.
python scripts/eval_all.py [--watch] [--runs thin|all]"""
import argparse, os, sys, time, traceback
os.environ.setdefault("M3_THREADS", "2")
import torch
torch.set_num_threads(2)
from m3.utils.common import ROOT
from m3.report.make_tables import runs
from m3.infer import predict as P
from m3.evalrun import eval_run
from m3.edge import compare as C
from m3.quantification import evaluate as QE
from m3.evalrun import RunData

P0 = [f"{a}-{r}-s{s}" for s in (0,) for a in ("tinycnn", "mnv3s") for r in ("maj", "single", "soft")]
P1 = [f"mnv3s-{r}-s0" for r in ("dm", "ra")]
EXPECT = P0 + P1
done = lambda rid, f: (ROOT / f"results/runs/{rid}/{f}").exists()


def corr_wanted(rid):
    a, g, s = P.parse_id(rid); return (s == 0 and g in ("maj", "single", "soft") and a in ("tinycnn", "mnv3s")) or rid == "mnv3s-ra-s0"


def quant_wanted(rid):
    a, g, s = P.parse_id(rid); return (g == "soft" and a in ("tinycnn", "mnv3s")) or rid == "mnv3s-maj-s0"


def int8_wanted(rid):
    a, g, s = P.parse_id(rid); return a in ("tinycnn", "mnv3s")


def step(rid):
    if not done(rid, "eval_conformal.csv") or (corr_wanted(rid) and not done(rid, "eval_corr.csv")):
        eval_run(rid, corr=corr_wanted(rid)); return True
    if int8_wanted(rid) and C.METHOD_FILE.exists() and not done(rid, "eval_int8.csv"):
        C.make_int8_logits(rid); C.compare(rid); print("int8 done", rid, flush=True); return True
    if quant_wanted(rid) and not done(rid, "quant_protocols.csv"):
        R = RunData(rid); o = ROOT / f"results/runs/{rid}"; t, ps = QE.protocols(R); t.to_csv(o / "quant_protocols.csv", index=False); ps.to_csv(o / "quant_app40_samples.csv", index=False)
        QE.alerts(R).to_csv(o / "quant_alerts.csv", index=False); QE.virtual_classrooms(R, QE.prov_fp32(R), "fp32").to_csv(o / "quant_classroom_fp32.csv", index=False); print("quant done", rid, flush=True); return True
    if quant_wanted(rid) and done(rid, "eval_int8.csv") and not done(rid, "quant_classroom_int8.csv") and P.parse_id(rid)[2] == 0:
        R = RunData(rid); QE.virtual_classrooms(R, QE.prov_int8(R), "int8").to_csv(ROOT / f"results/runs/{rid}/quant_classroom_int8.csv", index=False); print("int8 classroom", rid, flush=True); return True
    return False


def maybe_select_method():
    if C.METHOD_FILE.exists() and "tinycnn-soft-s0" in C.json.load(open(C.METHOD_FILE)).get("runs", []) and "mnv3s-soft-s0" in C.json.load(open(C.METHOD_FILE)).get("runs", []): return
    need = ["tinycnn-soft-s0", "mnv3s-soft-s0"]
    if all(r in runs() for r in need): print("INT8 calibration method:", C.select_method(need), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--watch", action="store_true"); ap.add_argument("--runs", default="all"); a = ap.parse_args()
    if a.runs == "thin":
        for r in ["mnv3s-soft-s9"]: eval_run(r); return
    while True:
        maybe_select_method(); progressed = False
        for rid in EXPECT:
            if rid in runs():
                try: progressed |= step(rid)
                except Exception: traceback.print_exc(); print("FAILED step", rid, flush=True)
        finished = all(r in runs() and not step_pending(r) for r in EXPECT)
        if not a.watch or finished: break
        if not progressed: time.sleep(120)


def step_pending(rid):
    if not done(rid, "eval_conformal.csv") or (corr_wanted(rid) and not done(rid, "eval_corr.csv")): return True
    if int8_wanted(rid) and not done(rid, "eval_int8.csv"): return True
    if quant_wanted(rid) and not done(rid, "quant_protocols.csv"): return True
    if quant_wanted(rid) and P.parse_id(rid)[2] == 0 and not done(rid, "quant_classroom_int8.csv"): return True
    return False


if __name__ == "__main__": main()
