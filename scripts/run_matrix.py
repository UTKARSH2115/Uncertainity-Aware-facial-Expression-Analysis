import argparse, ctypes, sys, subprocess
from m3.utils.common import ROOT
from m3.train.train import run, export_teacher_logits
TIERS = {
 "THIN": ["mnv3s-soft-s9"],
 "P0": [f"{a}-{r}-s{s}" for s in (0,) for a in ("tinycnn","mnv3s") for r in ("maj","single","soft")],
 "P1T": ["r18-soft-s0"],
 "P1": [f"mnv3s-{r}-s0" for r in ("dm","ra")],
}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tier", required=True); a = ap.parse_args()
    try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # keep system awake while running
    except Exception: pass
    for rid in TIERS[a.tier]:
        if (ROOT / f"results/runs/{rid}/logits/test.npy").exists(): print("skip", rid); continue
        ep = 3 if a.tier == "THIN" else None
        ok = run(rid, epochs=ep)
        if rid == "r18-soft-s0" and ok: export_teacher_logits(rid)
        if not ok: print("FAILED", rid); sys.exit(1)
