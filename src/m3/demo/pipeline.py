"""Aggregate-only end-to-end pipeline: detector -> INT8 model -> temperature -> Mondrian conformal set -> class-level PACC (seat-aware).
python -m m3.demo.pipeline --source mosaic --privacy aggregate [--run mnv3s-soft-s0] [--oracle]"""
import argparse, json, time
import cv2
import numpy as np
from m3.utils.common import ROOT, CLASSES, NEG, jdump
from m3.infer import predict as P
from m3.edge import onnx_tools as O
from m3.calib.temperature import fit_T, apply_T
from m3.conformal import cp
from m3.quantification import methods as Q
from m3.geometry import camera as G
from m3.evalrun import RunData, BIN_REPS, LEVELS
from m3.quantification.evaluate import prov_int8, bin_calib

K = 0.85


class Pipeline:
    def __init__(self, rid="mnv3s-soft-s0", alpha=0.1, threads=4, privacy="aggregate"):
        self.rid = rid; self.arch, self.reg, _ = P.parse_id(rid); self.size = P.run_size(rid); self.privacy = privacy
        self.sess = O.session(ROOT / f"results/runs/{rid}/onnx/int8.onnx", threads); R = RunData(rid); prov = prov_int8(R); rng = np.random.default_rng(3)
        self.cal = {}
        for b, reps in BIN_REPS.items():
            T, Mh, Ms = bin_calib(R, prov, reps, 11); yB = R.y["valB"]; pick = rng.integers(0, 2, len(yB))
            zB = np.stack([prov("valB", reps[0]), prov("valB", reps[1])])[pick, np.arange(len(yB))]
            thr = cp.calibrate(apply_T(P.eff_logits(self.reg, zB), T), yB, "LAC", alpha); self.cal[b] = dict(T=T, Ms=Ms, thr=thr)
        self.timers = {k: [] for k in ("detect", "preprocess", "infer", "conformal", "quantify")}
        self.window = {b: [] for b in BIN_REPS}; self.sets = []; self.log = []

    def crops(self, frame, boxes):
        out, fp = [], []
        for x, y, w, h in boxes:
            if w <= 0: continue
            m = int(round(0.1 * w)); x0, y0 = max(0, x - m), max(0, y - m); x1, y1 = min(frame.shape[1], x + w + m), min(frame.shape[0], y + h + m)
            c = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY); out.append(cv2.resize(c, (48, 48), interpolation=cv2.INTER_AREA)); fp.append(w)
        return np.array(out), np.array(fp)

    def step(self, frame, boxes):
        t = time.perf_counter(); crops, fpx = self.crops(frame, boxes); self.timers["preprocess"].append(time.perf_counter() - t)
        if len(crops) == 0: return
        t = time.perf_counter(); z = O.ort_logits(self.sess, crops, self.arch, self.size); self.timers["infer"].append(time.perf_counter() - t)
        t = time.perf_counter(); r = np.minimum(48, np.round(fpx / K)).astype(int); bins = np.array([G.bin_of(v) for v in r]); sets = np.zeros(len(z), int)
        for b in BIN_REPS:
            k = bins == b
            if not k.any(): continue
            q = apply_T(P.eff_logits(self.reg, z[k]), self.cal[b]["T"]); self.window[b].append(q)
            sets[k] = cp.predict_sets(q, self.cal[b]["thr"], "LAC").sum(1)
        self.sets.append(sets); self.timers["conformal"].append(time.perf_counter() - t)
        self.last = dict(bins=bins, z=z, sets=sets)   # per-face data kept in memory only

    def flush(self, tstamp, B=60):
        t = time.perf_counter(); n = {b: (np.concatenate(v) if v else np.zeros((0, 8))) for b, v in self.window.items()}; tot = sum(len(v) for v in n.values())
        if tot == 0: self.window = {b: [] for b in BIN_REPS}; return None
        def est(nb):
            pi = np.zeros(8); w = sum(len(v) for v in nb.values())
            for b, q in nb.items():
                if len(q): pi += len(q) / w * Q.pacc(q, self.cal[b]["Ms"])
            return pi
        pi = est(n); rng = np.random.default_rng(0); bs = []
        for _ in range(B): bs.append(est({b: (v[rng.integers(0, len(v), len(v))] if len(v) else v) for b, v in n.items()}))
        bs = np.array(bs); nas = bs[:, NEG].sum(1); wbi = bs[:, 1] - nas; self.timers["quantify"].append(time.perf_counter() - t)
        rec = dict(t=tstamp, n_face_obs=int(tot), prevalence={c: float(v) for c, v in zip(CLASSES, pi)}, NAS=float(pi[NEG].sum()), NAS_ci=[float(x) for x in np.quantile(nas, [.025, .975])],
                   WBI=float(pi[1] - pi[NEG].sum()), WBI_ci=[float(x) for x in np.quantile(wbi, [.025, .975])]); self.log.append(rec); self.window = {b: [] for b in BIN_REPS}; return rec

    def latency(self, fps_n=None):
        out = {k: dict(median_ms=float(np.median(v) * 1000), p90_ms=float(np.percentile(v, 90) * 1000)) for k, v in self.timers.items() if v}
        tot = sum(v["median_ms"] for v in out.values() if True); out["total_median_ms_per_frame_excl_quantify"] = float(sum(v["median_ms"] for k, v in out.items() if k not in ("quantify",) and isinstance(v, dict)))
        out["fps_est"] = 1000 / out["total_median_ms_per_frame_excl_quantify"] if out["total_median_ms_per_frame_excl_quantify"] else None; return out


def dashboard(rec, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from m3.report.style import OI
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [2, 1]})
    axs[0].bar(range(8), [rec["prevalence"][c] for c in CLASSES], color=OI[0]); axs[0].set_xticks(range(8)); axs[0].set_xticklabels([c[:5] for c in CLASSES], fontsize=7); axs[0].set_ylabel("estimated class share (PACC)")
    for i, (k, c) in enumerate((("NAS", OI[3]), ("WBI", OI[2]))):
        lo, hi = rec[k + "_ci"]; axs[1].errorbar(i, rec[k], yerr=[[rec[k] - lo], [hi - rec[k]]], fmt="o", color=c, capsize=4)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["negative-affect\nshare", "well-being\nindex"], fontsize=7); axs[1].axhline(0, color="k", lw=.5); axs[1].set_xlim(-.5, 1.5)
    fig.suptitle("Aggregate class-level view (no per-student output) — mosaic of real FER+ test faces", fontsize=8); plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_mosaic(rid, frames=300, oracle=True, window_frames=50, mode="C"):
    from m3.data.mosaic import seat_layout, render
    from m3.demo.detector import Detector
    pl = Pipeline(rid); R = RunData(rid); D = R.D; y = R.y["test"]; pi_test = np.bincount(y, minlength=8) / len(y); rng = np.random.default_rng(21)
    seats = seat_layout(mode); n = len(seats); by = [np.where(y == k)[0] for k in range(8)]
    cls = rng.permutation(np.repeat(np.arange(8), rng.multinomial(n, rng.dirichlet(20 * pi_test)))); ii = np.array([rng.choice(by[k]) for k in cls])
    faces = D["img"][R.idx["test"][ii]]; true = np.bincount(y[ii], minlength=8) / n; det = Detector(); hits = tot = 0; first_frame = None; t0 = time.time()
    for t, (frame, boxes) in enumerate(render(faces, seats, frames)):
        if t == 0: first_frame = frame
        td = time.perf_counter(); dboxes = det.detect(frame); pl.timers["detect"].append(time.perf_counter() - td)
        valid = boxes[boxes[:, 2] > 0]; tot += len(valid)
        for b in valid:
            if len(dboxes) and max(_iou(b, d) for d in dboxes) > 0.3: hits += 1
        pl.step(frame, boxes if oracle else dboxes)
        if (t + 1) % window_frames == 0: pl.flush(t + 1)
    # also accumulate pooled estimate over all windows
    est = np.mean([[r["prevalence"][c] for c in CLASSES] for r in pl.log], 0)
    out = dict(run=rid, mode=mode, seats=n, true_prevalence={c: float(v) for c, v in zip(CLASSES, true)}, est_prevalence_mean_over_windows={c: float(v) for c, v in zip(CLASSES, est)},
               AE=float(np.abs(est - true).mean()), true_NAS=float(true[NEG].sum()), est_NAS=float(est[NEG].sum()), detector=det.kind, detector_recall_on_mosaic=hits / max(tot, 1), oracle_boxes=oracle)
    d = ROOT / "results/demo"; d.mkdir(parents=True, exist_ok=True); jdump(out, d / "mosaic_replay.json"); jdump(pl.latency(), d / "latency_profile.json"); jdump(pl.log, d / "aggregate_log.json")
    dashboard(pl.log[-1], ROOT / "results/figures/fig_demo_dashboard.png"); cv2.imwrite(str(d / "mosaic_frame0.png"), first_frame); print(json.dumps(out, indent=1)); return out


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by_, bw, bh = b; ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by_ + bh) - max(ay, by_)); i = ix * iy
    return i / (aw * ah + bw * bh - i + 1e-9)


def run_webcam(rid, seconds=120, debug=False):
    from m3.demo.detector import Detector
    pl = Pipeline(rid); det = Detector(); cap = cv2.VideoCapture(0); t0 = time.time(); last = t0
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if not ok: break
        td = time.perf_counter(); b = det.detect(f); pl.timers["detect"].append(time.perf_counter() - td); pl.step(f, b)
        if debug and b is not None:
            for (x, y, w, h), s in zip(b, getattr(pl, "last", {}).get("sets", [])): cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 1); cv2.putText(f, f"|C|={s}", (x, y - 3), 0, .5, (0, 255, 0), 1)
        if time.time() - last >= 10: pl.flush(time.time() - t0); last = time.time()
        if debug: cv2.imshow("debug (self-test only)", f); cv2.waitKey(1)
    cap.release(); jdump(pl.latency(), ROOT / "results/demo/latency_profile_webcam.json"); jdump(pl.log, ROOT / "results/demo/aggregate_log_webcam.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="mosaic"); ap.add_argument("--run", default="mnv3s-soft-s0"); ap.add_argument("--privacy", default="aggregate")
    ap.add_argument("--debug-faces", action="store_true"); ap.add_argument("--detector-boxes", action="store_true"); ap.add_argument("--seconds", type=int, default=120); a = ap.parse_args()
    if a.source == "mosaic": run_mosaic(a.run, oracle=not a.detector_boxes)
    else: run_webcam(a.run, a.seconds, a.debug_faces and a.privacy != "aggregate" or a.debug_faces)
