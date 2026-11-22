"""Latency / size / complexity. python -m m3.edge.bench --archs tinycnn mnv3s [--run-of tinycnn=tinycnn-soft-s0 ...]"""
import argparse, json, platform, time
import cv2
import numpy as np, pandas as pd, torch
from m3.utils.common import ROOT
from m3.edge import onnx_tools as O
from m3.models.zoo import build
from m3.infer import predict as P

YUNET = ROOT / "models_ext/face_detection_yunet_2023mar.onnx"


def complexity(archs=("tinycnn", "mnv3s", "r18", "effb0_na")):
    from torchinfo import summary
    rows = []
    for a in archs:
        if a.endswith("_na"): continue
        size = 48 if a == "tinycnn" else 64; m = build(a, pretrained=False); ch = 1 if a == "tinycnn" else 3
        s = summary(m, input_size=(1, ch, size, size), verbose=0); rows.append(dict(arch=a, input=size, params=s.total_params, MACs=s.total_mult_adds, fp32_kib=s.total_params * 4 / 1024))
    return pd.DataFrame(rows)


def time_fn(fn, warm=50, runs=500):
    for _ in range(warm): fn()
    t = []
    for _ in range(runs): t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 90))


def detector_time(w, h, runs=100):
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h)); rng = np.random.default_rng(0); img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return time_fn(lambda: det.detect(img), 10, runs)[0]


def main(run_of, out="results/tables/tab_latency_budget.csv"):
    D = np.load(ROOT / "data/processed/ferplus.npz"); imgs = D["img"][:64]; rows = []
    det = {"640x480": detector_time(640, 480), "1280x720": detector_time(1280, 720)}
    for arch, rid in run_of.items():
        size = P.run_size(rid); d = O.onnx_dir(rid)
        if not (d / "fp32.onnx").exists(): O.export_fp32(rid)
        for prec in ("fp32", "int8"):
            f = d / f"{prec}.onnx"
            if not f.exists(): continue
            for th in (1, 4):
                s = O.session(f, th)
                for b in (1, 8):
                    x = O.preprocess_np(imgs[:b], arch, size); med, p90 = time_fn(lambda: s.run(None, {"input": x}))
                    pre = time_fn(lambda: O.preprocess_np(imgs[:1], arch, size), 20, 200)[0]
                    mpf = max(0, int((200 - det["640x480"]) // (pre + med / b))) if th else 0
                    rows.append(dict(device=platform.processor()[:30] or "laptop-cpu", arch=arch, precision=prec, threads=th, batch=b, median_ms=med, p90_ms=p90,
                                     size_kib=f.stat().st_size / 1024, det_ms_640=det["640x480"], det_ms_1280=det["1280x720"], pre_ms=pre, max_faces_at_5fps=mpf))
    df = pd.DataFrame(rows); df.to_csv(ROOT / out, index=False); complexity().to_csv(ROOT / "results/tables/model_complexity.csv", index=False); return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-of", nargs="+", default=["tinycnn=tinycnn-soft-s0", "mnv3s=mnv3s-soft-s0"]); a = ap.parse_args()
    print(main(dict(x.split("=") for x in a.run_of)).round(2).to_string())
