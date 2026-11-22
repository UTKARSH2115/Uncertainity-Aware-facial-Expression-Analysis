"""ONNX export, NumPy preprocessing, static INT8 PTQ (QDQ), ORT inference."""
import json, time
import cv2
import numpy as np, torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
from m3.utils.common import ROOT, load_npz
from m3.infer import predict as P
from m3.models.zoo import MEAN48, STD48

OPS = ["Conv", "Gemm", "MatMul"]
CAL = {"MinMax": CalibrationMethod.MinMax, "Entropy": CalibrationMethod.Entropy, "Percentile": CalibrationMethod.Percentile}


def preprocess_np(imgs_u8, arch, size=64):
    """NumPy twin of m3.models.zoo.prep. imgs_u8 [N,48,48] uint8 -> float32 NCHW."""
    x = imgs_u8.astype(np.float32) / 255
    if arch == "tinycnn": return ((x - MEAN48) / STD48)[:, None].astype(np.float32)
    if size != 48: x = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR) for im in x])
    x = np.repeat(x[:, None], 3, 1)
    m = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1); s = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    return ((x - m) / s).astype(np.float32)


def onnx_dir(rid): d = ROOT / f"results/runs/{rid}/onnx"; d.mkdir(parents=True, exist_ok=True); return d


def export_fp32(rid):
    arch, _, _ = P.parse_id(rid); size = P.run_size(rid); m = P.load_model(rid).float().eval()
    m = m.to(memory_format=torch.contiguous_format); f = onnx_dir(rid) / "fp32.onnx"
    torch.onnx.export(m, torch.randn(1, 1 if arch == "tinycnn" else 3, size, size), str(f), opset_version=17, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    return f


class Reader(CalibrationDataReader):
    def __init__(self, x): self.it = iter([{"input": x[i:i + 1]} for i in range(len(x))])
    def get_next(self): return next(self.it, None)


def calib_images(n=512, seed=0):
    D = load_npz(); tr = np.where((D["split"] == "train") & D["main"])[0]; rng = np.random.default_rng(seed); per = n // 8; sel = []
    for k in range(8):
        ii = tr[D["y_maj"][tr] == k]; sel += list(rng.choice(ii, min(per, len(ii)), replace=False))
    rest = n - len(sel); sel += list(rng.choice(np.setdiff1d(tr, sel), rest, replace=False)); return D["img"][np.array(sel)]


def quantize(rid, method="Percentile", tag="int8"):
    arch, _, _ = P.parse_id(rid); size = P.run_size(rid); d = onnx_dir(rid); src = d / "fp32.onnx"
    if not src.exists(): export_fp32(rid)
    pre = d / "fp32_pre.onnx"
    try: quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    except Exception as e: print("pre-process skipped:", e); pre = src
    x = preprocess_np(calib_images(), arch, size); out = d / f"{tag}.onnx"
    excl = fp32_nodes_to_keep(pre, arch)
    extra = {"CalibPercentile": 99.999} if method == "Percentile" else {}
    quantize_static(str(pre), str(out), Reader(x), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, calibrate_method=CAL[method], extra_options=extra, op_types_to_quantize=OPS, nodes_to_exclude=excl)
    return out


def fp32_nodes_to_keep(model_path, arch):
    """Pretrained MobileNet-style nets have per-channel activation outliers (first conv, depthwise convs) that break per-tensor
    INT8 activations; keep those Conv nodes in FP32 (spec 15.2: keep problematic ops in FP32). TinyCNN is fully quantised."""
    import onnx
    if arch == "tinycnn": return []
    m = onnx.load(str(model_path)); convs = [n for n in m.graph.node if n.op_type == "Conv"]
    grp = lambda n: next((a.i for a in n.attribute if a.name == "group"), 1)
    return [convs[0].name] + [n.name for n in convs if grp(n) > 1]


def session(path, threads=1):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(sess, imgs_u8, arch, size, bs=256):
    out = []
    for i in range(0, len(imgs_u8), bs): out.append(sess.run(None, {"input": preprocess_np(imgs_u8[i:i + bs], arch, size)})[0])
    return np.concatenate(out)
