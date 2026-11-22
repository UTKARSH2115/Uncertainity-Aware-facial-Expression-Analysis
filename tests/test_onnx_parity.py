import numpy as np, pytest
from m3.utils.common import ROOT, load_npz
from m3.edge.onnx_tools import preprocess_np, export_fp32, session, ort_logits
from m3.infer import predict as P
from m3.models.zoo import prep
import torch

RID = "mnv3s-soft-s9"


@pytest.mark.skipif(not (ROOT / f"results/runs/{RID}/ckpt_last.pt").exists(), reason="no run")
def test_parity():
    D = load_npz(); idx = P.split_idx(D)["test"][:256]; imgs = D["img"][idx]
    for arch in ("tinycnn", "mnv3s"):
        x = torch.from_numpy(imgs).float().div(255).unsqueeze(1)
        assert np.abs(prep(x, arch, 64).numpy() - preprocess_np(imgs, arch, 64)).max() < 1e-5
    f = export_fp32(RID); s = session(f); m = P.load_model(RID); ref = P.get_logits(m, "mnv3s", imgs, 64); got = ort_logits(s, imgs, "mnv3s", 64)
    assert np.abs(ref - got).max() < 1e-3 and (ref.argmax(1) == got.argmax(1)).all()
