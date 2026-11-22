import argparse, json, numpy as np, torch
import torch.nn.functional as F
from m3.utils.common import ROOT, load_npz
from m3.models.zoo import build, prep


def split_idx(d):
    return {"valA": np.where(d["split"] == "valA")[0], "valB": np.where(d["split"] == "valB")[0],
            "test": np.where((d["split"] == "test") & d["main"])[0]}


@torch.no_grad()
def get_logits(model, arch, imgs, size=64, bs=256, flip=False):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + bs])).float().div(255).unsqueeze(1)
        if flip: x = x.flip(-1)
        out.append(model(prep(x, arch, size)).float().numpy())
    return np.concatenate(out)


def parse_id(rid):
    a, r, s = rid.split("-"); return a, r, int(s[1:])


def load_model(rid):
    arch, reg, seed = parse_id(rid); m = build(arch, pretrained=False)
    ck = torch.load(ROOT / f"results/runs/{rid}/ckpt_last.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    if arch != "tinycnn": m = m.to(memory_format=torch.channels_last)
    return m.eval()


def run_size(rid):
    return json.load(open(ROOT / f"results/runs/{rid}/meta.json")).get("input_size", 64)


def probs_from_logits(regime, z):
    z = torch.as_tensor(np.asarray(z), dtype=torch.float64)
    if regime == "dm":
        a = F.softplus(z) + 1; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(z, 1).numpy()


def eff_logits(regime, z):
    """Logits usable for temperature scaling (DM: log q)."""
    if regime == "dm": return np.log(np.clip(probs_from_logits(regime, z), 1e-12, 1))
    return np.asarray(z, dtype=np.float64)


def save_run_logits(rid):
    d = load_npz(); arch, _, _ = parse_id(rid); m = load_model(rid); size = run_size(rid)
    o = ROOT / f"results/runs/{rid}/logits"; o.mkdir(parents=True, exist_ok=True)
    for k, idx in split_idx(d).items(): np.save(o / f"{k}.npy", get_logits(m, arch, d["img"][idx], size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); a = ap.parse_args(); save_run_logits(a.run)
