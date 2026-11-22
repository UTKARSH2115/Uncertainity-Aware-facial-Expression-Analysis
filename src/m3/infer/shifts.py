"""Logits under distribution shifts, TTA, MC-dropout. Saved under results/runs/<id>/logits_shift/<shift>/<split>.npy"""
import cv2
import numpy as np, torch
from m3.utils.common import ROOT, load_npz
from m3.data.degrade import resolution, corrupt, CORRUPTIONS
from m3.infer import predict as P
from m3.models.zoo import prep

RES = [40, 32, 24, 20, 16, 12, 8]
TTA_SHIFTS = [(0, 0), (2, 0), (0, 2), (-2, -2)]


def shifted_logits(rid, shift, split, model=None, D=None, idx=None):
    f = ROOT / f"results/runs/{rid}/logits_shift/{shift}/{split}.npy"
    if f.exists(): return np.load(f)
    D = D or load_npz(); idx = idx or P.split_idx(D); arch, _, _ = P.parse_id(rid); model = model or P.load_model(rid)
    ii = idx[split]; imgs = D["img"][ii]
    if shift.startswith("res"): imgs = resolution(imgs, int(shift[3:]))
    else:
        name, s = shift.rsplit("_s", 1); imgs = corrupt(imgs, name, int(s), ii)
    z = P.get_logits(model, arch, imgs, P.run_size(rid)); f.parent.mkdir(parents=True, exist_ok=True); np.save(f, z); return z


def tta_logits(rid, model=None, D=None, idx=None):
    f = ROOT / f"results/runs/{rid}/logits_tta_test.npy"
    if f.exists(): return np.load(f)
    D = D or load_npz(); idx = idx or P.split_idx(D); arch, _, _ = P.parse_id(rid); model = model or P.load_model(rid); size = P.run_size(rid)
    imgs = D["img"][idx["test"]]; out = []
    for flip in (False, True):
        for dx, dy in TTA_SHIFTS:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            sh = np.stack([cv2.warpAffine(im, M, (48, 48), borderMode=cv2.BORDER_REPLICATE) for im in imgs])
            out.append(P.get_logits(model, arch, sh, size, flip=flip))
    out = np.stack(out); np.save(f, out); return out


@torch.no_grad()
def mcd_logits(rid, passes=20, model=None, D=None, idx=None, seed=0):
    f = ROOT / f"results/runs/{rid}/logits_mcd_test.npy"
    if f.exists(): return np.load(f)
    D = D or load_npz(); idx = idx or P.split_idx(D); arch, _, _ = P.parse_id(rid); model = model or P.load_model(rid); size = P.run_size(rid)
    imgs = D["img"][idx["test"]]; model.eval(); feats = []
    for i in range(0, len(imgs), 256):
        x = prep(torch.from_numpy(imgs[i:i + 256]).float().div(255).unsqueeze(1), arch, size)
        if arch == "tinycnn": h = model.features(x).mean((2, 3))
        else:
            h = model.avgpool(model.features(x)).flatten(1); h = model.classifier[1](model.classifier[0](h))
        feats.append(h)
    h = torch.cat(feats); torch.manual_seed(seed); out = []
    drop, lin = (model.drop, model.fc) if arch == "tinycnn" else (model.classifier[2], model.classifier[3])
    drop.train()
    for _ in range(passes): out.append(lin(drop(h)).numpy())
    drop.eval(); out = np.stack(out); np.save(f, out); return out
