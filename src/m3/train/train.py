import argparse, math, time, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from m3.utils.common import ROOT, cfg, load_npz, jdump, seed_everything
from m3.models.zoo import build, prep
from m3.losses.losses import soft_ce, hard_ce, dm_nll, kd_kl
from m3.infer.predict import parse_id, get_logits, split_idx, save_run_logits
import m3.infer.predict as P

AUG = dict(rrc_scale=[0.85, 1.0], rrc_ratio=[0.9, 1.1], hflip=0.5, rotate_deg=10, brightness=0.2, contrast=0.2)


def augment(x, A=AUG, ra_levels=None, ra_prob=0.0):
    """Batched augmentation. x float [B,1,48,48] in [0,1]; returns (x_aug, flip_mask)."""
    B = x.shape[0]; u = lambda lo, hi: torch.empty(B).uniform_(lo, hi)
    area = u(*A["rrc_scale"]); ratio = torch.exp(u(math.log(A["rrc_ratio"][0]), math.log(A["rrc_ratio"][1])))
    sx = torch.sqrt(area * ratio).clamp(max=1); sy = torch.sqrt(area / ratio).clamp(max=1)
    th = torch.deg2rad(u(-A["rotate_deg"], A["rotate_deg"])); flip = torch.rand(B) < A["hflip"]
    tx = (torch.rand(B) * 2 - 1) * (1 - sx); ty = (torch.rand(B) * 2 - 1) * (1 - sy)
    fs = torch.where(flip, -1.0, 1.0)
    T = torch.zeros(B, 2, 3)
    T[:, 0, 0] = sx * torch.cos(th) * fs; T[:, 0, 1] = -sy * torch.sin(th); T[:, 0, 2] = tx
    T[:, 1, 0] = sx * torch.sin(th) * fs; T[:, 1, 1] = sy * torch.cos(th); T[:, 1, 2] = ty
    g = F.affine_grid(T, list(x.shape), align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
    if ra_levels and ra_prob > 0:
        sel = torch.rand(B) < ra_prob; lv = torch.tensor(ra_levels)[torch.randint(0, len(ra_levels), (B,))]
        for r in set(lv[sel].tolist()):
            m = sel & (lv == r)
            x[m] = F.interpolate(F.interpolate(x[m], size=(r, r), mode="area"), size=(48, 48), mode="bilinear", align_corners=False)
    mean = x.mean((1, 2, 3), keepdim=True)
    c = u(1 - A["contrast"], 1 + A["contrast"]).view(B, 1, 1, 1); b = u(1 - A["brightness"], 1 + A["brightness"]).view(B, 1, 1, 1)
    return (((x - mean) * c + mean) * b).clamp(0, 1), flip


def ece_np(p, y, bins=15):
    c = p.max(1); a = (p.argmax(1) == y); e = 0
    for i in range(bins):
        m = (c > i / bins) & (c <= (i + 1) / bins)
        if m.any(): e += m.mean() * abs(a[m].mean() - c[m].mean())
    return e


def run(rid, epochs=None, size=64, bs=128, threads=None):
    arch, reg, seed = parse_id(rid); C = cfg(); T = C["train"]
    epochs = epochs or T["epochs"][arch]; lr = T["lr"][arch]
    if arch == "tinycnn": size = 48
    rd = ROOT / f"results/runs/{rid}"; rd.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads or int(os.environ.get("M3_THREADS", 10)))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    seed_everything(seed); rng = np.random.default_rng(seed)
    d = load_npz(); tr = np.where((d["split"] == "train") & d["main"])[0]; sp = split_idx(d)
    X = torch.from_numpy(d["img"][tr]); p = torch.from_numpy(d["p"][tr]); n = torch.from_numpy(d["n"][tr].astype(np.float32))
    y = torch.from_numpy(d["y_maj"][tr].astype(np.int64))
    if reg == "single":
        r = np.random.default_rng(seed); cnt = d["n"][tr].astype(float)
        ys = np.array([r.choice(8, p=c / c.sum()) for c in cnt]); np.save(rd / "single_labels.npy", ys); y_s = torch.from_numpy(ys.astype(np.int64))
    if reg in ("kdct", "kdt"):
        TL = torch.from_numpy(np.load(ROOT / "results/runs/r18-soft-s0/teacher_train_logits.npy"))
    model = build(arch)
    if arch != "tinycnn": model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=T["weight_decay"])
    N = len(tr); spe = math.ceil(N / bs); total = epochs * spe; wu = T["warmup_epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, total - wu))))
    L = C["losses"]; log = []; ep0 = 0; t0 = time.time(); ck = rd / "ckpt_last.pt"
    if ck.exists():
        s = torch.load(ck, weights_only=False); model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        ep0 = s["epoch"]; log = s["log"]; rng.bit_generator.state = s["rng"]; torch.set_rng_state(s["trng"]); t0 -= s.get("elapsed", 0)
    jdump(dict(rid=rid, epochs=epochs, lr=lr, bs=bs, size=size, threads=torch.get_num_threads()), rd / "config.json")
    nan = False
    for ep in range(ep0, epochs):
        model.train(); perm = rng.permutation(N); te = time.time(); tl = 0
        for i in range(0, N, bs):
            idx = torch.from_numpy(perm[i:i + bs]); x = X[idx].float().div(255).unsqueeze(1)
            x, fl = augment(x, AUG, L["ra_levels"] if reg == "ra" else None, L["ra_prob"] if reg == "ra" else 0)
            z = model(prep(x, arch, size))
            if reg == "maj": loss = hard_ce(z, y[idx])
            elif reg == "single": loss = hard_ce(z, y_s[idx])
            elif reg in ("soft", "ra"): loss = soft_ce(z, p[idx])
            elif reg == "ls": loss = hard_ce(z, y[idx], L["label_smoothing"])
            elif reg == "dm": loss = dm_nll(z, n[idx])
            else:
                zt = TL[idx, fl.long()]; base = soft_ce(z, p[idx]) if reg == "kdct" else hard_ce(z, y[idx])
                loss = 0.5 * base + 0.5 * kd_kl(z, zt, L["kd_tau"])
            if not math.isfinite(loss.item()): nan = True; break
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        if nan: print("NaN loss"); break
        pv = P.probs_from_logits(reg, get_logits(model, arch, d["img"][sp["valA"]], size)); yv = d["y_maj"][sp["valA"]]; pt = d["p"][sp["valA"]]
        row = dict(epoch=ep + 1, train_loss=tl / N, val_acc=float((pv.argmax(1) == yv).mean()),
                   val_soft_nll=float(-(pt * np.log(np.clip(pv, 1e-12, 1))).sum(1).mean()), val_ece=float(ece_np(pv, yv)),
                   lr=sched.get_last_lr()[0], secs=time.time() - te)
        log.append(row); print(rid, {k: round(v, 4) for k, v in row.items()}, flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, log=log,
                        rng=rng.bit_generator.state, trng=torch.get_rng_state(), elapsed=time.time() - t0), ck)
    pd.DataFrame(log).to_csv(rd / "train_log.csv", index=False)
    jdump(dict(rid=rid, wall_s=time.time() - t0, nan=nan, input_size=size, torch=torch.__version__, device="cpu",
               final_train_loss=log[-1]["train_loss"] if log else None, end=time.ctime()), rd / "meta.json")
    if not nan: save_run_logits(rid)
    return not nan


def export_teacher_logits(rid="r18-soft-s0"):
    """Offline KD: teacher logits on clean and h-flipped train images, shape [N_train,2,8]."""
    d = load_npz(); tr = np.where((d["split"] == "train") & d["main"])[0]; arch, _, _ = parse_id(rid)
    m = P.load_model(rid); s = P.run_size(rid); imgs = d["img"][tr]
    out = np.stack([get_logits(m, arch, imgs, s), get_logits(m, arch, imgs, s, flip=True)], 1)
    np.save(ROOT / f"results/runs/{rid}/teacher_train_logits.npy", out.astype(np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--size", type=int, default=64); ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args(); run(a.run, a.epochs, a.size, a.bs)
