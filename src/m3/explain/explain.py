"""Phase 10: Grad-CAM (mnv3s last conv block) + region-occlusion sensitivity."""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from m3.utils.common import ROOT, CLASSES
from m3.infer import predict as P
from m3.models.zoo import prep
from m3.report.style import savefig, OI
from m3.evalrun import RunData

REGIONS = {"upper": (0, .45, 0, 1), "lower": (.55, 1, 0, 1), "left": (0, 1, 0, .5), "right": (0, 1, .5, 1), "eyes": (.25, .45, 0, 1), "mouth": (.65, .90, 0, 1)}


def gradcam(model, imgs, size=64, targets=None):
    acts, grads = {}, {}
    layer = model.features[-1]; h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o)); h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval(); x = prep(torch.from_numpy(imgs).float().div(255).unsqueeze(1), "mnv3s", size).requires_grad_(True); z = model(x)
    t = z.argmax(1) if targets is None else torch.as_tensor(targets); z.gather(1, t[:, None]).sum().backward(); h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True); cam = F.relu((w * acts["a"]).sum(1, keepdim=True)); cam = F.interpolate(cam, size=(48, 48), mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True); cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8); return cam.detach().numpy()


def occlude(imgs, region):
    r0, r1, c0, c1 = REGIONS[region]; x = imgs.copy(); m = x.mean((1, 2), keepdims=True).astype(np.uint8)
    x[:, int(r0 * 48):int(r1 * 48), int(c0 * 48):int(c1 * 48)] = np.broadcast_to(m, x.shape)[:, int(r0 * 48):int(r1 * 48), int(c0 * 48):int(c1 * 48)]; return x


def main(arch="mnv3s", seed=0):
    rids = {g: f"{arch}-{g}-s{seed}" for g in ("maj", "soft")}; R = {g: RunData(r) for g, r in rids.items()}; M = {g: P.load_model(r) for g, r in rids.items()}
    D = R["soft"].D; idx = R["soft"].idx["test"]; imgs = D["img"][idx]; y = R["soft"].y["test"]; p = R["soft"].p["test"]; m = p.max(1); size = P.run_size(rids["soft"]); rng = np.random.default_rng(0)
    # Grad-CAM grid: per class, 1 clear + 1 ambiguous face; rows: image / maj CAM / soft CAM
    fig, axs = plt.subplots(6, 8, figsize=(12, 9))
    for c in range(8):
        for gi, (grp, name) in enumerate(((m >= .8, "clear"), (m <= .5, "ambiguous"))):
            ii = np.where(grp & (y == c))[0]
            if len(ii) == 0: continue
            i = rng.choice(ii); x = imgs[i:i + 1]
            axs[3 * gi, c].imshow(x[0], cmap="gray"); axs[3 * gi, c].set_title(f"{CLASSES[c][:5]} ({name})", fontsize=7)
            for k, g in enumerate(("maj", "soft")):
                cam = gradcam(M[g], x, size, targets=[c])[0]; axs[3 * gi + 1 + k, c].imshow(x[0], cmap="gray"); axs[3 * gi + 1 + k, c].imshow(cam, cmap="jet", alpha=.5)
                if c == 0: axs[3 * gi + 1 + k, c].set_ylabel(g, fontsize=8)
    for a in axs.flat: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); savefig(fig, "fig_gradcam_grid")
    # region occlusion
    rows = []
    for g in ("maj", "soft"):
        q0 = R[g].q(R[g].z["test"]); base_acc = q0.argmax(1) == y
        for reg in REGIONS:
            q = R[g].q(P.get_logits(M[g], arch, occlude(imgs, reg), size))
            for c in range(8):
                k = y == c
                if k.sum() < 10: continue
                rows.append(dict(model=g, region=reg, **{"class": CLASSES[c]}, n=int(k.sum()), drop_q=float((q0[k, c] - q[k, c]).mean()), drop_acc=float(base_acc[k].mean() - (q.argmax(1) == y)[k].mean())))
    df = pd.DataFrame(rows); df.to_csv(ROOT / "results/tables/tab_region_importance.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, g in zip(axs, ("maj", "soft")):
        pv = df[df.model == g].pivot(index="class", columns="region", values="drop_q").reindex([c for c in CLASSES if c in set(df["class"])])[list(REGIONS)]
        im = ax.imshow(pv.values, cmap="viridis"); ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, rotation=30); ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=8); ax.set_title(f"{arch} {g}"); plt.colorbar(im, ax=ax, label="mean drop in q[y_maj]")
    plt.tight_layout(); savefig(fig, "fig_region_importance")


if __name__ == "__main__": main()
