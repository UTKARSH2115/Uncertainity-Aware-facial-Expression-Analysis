"""tab_geometry_modes.csv, fig_classroom_map; sensitivity table."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from m3.utils.common import ROOT
from m3.geometry import camera as G
from m3.report.style import savefig


def main():
    t = G.geometry_table(); t.to_csv(ROOT / "results/tables/tab_geometry_modes.csv", index=False); print(t.round(2))
    sens = []
    for W in (0.12, 0.14, 0.16):
        for k in (0.75, 0.85, 1.0):
            for m in "BCD":
                s = G.seats(m, W, k); v = s[s.visible]; sens.append(dict(W=W, k=k, mode=m, visible=len(v), r_min=v.r.min(), r_median=v.r.median(), r_max=v.r.max()))
    pd.DataFrame(sens).to_csv(ROOT / "results/tables/tab_geometry_sensitivity.csv", index=False)
    fig, axs = plt.subplots(1, 5, figsize=(12, 3), sharey=True); vmax = 48
    for ax, m in zip(axs, "ABCDE"):
        s = G.seats(m); v = s[s.visible]; h = s[~s.visible]
        sc = ax.scatter(v.x, v.dr, c=v.r, cmap="viridis", vmin=0, vmax=vmax, s=90, marker="s"); ax.scatter(h.x, h.dr, c="k", marker="x", s=50)
        for _, r in v.iterrows(): ax.text(r.x, r.dr, str(r.r), ha="center", va="center", fontsize=5.5, color="w")
        ax.scatter([0], [0], marker="^", c="r", s=70); ax.set_title(f"Mode {m} ({len(v)}/40 seats)", fontsize=9); ax.set_xlabel("lateral offset (m)"); ax.invert_yaxis()
    axs[0].set_ylabel("distance from camera (m)"); plt.colorbar(sc, ax=axs, label="effective face resolution r (px)"); savefig(fig, "fig_classroom_map")


if __name__ == "__main__": main()
