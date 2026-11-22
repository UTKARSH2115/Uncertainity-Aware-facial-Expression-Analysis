"""Pinhole geometry for the Raspberry Pi Camera v2 (IMX219)."""
import math
import numpy as np
import pandas as pd
from m3.utils.common import ROOT

F_MM = 3.04
MODES = {"A": (1920, 1.12), "B": (1640, 2.24), "C": (1280, 2.87), "D": (640, 5.74), "E": (1920, 1.913)}
ROWS = [2.0, 3.2, 4.4, 5.6, 6.8]
XS = [-3.15, -2.25, -1.35, -0.45, 0.45, 1.35, 2.25, 3.15]
BINS = {"B1": (40, 48), "B2": (24, 39), "B3": (16, 23), "B4": (1, 15)}


def face_px(d, p_um, W=0.14, f=F_MM):
    return (f * 1e-3 * W) / (p_um * 1e-6 * np.asarray(d, dtype=float))


def hfov_deg(N, p_um, f=F_MM):
    return math.degrees(2 * math.atan(N * p_um * 1e-3 / (2 * f)))


def eff_res(d, p_um, W=0.14, k=0.85):
    return np.minimum(48, np.round(face_px(d, p_um, W) / k)).astype(int)


def bin_of(r):
    r = int(r)
    for b, (lo, hi) in BINS.items():
        if lo <= r <= hi: return b
    return "B1" if r > 48 else "B4"


def seats(mode, W=0.14, k=0.85):
    N, p = MODES[mode]; half = hfov_deg(N, p) / 2; out = []
    for ri, dr in enumerate(ROWS):
        for x in XS:
            D = math.hypot(dr, x); vis = abs(math.degrees(math.atan2(abs(x), dr))) <= half
            out.append(dict(row=ri, x=x, dr=dr, D=D, visible=vis, r=int(eff_res(D, p, W, k))))
    return pd.DataFrame(out)


def geometry_table():
    rows = []
    for m, (N, p) in MODES.items():
        s = seats(m); v = s[s.visible]
        rows.append(dict(mode=m, N=N, p_eff_um=p, hfov_deg=hfov_deg(N, p), visible_seats=len(v), r_min=v.r.min(), r_median=v.r.median(), r_max=v.r.max()))
    return pd.DataFrame(rows)
