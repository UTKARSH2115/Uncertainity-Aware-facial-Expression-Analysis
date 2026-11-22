"""Deterministic degradations of real 48x48 FER+ faces (uint8 in, uint8 out)."""
import cv2
import numpy as np

SEV = {
    "gauss_noise": [0.02, 0.04, 0.06, 0.09, 0.12],
    "motion_blur": [3, 5, 7, 9, 11],
    "defocus_blur": [0.5, 1.0, 1.5, 2.0, 2.5],
    "jpeg": [80, 60, 40, 25, 10],
    "low_light": [(1.3, .01), (1.6, .02), (2.0, .03), (2.5, .04), (3.0, .05)],
    "contrast": [0.75, 0.60, 0.45, 0.30, 0.20],
    "occ_lower": [0.2, 0.3, 0.4, 0.5, 0.6],
    "occ_eyes": [0.08, 0.12, 0.16, 0.20, 0.24],
    "rotate": [5, 10, 15, 20, 30],
}
CORRUPTIONS = list(SEV)


def resolution(imgs, r):
    if r >= 48: return imgs.copy()
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        out[i] = cv2.resize(cv2.resize(im, (r, r), interpolation=cv2.INTER_AREA), (48, 48), interpolation=cv2.INTER_LINEAR)
    return out


def _disk(R):
    n = int(np.ceil(R)) * 2 + 1; c = n // 2; yy, xx = np.mgrid[:n, :n]; d = np.hypot(yy - c, xx - c)
    k = np.clip(R + 0.5 - d, 0, 1); return k / k.sum()


def _line(L, ang):
    k = np.zeros((L, L), np.float32); c = L // 2; dx, dy = np.cos(ang) * c, np.sin(ang) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1); return k / k.sum()


def corrupt(imgs, name, severity, idx=None):
    """severity 0 = identity. Seed per image = idx + 1e6*severity."""
    if severity == 0: return imgs.copy()
    v = SEV[name][severity - 1]; out = np.empty_like(imgs)
    idx = np.arange(len(imgs)) if idx is None else idx
    for j, im in enumerate(imgs):
        rng = np.random.default_rng(int(idx[j]) + 10 ** 6 * severity); x = im.astype(np.float32) / 255
        if name == "gauss_noise": x = x + rng.normal(0, v, x.shape)
        elif name == "motion_blur": x = cv2.filter2D(x, -1, _line(int(v), rng.uniform(0, np.pi)), borderType=cv2.BORDER_REFLECT)
        elif name == "defocus_blur": x = cv2.filter2D(x, -1, _disk(v).astype(np.float32), borderType=cv2.BORDER_REFLECT)
        elif name == "jpeg": x = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(v)])[1], 0).astype(np.float32) / 255
        elif name == "low_light": x = x ** v[0] + rng.normal(0, v[1], x.shape)
        elif name == "contrast": m = x.mean(); x = (x - m) * v + m
        elif name == "occ_lower": x = x.copy(); x[int(round(48 * (1 - v))):] = x.mean()
        elif name == "occ_eyes": x = x.copy(); h = int(round(48 * v)); c = int(round(0.35 * 48)); x[max(0, c - h // 2):c + (h - h // 2)] = x.mean()
        elif name == "rotate":
            M = cv2.getRotationMatrix2D((23.5, 23.5), v * rng.choice([-1, 1]), 1.0); x = cv2.warpAffine(x, M, (48, 48), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[j] = np.clip(np.round(x * 255), 0, 255).astype(np.uint8)
    return out
