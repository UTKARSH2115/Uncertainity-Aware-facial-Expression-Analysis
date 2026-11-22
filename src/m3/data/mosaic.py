"""Mosaic 'virtual classroom' frames built from REAL FER+ test faces placed at seat positions/sizes of the camera geometry."""
import cv2
import numpy as np
from m3.geometry import camera as G

CAM_H, HEAD_H = 2.5, 1.1   # assumed camera and seated-head heights (m); stated as assumption in the report


def seat_layout(mode="C", W=1280, H=720):
    N, p = G.MODES[mode]; s = G.seats(mode); s = s[s.visible].reset_index(drop=True); fpx = G.F_MM * 1e-3 / (p * 1e-6)
    s["u"] = W / 2 + fpx * s.x / s.dr; s["v"] = H / 2 + fpx * (CAM_H - HEAD_H) / s.dr * 0.5
    s["face_px"] = G.face_px(s.D, p); s["side"] = np.maximum(4, np.round(s.face_px / 0.85)).astype(int); return s


def render(faces, seats, frames=300, W=1280, H=720, jitter=2, seed=0):
    """Yield (frame_bgr, boxes[x,y,w,h]) for each frame. faces: uint8 [n,48,48] (one per seat)."""
    rng = np.random.default_rng(seed)
    for t in range(frames):
        img = np.full((H, W), 110, np.uint8); boxes = []
        for i, r in seats.iterrows():
            sd = int(r.side); f = cv2.resize(faces[i], (sd, sd), interpolation=cv2.INTER_AREA if sd < 48 else cv2.INTER_LINEAR)
            dx, dy = rng.integers(-jitter, jitter + 1, 2); x0 = int(r.u - sd / 2) + dx; y0 = int(r.v - sd / 2) + dy
            if x0 < 0 or y0 < 0 or x0 + sd > W or y0 + sd > H: boxes.append([0, 0, 0, 0]); continue
            img[y0:y0 + sd, x0:x0 + sd] = f; boxes.append([x0, y0, sd, sd])
        img = np.clip(img.astype(np.float32) + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
        yield cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), np.array(boxes)
