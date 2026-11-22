import numpy as np
FER2PLUS = {0:4, 1:5, 2:6, 3:1, 4:3, 5:2, 6:0}
def process_row(votes, name):
    """votes: 10 ints (8 emotions, unknown, NF). Returns dict(keep, reason, y, tied, p, n)."""
    v = np.asarray(votes, dtype=int); e = v[:8]; u, nf = v[8], v[9]
    if not isinstance(name, str) or name.strip() == "":
        return dict(keep=False, reason="F1")
    if e.sum() == 0: return dict(keep=False, reason="F2")
    if max(u, nf) >= e.max(): return dict(keep=False, reason="F3")
    tied = int((e == e.max()).sum()) > 1
    return dict(keep=True, reason="", y=int(e.argmax()), tied=tied, p=(e/e.sum()).astype(np.float32), n=e)
