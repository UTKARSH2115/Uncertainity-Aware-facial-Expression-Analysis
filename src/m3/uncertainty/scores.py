"""Confidence scores (higher = more confident). Each takes probabilities q [N,8]."""
import numpy as np
from m3.calib.metrics import entropy

COST = {"MSP": 1, "ENT": 1, "MARGIN": 1, "VAC": 1, "ENS3": 3, "MCD20": 20, "TTA8": 8}


def msp(q): return q.max(1)
def neg_ent(q): return -entropy(q)
def margin(q): s = np.sort(q, 1); return s[:, -1] - s[:, -2]
