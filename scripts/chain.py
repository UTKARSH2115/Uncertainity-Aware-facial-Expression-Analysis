"""Wait for the P0 matrix to finish, then run P1 (teacher first, then students)."""
import time, subprocess, sys
from m3.utils.common import ROOT
last = ROOT / "results/runs/mnv3s-soft-s2/logits/test.npy"
while not last.exists(): time.sleep(60)
time.sleep(30)
for tier in ("P1T", "P1"): subprocess.run([sys.executable, str(ROOT / "scripts/run_matrix.py"), "--tier", tier])
