import random, json, yaml
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
CLASSES = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
NEG = [3,4,5,6,7]  # sadness, anger, disgust, fear, contempt
def seed_everything(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
def cfg():
    return yaml.safe_load(open(ROOT/"configs/base.yaml"))
def load_npz():
    return dict(np.load(ROOT/"data/processed/ferplus.npz", allow_pickle=False))
def jdump(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    json.dump(o, open(p,"w"), indent=1, default=float)
