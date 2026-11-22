import json, platform, subprocess, sys, hashlib
import torch, onnxruntime, numpy, scipy, sklearn
from m3.utils.common import ROOT, jdump

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
v = {}
try: v = json.load(open(ROOT / "results/data_verify.json"))
except Exception: pass
env = dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(), torch=torch.__version__, cuda=torch.cuda.is_available(), gpu="none",
           ort=onnxruntime.__version__, numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch_threads=torch.get_num_threads(),
           fer2013_sha256=v.get("fer2013_sha256"), fer2013_md5=v.get("fer2013_md5"), pip_freeze=freeze)
jdump(env, ROOT / "results/env.json"); print("wrote results/env.json")
