import time, json, torch
from m3.models.zoo import build, prep
from m3.train.train import augment
from m3.losses.losses import soft_ce
from m3.utils.common import ROOT, jdump
torch.set_num_interop_threads(1)
res = {}
for th in [6, 10]:
    torch.set_num_threads(th)
    for arch, size in [("tinycnn", 48), ("mnv3s", 64), ("r18", 64)]:
        for bs in [128, 256]:
            m = build(arch)
            if arch != "tinycnn": m = m.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(m.parameters(), 1e-3); X = torch.rand(bs, 1, 48, 48); p = torch.softmax(torch.randn(bs, 8), 1)
            for i in range(23):
                if i == 3: t = time.time()
                x, _ = augment(X); l = soft_ce(m(prep(x, arch, size)), p); opt.zero_grad(); l.backward(); opt.step()
            s = (time.time() - t) / 20; e = s * 26940 / bs
            print(th, arch, bs, f"{s:.3f}s/it -> {e:.0f}s/epoch", flush=True); res[f"{th}-{arch}-{bs}"] = e
jdump(res, ROOT / "results/bench_speed.json")
