import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
MEAN48, STD48 = 0.5077, 0.2551   # approx FER train stats; recomputed in train if desired
class TinyCNN(nn.Module):
    def __init__(s, nc=8):
        super().__init__(); L=[]; c=1
        for w in [16,32,64,128]:
            L+=[nn.Conv2d(c,w,3,padding=1,bias=False),nn.BatchNorm2d(w),nn.ReLU(True),
                nn.Conv2d(w,w,3,padding=1,bias=False),nn.BatchNorm2d(w),nn.ReLU(True),nn.MaxPool2d(2)]; c=w
        s.features=nn.Sequential(*L); s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,nc)
    def forward(s,x): return s.fc(s.drop(s.features(x).mean((2,3))))
def build(name, nc=8, pretrained=True):
    W="DEFAULT" if pretrained else None
    if name=="tinycnn": return TinyCNN(nc)
    if name=="mnv3s":
        m=torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3]=nn.Linear(1024,nc); return m
    if name=="r18":
        m=torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None); m.fc=nn.Linear(512,nc); return m
    raise ValueError(name)
def prep(x, arch, size=64):
    """x: float [B,1,48,48] in [0,1] -> model input."""
    if arch=="tinycnn": return (x-MEAN48)/STD48
    if size!=x.shape[-1]: x=F.interpolate(x,size=(size,size),mode="bilinear",align_corners=False)
    x=x.repeat(1,3,1,1)
    m=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); s=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return ((x-m)/s).contiguous(memory_format=torch.channels_last)
def set_head_dropout_only(model): pass
