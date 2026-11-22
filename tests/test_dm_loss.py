import torch, numpy as np
from scipy.stats import dirichlet_multinomial as dmn
from scipy.special import gammaln
from m3.losses.losses import dm_nll, soft_ce
def test_dm():
    torch.manual_seed(0); z=torch.randn(6,8,dtype=torch.float64,requires_grad=True)
    n=torch.tensor(np.random.default_rng(0).multinomial(10,np.ones(8)/8,size=6),dtype=torch.float64)
    a=(torch.nn.functional.softplus(z)+1).detach().numpy()
    ref=np.mean([-dmn.logpmf(n[i].numpy(),a[i],10) + (gammaln(11)-gammaln(n[i].numpy()+1).sum()) for i in range(6)])
    l=dm_nll(z,n); assert abs(l.item()-ref)<1e-4
    l.backward(); assert torch.isfinite(z.grad).all()
def test_soft_ce_onehot():
    z=torch.randn(5,8); y=torch.randint(0,8,(5,)); p=torch.nn.functional.one_hot(y,8).float()
    assert abs(soft_ce(z,p).item()-torch.nn.functional.cross_entropy(z,y).item())<1e-6
