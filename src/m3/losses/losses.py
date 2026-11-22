import torch, torch.nn.functional as F
def soft_ce(z,p): return -(p*F.log_softmax(z,1)).sum(1).mean()
def hard_ce(z,y,ls=0.0): return F.cross_entropy(z,y,label_smoothing=ls)
def dm_nll(z,n):
    a=F.softplus(z)+1; S=a.sum(1); N=n.sum(1)
    ll=torch.lgamma(S)-torch.lgamma(N+S)+(torch.lgamma(n+a)-torch.lgamma(a)).sum(1)
    return -ll.mean()
def kd_kl(z,zt,tau):
    return tau*tau*F.kl_div(F.log_softmax(z/tau,1),F.softmax(zt/tau,1),reduction="batchmean")
