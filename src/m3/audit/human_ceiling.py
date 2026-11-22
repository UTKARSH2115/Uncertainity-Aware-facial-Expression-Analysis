import numpy as np
def loo(n):
    n=np.asarray(n,dtype=float); N=n.sum(); tot=0.0
    for k in range(len(n)):
        if n[k]<=0: continue
        m=n.copy(); m[k]-=1; w=np.flatnonzero(m==m.max())
        if k in w: tot+=n[k]/N/len(w)
    return tot
def loo_all(Nn): return np.array([loo(x) for x in Nn])
