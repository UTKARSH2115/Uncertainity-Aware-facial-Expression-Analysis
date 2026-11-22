import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from m3.utils.common import ROOT, CLASSES, load_npz, jdump
from m3.audit.human_ceiling import loo_all
from m3.report.style import savefig, OI
FERN=["angry","disgust","fear","happy","sad","surprise","neutral"]
def ent(p): return -(np.where(p>0,p*np.log(np.where(p>0,p,1)),0)).sum(1)/np.log(8)
def main():
    d=load_npz(); T=ROOT/"results/tables"; p=d["p"]; n=d["n"]; y=d["y_maj"]; sp=np.where(np.isin(d["split"],["valA","valB"]),"val",d["split"])
    m=p.max(1); H=ent(p); L=loo_all(n); agree=(d["fer_label_mapped"]==y)
    amb=m<=0.5; clr=m>=0.8
    rows_a=[];rows_h=[];rows_v=[]
    for s in ["all","train","val","test"]:
        for c in ["all"]+CLASSES:
            k=(np.ones(len(y),bool) if s=="all" else sp==s)&(np.ones(len(y),bool) if c=="all" else y==CLASSES.index(c) if c in CLASSES else False)
            if k.sum()==0: continue
            rows_a.append(dict(split=s,**{"class":c},n=int(k.sum()),fer_vs_ferplus_agreement=agree[k].mean()))
            rows_h.append({"split":s,"class":c,"loo_agreement_mean":L[k].mean(),"p_maj_mean":m[k].mean(),"frac_ambiguous":amb[k].mean(),"frac_clear":clr[k].mean()})
            rows_v.append({"split":s,"class":c,"max_share_mean":m[k].mean(),"Hn_mean":H[k].mean(),"classes_ge1_vote":(n[k]>=1).sum(1).mean(),"classes_ge2_votes":(n[k]>=2).sum(1).mean()})
    pd.DataFrame(rows_a).to_csv(T/"tab_audit_agreement.csv",index=False)
    pd.DataFrame(rows_h).to_csv(T/"tab_audit_human_ceiling.csv",index=False)
    pd.DataFrame(rows_v).to_csv(T/"tab_audit_votes.csv",index=False)
    # confusion FER(7) x FER+(8)
    fer=d["fer_label_mapped"]; inv={4:0,5:1,6:2,1:3,3:4,2:5,0:6}
    C=np.zeros((7,8))
    for a,b in zip(fer,y): C[inv[int(a)],int(b)]+=1
    Cn=C/C.sum(1,keepdims=True)
    off=[(FERN[i],CLASSES[j],int(C[i,j])) for i in range(7) for j in range(8) if not (inv.get(j,-1)==i) ]
    pd.DataFrame(sorted(off,key=lambda t:-t[2])[:10],columns=["fer2013","ferplus","count"]).to_csv(T/"tab_audit_top_confusions.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(7,2.8)); ax[0].hist(m,30,color=OI[0]); ax[0].set_xlabel("max vote share"); ax[0].set_ylabel("images")
    ax[1].hist(H,30,color=OI[1]); ax[1].set_xlabel("normalised vote entropy"); plt.tight_layout(); savefig(fig,"fig_ambiguity_hist")
    fig,ax=plt.subplots(figsize=(5,3.8)); im=ax.imshow(Cn,cmap="Blues"); ax.set_xticks(range(8)); ax.set_xticklabels(CLASSES,rotation=60,ha="right",fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels(FERN,fontsize=8); ax.set_xlabel("FER+ majority"); ax.set_ylabel("FER2013 label"); plt.colorbar(im); plt.tight_layout(); savefig(fig,"fig_fer_vs_ferplus_confusion")
    rng=np.random.default_rng(0)
    fig,axs=plt.subplots(6,8,figsize=(10,8))
    for c in range(8):
        for r,(grp) in enumerate([clr,amb]):
            idx=np.where(grp&(y==c))[0]
            for t in range(3):
                a=axs[r*3+t,c]; a.axis("off")
                if len(idx)>t:
                    i=rng.choice(idx); a.imshow(d["img"][i],cmap="gray"); a.set_title(f"{CLASSES[c][:4]} m={m[i]:.1f}",fontsize=6)
    plt.tight_layout(); savefig(fig,"fig_clear_vs_ambiguous")
    fig,axs=plt.subplots(8,8,figsize=(10,10))
    for c in range(8):
        idx=np.where((y==c)&(d["split"]=="train"))[0]
        for t in range(8):
            a=axs[t,c]; a.axis("off")
            if len(idx)>t:
                i=rng.choice(idx); a.imshow(d["img"][i],cmap="gray")
                if t==0:a.set_title(CLASSES[c],fontsize=7)
    plt.tight_layout(); savefig(fig,"fig_samples_grid")
    S=dict(loo_mean=L.mean(),fer_agree=agree.mean(),frac_amb=amb.mean(),frac_clear=clr.mean(),pmaj_mean=m.mean())
    print(S); jdump(S,ROOT/"results/audit_summary.json")
    assert 0.5<S["loo_mean"]<1 and 0.5<S["fer_agree"]<1
if __name__=="__main__": main()
