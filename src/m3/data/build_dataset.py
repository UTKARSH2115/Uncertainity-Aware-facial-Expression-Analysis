import json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from m3.utils.common import ROOT, CLASSES, jdump
from m3.data.labels import process_row, FER2PLUS
def main():
    raw=ROOT/"data/raw"
    f=pd.read_csv(raw/"fer2013.csv"); f.columns=[c.strip() for c in f.columns]
    g=pd.read_csv(raw/"fer2013new.csv")
    vc=g.columns[2:12]; V=g[vc].fillna(0).astype(int).values
    smap={"Training":"train","PublicTest":"val","PrivateTest":"test"}
    out={k:[] for k in ["img","p","n","y","fer","row","tied","split"]}
    counts={s:dict(n_raw=0,drop_F1=0,drop_F2=0,drop_F3=0,tied=0,n_final=0,**{c:0 for c in CLASSES}) for s in ["train","val","test"]}
    for i in range(len(f)):
        s=smap[f.Usage[i]]; c=counts[s]; c["n_raw"]+=1
        r=process_row(V[i], g["Image name"][i])
        if not r["keep"]: c["drop_"+r["reason"]]+=1; continue
        out["img"].append(np.array(f.pixels[i].split(),dtype=np.uint8).reshape(48,48))
        out["p"].append(r["p"]); out["n"].append(r["n"]); out["y"].append(r["y"]); out["tied"].append(r["tied"])
        out["fer"].append(FER2PLUS[int(f.emotion[i])]); out["row"].append(i); out["split"].append(s)
        if r["tied"]: c["tied"]+=1
    d=dict(img=np.stack(out["img"]),p=np.stack(out["p"]).astype(np.float32),n=np.stack(out["n"]).astype(np.int16),
           y_maj=np.array(out["y"],dtype=np.int8),split=np.array(out["split"]),fer_label_mapped=np.array(out["fer"],dtype=np.int8),
           row_idx=np.array(out["row"]),tied=np.array(out["tied"]))
    # main-experiment mask: not tied
    d["main"]=~d["tied"]
    for s in counts:
        m=(d["split"]==s)&d["main"]; counts[s]["n_final"]=int(m.sum())
        for k,c in enumerate(CLASSES): counts[s][c]=int((d["y_maj"][m]==k).sum())
    # val split A/B stratified
    vi=np.where((d["split"]=="val")&d["main"])[0]
    a,b=train_test_split(vi,test_size=0.5,stratify=d["y_maj"][vi],random_state=2026)
    sp=np.array(d["split"]); sp=sp.astype("<U8"); sp[a]="valA"; sp[b]="valB"
    d["split"]=sp
    np.savez_compressed(ROOT/"data/processed/ferplus.npz",**d)
    jdump({"valA":a.tolist(),"valB":b.tolist()},ROOT/"data/splits/val_split.json")
    df=pd.DataFrame(counts).T; df.index.name="split"; df.to_csv(ROOT/"results/tables/dataset_counts.csv")
    print(df.T); nt=counts["test"]["n_final"]; print("test n",nt)
    assert 3000<=nt<=3589
if __name__=="__main__": main()
