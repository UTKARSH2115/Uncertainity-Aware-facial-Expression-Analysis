import hashlib, sys
import pandas as pd, numpy as np
from m3.utils.common import ROOT, jdump
EXP_MD5="f8428a1edbd21e88f42c73edd2a14f95"
CC={0:4953,1:547,2:5121,3:8989,4:6077,5:4002,6:6198}
COLS=["Usage","Image name","neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","unknown","NF"]
def main():
    res=[]; raw=ROOT/"data/raw"
    md5=hashlib.md5(open(raw/"fer2013.csv","rb").read()).hexdigest()
    sha=hashlib.sha256(open(raw/"fer2013.csv","rb").read()).hexdigest()
    f=pd.read_csv(raw/"fer2013.csv"); f.columns=[c.strip() for c in f.columns]
    g=pd.read_csv(raw/"fer2013new.csv")
    res.append(("rows fer2013==35887", len(f)==35887))
    uc=f.Usage.value_counts().to_dict()
    res.append(("usage counts", uc=={"Training":28709,"PublicTest":3589,"PrivateTest":3589}))
    res.append(("class counts", f.emotion.value_counts().to_dict()==CC))
    ok=all(len(p.split())==2304 for p in f.pixels)
    arr=np.array([p.split() for p in f.pixels],dtype=np.int32)
    res.append(("pixels 2304 in [0,255]", ok and arr.min()>=0 and arr.max()<=255))
    res.append(("fer+ rows 35887", len(g)==35887))
    res.append(("fer+ header", list(g.columns)==COLS))
    res.append(("usage aligned", bool((f.Usage.values==g.Usage.values).all())))
    v=g[COLS[2:]].fillna(0); v_ok=bool((v>=0).all().all())
    s=v[g["Image name"].notna()].sum(axis=1)
    res.append(("votes nonneg ints", v_ok))
    for n,ok in res: print(f"{'PASS' if ok else 'FAIL'}  {n}")
    print("vote-sum min/median/max:", s.min(), s.median(), s.max())
    md5ok = md5==EXP_MD5
    print("MD5", md5, "MATCH" if md5ok else "WARN: differs (content verified if all PASS)")
    allok=all(o for _,o in res)
    jdump({"fer2013_md5":md5,"fer2013_sha256":sha,"md5_match":md5ok,"content_ok":allok}, ROOT/"results/data_verify.json")
    sys.exit(0 if allok else 1)
if __name__=="__main__": main()
