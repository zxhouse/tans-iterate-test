# S4: real byte data (files on this machine), L >= alphabet
import numpy as np, glob, os, json, time
from tanslib import *
def hist(paths, cap=4_000_000):
    b=b"".join(open(f,"rb").read() for f in paths)[:cap]; c=np.bincount(np.frombuffer(b,np.uint8),minlength=256).astype(float); c=c[c>0]; return c/c.sum()
D={"text (licences)":sorted(glob.glob("/usr/share/common-licenses/*")),"source code (.py)":sorted(glob.glob(os.path.join(os.path.dirname(os.__file__),"*.py"))),"executable (bash)":["/bin/bash"]}
out=[]
for name,paths in D.items():
    p=hist(paths); H=entropy(p); n=len(p)
    for L in (256,512,1024,2048):
        if L<n: continue
        l=quantize(p,L); r=dict(data=name,n=n,L=L,H=H,fast=exact_rate(spread_fast(l),p)-H,tuned=exact_rate(spread_tuned(l,p),p)-H,iter=spread_iterated_tuned(l,p,8)[1]-H,dy=spread_dube_yokoo(l,p,30)[1]-H)
        t=time.time(); r["dt"]=dt_tans(l,p,50,np.random.default_rng(0))[1]-H; r["t_dt"]=time.time()-t
        r["hc"]=hill_climb(spread_tuned(l,p),p,50,rng=np.random.default_rng(0))[1]-H
        if L<=1024:
            t=time.time(); s,rl,ev,_=local_search(spread_tuned(l,p),p,w=4,reassign=False); r["ls"]=rl-H; r["ev_ls"]=ev; r["t_ls"]=time.time()-t
        out.append(r); print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
json.dump(out,open("../results/s4.json","w"))
