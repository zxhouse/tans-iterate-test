import json, numpy as np
from tanslib import *
S=sources(); rng=np.random.default_rng(3); out=[]
for name,L in [("ttl-32",128),("pktsize-16",64),("zipf1.5-64",256),("geom0.8-32",128)]:
    p=S[name]; n=len(p); s=spread_tuned(quantize(p,L),p); r0=exact_rate(s,p); d=[]
    b=[trace_bits(s,n,rng.choice(n,size=2000,p=p).astype(np.int32))[0]/2000 for _ in range(2000)]
    for i in range(L):
        for j in range(i+1,min(L,i+5)):
            if s[i]!=s[j]:
                c=s.copy(); c[i],c[j]=c[j],c[i]; d.append(exact_rate(c,p)-r0)
    d=np.array(d); nz=np.abs(d)>1e-11; imp=-d[d<-1e-11]; sd=float(np.std(b))
    rec=dict(source=name,L=L,moves=len(d),tie_frac=float(1-nz.mean()),improving_frac=float((d<-1e-11).mean()),median_abs_nonzero=float(np.median(np.abs(d[nz]))),best_improvement=float(imp.max()) if len(imp) else 0.0,
             median_improvement=float(np.median(imp)) if len(imp) else 0.0, batch_sd=sd, batch_bias=float(np.mean(b)-r0))
    rec["tests_best"]=(sd/rec["best_improvement"])**2 if rec["best_improvement"] else None; rec["tests_median"]=(sd/rec["median_improvement"])**2 if rec["median_improvement"] else None
    out.append(rec); print(rec)
json.dump(out,open("../results/ties.json","w"),indent=1)
