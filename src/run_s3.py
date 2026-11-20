# S3: anytime curves - redundancy vs number of exact evaluations
import numpy as np, json
from tanslib import *
rng=np.random.default_rng(77); out=[]
for kind,n,m in [("uniform",32,4),("dirichlet0.3",32,2),("zipf",64,2),("dirichlet0.3",16,4)]:
    L=m*n; rows=[]
    for k in range(40):
        p=random_dists(kind,n,rng); H=entropy(p); l=quantize(p,L); t=exact_rate(spread_tuned(l,p),p)-H
        row=dict(tuned=t, iter=spread_iterated_tuned(l,p,8)[1]-H, dy=spread_dube_yokoo(l,p,30)[1]-H)
        for B in (10,20,30,50,100,200):
            row[f"dt{B}"]=dt_tans(l,p,B,np.random.default_rng(k))[1]-H
            row[f"dtnops{B}"]=dt_tans(l,p,B,np.random.default_rng(k),prescreen=False)[1]-H
            row[f"hc{B}"]=hill_climb(spread_tuned(l,p),p,B,rng=np.random.default_rng(k))[1]-H
        row["ls"]=local_search(spread_tuned(l,p),p,w=4,reassign=False)[1]-H; rows.append(row)
    out.append(dict(kind=kind,n=n,L=L,rows=rows))
    rel=lambda key: float(np.median([100*(1-r[key]/r["tuned"]) for r in rows]))
    print(kind,n,L,"iter %.1f DY %.1f LS %.1f |"%(rel("iter"),rel("dy"),rel("ls")),{B:(round(rel(f"dt{B}"),1),round(rel(f"dtnops{B}"),1),round(rel(f"hc{B}"),1)) for B in (10,20,30,50,100,200)},flush=True)
json.dump(out,open("../results/s3.json","w"))
