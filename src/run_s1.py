# S1: benchmark of table constructions on random distributions (as in Duda 2021, Fig. 2: uniform-then-normalised and permuted Zipf)
import numpy as np, json, time, sys
from tanslib import *
rng=np.random.default_rng(2026); out=[]
CFG=[(8,(2,4,8)),(16,(2,4,8)),(32,(2,4,8)),(64,(2,4,8)),(256,(2,4,8))]
NS={8:200,16:200,32:150,64:100,256:40}
for kind in ("uniform","zipf","dirichlet0.3"):
    for n,mults in CFG:
        for m in mults:
            L=m*n; recs=[]
            for k in range(NS[n]):
                p=random_dists(kind,n,rng); H=entropy(p); l=quantize(p,L)
                r=dict(fast=exact_rate(spread_fast(l),p)-H, precise=exact_rate(spread_precise(l),p)-H, tuned=exact_rate(spread_tuned(l,p),p)-H)
                t=time.time(); r["iter"]=spread_iterated_tuned(l,p,8)[1]-H; r["t_iter"]=time.time()-t
                r["dy"]=spread_dube_yokoo(l,p,30)[1]-H
                t=time.time(); s,rs,ev,st=spread_safeguarded_tuned(l,p); r["safe"]=rs-H; r["t_safe"]=time.time()-t; r["ev_safe"]=ev
                if L<=256 and k<40:
                    t=time.time(); r["ls"]=local_search(spread_tuned(l,p),p,w=4)[1]-H; r["t_ls"]=time.time()-t
                recs.append(r)
            g=lambda a,b: np.array([100*(1-x[a]/x[b]) for x in recs if b in x and a in x and x[b]>1e-12])
            sm=dict(kind=kind,n=n,L=L,N=len(recs),**{f"mean_{k}":float(np.mean([x[k] for x in recs])) for k in ("fast","precise","tuned","iter","dy","safe")})
            sm["gain_safe_vs_tuned_median"]=float(np.median(g("safe","tuned"))); sm["gain_safe_vs_tuned_mean"]=float(np.mean(g("safe","tuned")))
            sm["frac_safe_better"]=float(np.mean([x["safe"]<x["tuned"]-1e-12 for x in recs]))
            sm["gain_iter_vs_tuned_median"]=float(np.median(g("iter","tuned"))); sm["frac_iter_better"]=float(np.mean([x["iter"]<x["tuned"]-1e-12 for x in recs]))
            sm["frac_dy_worse_than_tuned"]=float(np.mean([x["dy"]>x["tuned"]+1e-12 for x in recs]))
            sm["safe_vs_dy_median"]=float(np.median(g("safe","dy"))); sm["frac_safe_le_dy"]=float(np.mean([x["safe"]<=x["dy"]+1e-12 for x in recs]))
            if "ls" in recs[0]:
                ls=[x for x in recs if "ls" in x]; sm["mean_ls"]=float(np.mean([x["ls"] for x in ls]))
                sm["gap_closed_by_safe"]=float(np.median([ (x["tuned"]-x["safe"])/(x["tuned"]-x["ls"]) for x in ls if x["tuned"]-x["ls"]>1e-9] or [np.nan]))
                sm["t_ls_mean"]=float(np.mean([x["t_ls"] for x in ls]))
            sm["t_safe_mean"]=float(np.mean([x["t_safe"] for x in recs])); sm["ev_safe_mean"]=float(np.mean([x["ev_safe"] for x in recs]))
            out.append(dict(summary=sm,records=recs)); print(json.dumps(sm),flush=True)
json.dump(out,open("../results/s1.json","w"))
