import numpy as np, json, time
from tanslib import *
rng=np.random.default_rng(abs(hash(sys.argv[1]))%(2**32) if False else 2026); out=[]; B=50
import sys
NS={8:200,16:200,32:150,64:100}
KINDS=sys.argv[1].split(",")
for kind in KINDS:
    for n in (8,16,32,64):
        for m in (2,4,8):
            L=m*n; recs=[]
            for k in range(NS[n]):
                p=random_dists(kind,n,rng); H=entropy(p); l=quantize(p,L); sd=int(rng.integers(1<<30))
                r=dict(tuned=exact_rate(spread_tuned(l,p),p)-H, iter=spread_iterated_tuned(l,p,8)[1]-H, dy=spread_dube_yokoo(l,p,30)[1]-H)
                t=time.time(); r["dt"]=dt_tans(l,p,B,np.random.default_rng(sd))[1]-H; r["t_dt"]=time.time()-t
                r["dt_noPS"]=dt_tans(l,p,B,np.random.default_rng(sd),prescreen=False)[1]-H
                r["dt_noRet"]=dt_tans(l,p,B,np.random.default_rng(sd),retain=0)[1]-H
                if L<=256 and k<60:
                    s,rl,ev,_=local_search(spread_tuned(l,p),p,w=4,reassign=False); r["ls"]=rl-H; r["ev_ls"]=ev
                recs.append(r)
            def gain(a,b): 
                v=[100*(1-x[a]/x[b]) for x in recs if x[b]>1e-12]; return float(np.median(v)), float(np.mean(v))
            sm=dict(kind=kind,n=n,L=L,N=len(recs),B=B)
            for a in ("iter","dy","dt","dt_noPS","dt_noRet"): sm[f"{a}_vs_tuned_med"],sm[f"{a}_vs_tuned_mean"]=gain(a,"tuned")
            sm["dt_vs_iter_med"],sm["dt_vs_iter_mean"]=gain("dt","iter"); sm["dt_vs_dy_med"],sm["dt_vs_dy_mean"]=gain("dt","dy")
            sm["frac_dt_le_iter"]=float(np.mean([x["dt"]<=x["iter"]+1e-12 for x in recs])); sm["frac_dt_le_dy"]=float(np.mean([x["dt"]<=x["dy"]+1e-12 for x in recs]))
            sm["frac_dt_lt_dy"]=float(np.mean([x["dt"]<x["dy"]-1e-12 for x in recs]))
            ls=[x for x in recs if "ls" in x]
            if ls:
                sm["ls_vs_tuned_med"]=float(np.median([100*(1-x["ls"]/x["tuned"]) for x in ls if x["tuned"]>1e-12])); sm["ev_ls_mean"]=float(np.mean([x["ev_ls"] for x in ls]))
                gc=[(x["tuned"]-x["dt"])/(x["tuned"]-x["ls"]) for x in ls if x["tuned"]-x["ls"]>1e-9]; sm["dt_gap_closed_med"]=float(np.median(gc)) if gc else None
                gi=[(x["tuned"]-x["iter"])/(x["tuned"]-x["ls"]) for x in ls if x["tuned"]-x["ls"]>1e-9]; sm["iter_gap_closed_med"]=float(np.median(gi)) if gi else None
                sm["frac_dt_le_ls"]=float(np.mean([x["dt"]<=x["ls"]+1e-12 for x in ls]))
            sm["t_dt_mean"]=float(np.mean([x["t_dt"] for x in recs]))
            out.append(dict(summary=sm,records=recs)); print(json.dumps(sm),flush=True)
json.dump(out,open("../results/s2_"+sys.argv[1].replace(",","_")+".json","w"))
