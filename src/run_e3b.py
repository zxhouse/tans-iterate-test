# E3b: the baseline loop of the preprint (Sec. 5.1) on tANS tables, with and without model-based pre-screening
import json, numpy as np
from tanslib import *
from run_e3 import neighbour
S=sources(); B=300; m=2000; SEEDS=100; out=[]
def run(name,L,screen,seed):
    p=S[name]; n=len(p); H=entropy(p); rng=np.random.default_rng(seed); cnt=np.full(n,0.5)
    def batch():
        b=rng.choice(n,size=m,p=p).astype(np.int32); cnt[:]+=np.bincount(b,minlength=n); return b
    batch(); ph=lambda: cnt/cnt.sum(); s0=spread_tuned(quantize(ph(),L),ph())
    test=lambda a: a["obs"].append(trace_bits(a["s"],n,batch())[0]/m-H); mean=lambda a: float(np.mean(a["obs"]))
    arch=[dict(s=s0,obs=[])]; test(arch[0]); used=1; rot=0
    while used+5<=B:
        top=sorted(arch,key=mean)[:3]; par=top[int(rng.integers(len(top)))]; ideas=[neighbour(par["s"],rng) for _ in range(5)]
        if screen: q=ph(); c=min(ideas,key=lambda s: exact_rate(s,q))
        else: c=ideas[0]
        a=dict(s=c,obs=[]); test(a); arch.append(a)
        for t in sorted(arch,key=mean)[:3]: test(t)
        test(arch[rot%len(arch)]); rot+=1; used+=5
    true=lambda s: exact_rate(s,p)-H; fav=min(arch,key=mean); q=ph(); mod=min(arch,key=lambda a: exact_rate(a["s"],q)); tr=[true(a["s"]) for a in arch]
    return dict(true=true(fav["s"]),curse=true(fav["s"])-mean(fav),ship_by_model=true(mod["s"]),best_in_archive=min(tr),built=len(arch))
if __name__=="__main__":
    for name,L in [("ttl-32",128),("pktsize-16",64),("zipf1.5-64",256)]:
        for screen in (False,True):
            rs=[run(name,L,screen,5000+s) for s in range(SEEDS)]
            rec=dict(source=name,L=L,screen=screen,**{k:(float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs])/np.sqrt(SEEDS))) for k in rs[0]})
            out.append(rec); print(json.dumps(rec),flush=True)
    json.dump(out,open("../results/e3b.json","w"),indent=1)
