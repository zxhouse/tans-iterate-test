# E3: tuning a table from noisy tests on sampled traffic (source unknown to the tuner)
import json, numpy as np
from tanslib import *
S=sources(); B=300; m=2000; SEEDS=100; out=[]
def neighbour(sym,rng,w=4):
    L=len(sym); c=sym.copy()
    while True:
        i=int(rng.integers(L)); j=int(np.clip(i+rng.integers(-w,w+1),0,L-1))
        if c[i]==c[j]: continue
        if rng.random()<0.5: c[i],c[j]=c[j],c[i]; return c
        if np.count_nonzero(c==c[i])>1: c[i]=c[j]; return c
def run(name,L,policy,seed):
    p=S[name]; n=len(p); H=entropy(p); rng=np.random.default_rng(seed)
    batch=lambda k=m: rng.choice(n,size=k,p=p).astype(np.int32)
    ph=(np.bincount(batch(),minlength=n)+0.5); ph/=ph.sum()
    s0=spread_tuned(quantize(ph,L),ph); true=lambda s: exact_rate(s,p)-H
    test=lambda s,b=None: trace_bits(s,n,batch() if b is None else b)[0]/m - H
    if policy=="pilot": return dict(true=true(s0))
    if policy=="plugin":
        pool=np.bincount(batch(B*m),minlength=n)+ph*m; ph2=pool/pool.sum()
        s=local_search(spread_tuned(quantize(ph2,L),ph2),ph2,w=4)[0]; return dict(true=true(s))
    if policy.startswith("paired"):
        r=int(policy[6:]); inc=s0; used=0; acc=0
        while used+2*r<=B:
            c=neighbour(inc,rng); d=0
            for _ in range(r): b=batch(); d+=test(c,b)-test(inc,b)
            used+=2*r
            if d<0: inc=c; acc+=1
        return dict(true=true(inc),accepted=acc)
    arch=[dict(s=s0,obs=[test(s0)])]; used=1; mean=lambda a: float(np.mean(a["obs"]))
    if policy=="once":
        inc=arch[0]
        while used<B:
            c=dict(s=neighbour(inc["s"],rng),obs=[]); c["obs"].append(test(c["s"])); used+=1; arch.append(c)
            if mean(c)<mean(inc): inc=c
    else:
        b=int(policy[2:]); 
        def leader():
            c=[a for a in arch if len(a["obs"])>=2] or arch; return min(c,key=mean)
        while used<B:
            for t in range(5):
                if used>=B: break
                if t<b: c=dict(s=neighbour(leader()["s"],rng),obs=[]); c["obs"].append(test(c["s"])); arch.append(c)
                else:
                    top=sorted(arch,key=mean)[:3]; a=min(top,key=lambda a:len(a["obs"])); a["obs"].append(test(a["s"]))
                used+=1
        inc=leader()
    tr=[true(a["s"]) for a in arch]; ti=true(inc["s"])
    return dict(true=ti,est=mean(inc),curse=ti-mean(inc),rejected_better=float(min(tr)<ti-1e-12),best_in_archive=min(tr),ntests=len(inc["obs"]))
if __name__=='__main__':
  for name,L in [("ttl-32",128),("pktsize-16",64),("zipf1.5-64",256)]:
      p=S[name]; H=entropy(p); l=quantize(p,L); orc=spread_tuned(l,p)
      ref=dict(oracle_tuned=exact_rate(orc,p)-H, oracle_ls=local_search(orc,p,w=4)[1]-H)
      for pol in ["pilot","once","dt4","dt3","dt2","dt1","paired1","paired3","plugin"]:
          rs=[run(name,L,pol,1000+s) for s in range(SEEDS)]
          agg={k:(float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs])/np.sqrt(SEEDS))) for k in rs[0]}
          rec=dict(source=name,L=L,policy=pol,**agg,**ref); out.append(rec); print(json.dumps(rec),flush=True)
  json.dump(out,open("../results/e3.json","w"),indent=1)
