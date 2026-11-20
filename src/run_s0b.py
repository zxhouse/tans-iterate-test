import json,glob,numpy as np
from tanslib import *
for f in sorted(glob.glob("../results/enum_?.json")):
    e=json.loads(open(f).read()); p=np.array(e["p"]); p/=p.sum(); H=entropy(p); l=quantize(p,16)
    t=exact_rate(spread_tuned(l,p),p)-H; it=spread_iterated_tuned(l,p,8)[1]-H; dy=spread_dube_yokoo(l,p)[1]-H
    dt=[dt_tans(l,p,30,np.random.default_rng(s))[1]-H for s in range(20)]
    print(f"p={np.round(p,3)} opt {e['best']:.6f} tuned {t:.6f} iter {it:.6f} DY {dy:.6f} DT(30) mean {np.mean(dt):.6f} hit-opt {np.mean(np.array(dt)<e['best']+1e-8):.2f}")
