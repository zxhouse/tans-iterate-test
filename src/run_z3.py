# Z3: decoder-reproducible variants (built from the transmitted counts only) and per-stream saving vs side-information cost
import json, numpy as np, zstandard as z, sys, time
from tanslib import *
import zstd_fse as Z
f=sys.argv[1]; d=open(f,"rb").read(); c=z.ZstdCompressor(level=3,write_content_size=True).compress(d)
st,fcs,end,nb,sb=Z.parse_frame(c); rows=[]
for s in st:
    if s["mode"]!="fse": continue
    norm=np.array(s["norm"]); used=np.flatnonzero(norm!=0); l=np.where(norm[used]==-1,1,norm[used]).astype(int)
    remap=np.full(len(norm),-1); remap[used]=np.arange(len(used)); sy=remap[s["symbols"]].astype(np.int32); n=len(used); L=int(l.sum())
    cnt=np.bincount(sy,minlength=n).astype(float); p=(cnt+1e-9)/(cnt+1e-9).sum(); q=l/L
    zs=remap[Z.zstd_spread(list(norm),s["al"])].astype(np.int32); b=lambda sp: trace_bits(sp,n,sy[::-1].copy())[0]
    t=time.time(); dq=dt_tans(l,q,50,np.random.default_rng(0))[0]; tq=time.time()-t
    rows.append(dict(stream=s["stream"],n=n,L=L,nseq=len(sy),fast=b(zs),dt_p=b(dt_tans(l,p,50,np.random.default_rng(0))[0]),dt_q=b(dq),iter_q=b(spread_iterated_tuned(l,q,8)[0]),t_dtq=tq))
R={k:sum(r[k] for r in rows) for k in ("fast","dt_p","dt_q","iter_q")}
sav=np.array([r["fast"]-r["dt_p"] for r in rows]); savq=np.array([r["fast"]-r["dt_q"] for r in rows]); ns=np.array([r["n"] for r in rows])
out=dict(file=f,streams=len(rows),totals=R,save_dt_p_bits_per_stream=dict(mean=float(sav.mean()),median=float(np.median(sav)),p90=float(np.percentile(sav,90))),
         save_dt_q_bits_per_stream=dict(mean=float(savq.mean()),median=float(np.median(savq))),symbols_per_table_mean=float(ns.mean()),
         frac_streams_saving_exceeds_n_bits=float(np.mean(sav>ns)),frac_streams_saving_exceeds_2n_bits=float(np.mean(sav>2*ns)),t_dtq_mean=float(np.mean([r["t_dtq"] for r in rows])))
print(json.dumps(out)); json.dump(dict(summary=out,rows=rows),open("../results/z3_"+f.split("/")[-1]+".json","w"))
