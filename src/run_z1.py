# Z1: real libzstd 1.5.7 frames; re-encode every FSE-compressed sequence stream with alternative spreads built
# from the SAME normalised counts (zstd's own quantisation), measure exact bits with a real tANS encoder.
import sys, json, glob, os, time, numpy as np, zstandard as z
from tanslib import *
import zstd_fse as Z
level=int(sys.argv[1]); files=sorted(glob.glob("../corpus/*.bin")); res=[]
def bits(sym, n, s): return trace_bits(sym, n, s[::-1].copy())[0]   # ANS encodes in reverse order
for f in files:
    d=open(f,"rb").read(); c=z.ZstdCompressor(level=level,write_content_size=True).compress(d)
    t=time.time(); streams,fcs,end,nb,seqb=Z.parse_frame(c); assert end==len(c)
    agg=dict(file=os.path.basename(f),level=level,raw=len(d),comp=len(c),seq_bytes=seqb,blocks=nb,
             fse_streams=0,state_bits_actual=0,b_fast=0.0,b_tunedq=0.0,b_tunedp=0.0,b_iter=0.0,b_dt=0.0,entropy_bits=0.0,
             nonfse_state_bits=0,t_dt=0.0)
    for s in streams:
        if s["mode"]!="fse": agg["nonfse_state_bits"]+=s["state_bits"]; continue
        norm=np.array(s["norm"]); used=np.flatnonzero(norm!=0); l=np.where(norm[used]==-1,1,norm[used]).astype(int)
        remap=np.full(len(norm),-1); remap[used]=np.arange(len(used)); sy=remap[s["symbols"]].astype(np.int32); assert (sy>=0).all()
        n=len(used); L=int(l.sum()); assert L==1<<s["al"]
        cnt=np.bincount(sy,minlength=n).astype(float); p=(cnt+1e-9)/ (cnt+1e-9).sum(); q=l/L
        zs=Z.zstd_spread(list(norm),s["al"]); zs=remap[zs].astype(np.int32)
        agg["fse_streams"]+=1; agg["state_bits_actual"]+=s["state_bits"]
        agg["entropy_bits"]+=float(-(cnt[cnt>0]*np.log2(cnt[cnt>0]/cnt.sum())).sum())
        agg["b_fast"]+=bits(zs,n,sy)
        agg["b_tunedq"]+=bits(spread_tuned(l,q),n,sy)
        agg["b_tunedp"]+=bits(spread_tuned(l,p),n,sy)
        agg["b_iter"]+=bits(spread_iterated_tuned(l,p,8)[0],n,sy)
        t1=time.time(); dts=dt_tans(l,p,50,np.random.default_rng(0))[0]; agg["t_dt"]+=time.time()-t1
        agg["b_dt"]+=bits(dts,n,sy)
    res.append(agg); print(json.dumps(agg),flush=True)
json.dump(res,open(f"../results/z1_L{level}.json","w"),indent=1)
