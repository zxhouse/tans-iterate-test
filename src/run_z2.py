# Z2: small independent frames (as in RPC / key-value / log shipping). Streams in zstd's predefined mode use the fixed
# default distributions and the fixed FSE spread. A static table can be optimised offline: train p on even frames,
# build a DT-tANS spread for the SAME default counts, test on odd frames (out of sample).
import sys, json, glob, os, numpy as np, zstandard as z
from tanslib import *
import zstd_fse as Z
fsz=int(sys.argv[1]); files=sorted(glob.glob("../corpus/*.bin"))
DEF={"LL":Z.LL_DEF,"OF":Z.OF_DEF,"ML":Z.ML_DEF}
frames=[]
for f in files:
    d=open(f,"rb").read(); cc=z.ZstdCompressor(level=3,write_content_size=True)
    for k,off in enumerate(range(0,len(d)-fsz+1,fsz*4)):        # every 4th chunk to bound runtime
        c=cc.compress(d[off:off+fsz]); st,fcs,end,nb,sb=Z.parse_frame(c); frames.append((os.path.basename(f),k,len(c),sb,st))
tot_comp=sum(fr[2] for fr in frames); tot_seq=sum(fr[3] for fr in frames)
modes={}
for fr in frames:
    for s in fr[4]: modes[(s["stream"],s["mode"])]=modes.get((s["stream"],s["mode"]),0)+s["state_bits"]
out=dict(frame_size=fsz,frames=len(frames),comp_bytes=tot_comp,seq_bytes=tot_seq,state_bits_by_mode={f"{a}:{b}":v for (a,b),v in modes.items()})
for name in ("LL","OF","ML"):
    norm,al=DEF[name]; norm=np.array(norm); used=np.flatnonzero(norm!=0); l=np.where(norm[used]==-1,1,norm[used]).astype(int)
    remap=np.full(len(norm),-1); remap[used]=np.arange(len(used)); zs=remap[Z.zstd_spread(list(norm),al)].astype(np.int32); n=len(used)
    tr=[];te=[]
    for fr in frames:
        for s in fr[4]:
            if s["stream"]==name and s["mode"]=="pre": (tr if fr[1]%2==0 else te).append(remap[s["symbols"]].astype(np.int32))
    if not tr or not te: continue
    cnt=np.bincount(np.concatenate(tr),minlength=n).astype(float)+0.5; p=cnt/cnt.sum()
    sp={"zstd_default":zs,"tuned":spread_tuned(l,p),"iter":spread_iterated_tuned(l,p,8)[0],"dt50":dt_tans(l,p,50,np.random.default_rng(0))[0],"dt400":dt_tans(l,p,400,np.random.default_rng(0))[0]}
    for k,v in sp.items():
        out[f"{name}_{k}_test_bits"]=float(sum(trace_bits(v,n,s[::-1].copy())[0] for s in te))
        out[f"{name}_{k}_train_rate"]=exact_rate(v,p)
    out[f"{name}_test_streams"]=len(te); out[f"{name}_test_entropy_bits"]=float(sum(-(np.log2(p[s])).sum() for s in te))
print(json.dumps(out,indent=1)); json.dump(out,open(f"../results/z2_{fsz}.json","w"),indent=1)
