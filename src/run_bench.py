import numpy as np, zstandard as z, struct, subprocess, json, time, ctypes
from tanslib import *
import zstd_fse as Z
d=open("../corpus/py.bin","rb").read(); c=z.ZstdCompressor(level=3,write_content_size=True).compress(d)
st,*_=Z.parse_frame(c); out=[]
for name in ("LL","ML","OF"):
    S=[s for s in st if s["stream"]==name and s["mode"]=="fse"]; s=max(S,key=lambda s:s["nseq"])
    norm=np.array(s["norm"]); used=np.flatnonzero(norm!=0); l=np.where(norm[used]==-1,1,norm[used]).astype(int)
    remap=np.full(len(norm),-1); remap[used]=np.arange(len(used)); sy=remap[s["symbols"]].astype(np.int32); n=len(used); L=int(l.sum())
    cnt=np.bincount(sy,minlength=n)+1e-9; p=cnt/cnt.sum(); zs=remap[Z.zstd_spread(list(norm),s["al"])].astype(np.int32)
    dts=dt_tans(l,p,50,np.random.default_rng(0))[0]
    msg=np.tile(sy,int(np.ceil(20_000_000/len(sy))))[:20_000_000].astype(np.int32)
    open("/tmp/b.bin","wb").write(struct.pack("<iiq",L,n,len(msg))+zs.astype("<i4").tobytes()+dts.astype("<i4").tobytes()+msg.astype("<i4").tobytes())
    r=[json.loads(x) for x in subprocess.run(["./bench","/tmp/b.bin"],capture_output=True,text=True).stdout.split("\n") if x]
    # C cost of one exact rate evaluation (the unit of the DT-tANS budget)
    t=time.time(); [exact_rate(dts,p) for _ in range(200)]; ev=(time.time()-t)/200
    rec=dict(stream=name,L=L,n=n,zstd=r[0],dt=r[1],exact_eval_ms=1e3*ev); out.append(rec); print(json.dumps(rec))
json.dump(out,open("../results/bench.json","w"),indent=1)
