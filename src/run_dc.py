# Data-centre scaling (Monte Carlo). Volumes and unit value from Duda's 2026 report "Kodowanie entropijne w skali planety" v4.2:
# ANS codec output 186 EB/yr (reference), P10-P90 150-343 EB, 93% via zstd; Huffman-replacement scenario 1.9 EB <-> 45 MUSD
# (range 26-120 MUSD for 1.5-3.3 EB). Savings per byte measured here on real libzstd 1.5.7 frames (Z1).
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(0); N=200000
def z1(lv):
    R=[json.loads(x) for x in open(f"../results/z1_{lv}.log")]; c=sum(r["comp"] for r in R)
    return {k:sum(r["b_fast"]-r[k] for r in R)/8/c for k in ("b_tunedp","b_dt")}
F={lv:z1(lv) for lv in (1,3,19)}; print("measured saving fraction of zstd output:",{lv:{k:f"{100*v:.4f}%" for k,v in d.items()} for lv,d in F.items()})
mu=np.log(np.sqrt(150*343)); sig=(np.log(343)-np.log(150))/(2*1.2816)
V=np.exp(rng.normal(mu,sig,N))*0.93                       # zstd output, EB/yr
w19=rng.uniform(0.02,0.15,N); w1=rng.uniform(0.2,0.5,N)*(1-w19); w3=1-w19-w1   # byte share by level class (assumed)
f=w1*F[1]["b_dt"]+w3*F[3]["b_dt"]+w19*F[19]["b_dt"]
usd=rng.triangular(26/3.3,45/1.9,120/1.5,N)*1e6          # USD per EB of avoided bytes, implied by the report
scen={"A: technical ceiling (all zstd, new format, side-information free)":np.ones(N),
      "B: archival + static tables only (level 19 class, dictionaries)":w19/np.maximum(1,1)+rng.uniform(0,0.02,N),
      "C: today's zstd format":np.zeros(N)}
out={}
for k,a in scen.items():
    if k.startswith("B"): eb=V*w19*F[19]["b_dt"]+V*rng.uniform(0,0.02,N)*F[3]["b_dt"]
    else: eb=V*f*a
    pb=eb*1000; m=pb/1000*usd/1e6
    out[k]=dict(PB_per_year=[float(np.percentile(pb,q)) for q in (10,50,90)],MUSD_per_year=[float(np.percentile(m,q)) for q in (10,50,90)])
    print(k,"PB/yr P10/50/90 %.1f / %.1f / %.1f ; MUSD/yr %.2f / %.2f / %.2f"%(*out[k]["PB_per_year"],*out[k]["MUSD_per_year"]))
out["reference: FSE vs Huffman contribution (report)"]=dict(PB_per_year=[1500,1900,3300],MUSD_per_year=[26,45,120])
out["measured_fraction"]=F
json.dump(out,open("../results/dc_sim.json","w"),indent=1)
fig,ax=plt.subplots(figsize=(6.4,2.6)); labs=["A: ceiling,\nall zstd","B: archival /\nstatic tables","FSE vs Huffman\n(whole tANS gain,\nDuda 2026 report)"]
keys=[list(scen)[0],list(scen)[1],"reference: FSE vs Huffman contribution (report)"]
for i,k in enumerate(keys):
    lo,md,hi=out[k]["PB_per_year"]; ax.barh(i,md,color=["C0","C2","0.6"][i]); ax.errorbar(md,i,xerr=[[md-lo],[hi-md]],fmt="none",ecolor="k",capsize=3)
    ax.text(hi*1.15,i,f"{md:.0f} PB/yr",va="center",fontsize=7)
ax.set_xscale("log"); ax.set_yticks(range(3)); ax.set_yticklabels(labs,fontsize=7); ax.set_xlabel("bytes avoided per year [PB/yr], log scale (P10–P90 bars)"); ax.set_ylabel("scenario [–]")
ax.set_xlim(0.5,2e4); fig.tight_layout(); fig.savefig("../fig/dc_scaling.pdf"); fig.savefig("../fig/dc_scaling.png",dpi=170)
