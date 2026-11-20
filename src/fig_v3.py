import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.grid":True,"grid.alpha":.3})
S=[]; [S.extend(json.load(open(f))) for f in sorted(glob.glob("../results/s2_*.json"))]
M=[("iter","iterated tuned (Duda 2021, proposal)","v--"),("dy","Dubé–Yokoo 2019","^--"),("dt_noRet","DT-tANS, no retention","x:"),("dt_noPS","DT-tANS, no pre-screening","s:"),("dt","DT-tANS (ours, 50 evaluations)","D-"),("ls","full local search (hundreds–thousands of evaluations)","o-")]
kinds=["uniform","zipf","dirichlet0.3"]; titles={"uniform":"uniform p, normalised","zipf":"permuted Zipf","dirichlet0.3":"Dirichlet(0.3), skewed"}
fig,ax=plt.subplots(1,3,figsize=(7.4,3.1),sharey=False)
for a,kd in zip(ax,kinds):
    for key,lab,st in M:
        xs=[];ys=[]
        for mult in (2,4,8):
            v=[]
            for blk in S:
                s=blk["summary"]
                if s["kind"]!=kd or s["L"]!=mult*s["n"]: continue
                rr=[100*(1-r[key]/r["tuned"]) for r in blk["records"] if key in r and r["tuned"]>1e-12]
                if rr: v.append(np.median(rr))
            if v: xs.append(mult); ys.append(np.mean(v))
        a.plot(xs,ys,st,ms=3.5,lw=1.1,label=lab)
    a.set_xscale("log",base=2); a.set_xticks([2,4,8]); a.set_xticklabels(["2","4","8"]); a.set_title(titles[kd])
    a.set_xlabel("states per symbol L/n [–]"); a.set_ylabel("redundancy reduction vs tuned [%]")
h,lb=ax[0].get_legend_handles_labels(); fig.legend(h,lb,loc="lower center",ncol=3,fontsize=6,frameon=False); fig.tight_layout(rect=(0,0.13,1,1)); fig.savefig("../fig/v3_benchmark.pdf"); fig.savefig("../fig/v3_benchmark.png",dpi=170)
S3=json.load(open("../results/s3.json")); fig,ax=plt.subplots(1,4,figsize=(7.4,2.5)); Bs=[10,20,30,50,100,200]
for a,blk in zip(ax,S3):
    rows=blk["rows"]; rel=lambda k: np.median([100*(1-r[k]/r["tuned"]) for r in rows])
    a.plot(Bs,[rel(f"dt{B}") for B in Bs],"D-",ms=3,lw=1.1,label="DT-tANS")
    a.plot(Bs,[rel(f"dtnops{B}") for B in Bs],"s:",ms=3,lw=1.1,label="DT-tANS, no pre-screening")
    a.plot(Bs,[rel(f"hc{B}") for B in Bs],"o--",ms=3,lw=1.1,label="stochastic hill climbing")
    a.axhline(rel("ls"),color="k",ls=":",lw=1,label="full local search")
    a.axhline(rel("dy"),color="C3",ls="-.",lw=1,label="Dubé–Yokoo")
    a.set_xscale("log"); a.set_xticks(Bs); a.set_xticklabels([str(b) for b in Bs],fontsize=6)
    a.set_title(f'{blk["kind"].replace("dirichlet0.3","Dirichlet(0.3)")}, n={blk["n"]}, L={blk["L"]}',fontsize=7)
    a.set_xlabel("exact evaluations [–]"); a.set_ylabel("reduction vs tuned [%]")
ax[0].legend(fontsize=5); fig.tight_layout(); fig.savefig("../fig/v3_anytime.pdf"); fig.savefig("../fig/v3_anytime.png",dpi=170)
# numbers for text
allr=[]
for blk in S:
    for r in blk["records"]: allr.append((blk["summary"]["kind"],r))
for kd in kinds:
    rs=[r for k,r in allr if k==kd]
    print(kd,len(rs),"DT<=DY %.3f DT<DY %.3f DT<=iter %.3f bestofDTDY vs tuned med %.2f"%(np.mean([r["dt"]<=r["dy"]+1e-12 for r in rs]),np.mean([r["dt"]<r["dy"]-1e-12 for r in rs]),np.mean([r["dt"]<=r["iter"]+1e-12 for r in rs]),np.median([100*(1-min(r["dt"],r["dy"])/r["tuned"]) for r in rs if r["tuned"]>1e-12])))
print("total distributions",len(allr))
