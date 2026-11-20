import numpy as np, sys
exec(open('iot2.py').read().split("rng = np.random.default_rng(3)")[0])
# stakeholders = sensors (Prop. 2 of the preprint): one firmware table serves all of them
ids = [i for i in np.unique(ins_of) if (ins_of == i).sum() > 100]
Qs = {i: posdist(P[tr & (ins_of == i)]) for i in ids}; Qp = posdist(Ptr)
pall = np.bincount(Ptr.ravel(), minlength=8) / Ptr.size; l0 = quantize(pall, L)
fpool = lambda s: packet_cost(s, Qp)
fmax = lambda s: max(packet_cost(s, Qs[i]) for i in ids)
def dtp(f, B, seed=1):
    global Q
    Q = Qp; return dt_packet(spread_tuned(l0, pall), f, B, rng=np.random.default_rng(seed))[0]
sp = dtp(fpool, 50); sm = dtp(fmax, 50)
lsm = swap_search(sm, fmax)[0]
print('sensor  packets  p(diff=0)  deployed  pooled-DT  minimax-DT  minimax-LS')
for nm, s in []: pass
rows = []
for i in ids:
    Pt = P[~tr & (ins_of == i)]
    b = [enc_bits(s, Pt) / len(Pt) for s in (SENSOR, sp, sm, lsm)]; rows.append(b)
    print('%5d %7d   %.3f   %.3f   %.3f   %.3f   %.3f' % (i, len(Pt), (Pt[:, 1:] == 3).mean(), *b))
RR = np.array(rows); print('worst sensor:', np.round(RR.max(0), 3), ' mean:', np.round(RR.mean(0), 3))
lsp = swap_search(sp, fpool)[0]; print(type(L), lsp.dtype, type(k) if "k" in dir() else "")
R2 = np.array([[enc_bits(s, P[~tr & (ins_of == i)]) / (~tr & (ins_of == i)).sum() for s in (lsp, lsm)] for i in ids])
print('pooled-LS vs minimax-LS worst %.3f %.3f mean %.3f %.3f' % (R2[:, 0].max(), R2[:, 1].max(), R2[:, 0].mean(), R2[:, 1].mean()))
