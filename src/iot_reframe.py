import numpy as np, sys
exec(open('iot2.py').read().split("rng = np.random.default_rng(3)")[0])
from reframe import corrected_counts, neighbour_frames
def run(Qx, Ptest, pbar, tag):
    global Q
    Q = Qx; f = lambda s: packet_cost(s, Qx)
    def dtp(l, B, seed): return dt_packet(spread_tuned(l, pbar), f, B, rng=np.random.default_rng(seed))
    base = enc_bits(SENSOR, Ptest) / len(Ptest); l0 = quantize(pbar, L)
    s50, c50 = dtp(l0, 50, 1)
    sA, cA = dtp(l0, 25, 1)
    cands = [corrected_counts(sA, pbar)] + neighbour_frames(l0, pbar, 6)
    sc = sorted(((f(spread_tuned(m, pbar)), i, m) for i, m in enumerate(cands) if m.tobytes() != l0.tobytes()), key=lambda t: t[0])[:4]
    lB = sc[0][2]; sB, cB = dtp(lB, 50 - 25 - len(sc), 1)
    sR = sB if cB < cA else sA
    for name, s in [('KL+DTpacket50', s50), ('reframe50', sR)]:
        b = enc_bits(s, Ptest) / len(Ptest); print(tag, name, 'counts', np.bincount(s, minlength=8), 'test %.3f bits/packet, %+.2f%% vs deployed' % (b, 100 * (b - base) / base))
pall = np.bincount(Ptr.ravel(), minlength=8) / Ptr.size
run(posdist(Ptr), Pte, pall, '7-symbol packets:')
Q6 = posdist(Ptr)[1:]; pd = np.bincount(Ptr[:, 1:].ravel(), minlength=8) / Ptr[:, 1:].size
SENSOR_bits = None
run(Q6, Pte[:, 1:], pd, '6 differences:')
