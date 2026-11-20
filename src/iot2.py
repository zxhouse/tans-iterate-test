import numpy as np, json, math, sys
sys.path.insert(0, '.')
from tanslib import *
exec(open('run_iot.py').read().split("lS = np.bincount")[0])  # reuse loading, packets, enc_bits
# time split per sensor: first half of each sensor's packets = design (train), second half = evaluation (test)
from itertools import groupby
P = np.array(packets, np.int32)
ins_of = []
for ins, grp in groupby(rows, key=lambda r: r[0]):
    n = len(list(grp)); ins_of += [ins] * ((n - 6 + 6) // 7 if n >= 7 else 0)
ins_of = np.array(ins_of[:len(P)])
tr = np.zeros(len(P), bool)
for i in np.unique(ins_of):
    idx = np.flatnonzero(ins_of == i); tr[idx[:len(idx) // 2]] = True
Ptr, Pte = P[tr], P[~tr]
def posdist(PP): return np.array([np.bincount(PP[:, t], minlength=8) / len(PP) for t in range(7)])
Q = posdist(Ptr)
def packet_cost(sym, Q):
    T, B, _ = chain(sym, np.ones(8) / 8); d = np.zeros(L); d[0] = 1.0; tot = 0.0
    for t in range(Q.shape[0]):
        nd = np.zeros(L)
        for s in range(8):
            if Q[t, s] == 0: continue
            tot += Q[t, s] * (d * B[s]).sum(); np.add.at(nd, T[s], Q[t, s] * d)
        d = nd
    return tot
def swap_search(sym, f, w=4, evals=None):
    sym = sym.copy(); best = f(sym); ne = 1; imp = True
    while imp:
        imp = False
        for i in range(L):
            for j in range(i + 1, min(L, i + w + 1)):
                if sym[i] == sym[j]: continue
                sym[i], sym[j] = sym[j], sym[i]; c = f(sym); ne += 1
                if c < best - 1e-12: best = c; imp = True
                else: sym[i], sym[j] = sym[j], sym[i]
    return sym, best, ne
def dt_packet(sym0, f, budget=50, w=4, rng=None):
    # DT-tANS loop with the packet objective as the test; pre-screen swaps with stationary value-function score of the position-averaged distribution
    pbar = Q[1:].mean(0); arch = [(f(sym0), sym0)]; used = 1; seen = {sym0.tobytes()}
    while used < budget:
        arch.sort(key=lambda t: t[0]); r0, par = arch[0] if rng.random() < 0.6 or len(arch) == 1 else arch[int(rng.integers(1, min(3, len(arch))))]
        g, h, rho = value_function(par, pbar)
        ideas = sorted(((rho[i] - rho[j]) * (h[j] - h[i]), i, j) for i in range(L) for j in range(i + 1, min(L, i + w + 1)) if par[i] != par[j])[::-1]
        k = 0
        for sc, i, j in ideas:
            c = par.copy(); c[i], c[j] = c[j], c[i]
            if c.tobytes() in seen: continue
            seen.add(c.tobytes()); arch.append((f(c), c)); used += 1; k += 1
            if k == 2 or used >= budget: break
        if k == 0: break
    arch.sort(key=lambda t: t[0]); return arch[0][1], arch[0][0]
rng = np.random.default_rng(3)
f = lambda s: packet_cost(s, Q)
out = {'n_train': int(tr.sum()), 'n_test': int((~tr).sum())}
pall = np.bincount(Ptr.ravel(), minlength=8) / Ptr.size
for tag, l in [('sensor_counts', np.bincount(SENSOR, minlength=8)), ('kl_counts', quantize(pall, L))]:
    c = {'sensor_table': SENSOR} if tag == 'sensor_counts' else {}
    c['fast'] = spread_fast(l); c['tuned'] = spread_tuned(l, pall); c['iter_tuned'] = spread_iterated_tuned(l, pall, 8)[0]
    c['dt_stationary'] = dt_tans(l, pall, 50, rng=rng)[0]
    c['dt_packet'] = dt_packet(c['tuned'], f, 50, rng=rng)[0]
    c['ls_packet'], _, ne = swap_search(c['dt_packet'], f)
    out[tag] = {'l': l.tolist(), 'ls_evals': ne}
    for k, s in c.items():
        out[tag][k] = {'E_train': f(s), 'bits_test': enc_bits(s, Pte) / len(Pte)}
json.dump(out, open('../results/iot2.json', 'w'), indent=1)
base = out['sensor_counts']['sensor_table']['bits_test']
other = 6 + 14 * (Pte == 7).sum() / len(Pte)
print('test packets', len(Pte), 'baseline ANS bits/packet %.3f, flush+escape payload %.3f' % (base, other))
for tag in ['sensor_counts', 'kl_counts']:
    print(tag, out[tag]['l'], 'LS evals', out[tag]['ls_evals'])
    for k, v in out[tag].items():
        if isinstance(v, dict): print('  %-14s E_tr %.3f test %.3f  Δans %+.2f%%  Δpayload %+.2f%%' % (k, v['E_train'], v['bits_test'], 100 * (v['bits_test'] - base) / base, 100 * (v['bits_test'] - base) / (base + other)))
