# Real deployment 1: battery temperature sensor (Markiewicz et al., Energy & Buildings 2025, github mmr01/ans-iot)
import numpy as np, json, math, sys
sys.path.insert(0, '.')
from tanslib import *
rows = []
for ln in open('../external/ans-iot/data/temperatures.txt'):
    f = ln.split()
    if len(f) >= 4 and f[0].isdigit():
        rows.append((int(f[0]), f[2] + ' ' + f[3], float(f[1])))
rows.sort()
L, R = 64, 6
SENSOR = np.array([2,4,3,3,3,3,3,3,2,4,3,3,3,3,3,3,3,2,4,3,3,3,3,3,3,3,2,3,4,3,3,3,3,3,3,2,3,4,3,3,3,3,3,3,2,3,4,3,3,3,3,3,3,3,3,3,3,3,3,5,1,7,6,0], np.int32)
def raw14(t): return int(round((t + 46.85) * 16384 / 175.72))  # SHT2x conversion, 16-bit raw >> 2
# packets of 7 consecutive readings per sensor, first absolute (symbol 7)
packets = []
from itertools import groupby
for ins, grp in groupby(rows, key=lambda r: r[0]):
    v = [raw14(t) for _, _, t in grp]
    for k in range(0, len(v) - 6, 7):
        s = [7]
        for i in range(1, 7):
            d = v[k + i] - v[k + i - 1]; s.append(d + 3 if abs(d) <= 3 else 7)
        packets.append(s)
P = np.array(packets, np.int32)
cnt = np.bincount(P.ravel(), minlength=8); p = cnt / cnt.sum()
pd = np.bincount(P[:, 1:].ravel(), minlength=8); pdiff = pd / pd.sum()
def enc_bits(sym, pk):
    n = 8; l = np.bincount(sym, minlength=n); order = [np.flatnonzero(sym == s) for s in range(n)]
    tot = 0
    for msg in pk:
        x = L
        for s in msg:
            k = R - int(math.floor(math.log2(l[s]))); nb = k if (x >> k) >= l[s] else k - 1
            tot += nb; x = L + order[s][(x >> nb) - l[s]]
    return tot
lS = np.bincount(SENSOR, minlength=8)
res = {'n_packets': len(packets), 'p_all': p.tolist(), 'p_diff': pdiff.tolist(), 'H_all': entropy(p), 'l_sensor': lS.tolist()}
rng = np.random.default_rng(1)
def evalset(l, tag):
    out = {}
    cands = {'fast': spread_fast(l), 'precise': spread_precise(l), 'tuned': spread_tuned(l, p)}
    s_it, _, _ = spread_iterated_tuned(l, p, 8); cands['iter_tuned'] = s_it
    s_dt = dt_tans(l, p, 50, rng=rng)[0]; cands['dt_tans'] = s_dt
    s_ls = local_search(s_dt, p)[0]; cands['local_search'] = s_ls
    for k, s in cands.items():
        out[k] = {'rate': exact_rate(s, p), 'bits_ans': enc_bits(s, packets)}
    return out
res['sensor_table'] = {'rate': exact_rate(SENSOR, p), 'bits_ans': enc_bits(SENSOR, packets)}
res['same_counts'] = evalset(lS, 'sens')
lq = quantize(p, L); res['l_kl'] = lq.tolist(); res['requantized'] = evalset(lq, 'kl')
esc = int((P == 7).sum()); res['escape_payload_bits'] = 14 * esc; res['flush_bits'] = 6 * len(packets)
json.dump(res, open('../results/iot.json', 'w'), indent=1)
b0 = res['sensor_table']['bits_ans']
print('packets', len(packets), 'p', np.round(p, 4), 'H', res['H_all'], 'lS', lS, 'lq', lq)
print('sensor', res['sensor_table'], 'per packet', b0 / len(packets))
tot0 = b0 + res['escape_payload_bits'] + res['flush_bits']
for g in ['same_counts', 'requantized']:
    for k, v in res[g].items():
        print(g, k, round(v['rate'], 4), v['bits_ans'], 'Δans %.2f%%' % (100 * (v['bits_ans'] - b0) / b0), 'Δtotal payload %.2f%%' % (100 * (v['bits_ans'] - b0) / tot0))
