# Real deployment 3: PivCo-Huffman PHA prebuilt FSE tables (50 static byte tables, tableLog 12; FSE fast spread)
import numpy as np, json, sys
sys.path.insert(0, '.')
from tanslib import *
def normalize_counts(probs, total):  # re-implementation of extras/gen_fse_tables.py (largest remainder, min 1)
    raw = [p * total for p in probs]; fl = [int(x) for x in raw]; frac = [(raw[v] - fl[v], v) for v in range(256)]
    c = list(fl)
    for v in range(256):
        if probs[v] > 0 and c[v] == 0: c[v] = 1
    rem = total - sum(c)
    if rem >= 0:
        frac.sort(reverse=True)
        for _, v in frac[:rem]: c[v] += 1
    else:
        order = sorted(range(256), key=lambda v: c[v], reverse=True); i = 0; d = -rem
        while d > 0:
            v = order[i % 256]
            if c[v] > 1: c[v] -= 1; d -= 1
            i += 1
    return np.array(c)
pop = np.array([bin(v).count('1') for v in range(256)])
rng = np.random.default_rng(6); out = []
for i in range(1, 51):
    q = (49 + i) / 100; p = (1 - q) ** pop * q ** (8 - pop); p /= p.sum()
    for Llog in (12, 10):
        l = normalize_counts(p, 1 << Llog)
        r = {'p0': q, 'Llog': Llog, 'H': entropy(p), 'fse': exact_rate(spread_fast(l), p), 'tuned': exact_rate(spread_tuned(l, p), p)}
        r['iter'] = spread_iterated_tuned(l, p, 8)[1]
        r['dt'] = exact_rate(dt_tans(l, p, 50, rng=rng)[0], p) if (i % 7 == 0 or i in (31, 41, 46, 50)) else float('nan')
        r['kl_tuned'] = exact_rate(spread_tuned(quantize(p, 1 << Llog), p), p)
        out.append(r)
        print('p0=%.2f L=2^%d H=%.4f fse=%.4f tuned=%.4f iter=%.4f dt=%.4f kl+tuned=%.4f  gain(best spread) %.2f%%  gain(+kl) %.2f%%' % (q, Llog, r['H'], r['fse'], r['tuned'], r['iter'], r['dt'], r['kl_tuned'],
              100 * (r['fse'] - min(r['tuned'], r['iter'])) / r['fse'], 100 * (r['fse'] - r['kl_tuned']) / r['fse']))
json.dump(out, open('../results/pha.json', 'w'), indent=1)
