# Observation 1: is the relative value function h of the tANS state chain nondecreasing in the state x?
# 3 spreads (tuned, fast, random) x 3 sources x n in {8,16,32,64} x L/n in {2,4,8} x 15 distributions = 1620 spreads
import numpy as np, json, math
from tanslib import *
rng = np.random.default_rng(1620); out = []
for kind in ['uniform', 'zipf', 'dirichlet0.3']:
    for n in [8, 16, 32, 64]:
        for m in [2, 4, 8]:
            L = n * m
            for t in range(15):
                p = random_dists(kind, n, rng); l = quantize(p, L)
                rnd = spread_fast(l).copy(); rng.shuffle(rnd)
                for name, s in [('tuned', spread_tuned(l, p)), ('fast', spread_fast(l)), ('random', rnd)]:
                    g, h, _ = value_function(s, p); x = np.arange(L, 2 * L)
                    mono = bool(np.all(np.diff(h) >= -1e-9)); a = float(np.polyfit(np.log2(x), h, 1)[0])
                    out.append(dict(kind=kind, n=n, L=L, spread=name, monotone=mono, slope=a))
mono = sum(o['monotone'] for o in out); sl = np.array([o['slope'] for o in out])
print(len(out), 'spreads,', mono, 'with nondecreasing h; slope median %.2f (5-95%%: %.2f-%.2f)' % (np.median(sl), *np.percentile(sl, [5, 95])))
json.dump(out, open('../results/monotone.json', 'w'))
