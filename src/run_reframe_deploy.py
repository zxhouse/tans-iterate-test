import numpy as np, json, sys
sys.path.insert(0, '.')
from tanslib import *; from reframe import *
exec(open('run_pha.py').read().split("pop = np.array")[0].split("from tanslib import *")[1])  # normalize_counts
pop = np.array([bin(v).count('1') for v in range(256)]); out = {'pha': [], 'loco': []}
for q in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]:
    p = (1 - q) ** pop * q ** (8 - pop); p /= p.sum()
    for Llog in (10, 12):
        L = 1 << Llog; l_dep = normalize_counts(p, L); r_dep = exact_rate(spread_fast(l_dep), p)
        a = dt_tans(quantize(p, L), p, 50, rng=np.random.default_rng(1))[1]; b = dt_reframe(p, L, 50, rng=np.random.default_rng(1))
        out['pha'].append(dict(p0=q, Llog=Llog, H=entropy(p), deployed=r_dep, kl_dt50=a, reframe50=b[1], tag=b[2]))
        print('PHA p0=%.2f L=2^%d deployed %.4f KL+DT50 %.4f (%.2f%%) reframe50 %.4f (%.2f%%) %s' % (q, Llog, r_dep, a, 100 * (r_dep - a) / r_dep, b[1], 100 * (r_dep - b[1]) / r_dep, b[2]), flush=True)
lo = json.load(open('../results/loco.json'))
from run_loco_helpers import geo_src
for r in lo:
    p = geo_src(r['par'] / (r['par'] + 1)) if r['kind'] == 'z' else np.array([1 - r['par'], r['par']])
    if len(p) < 3: continue
    a = dt_tans(quantize(p, 128), p, 50, rng=np.random.default_rng(1))[1]; b = dt_reframe(p, 128, 50, rng=np.random.default_rng(1))[1]
    out['loco'].append(dict(mode=r['mode'], loco=r['loco'], H=r['H'], kl_dt50=a, reframe50=b))
x = np.array([(o['kl_dt50'] - o['reframe50']) / o['kl_dt50'] * 100 for o in out['loco']]); y = np.array([(o['loco'] - o['reframe50']) / o['loco'] * 100 for o in out['loco']])
print('LOCO z tables', len(x), 'reframe vs KL+DT50 mean %.3f%% max %.3f%%; vs deployed max %.3f%%' % (x.mean(), x.max(), y.max()))
json.dump(out, open('../results/reframe_deploy.json', 'w'), indent=1)
