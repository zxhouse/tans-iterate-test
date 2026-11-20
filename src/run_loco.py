# Real deployment 4: LOCO-ANS (Alonso, Sutter, Lopez de Vergara, IEEE Access 2021) static tANS tables, L=128, 32 z-modes + 32 y-modes
import numpy as np, re, json, sys
sys.path.insert(0, '.')
from tanslib import *
D = '../external/LOCO-ANS/codec/src/ANS_tables/'
def parse(fn):
    modes = []; cur = None
    for ln in open(fn):
        m = re.search(r'mode_id:\s*(\d+)\s*\|\s*parameter:\s*([0-9.]+)', ln)
        if m: cur = {'id': int(m.group(1)), 'par': float(m.group(2)), 'sym': []}; modes.append(cur)
        e = re.search(r'\{\s*(-?\d+),\s*(-?\d+)\}', ln.split('//')[0] if m is None else ln.split('//')[0] + '')
        if e and cur is not None and m is None: cur['sym'].append(int(e.group(1)))
        elif m:
            e2 = re.search(r'\{\s*(-?\d+),\s*(-?\d+)\}', ln.split('mode_id')[1])
            if e2: cur['sym'].append(int(e2.group(1)))
    return modes
def geo_src(theta, max_code_len=7, max_symbols=8, min_symbols=2):
    probs = []
    for i in range(max_symbols):
        q = (1 - theta) * theta ** i; rem = 1 - (sum(probs) + q)
        if -np.log2(rem) > max_code_len and i >= min_symbols - 1: break
        probs.append(q)
    probs = probs[:int(2 ** np.floor(np.log2(len(probs))))]
    return np.array(probs + [1 - sum(probs)])
rng = np.random.default_rng(8); out = []
for kind, fn in [('z', 'tANS_z_decoder_table.dat'), ('y', 'tANS_y_decoder_table.dat')]:
    for md in parse(D + fn):
        sym = np.array(md['sym'], np.int32); assert len(sym) == 128, (kind, md['id'], len(sym))
        p = geo_src(md['par'] / (md['par'] + 1)) if kind == 'z' else np.array([1 - md['par'], md['par']])
        n = len(p); l = np.bincount(sym, minlength=n)
        if len(l) != n or (l == 0).any() or n < 2:
            continue
        r = {'kind': kind, 'mode': md['id'], 'par': md['par'], 'n': n, 'H': entropy(p), 'loco': exact_rate(sym, p),
             'tuned': exact_rate(spread_tuned(l, p), p), 'iter': spread_iterated_tuned(l, p, 8)[1],
             'dt': exact_rate(dt_tans(l, p, 50, rng=rng)[0], p)}
        lq = quantize(p, 128); r['kl_tuned'] = exact_rate(spread_tuned(lq, p), p); r['kl_dt'] = exact_rate(dt_tans(lq, p, 50, rng=rng)[0], p)
        r['best'] = min(r['tuned'], r['iter'], r['dt'], r['kl_tuned'], r['kl_dt'])
        out.append(r)
        print('%s%2d par=%.4f n=%d H=%.4f loco=%.4f tuned=%.4f iter=%.4f dt=%.4f kl+dt=%.4f  ΔH loco %.3f%% -> best %.3f%%  gain %.2f%%' % (
            kind, md['id'], md['par'], n, r['H'], r['loco'], r['tuned'], r['iter'], r['dt'], r['kl_dt'],
            100 * (r['loco'] / r['H'] - 1), 100 * (r['best'] / r['H'] - 1), 100 * (r['loco'] - r['best']) / r['loco']))
json.dump(out, open('../results/loco.json', 'w'), indent=1)
