import numpy as np, json, sys, time
sys.path.insert(0, '.')
from tanslib import *; from reframe import *
prng = np.random.default_rng(21); out = []
cfg = [(k, n, m) for k in ['uniform', 'zipf', 'dirichlet0.3'] for n in [8, 16, 32, 64] for m in [2, 4, 8]]
for kind, n, m in cfg:
    L = n * m
    for t in range(30):
        p = random_dists(kind, n, prng); l0 = quantize(p, L); R = lambda: np.random.default_rng(1000 + t)
        r = dict(kind=kind, n=n, L=L, H=entropy(p), tuned=exact_rate(spread_tuned(l0, p), p),
                 kl50=dt_tans(l0, p, 50, rng=R())[1], kl100=dt_tans(l0, p, 100, rng=R())[1],
                 rf50=dt_reframe(p, L, 50, rng=R())[1], scr50=dt_frames(p, L, 50, rng=R(), midpoint=False)[1],
                 scrmid50=dt_frames(p, L, 50, rng=R(), midpoint=True)[1], scrmid100=dt_frames(p, L, 100, rng=R(), midpoint=True)[1])
        if L <= 128:
            _, r['joint_ls'], r['ls_evals'], _ = local_search(spread_tuned(l0, p), p, reassign=True)
        out.append(r)
    print(kind, n, L, flush=True)
json.dump(out, open('../results/reframe_synth2.json', 'w'), indent=1)
