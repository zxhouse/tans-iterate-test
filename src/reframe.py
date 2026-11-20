"""Reframing for tANS tables (Kuras et al., arXiv:2611.03034, Secs. 4.6 and 5.7): the quantisation L_s is the frame.
More iteration cannot repair a poor frame; the theory's finding is that one reframe at the midpoint of the budget is
better than none. The new frame is chosen by frame-level pre-screening, informed by what the first frame taught:
the symbol costs e_s measured in the best table of frame 0."""
import numpy as np, math
from tanslib import *

def symbol_costs(sym, p):
    T, B, _ = chain(sym, p); pi, _ = stationary_pi(sym, p)
    return (B * pi[None, :]).sum(1)                       # e_s = mean bits charged for s

def corrected_counts(sym, p):
    """the table behaves as if s had probability q_s = 2^-e_s; aim for q_s = p_s by rescaling counts L_s -> L_s p_s / q_s"""
    l = np.bincount(sym, minlength=len(p)); e = symbol_costs(sym, p); t = l * p * 2.0 ** e
    return quantize(t / t.sum(), int(l.sum()))

def neighbour_frames(l, p, k=3):
    """single-state transfers a->b ranked by the KL proxy (smallest increase first)"""
    c = []
    for a in range(len(l)):
        if l[a] < 2: continue
        for b in range(len(l)):
            if a == b: continue
            d = p[a] * math.log2(l[a] / (l[a] - 1)) - p[b] * math.log2((l[b] + 1) / l[b])
            c.append((d, a, b))
    c.sort(); out = []
    for d, a, b in c[:k]:
        m = l.copy(); m[a] -= 1; m[b] += 1; out.append(m)
    return out

def dt_reframe(p, L, budget=50, rng=None, n_screen=4, l0=None):
    """budget/2 in the KL frame, then frame pre-screening (n_screen tuned-spread tests), then the rest in the chosen frame."""
    rng = rng or np.random.default_rng(0); l0 = quantize(p, L) if l0 is None else l0
    b0 = budget // 2; s0, r0, _ = dt_tans(l0, p, b0, rng=rng)
    cands = [corrected_counts(s0, p)] + neighbour_frames(l0, p, n_screen)
    seen = {l0.tobytes()}; scored = []
    for m in cands:
        if m.tobytes() in seen or len(scored) >= n_screen: continue
        seen.add(m.tobytes()); scored.append((exact_rate(spread_tuned(m, p), p), m))
    used = b0 + len(scored)
    best = (r0, s0, 'kl')
    if scored:
        rf, lf = min(scored, key=lambda t: t[0])
        s1, r1, _ = dt_tans(lf, p, max(budget - used, 11), rng=rng)
        if r1 < best[0]: best = (r1, s1, 'reframed')
    return best[1], best[0], best[2]

def frame_screen(p, L, l0=None, k=4):
    """frame-level pre-screening up front: KL frame, its corrected frame and the k-1 nearest KL neighbours,
    each tested once with the tuned spread (cheap concept test)."""
    l0 = quantize(p, L) if l0 is None else l0
    st = spread_tuned(l0, p); cands = [l0, corrected_counts(st, p)] + neighbour_frames(l0, p, k - 1)
    seen = set(); out = []
    for m in cands:
        if m.tobytes() in seen: continue
        seen.add(m.tobytes()); out.append((exact_rate(spread_tuned(m, p), p), m))
    out.sort(key=lambda t: t[0]); return out, len(out) + 1   # +1: stationary solve for the corrected frame

def dt_frames(p, L, budget=50, rng=None, midpoint=True):
    """DT-tANS with frames: screen frames up front, iterate in the best; optionally one reframe at the midpoint
    to the frame suggested by the best table so far (corrected counts) or the runner-up of the screening."""
    rng = rng or np.random.default_rng(0); scr, used = frame_screen(p, L)
    l1 = scr[0][1]
    if not midpoint:
        s, r, _ = dt_tans(l1, p, max(budget - used, 11), rng=rng); return s, r, 'screened'
    b1 = (budget - used) // 2; s1, r1, _ = dt_tans(l1, p, max(b1, 11), rng=rng); used += max(b1, 11)
    nxt = corrected_counts(s1, p); used += 1
    if nxt.tobytes() == l1.tobytes() and len(scr) > 1: nxt = scr[1][1]
    best = (r1, s1, 'screened')
    if nxt.tobytes() != l1.tobytes():
        s2, r2, _ = dt_tans(nxt, p, max(budget - used, 11), rng=rng)
        if r2 < r1: best = (r2, s2, 'reframed')
    return best[1], best[0], best[2]
