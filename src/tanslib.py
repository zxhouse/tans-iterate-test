import ctypes, numpy as np, heapq, os, math
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtans.so"))
_ip = np.ctypeslib.ndpointer(np.int32, flags="C"); _dp = np.ctypeslib.ndpointer(np.float64, flags="C")
_lib.exact_rate.restype = ctypes.c_double
_lib.exact_rate.argtypes = [_ip, ctypes.c_int, ctypes.c_int, _dp, _dp, ctypes.c_int]
_lib.trace_bits.restype = ctypes.c_double
_lib.trace_bits.argtypes = [_ip, ctypes.c_int, ctypes.c_int, _ip, ctypes.c_long, ctypes.POINTER(ctypes.c_int)]
_lib.local_search.restype = ctypes.c_double
_lib.local_search.argtypes = [_ip, ctypes.c_int, ctypes.c_int, _dp, ctypes.c_int, _dp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_int)]

def entropy(p):
    p = np.asarray(p); p = p[p > 0]; return float(-(p * np.log2(p)).sum())

def exact_rate(sym, p):
    sym = np.ascontiguousarray(sym, np.int32); p = np.ascontiguousarray(p, np.float64)
    pi = np.zeros(len(sym)); return _lib.exact_rate(sym, len(sym), len(p), p, pi, 0)

def trace_bits(sym, n, trace, state=0):
    sym = np.ascontiguousarray(sym, np.int32); trace = np.ascontiguousarray(trace, np.int32)
    st = ctypes.c_int(state); b = _lib.trace_bits(sym, len(sym), n, trace, len(trace), ctypes.byref(st)); return b, st.value

def local_search(sym, P, wts=None, w=4, reassign=True, maxsweeps=200):
    """P: (ncls, n) array of class distributions (or 1-D). Returns (sym, objective, evals, sweeps)."""
    P = np.atleast_2d(np.ascontiguousarray(P, np.float64)); ncls, n = P.shape
    wts = np.ones(ncls) / ncls if wts is None else np.ascontiguousarray(wts, np.float64)
    sym = np.ascontiguousarray(sym, np.int32).copy(); ev = ctypes.c_long(0); sw = ctypes.c_int(0)
    r = _lib.local_search(sym, len(sym), n, np.ascontiguousarray(P.ravel()), ncls, wts, w, int(reassign), maxsweeps, ctypes.byref(ev), ctypes.byref(sw))
    return sym, r, ev.value, sw.value

def quantize(p, L):
    """KL-optimal greedy quantization: l_s >= 1, sum l_s = L."""
    n = len(p); l = np.ones(n, dtype=int)
    h = [(-p[s] * math.log2(2.0), s) for s in range(n)]; heapq.heapify(h)
    for _ in range(L - n):
        g, s = heapq.heappop(h); l[s] += 1
        heapq.heappush(h, (-p[s] * math.log2((l[s] + 1) / l[s]), s))
    return l

def spread_fast(l):
    L = int(l.sum()); step = (L >> 1) + (L >> 3) + 3; sym = np.zeros(L, np.int32); pos = 0
    for s, c in enumerate(l):
        for _ in range(c): sym[pos] = s; pos = (pos + step) & (L - 1)
    return sym

def spread_precise(l):
    L = int(l.sum()); keys = [((i + 0.5) * L / c, s) for s, c in enumerate(l) for i in range(c)]
    keys.sort(); return np.array([s for _, s in keys], np.int32)

def spread_tuned(l, p):
    keys = [(1.0 / (p[s] * math.log(1 + 1.0 / i)), s) for s, c in enumerate(l) for i in range(c, 2 * c)]
    keys.sort(); return np.array([s for _, s in keys], np.int32)

def huffman_rate(p):
    h = [(pi, i, 0) for i, pi in enumerate(p)]; heapq.heapify(h); tot = 0.0; c = len(p)
    while len(h) > 1:
        a = heapq.heappop(h); b = heapq.heappop(h); tot += a[0] + b[0]; heapq.heappush(h, (a[0] + b[0], c, 0)); c += 1
    return tot

def sources():
    d = {}
    z = lambda n, a: (lambda v: v / v.sum())(1.0 / np.arange(1, n + 1) ** a)
    d["zipf1.0-256"] = z(256, 1.0); d["zipf1.5-64"] = z(64, 1.5)
    g = 0.8 ** np.arange(32); d["geom0.8-32"] = g / g.sum()
    t = np.full(32, 0.002); t[[3, 11, 27]] = [0.52, 0.27, 0.09]; t[[2, 4, 10, 12, 26]] = [0.03, 0.02, 0.015, 0.01, 0.01]; d["ttl-32"] = t / t.sum()
    s = np.array([.38, .05, .03, .02, .02, .02, .03, .04, .02, .02, .02, .03, .04, .05, .06, .15]); d["pktsize-16"] = s / s.sum()
    f = np.full(16, 0.1 / 15); f[0] = 0.9; d["flag0.9-16"] = f
    return d

def huffman_lengths(p):
    n = len(p); h = [(p[i], i, (i,)) for i in range(n)]; heapq.heapify(h); ln = np.zeros(n, int); c = n
    while len(h) > 1:
        a = heapq.heappop(h); b = heapq.heappop(h)
        for i in a[2] + b[2]: ln[i] += 1
        heapq.heappush(h, (a[0] + b[0], c, a[2] + b[2])); c += 1
    return ln

def quantize_pow2(p, L):
    """Huffman-induced power-of-two quantization, length-limited to R=log2 L by a greedy Kraft repair."""
    R = int(math.log2(L)); ln = np.minimum(huffman_lengths(p), R); n = len(p)
    kraft = lambda: sum(2 ** (R - int(x)) for x in ln)
    while kraft() > L:
        c = [s for s in range(n) if ln[s] < R]; s = min(c, key=lambda s: p[s]); ln[s] += 1
    while True:
        slack = L - kraft(); c = [s for s in range(n) if ln[s] > 1 and 2 ** (R - int(ln[s])) <= slack]
        if not c: break
        s = max(c, key=lambda s: p[s]); ln[s] -= 1
    l = np.array([2 ** (R - int(x)) for x in ln]); assert l.sum() == L, (l.sum(), L); return l

def portfolio(p, L):
    l = quantize(p, L); l2 = quantize_pow2(p, L)
    return {"fast": spread_fast(l), "precise": spread_precise(l), "tuned": spread_tuned(l, p), "huffq": spread_tuned(l2, p)}

def ils(sym, p, rounds, rng, w=4, k=3):
    best, rb, ev, _ = local_search(sym, p, w=w); L = len(best); n = len(p)
    for _ in range(rounds):
        c = best.copy()
        for _ in range(k):
            i, j = rng.integers(L, size=2)
            if rng.random() < 0.5: c[i], c[j] = c[j], c[i]
            elif np.count_nonzero(c == c[i]) > 1: c[i] = c[j]
        c, r, e, _ = local_search(c, p, w=w); ev += e
        if r < rb - 1e-11: best, rb = c, r
    return best, rb, ev

def markov_trace(T, m, rng, x0=0):
    n = T.shape[0]; cdf = np.cumsum(T, axis=1); u = rng.random(m); out = np.empty(m, np.int32); x = x0
    for i in range(m):
        x = min(int(np.searchsorted(cdf[x], u[i])), n - 1); out[i] = x
    return out

def stationary(T):
    w, v = np.linalg.eig(T.T); pi = np.real(v[:, np.argmax(np.real(w))]); return pi / pi.sum()

def cond_entropy(T):
    pi = stationary(T); Tm = np.where(T > 0, T, 1); return float(-(pi[:, None] * T * np.log2(Tm)).sum())

def trace_ls(sym, n, trace, w=4, maxsweeps=50, reassign=True):
    """first-improvement sweeps with the empirical bit count on a fixed trace as (deterministic) objective"""
    sym = sym.copy(); L = len(sym); best = trace_bits(sym, n, trace)[0]; ev = 0
    for _ in range(maxsweeps):
        imp = False
        for i in range(L):
            for j in range(max(0, i - w), min(L, i + w + 1)):
                if sym[i] == sym[j]: continue
                if j > i:
                    sym[i], sym[j] = sym[j], sym[i]; b = trace_bits(sym, n, trace)[0]; ev += 1
                    if b < best: best = b; imp = True
                    else: sym[i], sym[j] = sym[j], sym[i]
                if reassign and sym[i] != sym[j] and np.count_nonzero(sym == sym[i]) > 1:
                    a = sym[i]; sym[i] = sym[j]; b = trace_bits(sym, n, trace)[0]; ev += 1
                    if b < best: best = b; imp = True
                    else: sym[i] = a
        if not imp: break
    return sym, best / len(trace), ev

def spread_precise_realp(l, p):
    """Precise initialization of Duda (2013, Sec. 4.1) with real probabilities as suggested in Sec. 4.3: keys 1/(2p_s)+i/p_s."""
    keys = [((i + 0.5) / p[s], s) for s, c in enumerate(l) for i in range(c)]
    keys.sort(); return np.array([s for _, s in keys], np.int32)

def stationary_pi(sym, p):
    sym = np.ascontiguousarray(sym, np.int32); p = np.ascontiguousarray(p, np.float64)
    pi = np.zeros(len(sym)); r = _lib.exact_rate(sym, len(sym), len(p), p, pi, 0); return pi, r

def slot_intervals(l, L):
    """for each symbol s and occurrence y in [l_s,2l_s): predecessor interval [lo,hi) in X coords and nbits"""
    R = int(math.log2(L)); out = []
    for s, c in enumerate(l):
        k = R - int(math.floor(math.log2(c)))
        for y in range(c, 2 * c):
            nb = k if (y << k) < 2 * L else k - 1
            out.append((s, y, (y << nb) - L, ((y + 1) << nb) - L, nb))
    return out

def spread_from_keys(keys):
    """keys: list of (key, s); smaller key -> lower state"""
    keys = sorted(keys); return np.array([s for _, s in keys], np.int32)

def spread_iterated_tuned(l, p, iters=8, start=None):
    """Duda (2021, Sec. IV-B) 'iterated tuned symbol spread': rho_{t+1} = stationary distribution of current spread;
    new spread = slots sorted by decreasing inflow mass p_s * rho(interval). Returns best spread seen (exact rate)."""
    L = int(l.sum()); sym = spread_tuned(l, p) if start is None else start.copy(); slots = slot_intervals(l, L)
    best = sym; bestr = exact_rate(sym, p); hist = [bestr]
    for _ in range(iters):
        pi, _ = stationary_pi(sym, p); P = np.concatenate([[0.0], np.cumsum(pi)])
        keys = [(-(p[s] * (P[hi] - P[lo])), s) for s, y, lo, hi, nb in slots]
        sym = spread_from_keys(keys); r = exact_rate(sym, p); hist.append(r)
        if r < bestr - 1e-13: best, bestr = sym, r
    return best, bestr, hist

def spread_ranged(l):
    return np.repeat(np.arange(len(l)), l).astype(np.int32)

def spread_dube_yokoo(l, p, iters=30):
    """Dube-Yokoo (ISIT 2019) as described by Steiner et al. 2025: start from ranged table, repeatedly re-sort by stationary state probabilities."""
    return spread_iterated_tuned(l, p, iters=iters, start=spread_ranged(l))

import scipy.sparse as _sp, scipy.sparse.linalg as _spl
def chain(sym, p):
    """transition targets T[s,X] (X coords), bits B[s,X], cost b(X)=sum_s p_s B[s,X]"""
    L = len(sym); n = len(p); R = int(math.log2(L)); l = np.bincount(sym, minlength=n)
    order = [np.flatnonzero(sym == s) for s in range(n)]
    x = np.arange(L, 2 * L); T = np.zeros((n, L), np.int64); B = np.zeros((n, L), np.int64)
    for s in range(n):
        k = R - int(math.floor(math.log2(l[s]))); nb = np.where((x >> k) >= l[s], k, k - 1)
        y = x >> nb; T[s] = order[s][y - l[s]]; B[s] = nb
    return T, B, (p[:, None] * B).sum(0)
def value_function(sym, p):
    """average cost g and relative values h (h[0]=0) of the state chain with per-state cost b(x)"""
    T, B, b = chain(sym, p); L = len(sym); n = len(p)
    rows = np.repeat(np.arange(L), n); cols = T.T.reshape(-1); vals = np.tile(p, L)
    P = _sp.csr_matrix((vals, (rows, cols)), shape=(L, L))
    pi, g = stationary_pi(sym, p)
    A = (_sp.identity(L) - P).tolil(); rhs = b - g
    A[0, :] = 0; A[0, 0] = 1; rhs = rhs.copy(); rhs[0] = 0
    h = _spl.spsolve(A.tocsr(), rhs); return g, h, pi
def spread_policy_iteration(l, p, iters=8, start=None):
    """ours: improvement step = slots sorted by decreasing inflow mass, positions sorted by increasing relative value h"""
    L = int(l.sum()); sym = spread_tuned(l, p) if start is None else start.copy(); slots = slot_intervals(l, L)
    best = sym; bestr = exact_rate(sym, p); hist = [bestr]
    for _ in range(iters):
        g, h, pi = value_function(sym, p); P = np.concatenate([[0.0], np.cumsum(pi)])
        mass = np.array([p[s] * (P[hi] - P[lo]) for s, y, lo, hi, nb in slots]); ss = np.array([s for s, *_ in slots])
        pos = np.argsort(h, kind="stable"); new = np.empty(L, np.int32); new[pos] = ss[np.argsort(-mass, kind="stable")]
        sym = new; r = exact_rate(sym, p); hist.append(r)
        if r < bestr - 1e-13: best, bestr = sym, r
    return best, bestr, hist

def spread_safeguarded_tuned(l, p, max_iters=30, alphas=(1.0, 0.5, 0.25, 0.125, 0.0625)):
    """ours: Duda's iterated tuned spread made monotone. Each step proposes the policy-improvement ordering
    (slots by decreasing inflow mass under the current stationary distribution) and, if the exact rate does not
    improve, interpolates between the current and proposed slot positions with step alpha (backtracking).
    Stops when no alpha improves. Never worse than its start (the tuned spread)."""
    L = int(l.sum()); n = len(l); slots = slot_intervals(l, L); ss = np.array([s for s, *_ in slots])
    off = np.concatenate([[0], np.cumsum(l)[:-1]])          # slot index of (s, y) = off[s] + (y - l_s)
    cur = spread_tuned(l, p); rcur = exact_rate(cur, p); evals = 1; steps = 0
    for _ in range(max_iters):
        pi, _ = stationary_pi(cur, p); P = np.concatenate([[0.0], np.cumsum(pi)])
        mass = np.array([p[s] * (P[hi] - P[lo]) for s, y, lo, hi, nb in slots])
        target = np.empty(L); target[np.argsort(-mass, kind="stable")] = np.arange(L)
        oldpos = np.empty(L)
        for s in range(n): oldpos[off[s]:off[s] + l[s]] = np.flatnonzero(cur == s)
        ok = False
        for a in alphas:
            key = a * target + (1 - a) * oldpos
            cand = np.empty(L, np.int32); cand[np.argsort(key, kind="stable")] = ss
            if np.array_equal(cand, cur): continue
            r = exact_rate(cand, p); evals += 1
            if r < rcur - 1e-13: cur, rcur, ok = cand, r, True; steps += 1; break
        if not ok: break
    return cur, rcur, evals, steps

def random_dists(kind, n, rng):
    if kind == "uniform": v = rng.random(n)
    elif kind == "zipf": v = 1.0 / rng.permutation(np.arange(1, n + 1))
    elif kind == "dirichlet0.3": v = rng.dirichlet(np.full(n, 0.3)) + 1e-9
    elif kind == "geometric": v = rng.permutation(0.85 ** np.arange(n))
    return v / v.sum()

def dt_loop(l, p, budget=40, w=4, k_build=2, prescreen=True, retain=3, rng=None, start=None):
    """Iterate-test loop of Kuras et al. (arXiv:2611.03034) applied to tANS tables.
    prototype = symbol spread; ideation = swaps within window w of a leader (local ideation, Assumption 2);
    pre-screening = first-order predicted change dR ~ (rho_i - rho_j)(h_j - h_i) from the leader's stationary
    distribution rho and relative value function h (h is nondecreasing, Lemma); test = exact rate (unbiased, noise-free);
    retention = archive, ideas drawn from the `retain` best (Assumption 5). Budget counts exact rate evaluations
    plus value-function solves."""
    rng = rng or np.random.default_rng(0); L = int(l.sum())
    cur = spread_tuned(l, p) if start is None else start.copy()
    arch = [(exact_rate(cur, p), cur)]; used = 1; cache = {}
    seen = {cur.tobytes()}
    while used < budget:
        arch.sort(key=lambda t: t[0]) if retain else None; top = arch[:max(retain, 1)]
        r0, par = top[0] if rng.random() < 0.6 or len(top) == 1 else top[int(rng.integers(1, len(top)))]
        key = par.tobytes()
        if key not in cache:
            g, h, rho = value_function(par, p); cache[key] = (h, rho); used += 1
        h, rho = cache[key]
        ideas = [(i, j) for i in range(L) for j in range(i + 1, min(L, i + w + 1)) if par[i] != par[j]]
        if prescreen:
            sc = np.array([(rho[i] - rho[j]) * (h[j] - h[i]) for i, j in ideas]); order = np.argsort(sc)
            pick = [ideas[o] for o in order]
        else:
            pick = [ideas[o] for o in rng.permutation(len(ideas))]
        built = 0
        for i, j in pick:
            if built >= k_build or used >= budget: break
            c = par.copy(); c[i], c[j] = c[j], c[i]; kb = c.tobytes()
            if kb in seen: continue
            seen.add(kb); r = exact_rate(c, p); used += 1; built += 1
            if retain: arch.append((r, c))
            else: arch = [(r, c)]                        # no retention: always move to the newest prototype
        if built == 0: break                              # no unseen idea left near the leaders
    best = min(arch, key=lambda t: t[0]) if retain else arch[0]
    return best[1], best[0], used

def dt_tans(l, p, budget=50, rng=None, prescreen=True, retain=3):
    """DT-tANS: concept pre-screening over {tuned spread, Duda's iterated tuned spread}, then the iterate-test loop."""
    s_it, r_it, hist = spread_iterated_tuned(l, p, 8); used = len(hist)
    return dt_loop(l, p, budget=max(budget - used, 2), rng=rng, start=s_it, prescreen=prescreen, retain=retain)[:2] + (budget,)

def hill_climb(sym, p, budget, w=4, rng=None):
    """stochastic hill climbing (Blanes et al. 2020; Pieprzyk et al. 2023 Alg. 7, windowed): random swap, exact test, keep if better"""
    rng = rng or np.random.default_rng(0); cur = sym.copy(); r = exact_rate(cur, p); used = 1; L = len(sym)
    while used < budget:
        i = int(rng.integers(L)); j = min(L - 1, i + 1 + int(rng.integers(w)))
        if cur[i] == cur[j]: continue
        c = cur.copy(); c[i], c[j] = c[j], c[i]; rc = exact_rate(c, p); used += 1
        if rc < r: cur, r = c, rc
    return cur, r
