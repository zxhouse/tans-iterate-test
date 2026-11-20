# Parse Pcodec (pco 1.x) standalone files per docs/format.md: tANS size, bin weights, bin bounds; reconstruct bin indices of latents
import numpy as np
class BR:
    def __init__(s, b): s.b = b; s.p = 0
    def read(s, n):
        v = 0
        for k in range(n):
            byte = s.b[s.p >> 3]; v |= ((byte >> (s.p & 7)) & 1) << k; s.p += 1
        return v
    def align(s): s.p = (s.p + 7) & ~7
MID = 1 << 63; M64 = (1 << 64) - 1
def parse(buf, x_int64):
    r = BR(buf); assert r.read(32) == int.from_bytes(b'pco!', 'little'); r.read(8); r.read(8)
    nh = r.read(6) + 1; r.read(nh); r.align(); r.read(8); r.read(8)   # wrapped header
    out = []; off = 0
    while True:
        dtype = r.read(8)
        if dtype == 0: break
        n = r.read(24) + 1; mode = r.read(4)
        if mode != 0: return None, 'mode%d' % mode
        delta = r.read(4); order = 0
        if delta == 1: order = r.read(3); r.read(1)
        elif delta != 0: return None, 'delta%d' % delta
        size_log = r.read(4); nb = r.read(15); w = []; lo = []; ob = []
        for _ in range(nb): w.append(r.read(size_log) + 1); lo.append(r.read(64)); ob.append(r.read(7))
        assert sum(w) == (1 << size_log) or (size_log == 0 and nb == 1)
        lat = [(int(v) & M64) ^ MID for v in x_int64[off:off + n]]
        for _ in range(order): lat = [(lat[i] - lat[i - 1]) & M64 for i in range(1, len(lat))]
        if order: lat = [v ^ MID for v in lat]
        lo_a = np.array(lo, dtype=object); idx = []
        order_lo = sorted(range(nb), key=lambda i: lo[i]); los = [lo[i] for i in order_lo]
        import bisect
        for v in lat:
            j = order_lo[bisect.bisect_right(los, v) - 1]; assert lo[j] <= v < lo[j] + (1 << ob[j]) or ob[j] == 64, 'bin check'; idx.append(j)
        out.append({'n': n, 'order': order, 'size_log': size_log, 'weights': w, 'bins': np.bincount(idx, minlength=nb).tolist()})
        # skip page: we do not need it; locate next chunk impossible without decoding -> single-chunk files only
        break
    return out, 'ok'
