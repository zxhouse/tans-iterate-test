# Minimal Zstandard (RFC 8878) frame parser that exactly decodes every FSE-coded sequence stream.
# Self-checks: every sequence bitstream must be consumed to the last bit, and literal + match lengths
# summed over the frame must equal the frame content size.
import numpy as np

LL_BITS = [0]*16 + [1,1,1,1,2,2,3,3,4,6,7,8,9,10,11,12,13,14,15,16]
LL_BASE = list(range(16)) + [16,18,20,22,24,28,32,40,48,64,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000]
ML_BITS = [0]*32 + [1,1,1,1,2,2,3,3,4,4,5,7,8,9,10,11,12,13,14,15,16]
ML_BASE = list(range(3, 35)) + [35,37,39,41,43,47,51,59,67,83,99,0x83,0x103,0x203,0x403,0x803,0x1003,0x2003,0x4003,0x8003,0x10003]
LL_DEF = ([4,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,3,2,1,1,1,1,1,-1,-1,-1,-1], 6)
ML_DEF = ([1,4,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1], 6)
OF_DEF = ([1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1], 5)
MAXAL = {"LL": 9, "OF": 8, "ML": 9}

def read_ncount(buf, pos, maxsym):
    """FSE_readNCount; returns (norm list, accuracy log, bytes consumed)"""
    v = int.from_bytes(buf[pos:pos + 600], "little"); bit = 0
    def peek(n): return (v >> bit) & ((1 << n) - 1)
    al = peek(4) + 5; bit += 4; remaining = (1 << al) + 1; threshold = 1 << al; nb = al + 1
    norm = []; prev0 = False
    while remaining > 1 and len(norm) <= maxsym:
        if prev0:
            n0 = len(norm)
            while peek(16) == 0xFFFF: n0 += 24; bit += 16
            while peek(2) == 3: n0 += 3; bit += 2
            n0 += peek(2); bit += 2
            norm += [0] * (n0 - len(norm))
            if len(norm) > maxsym: break
        mx = (2 * threshold - 1) - remaining
        if peek(nb - 1) < mx: c = peek(nb - 1); bit += nb - 1
        else:
            c = peek(nb)
            if c >= threshold: c -= mx
            bit += nb
        c -= 1; remaining -= abs(c); norm.append(c); prev0 = (c == 0)
        while remaining < threshold: nb -= 1; threshold >>= 1
    assert remaining == 1, "bad ncount"
    return norm, al, (bit + 7) // 8

def zstd_spread(norm, al):
    """the decoder spread of FSE (zstd): -1 symbols at the top, others with step 5/8 L + 3"""
    L = 1 << al; sym = [None] * L; hi = L - 1
    for s, c in enumerate(norm):
        if c == -1: sym[hi] = s; hi -= 1
    pos = 0; step = (L >> 1) + (L >> 3) + 3
    for s, c in enumerate(norm):
        for _ in range(max(c, 0)):
            sym[pos] = s; pos = (pos + step) & (L - 1)
            while pos > hi: pos = (pos + step) & (L - 1)
    return np.array(sym, np.int32)

def dtable(sym, norm, al):
    L = 1 << al; nxt = [c if c > 0 else 1 for c in norm]; t = []
    for u in range(L):
        s = int(sym[u]); ns = nxt[s]; nxt[s] += 1; nb = al - (ns.bit_length() - 1)
        t.append((s, nb, (ns << nb) - L))
    return t

class RevBits:
    def __init__(self, b):
        self.v = int.from_bytes(b, "little"); last = b[-1]; assert last != 0
        self.pos = 8 * len(b) - (8 - last.bit_length()) - 1
    def read(self, n):
        if n == 0: return 0
        self.pos -= n; assert self.pos >= 0, "overread"
        return (self.v >> self.pos) & ((1 << n) - 1)

def parse_frame(buf):
    """yields dicts per FSE-compressed (or any) sequence stream; returns totals"""
    assert int.from_bytes(buf[:4], "little") == 0xFD2FB528; p = 4; fhd = buf[p]; p += 1
    fcs_flag = fhd >> 6; single = (fhd >> 5) & 1; csum = (fhd >> 2) & 1; did = fhd & 3
    if not single: p += 1
    p += [0, 1, 2, 4][did]; fcs_size = [1 if single else 0, 2, 4, 8][fcs_flag]
    fcs = int.from_bytes(buf[p:p + fcs_size], "little") + (256 if fcs_size == 2 else 0); p += fcs_size
    prev = {"LL": None, "OF": None, "ML": None}; streams = []; total_len = 0; seq_bytes = 0; nblocks = 0
    while True:
        h = int.from_bytes(buf[p:p + 3], "little"); p += 3; last = h & 1; btype = (h >> 1) & 3; bsize = h >> 3; nblocks += 1
        if btype == 0: total_len += bsize; p += bsize
        elif btype == 1: total_len += bsize; p += 1
        elif btype == 2:
            end = p + bsize; q = p; b0 = buf[q]; lt = b0 & 3; sf = (b0 >> 2) & 3
            if lt in (0, 1):
                if sf in (0, 2): regen = b0 >> 3; q += 1
                elif sf == 1: regen = (b0 >> 4) + (buf[q + 1] << 4); q += 2
                else: regen = (b0 >> 4) + (buf[q + 1] << 4) + (buf[q + 2] << 12); q += 3
                q += regen if lt == 0 else 1
            else:
                hl, bits = {0: (3, 10), 1: (3, 10), 2: (4, 14), 3: (5, 18)}[sf]
                hv = int.from_bytes(buf[q:q + hl], "little"); regen = (hv >> 4) & ((1 << bits) - 1); comp = (hv >> (4 + bits)) & ((1 << bits) - 1)
                q += hl + comp
            b = buf[q]
            if b == 0: nseq = 0; q += 1
            elif b < 128: nseq = b; q += 1
            elif b < 255: nseq = ((b - 128) << 8) + buf[q + 1]; q += 2
            else: nseq = buf[q + 1] + (buf[q + 2] << 8) + 0x7F00; q += 3
            if nseq == 0: total_len += regen; p = end
            else:
                seq_start = q; modes = buf[q]; q += 1; tabs = {}
                for name, sh, dflt, maxsym in (("LL", 6, LL_DEF, 35), ("OF", 4, OF_DEF, 31), ("ML", 2, ML_DEF, 52)):
                    m = (modes >> sh) & 3
                    if m == 0: norm, al = dflt; tabs[name] = ("pre", norm, al)
                    elif m == 1: tabs[name] = ("rle", buf[q], 0); q += 1
                    elif m == 2: norm, al, used = read_ncount(buf, q, maxsym); q += used; tabs[name] = ("fse", norm, al)
                    else: tabs[name] = prev[name] if prev[name][0] != "fse" else ("rep",) + prev[name][1:]
                    prev[name] = tabs[name] if tabs[name][0] != "rep" else ("fse",) + tabs[name][1:]
                br = RevBits(buf[q:end]); dec = {}; st = {}
                for name in ("LL", "OF", "ML"):
                    t = tabs[name]
                    if t[0] == "rle": dec[name] = None
                    else:
                        sp = zstd_spread(t[1], t[2]); dec[name] = dtable(sp, t[1], t[2]); st[name] = br.read(t[2])
                syms = {"LL": [], "OF": [], "ML": []}; sbits = {"LL": 0, "OF": 0, "ML": 0}; xbits = 0; lens = 0; mlsum = 0; llsum = 0
                for i in range(nseq):
                    cur = {}
                    for name in ("LL", "OF", "ML"):
                        cur[name] = tabs[name][1] if dec[name] is None else dec[name][st[name]][0]
                        syms[name].append(cur[name])
                    of = cur["OF"]; br.read(of); xbits += of
                    ml = ML_BASE[cur["ML"]] + br.read(ML_BITS[cur["ML"]]); xbits += ML_BITS[cur["ML"]]
                    ll = LL_BASE[cur["LL"]] + br.read(LL_BITS[cur["LL"]]); xbits += LL_BITS[cur["LL"]]
                    lens += ll + ml; mlsum += ml; llsum += ll
                    if i < nseq - 1:
                        for name in ("LL", "ML", "OF"):
                            if dec[name] is None: continue
                            s, nb, base = dec[name][st[name]]; st[name] = base + br.read(nb); sbits[name] += nb
                assert br.pos == 0, f"stream not consumed: {br.pos} bits left"
                assert llsum <= regen, "literal lengths exceed literals"
                total_len += regen + mlsum
                for name in ("LL", "OF", "ML"):
                    t = tabs[name]
                    streams.append(dict(stream=name, mode=t[0], norm=list(t[1]) if t[0] != "rle" else None, al=t[2],
                                        symbols=np.array(syms[name], np.int32), state_bits=sbits[name], nseq=nseq, xbits=xbits))
                seq_bytes += end - seq_start
                p = end
        else: raise ValueError("reserved block")
        if last: break
    if csum: p += 4
    assert total_len == fcs, f"content size mismatch {total_len} != {fcs}"
    return streams, fcs, p, nblocks, seq_bytes
