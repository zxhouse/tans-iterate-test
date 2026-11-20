import numpy as np, json, sys, math
sys.path.insert(0, '.')
from tanslib import *
from pco_parse import parse
from pcodec import standalone, ChunkConfig, ModeSpec
from vega_datasets import local_data as V
cols = {}
sw = V.seattle_weather(); cols['seattle_weather.temp_max(0.1C)'] = (sw.temp_max * 10).round(); cols['seattle_weather.precip(0.1mm)'] = (sw.precipitation * 10).round(); cols['seattle_weather.wind(0.1)'] = (sw.wind * 10).round()
cols['seattle_temps.hourly(0.1F)'] = (V.seattle_temps().temp * 10).round(); cols['sf_temps.hourly(0.1F)'] = (V.sf_temps().temp * 10).round()
ap = V.airports(); cols['airports.latitude(1e-6)'] = (ap.latitude * 1e6).round(); cols['airports.longitude(1e-6)'] = (ap.longitude * 1e6).round()
st = V.stocks(); cols['stocks.price(0.01USD)'] = (st.price * 100).round()
ie = V.iowa_electricity(); cols['iowa_electricity.net_generation'] = ie.net_generation
ue = V.us_employment(); cols['us_employment.nonfarm(k)'] = ue.nonfarm.round()
oh = V.ohlc(); cols['ohlc.close(0.01)'] = (oh.close * 100).round()
# sensor deployment readings (all sensors, centi-degC), from Markiewicz et al. data
t = []
for ln in open('../external/ans-iot/data/temperatures.txt'):
    f = ln.split()
    if len(f) >= 4 and f[0].isdigit(): t.append((int(f[0]), f[2] + f[3], round(float(f[1]) * 100)))
t.sort(); cols['ans_iot.temperature(0.01C)'] = np.array([v for _, _, v in t])
res = {}; rng = np.random.default_rng(5)
for name, x in cols.items():
    x = np.asarray(x, np.int64)[:2 ** 20]
    for lvl in [8, 12]:
        buf = standalone.simple_compress(x, ChunkConfig(compression_level=lvl, mode_spec=ModeSpec.classic()))
        info, st_ = parse(buf, x)
        if info is None: print(name, lvl, st_); continue
        c = info[0]; l = np.array(c['weights']); cnt = np.array(c['bins'], float); p = cnt / cnt.sum(); n = len(l)
        if n < 2: continue
        L = 1 << c['size_log']; stride = (3 * L) // 5; stride += (stride % 2 == 0)
        pco = np.zeros(L, np.int32); step = 0
        for s, wgt in enumerate(l):
            for _ in range(wgt): pco[(stride * step) & (L - 1)] = s; step += 1
        lb = huffman_rate(p)
        r = {'n': c['n'], 'L': L, 'bins': n, 'order': c['order'], 'H': entropy(p), 'bytes': len(buf),
             'pco': exact_rate(pco, p), 'tuned': exact_rate(spread_tuned(l, p), p)}
        r['iter'] = spread_iterated_tuned(l, p, 8)[1]
        r['dt'] = exact_rate(dt_tans(l, p, 50, rng=rng)[0], p) if L <= 1024 else float('nan')
        # bits saved in the chunk: n * (pco - best)
        best = min(r['tuned'], r['iter'], r['dt'] if r['dt'] == r['dt'] else 9e9)
        r['saved_pct_of_file'] = 100 * c['n'] * (r['pco'] - best) / 8 / len(buf)
        res['%s@%d' % (name, lvl)] = r
        print('%-34s lvl%2d n=%7d L=%5d bins=%3d H=%.3f pco=%.4f tuned=%.4f iter=%.4f dt=%.4f  saved %.3f%% of file' % (name, lvl, c['n'], L, n, r['H'], r['pco'], r['tuned'], r['iter'], r['dt'], r['saved_pct_of_file']))
json.dump(res, open('../results/pco.json', 'w'), indent=1)
