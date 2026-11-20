import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '../results/'
pts = {}
pha = json.load(open(R + 'pha.json'))
for Llog, lab in [(12, 'PivCo PHA bitmaps, L=4096'), (10, 'PivCo PHA bitmaps, L=1024')]:
    rr = [r for r in pha if r['Llog'] == Llog]
    pts[lab] = ([r['fse'] / r['H'] - 1 for r in rr], [(r['fse'] - min(r['tuned'], r['iter'])) / r['fse'] for r in rr])
lo = json.load(open(R + 'loco.json'))
pts['LOCO-ANS image coder, L=128'] = ([r['loco'] / r['H'] - 1 for r in lo], [(r['loco'] - min(r['tuned'], r['iter'], r['dt'])) / r['loco'] for r in lo])
pc = json.load(open(R + 'pco.json')).values()
pts['Pcodec numeric columns, L=64-4096'] = ([r['pco'] / r['H'] - 1 for r in pc], [(r['pco'] - min(r['tuned'], r['iter'])) / r['pco'] for r in pc])
iot = json.load(open(R + 'iot.json'))
pts['Temperature sensor (LoRa), L=64'] = ([iot['sensor_table']['rate'] / iot['H_all'] - 1], [(iot['sensor_table']['rate'] - iot['same_counts']['dt_tans']['rate']) / iot['sensor_table']['rate']])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
st = {'PivCo PHA bitmaps, L=4096': ('o', '#1f77b4'), 'PivCo PHA bitmaps, L=1024': ('s', '#9ecae1'), 'LOCO-ANS image coder, L=128': ('^', '#2ca02c'),
      'Pcodec numeric columns, L=64-4096': ('D', '#ff7f0e'), 'Temperature sensor (LoRa), L=64': ('*', '#d62728')}
for k, (x, y) in pts.items():
    m, c = st[k]; ax.scatter(np.maximum(100 * np.array(x), 1e-4), 100 * np.array(y), marker=m, color=c, s=90 if m == '*' else 22, label=k, edgecolor='k' if m == '*' else 'none', zorder=3)
xx = np.logspace(-4, 2, 50); ax.plot(xx, xx / (1 + xx / 100), 'k:', lw=0.8, label='all excess removed (upper bound)')
ax.set_xscale('log'); ax.set_xlim(1e-4, 100); ax.set_ylim(-0.3, 10)
ax.set_xlabel('Excess of the deployed table over entropy [%]'); ax.set_ylabel('Rate reduction by a better spread,\nsame counts [%]')
ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]); ax.set_xticklabels(['0.0001', '0.001', '0.01', '0.1', '1', '10', '100'])
ax.set_yticks([0, 2, 4, 6, 8, 10]); ax.grid(alpha=0.3, which='major'); ax.legend(fontsize=7, loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig('../fig/deploy_map.pdf'); fig.savefig('../fig/deploy_map.png', dpi=150)
