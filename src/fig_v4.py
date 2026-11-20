import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '../results/'; P = '../fig/'
plt.rcParams.update({'font.size': 9})
# ---- Fig: sensor
iot = json.load(open(R + 'iot2.json')); base = iot['sensor_counts']['sensor_table']['bits_test']
meth = [('fast', 'FSE step'), ('tuned', 'tuned'), ('iter_tuned', 'iterated\ntuned'), ('dt_stationary', 'DT-tANS\n(stat.)'), ('dt_packet', 'DT-tANS\n(packet)'), ('ls_packet', 'local\nsearch\n(packet)')]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
x = np.arange(len(meth)); w = 0.38
for k, (tag, lab, col) in enumerate([('sensor_counts', 'deployed counts (1,1,6,47,6,1,1,1)', '#9ecae1'), ('kl_counts', 'KL counts (1,1,6,39,6,1,1,9)', '#08519c')]):
    v = [100 * (iot[tag][m]['bits_test'] - base) / base for m, _ in meth]
    ax[0].bar(x + (k - 0.5) * w, v, w, color=col, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in meth], fontsize=7); ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylabel('Coded bits vs deployed table [%]'); ax[0].set_ylim(-25, 2); ax[0].set_yticks([-25, -20, -15, -10, -5, 0])
ax[0].set_title('(a) 7-reading packets (as deployed)', fontsize=9)
m6 = ['tuned', 'iterated\ntuned', 'DT-tANS\n(packet)', 'local search\n(packet)']
dep6 = [-1.44, -1.34, -1.98, -2.13]; kl6 = [2.83, 3.05, -2.61, -2.72]; x6 = np.arange(4)
ax[1].bar(x6 - w / 2, dep6, w, color='#9ecae1'); ax[1].bar(x6 + w / 2, kl6, w, color='#08519c')
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xticks(x6); ax[1].set_xticklabels(m6, fontsize=7); ax[1].set_ylim(-4, 4); ax[1].set_yticks([-4, -2, 0, 2, 4])
ax[1].set_ylabel('Coded bits vs deployed table [%]'); ax[1].set_title('(b) first reading sent raw, 6 differences coded', fontsize=9)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, ['deployed counts', 'KL-optimal counts'], loc='lower center', ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(P + 'fig_sensor.pdf')
# ---- Fig: PHA
pha = json.load(open(R + 'pha.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for a, Llog in zip(ax, (12, 10)):
    rr = [r for r in pha if r['Llog'] == Llog]; q = [r['p0'] for r in rr]
    ex = lambda k: [100 * (r[k] / r['H'] - 1) for r in rr]
    a.plot(q, ex('fse'), 'o-', ms=2.5, color='#d62728', label='deployed (FSE step spread)')
    a.plot(q, [100 * (min(r['tuned'], r['iter']) / r['H'] - 1) for r in rr], 's-', ms=2.5, color='#1f77b4', label='better spread, same counts')
    a.plot(q, ex('kl_tuned'), '^-', ms=2.5, color='#2ca02c', label='better spread + KL counts')
    a.axvspan(0.625, 1.0, color='0.92', zorder=0); a.text(0.635, 2 if Llog == 12 else 30, 'FSE path used', fontsize=7)
    a.set_yscale('symlog', linthresh=1); a.set_ylim(0, 100); a.set_yticks([0, 1, 10, 100]); a.set_yticklabels(['0', '1', '10', '100'])
    a.set_xlim(0.5, 1.0); a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); a.set_xlabel('Bit skew $p_0$ [-]'); a.set_title('(%s) L = %d states' % ('a' if Llog == 12 else 'b', 1 << Llog), fontsize=9)
ax[0].set_ylabel('Excess over entropy [%]'); ax[0].legend(fontsize=7, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(P + 'fig_pha.pdf')
# ---- Fig: reframing
d = json.load(open(R + 'reframe_synth2.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
kinds = [('uniform', 'uniform'), ('zipf', 'Zipf'), ('dirichlet0.3', 'Dirichlet(0.3)')]
xs = np.arange(3); w = 0.26
for k, (key, lab, col) in enumerate([('kl100', '+50 evaluations, same frame', '#bbbbbb'), ('joint_ls', 'joint local search (~1000+ eval.)', '#ff7f0e'), ('rf50', 'reframe once (50 eval. in total)', '#08519c')]):
    v = []
    for m in [2, 4, 8]:
        S = [r for r in d if r['kind'] == 'dirichlet0.3' and r['L'] == m * r['n'] and key in r]
        v.append(np.mean([(r['kl50'] - r[key]) / r['kl50'] * 100 for r in S]))
    ax[0].bar(xs + (k - 1) * w, v, w, color=col, label=lab)
ax[0].set_xticks(xs); ax[0].set_xticklabels(['2', '4', '8']); ax[0].set_xlabel('States per symbol $L/n$ [-]'); ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylabel('Mean rate gain vs DT-tANS(50) [%]'); ax[0].set_ylim(-0.1, 0.7); ax[0].set_yticks([0, 0.2, 0.4, 0.6]); ax[0].set_title('(a) Dirichlet(0.3) sources', fontsize=9); ax[0].legend(fontsize=6.5, frameon=False, loc='upper right')
S = [r for r in d if r['kind'] == 'dirichlet0.3']
g = np.sort([(r['kl50'] - r['rf50']) / r['kl50'] * 100 for r in S])[::-1]; g2 = np.sort([(r['kl50'] - r['kl100']) / r['kl50'] * 100 for r in S])[::-1]
ax[1].plot(np.arange(1, len(g) + 1), g, color='#08519c', label='reframe once'); ax[1].plot(np.arange(1, len(g2) + 1), g2, color='#888888', label='+50 evaluations, same frame')
ax[1].set_xscale('log'); ax[1].set_yscale('symlog', linthresh=0.01); ax[1].set_ylim(-1, 20); ax[1].set_yticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10]); ax[1].set_yticklabels(['-1', '-0.1', '-0.01', '0', '0.01', '0.1', '1', '10'])
ax[1].set_xticks([1, 10, 100]); ax[1].set_xticklabels(['1', '10', '100']); ax[1].set_xlabel('Rank of the distribution [-]'); ax[1].set_ylabel('Rate gain vs DT-tANS(50) [%]')
ax[1].set_title('(b) all 360 Dirichlet(0.3) distributions, sorted', fontsize=9); ax[1].legend(fontsize=7, frameon=False, loc='lower left'); ax[1].axhline(0, color='k', lw=0.6)
fig.tight_layout(); fig.savefig(P + 'fig_reframe.pdf')
for f in ['fig_sensor', 'fig_pha', 'fig_reframe']:
    pass
