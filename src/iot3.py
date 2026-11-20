import numpy as np, sys
exec(open('iot2.py').read().split("rng = np.random.default_rng(3)")[0])
# variant: first reading sent raw (not ANS-coded); only the 6 differences are coded
Q = posdist(Ptr)[1:]; f = lambda s: packet_cost(s, Q); Pte6 = Pte[:, 1:]
pd = np.bincount(Ptr[:, 1:].ravel(), minlength=8) / Ptr[:, 1:].size; rng = np.random.default_rng(4)
print('p_diff', np.round(pd, 4))
for tag, l in [('sensor_counts', np.bincount(SENSOR, minlength=8)), ('kl_counts', quantize(pd, L))]:
    c = {'sensor_table': SENSOR} if tag == 'sensor_counts' else {}
    c['tuned'] = spread_tuned(l, pd); c['iter_tuned'] = spread_iterated_tuned(l, pd, 8)[0]
    c['dt_packet'] = dt_packet(c['tuned'], f, 50, rng=rng)[0]; c['ls_packet'] = swap_search(c['dt_packet'], f)[0]
    for k, s in c.items():
        b = enc_bits(s, Pte6) / len(Pte6)
        if k == 'sensor_table': base = b
        print(tag, l, k, '%.3f' % b, '%+.2f%%' % (100 * (b - base) / base))
