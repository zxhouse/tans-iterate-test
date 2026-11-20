import numpy as np
def geo_src(theta, max_code_len=7, max_symbols=8, min_symbols=2):
    probs = []
    for i in range(max_symbols):
        q = (1 - theta) * theta ** i; rem = 1 - (sum(probs) + q)
        if -np.log2(rem) > max_code_len and i >= min_symbols - 1: break
        probs.append(q)
    probs = probs[:int(2 ** np.floor(np.log2(len(probs))))]
    return np.array(probs + [1 - sum(probs)])
