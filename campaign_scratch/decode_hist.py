import numpy as np
d = np.load('/Users/assmuth/dsnn/decode_ds_192.npz')
n = d['ntok']; m = d['mode']
print('all', np.percentile(n, [0, 25, 50, 75, 90, 95, 99, 100]).astype(int))
print('rev', np.percentile(n[m == 0], [0, 50, 100]).astype(int))
print('rnd', np.percentile(n[m == 1], [0, 50, 100]).astype(int))
for W in (8192, 16384, 32768):
    b = np.minimum(((n + W - 1) // W) * W, int(n.max()))
    u, c = np.unique(b, return_counts=True)
    print('W', W, dict(zip(u.tolist(), c.tolist())))
