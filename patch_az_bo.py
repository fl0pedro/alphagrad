#!/usr/bin/env python3
"""Add `--optimizer bo`: a self-contained surrogate-based Bayesian optimizer over
the (order, per-vertex QUANT) genome. kNN surrogate + UCB acquisition (exploit
predicted objective + explore under-sampled regions). Reuses _feat/_mf_measure
from the v2 optimizer branch. No policy, no palimpsa encoder -> honest 'surrogate-
only' arm, approx-capable (order_ctx can't encode explicit quant; sklearn absent)."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if 'elif _opt == "bo":' in s:
    print("ALREADY PATCHED"); sys.exit(0)

# 1) add "bo" to the --optimizer choices
old_ch = 'ap.add_argument("--optimizer", default="none", choices=["none", "random", "sa", "ga"],'
new_ch = 'ap.add_argument("--optimizer", default="none", choices=["none", "random", "sa", "ga", "bo"],'
assert old_ch in s, "optimizer choices anchor"; s = s.replace(old_ch, new_ch, 1)

# 2) a per-vertex feature helper (order-position + quant-index), inserted right
#    after _mf_rand_cand is defined
anchor_feat = "    def _mf_seq(cand):"
feat = ('''    def _mf_feat(cand):
        o, q = cand
        pos = np.zeros(NV, dtype=np.float64)
        for rank, a in enumerate(o):
            pos[int(a)] = rank / max(NV - 1, 1)
        qi = np.zeros(NV, dtype=np.float64)
        if q is not None:
            for k, a in enumerate(o):
                qi[int(a)] = (_QD.index(q[k]) if q[k] in _QD else 0) / max(len(_QD) - 1, 1)
        return np.concatenate([pos, qi])
''')
assert anchor_feat in s, "feat anchor"; s = s.replace(anchor_feat, feat + anchor_feat, 1)

# 3) the BO loop, inserted before the DONE json dump
anchor_bo = "    import json as _json_mf"
bo = r'''    if _opt == "bo":
        _bo_X, _bo_R = [], []   # features, raw 4-tuples (rescored under current PopArt)
        def _bo_add(cands, raws):
            for c, r in zip(cands, raws):
                if r is not None:
                    _bo_X.append(_mf_feat(c)); _bo_R.append(r)
        _bo_add_init = [_mf_rand_cand(_orng) for _ in range(max(A.topk, 8))]
        _bo_add(_bo_add_init, _mf_measure(_bo_add_init))
        _beta = float(_os_mf.environ.get("ALPHAGRAD_BO_BETA", "0.3"))
        _kk = 5
        while _n[0] < _budget and _bo_X:
            Xa = np.asarray(_bo_X); ya = np.asarray([scalarize(r) for r in _bo_R])
            cands = [_mf_rand_cand(_orng) for _ in range(max(A.pool, A.topk))]
            scores = []
            for c in cands:
                d = np.linalg.norm(Xa - _mf_feat(c), axis=1)
                idx = np.argsort(d)[:min(_kk, len(d))]
                w = 1.0 / (d[idx] + 1e-6)
                mu = float(np.sum(w * ya[idx]) / np.sum(w))
                scores.append(mu + _beta * float(d[idx].min()))  # UCB
            top = [cands[i] for i in np.argsort(scores)[::-1][:A.topk]]
            _bo_add(top, _mf_measure(top))
            _mf_log("bo")

'''
assert anchor_bo in s, "bo insert anchor"
s = s.replace(anchor_bo, bo + anchor_bo, 1)
p.write_text(s)
print("added --optimizer bo (kNN-UCB surrogate over order+quant)")
