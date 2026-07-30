#!/usr/bin/env python3
"""ALPHAGRAD_AZ_NO_SEED=1 -> AZ cold start (empty buffer, no 800-sample offline
cost-head seed), for a fair comparison vs cold-start PPO. Default OFF = byte-identical.
5 empty-buffer guards."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "_AZ_NO_SEED" in s:
    print("ALREADY PATCHED"); sys.exit(0)

o1 = ('d = np.load(A.seed_dataset, allow_pickle=True)\n'
      'bufX = list(np.asarray(d["X"], np.float64))            # pooled encoder ctx\n'
      'bufY = list(np.asarray(d["Y"], np.float64))            # measured 4-tuple')
n1 = ('_AZ_NO_SEED = os.environ.get("ALPHAGRAD_AZ_NO_SEED", "0") == "1"\n'
      'if _AZ_NO_SEED:\n'
      '    bufX, bufY = [], []            # COLD START (fair vs cold PPO): no offline cost-head seed\n'
      '    print("[loop] AZ_NO_SEED=1: COLD START (empty buffer, no offline seed)", flush=True)\n'
      'else:\n'
      '    d = np.load(A.seed_dataset, allow_pickle=True)\n'
      '    bufX = list(np.asarray(d["X"], np.float64))        # pooled encoder ctx\n'
      '    bufY = list(np.asarray(d["Y"], np.float64))        # measured 4-tuple')
assert o1 in s, "anchor 1 (seed load)"; s = s.replace(o1, n1, 1)

o2 = 'HOX = np.array([bufX[i] for i in _ho]); HOS = np.array([bufS[i] for i in _ho])'
n2 = ('HOX = np.array([bufX[i] for i in _ho]) if len(_ho) else np.zeros((0, EMBD))\n'
      'HOS = np.array([bufS[i] for i in _ho]) if len(_ho) else np.zeros((0,))')
assert o2 in s, "anchor 2 (held-out)"; s = s.replace(o2, n2, 1)

o3 = ('stds = None\n'
      '# warm-start the head with a quick ranking fit on the seed buffer\n'
      'cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)')
n3 = ('stds = (np.zeros(EMBD), np.ones(EMBD), np.zeros(4), np.ones(4))  # default (cold) standardization\n'
      '# warm-start the head with a quick ranking fit on the seed buffer (skip if cold/empty)\n'
      'if len(bufX) >= 2:\n'
      '    cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)')
assert o3 in s, "anchor 3 (pre-round train)"; s = s.replace(o3, n3, 1)

o4 = ('    _tr0 = time.time()\n'
      '    cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)\n'
      '    _t_rank = time.time() - _tr0')
n4 = ('    _tr0 = time.time()\n'
      '    if len(bufX) >= 2:\n'
      '        cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)\n'
      '    _t_rank = time.time() - _tr0')
assert o4 in s, "anchor 4 (per-round train)"; s = s.replace(o4, n4, 1)

o5 = ('    ho_pred = _pred_scalar_np(cost_head, HOX, *stds)\n'
      '    sp_ho = float(spearmanr(ho_pred, HOS).correlation)')
n5 = ('    if len(HOX) > 2:\n'
      '        ho_pred = _pred_scalar_np(cost_head, HOX, *stds)\n'
      '        sp_ho = float(spearmanr(ho_pred, HOS).correlation)\n'
      '    else:\n'
      '        sp_ho = float("nan")')
assert o5 in s, "anchor 5 (sp_ho)"; s = s.replace(o5, n5, 1)

p.write_text(s)
print("5 cold-start guards applied OK")
