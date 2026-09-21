"""ALPHAGRAD_FACE_DIAG=<prefix>: host-side dumps for dsnn-dfw.95.

Three JSON files per episode, written from `jax.debug.callback`. Pure
instrumentation: nothing here is read by the trainer and the whole module is
dead unless the variable is set.
"""

from __future__ import annotations

import json
import os

import numpy as np

PREFIX = os.environ.get("ALPHAGRAD_FACE_DIAG", "")
ON = bool(PREFIX)

_N: dict[str, int] = {}


def _dump(name, payload):
    k = _N.get(name, 0)
    _N[name] = k + 1
    payload["index"] = k
    path = "%s.%s.%04d.json" % (PREFIX, name, k)
    with open(path, "w") as fh:
        json.dump(payload, fh, sort_keys=True, indent=1, default=float)
    print("[facediag] wrote %s" % path, flush=True)


def _stats(x, live=None):
    x = np.asarray(x, np.float64).reshape(-1)
    m = np.isfinite(x)
    if live is not None:
        w = np.asarray(live, np.float64).reshape(-1)
        if w.size == x.size:
            m = m & (w > 0.5)
    n = int(m.sum())
    out = {"n": n}
    if n == 0:
        return out
    y = x[m]
    out.update({"mean": float(y.mean()), "std": float(y.std()),
                "min": float(y.min()), "max": float(y.max()),
                "absmax": float(np.abs(y).max())})
    return out


def ratio_cb(log_ratio, lr_face, lr_rest, ratio, w_live, norm_adv,
             s_has_op=None, s_n_op=None):
    """Check 1: the replay alignment of the first minibatch of epoch 0."""
    lr = np.asarray(log_ratio, np.float64).reshape(-1)
    live = np.asarray(w_live, np.float64).reshape(-1)
    r = np.asarray(ratio, np.float64).reshape(-1)
    lrf = np.asarray(lr_face, np.float64).reshape(-1)
    na = np.asarray(norm_adv, np.float64).reshape(-1)
    k3 = (r - 1.0) - lr
    sel = live > 0.5 if live.size == lr.size else np.ones(lr.size, bool)
    out = {
        "n_samples": int(lr.size),
        "n_live": int(sel.sum()),
        "log_ratio_total": _stats(lr, live),
        "log_ratio_face": _stats(lr_face, live),
        "log_ratio_rest": _stats(lr_rest, live),
        "k3_kl_per_sample": _stats(k3, live),
        "n_live_abs_gt_1e_6": int((sel & (np.abs(lr) > 1e-6)).sum()),
        "n_live_abs_gt_1e_3": int((sel & (np.abs(lr) > 1e-3)).sum()),
        "n_live_abs_gt_1e_1": int((sel & (np.abs(lr) > 1e-1)).sum()),
        "norm_adv": _stats(norm_adv, live),
        "n_live_adv_negative": int((sel & (na < 0.0)).sum()) if na.size
        == lr.size else -1,
        "n_live_inside_clip": int((sel & (np.abs(lr) <= 0.1823)).sum()),
    }
    if s_has_op is not None:
        op = np.asarray(s_has_op, np.float64).reshape(-1) > 0.5
        if op.size == lr.size:
            nop = np.asarray(s_n_op, np.float64).reshape(-1)
            for gname, m in (("sample_has_op", sel & op),
                             ("sample_no_op", sel & ~op)):
                rec = {"n": int(m.sum())}
                if rec["n"]:
                    rec["log_ratio_face"] = _stats(lrf[m])
                    rec["norm_adv"] = _stats(na[m]) if na.size == lr.size \
                        else {"n": 0}
                    rec["n_op_slots"] = _stats(nop[m])
                out[gname] = rec
    _dump("check1", out)


def adv_cb(head_names, nac, norm_adv, op_step, skip_step, live):
    """Check 2: the advantage split by plan class, per channel."""
    nac = np.asarray(nac, np.float64)
    na = np.asarray(norm_adv, np.float64)
    op = np.asarray(op_step, np.float64) > 0.5
    sk = np.asarray(skip_step, np.float64) > 0.5
    lvr = np.asarray(live, np.float64)
    lv = (lvr.reshape(na.shape) > 0.5 if lvr.size == na.size
          else np.ones(na.shape, bool))
    plan_op = op.any(axis=1)
    plan_op_t = np.broadcast_to(plan_op[:, None], op.shape)
    groups = {
        "step_has_op": op & lv,
        "step_no_op": (~op) & lv,
        "plan_has_op": plan_op_t & lv,
        "plan_no_op": (~plan_op_t) & lv,
        "step_has_skip": sk & lv,
    }
    out = {
        "head_names": list(head_names),
        "n_env": int(op.shape[0]),
        "n_step": int(op.shape[1]),
        "n_plan_with_op": int(plan_op.sum()),
        "n_live_steps": int(lv.sum()),
        "frac_live_steps_with_op": (float(op[lv].mean()) if lv.any()
                                    else 0.0),
    }
    for gname, m in groups.items():
        rec = {"n": int(m.sum())}
        if rec["n"]:
            rec["norm_adv"] = {"mean": float(na[m].mean()),
                               "std": float(na[m].std())}
            for c, hn in enumerate(head_names):
                if c < nac.shape[-1]:
                    v = nac[..., c][m]
                    rec[hn] = {"mean": float(v.mean()),
                               "std": float(v.std())}
        out[gname] = rec
    _dump("check2", out)


def grad_cb(term_names, op_names, *packs):
    """Check 3: per loss term, the gradient on the face head's op logit bias.

    A POSITIVE gradient lowers the bias, because the optimizer steps against
    the gradient. So a positive number on OP_NONE is a push AWAY from none.
    """
    out = {"note": "positive gradient = the update lowers that logit bias",
           "op_names": list(op_names)}
    k = 0
    for nm in term_names:
        g_ops = np.asarray(packs[k], np.float64)
        k += 1
        g_skip = float(np.asarray(packs[k], np.float64))
        k += 1
        g_all = np.asarray(packs[k], np.float64)
        k += 1
        rec = {"skip_bias_grad": g_skip,
               "head_bias_grad_l2": float(np.sqrt((g_all ** 2).sum()))}
        for o, on in enumerate(op_names):
            col = g_ops[:, o]
            rec[on] = {"per_slot": [float(v) for v in col],
                       "sum": float(col.sum()),
                       "l2": float(np.sqrt((col ** 2).sum()))}
        out[nm] = rec
    _dump("check3", out)
