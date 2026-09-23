#!/usr/bin/env python3
# Replays recorded TLM plans through LiveFaceStream.chunk_ex twice, once per
# face elimination (the per-face path) and once per vertex, and compares every
# output of every face of every vertex. Run in a fresh interpreter with
# ALPHAGRAD_MAX_FACES=128 and the TLM shape pins; see
# tests/onelim_face_chunk_test.py.
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

FIELDS = ("tokens", "count", "n_faces", "ends", "ekey", "cvx", "head", "wrok")


def build_env():
    import alphagrad.approx.tools.landscape_map as lm
    args = lm.make_argparser().parse_args(
        ["--example", "TransformerLM", "--dataset", "wikitext2",
         "--seed", "250197"])
    env, _eval, _cj = lm.build_env(args)
    return env


def stream(env, one_elim):
    import alphagrad.approx.env as envmod
    from alphagrad.approx.common.face_driver import build_live_face_stream
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    cfg = env.config
    s = build_live_face_stream(
        cfg.jaxpr, cfg.argnums, env.consts, env.args,
        vocab=incr_token_vocab(), max_faces=envmod.MAX_FACES,
        max_axes=envmod.MAX_AXES_PER_VERTEX, window=envmod.MAX_DELTA_TOKENS,
        cache=64)
    s.one_elim = bool(one_elim)
    return s


def walk(env, rec, one_elim):
    from alphagrad.approx.common.plan_log import decode_wires
    order, specs, fspecs, fskips = decode_wires(rec)
    s = stream(env, one_elim)
    out = []
    t0 = time.perf_counter()
    for n in range(int(order.shape[0])):
        v = int(order[n])
        f = 0
        while True:
            r = s.chunk_ex(order, specs, n, v, specs[n], fspecs[n], fskips[n],
                           f, fspecs, fskips)
            out.append((n, v, f, tuple(np.array(x, copy=True) for x in r)))
            f += 1
            if f >= int(r[2]):
                break
    secs = time.perf_counter() - t0
    return out, secs, dict(s.stats), s.last_onelim_error


def first_difference(a, b):
    if len(a) != len(b):
        return f"{len(a)} chunks against {len(b)}"
    for (n, v, f, ra), (n2, v2, f2, rb) in zip(a, b):
        if (n, v, f) != (n2, v2, f2):
            return f"walk order differs at {(n, v, f)} / {(n2, v2, f2)}"
        for name, x, y in zip(FIELDS, ra, rb):
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                return (f"step {n} vertex {v} face {f}: {name} differs "
                        f"({x.dtype}{x.shape} vs {y.dtype}{y.shape})")
    return None


def main(plans_path):
    import alphagrad.approx.env as envmod
    if envmod.MAX_FACES != 128:
        raise RuntimeError(
            f"MAX_FACES is {envmod.MAX_FACES}; the plans were recorded at 128. "
            f"Set ALPHAGRAD_MAX_FACES=128 before env.py is imported.")
    recs = [json.loads(line) for line in open(plans_path) if line.strip()]
    env = build_env()
    bad = 0
    for pi, rec in enumerate(recs):
        old, t_old, st_old, _e = walk(env, rec, False)
        new, t_new, st_new, err = walk(env, rec, True)
        diff = first_difference(old, new)
        decided = sum(1 for _n, _v, f, r in new if f > 0)
        print(f"[plan {pi}] chunks={len(new)} faces_after_first={decided} "
              f"old_s={t_old:.2f} new_s={t_new:.2f} "
              f"ratio={t_old / max(t_new, 1e-9):.2f} "
              f"old_elims={st_old['elims']} new_elims={st_new['elims']} "
              f"face_elims={st_new['face_elims']} "
              f"fallback={st_new['onelim_fallback']} "
              f"failures={st_old['failures']}/{st_new['failures']} "
              f"diff={diff}", flush=True)
        if diff is not None or st_new["onelim_fallback"]:
            bad += 1
            print(f"[plan {pi}] last one-elimination error: {err}", flush=True)
    if bad:
        print(f"PROBE-FAIL {bad} of {len(recs)} plans", flush=True)
        return 1
    print(f"PROBE-OK {len(recs)} plans identical", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
