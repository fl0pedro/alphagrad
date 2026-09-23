#!/usr/bin/env python3
# Replays recorded TLM plans through the merged count + slot-legality callback
# twice, once with the recording elimination per vertex (old) and once read off
# the chunk stream's one elimination (new), and compares every output for every
# vertex, then the chunks of every face. Run in a fresh interpreter with
# ALPHAGRAD_MAX_FACES=128 and the TLM shape pins; see
# tests/onelim_slot_legality_test.py.
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

LEG_FIELDS = ("sizes", "quant", "pair", "comp", "n_out", "n_faces")
CHUNK_FIELDS = ("tokens", "count", "n_faces", "ends", "ekey", "cvx", "head",
                "wrok")


def build_env():
    import alphagrad.approx.tools.landscape_map as lm
    args = lm.make_argparser().parse_args(
        ["--example", "TransformerLM", "--dataset", "wikitext2",
         "--seed", "250197"])
    env, _eval, _cj = lm.build_env(args)
    return env


def callbacks(jaxpr, argnums, consts, args, slot_one_elim, max_faces,
              max_axes, window):
    from alphagrad.approx.common.face_driver import (
        build_live_face_stream, make_face_slot_legality_callback)
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    s = build_live_face_stream(
        jaxpr, argnums, consts, args, vocab=incr_token_vocab(),
        max_faces=max_faces, max_axes=max_axes, window=window, cache=64)
    s.slot_one_elim = bool(slot_one_elim)
    cb = make_face_slot_legality_callback(
        s, max_faces=max_faces, max_axes=max_axes, with_count=True)
    return s, cb


def walk(s, cb, order, specs, fspecs, fskips, decide=None):
    # Rollout order per step: the legality callback, then every face chunk.
    # `decide(n, nf)` writes step n's decisions before the chunks are read.
    leg, chunks = [], []
    t_leg = t_chunk = 0.0
    for n in range(int(order.shape[0])):
        v = int(order[n])
        t0 = time.perf_counter()
        out = cb(order, specs, np.int32(n), np.int32(v - 1), fspecs, fskips)
        out = tuple(np.array(x, copy=True) for x in out)
        t_leg += time.perf_counter() - t0
        leg.append((n, v, out))
        nf = int(out[5])
        if decide is not None:
            decide(n, nf)
        t0 = time.perf_counter()
        f = 0
        while True:
            r = s.chunk_ex(order, specs, n, v, specs[n], fspecs[n], fskips[n],
                           f, fspecs, fskips)
            chunks.append((n, v, f, tuple(np.array(x, copy=True) for x in r)))
            f += 1
            if f >= int(r[2]):
                break
        t_chunk += time.perf_counter() - t0
    return leg, chunks, t_leg, t_chunk


def first_difference(a, b, fields):
    if len(a) != len(b):
        return f"{len(a)} entries against {len(b)}"
    for ea, eb in zip(a, b):
        if ea[:-1] != eb[:-1]:
            return f"walk order differs at {ea[:-1]} / {eb[:-1]}"
        for name, x, y in zip(fields, ea[-1], eb[-1]):
            if (x.dtype != y.dtype or x.shape != y.shape
                    or not np.array_equal(x, y)):
                return (f"at {ea[:-1]}: {name} differs "
                        f"({x.dtype}{x.shape} vs {y.dtype}{y.shape})")
    return None


def run_plan(env, rec, slot_one_elim):
    import alphagrad.approx.env as envmod
    from alphagrad.approx.common.plan_log import decode_wires
    order, specs, fspecs, fskips = decode_wires(rec)
    cfg = env.config
    s, cb = callbacks(cfg.jaxpr, cfg.argnums, env.consts, env.args,
                      slot_one_elim, envmod.MAX_FACES,
                      envmod.MAX_AXES_PER_VERTEX, envmod.MAX_DELTA_TOKENS)
    leg, chunks, t_leg, t_chunk = walk(s, cb, order, specs, fspecs, fskips)
    return leg, chunks, t_leg, t_chunk, dict(s.stats), s


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
        lo, co, tlo, tco, sto, _s = run_plan(env, rec, False)
        ln, cn, tln, tcn, stn, s = run_plan(env, rec, True)
        dl = first_difference(lo, ln, LEG_FIELDS)
        dc = first_difference(co, cn, CHUNK_FIELDS)
        print(f"[plan {pi}] vertices={len(ln)} chunks={len(cn)} "
              f"old_leg_s={tlo:.2f} new_leg_s={tln:.2f} "
              f"old_chunk_s={tco:.2f} new_chunk_s={tcn:.2f} "
              f"old_elims={sto['elims']}+{sto['slot_probe']} "
              f"new_elims={stn['elims']}+{stn['slot_probe']} "
              f"slot_onelim={stn['slot_onelim']} "
              f"fallback={stn['slot_onelim_fallback']} "
              f"count_onelim={stn['count_onelim']} "
              f"failures={sto['failures']}/{stn['failures']} "
              f"leg_diff={dl} chunk_diff={dc}", flush=True)
        if stn["slot_onelim_fallback"]:
            print(f"[plan {pi}] last fallback: {s.last_slot_onelim_error}",
                  flush=True)
        if dl is not None or dc is not None or not stn["slot_onelim"]:
            bad += 1
    if bad:
        print(f"REPLAY-FAIL {bad} of {len(recs)} plans", flush=True)
        return 1
    print(f"REPLAY-OK {len(recs)} plans identical", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
