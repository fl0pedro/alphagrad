#!/usr/bin/env python3
# dsnn-dfw.189: whole episodes with the shared prefix graph off, then on.
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

COMMON = [
    "--seed", "250197", "--measure-latency", "--latency-inner-reps", "50",
    "--num-data-points", "5", "--reps-per-point", "4",
    "--ref-num-data-points", "5", "--ref-reps-per-point", "32",
    "--measure-budget-secs", "1.0", "--measure-window-secs", "0.05",
    "--incremental-encode", "--cmp-type", "latency",
    "--mem-type", "peak_memory", "--terminal-rewards-only",
    "--rewards", "cmp", "mem", "acc", "--reward-mode", "lagrangian",
    "--quality-metric", "grad_cosine", "--approx-add", "lossless",
    "--fixed-order", "free", "--cost-form", "paired-log",
    "--mem-channel", "watermark", "--reduce-axis-space", "physical",
    "--face-read", "last-row", "--set-pointer", "--face-actions",
    "--per-face-masks", "--unified-face-head", "--live-faces",
    "--dynamic-substeps", "--hidden-dim", "256", "--vocab-size", "256",
    "--num-layers", "3", "--tokenize-where", "local",
    "--approx-profile", "all", "--wandb", "disabled",
]
TARGETS = {
    "tlm": ["--example", "TransformerLM", "--dataset", "wikitext2",
            "--face-wire-faces", "128"],
    "nn256": ["--example", "NeuralNetwork", "--dataset", "mnist",
              "--face-wire-faces", "64"],
    "rtrl": ["--example", "RSNN_SHD", "--dataset", "shd",
             "--temporal-rule", "rtrl", "--face-wire-faces", "64"],
}


def build(target):
    from alphagrad.approx import ppo
    from alphagrad.approx.cpu_approx_worker import _build_env_from_args
    import alphagrad.approx.env as E
    from alphagrad.approx.common.masks import (
        set_diag_per_face, set_per_face_masks, set_reduce_axis_space)
    ns, unknown = ppo.make_argparser().parse_known_args(
        TARGETS[target] + COMMON)
    assert unknown == [], unknown
    d = dict(vars(ns))
    d["exec_on_gpu"] = False
    os.environ["ALPHAGRAD_APPROX_ADD"] = str(d["approx_add"])
    os.environ["ALPHAGRAD_VOCAB_SIZE"] = str(int(d["vocab_size"]))
    set_diag_per_face(False)
    set_per_face_masks(True)
    set_reduce_axis_space("physical")
    env = _build_env_from_args(d, None, seed=int(d["seed"]))
    E.configure_face_wire_faces(int(d["face_wire_faces"]))
    return env


def plan_orders(env, kind, n_envs, seed):
    if kind == "markowitz":
        from alphagrad.approx.common.order import markowitz_order
        o = markowitz_order(env.config.jaxpr, env.config.argnums,
                            list(env.consts), list(env.args),
                            env.valid_vertices)
        o = np.asarray([int(x) for x in o], np.int32)
        return [o.copy() for _ in range(n_envs)]
    vv = [int(v) for v in env.valid_vertices]
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.permutation(vv), np.int32) for _ in range(n_envs)]


def _slot_row(pair, comp, nout, rng):
    from alphagrad.approx.env import COMPRESS_SENTINEL
    ii, jj = np.nonzero(np.asarray(pair) > 0)
    keep = [(int(i), int(j)) for i, j in zip(ii, jj)
            if int(i) < int(nout) <= int(j)]
    aa = [int(a) for a in np.nonzero(np.asarray(comp) > 0)[0]]
    kinds = (["diag"] if keep else []) + (["comp"] if aa else [])
    if not kinds or rng.random() < 0.55:
        return None
    k = kinds[int(rng.integers(len(kinds)))]
    if k == "diag":
        i, j = keep[int(rng.integers(len(keep)))]
        return (i, j - int(nout), -1)
    return (COMPRESS_SENTINEL, aa[int(rng.integers(len(aa)))], 0)


def draw_stage1(leg, nf, W, S, rng):
    from alphagrad.approx.env import QUANT_SENTINEL
    from alphagrad.approx.common.masks import FACE_QUANT_DTYPES
    from alphagrad.approx.common.plan_log import quant_dtype_id
    quant, pair, comp, nout = leg[1], leg[2], leg[3], leg[4]
    rows = -np.ones((W, S, 3), np.int32)
    rows[..., 2] = 0
    skips = np.zeros((W,), np.int32)
    for f in range(int(nf)):
        u = rng.random()
        if u < 0.15:
            skips[f] = 1
            continue
        if u < 0.35:
            both = np.nonzero((quant[f, 0] > 0) & (quant[f, 1] > 0))[0]
            if len(both):
                q = quant_dtype_id(
                    FACE_QUANT_DTYPES[int(both[int(rng.integers(len(both)))])])
                rows[f, 0] = (QUANT_SENTINEL, q, 0)
                rows[f, 1] = (QUANT_SENTINEL, q, 0)
                continue
        for s in (0, 1):
            r = _slot_row(pair[f, s], comp[f, s], nout[f, s], rng)
            if r is not None:
                rows[f, s] = r
    return rows, skips


def draw_stage2(dec, rows1, skips, nf, rng):
    rows = np.array(rows1, copy=True)
    for f in range(int(nf)):
        if not skips[f]:
            r = _slot_row(dec.pair[f, 2], dec.comp[f, 2], dec.nout[f, 2], rng)
            if r is not None:
                rows[f, 2] = r
    return rows


def _snap(x):
    if isinstance(x, tuple):
        return tuple(_snap(y) for y in x)
    return np.array(x, copy=True)


def _entry_sig(val):
    from graphax import SKIP_FACE
    if val is SKIP_FACE:
        return "SKIP"
    if isinstance(val, tuple):
        return tuple(_entry_sig(x) for x in val)
    return None if val is None else type(val).__name__


def _ft_sig(ft):
    if ft is None:
        return None
    return {int(v): {str(k): _entry_sig(val) for k, val in per.items()}
            for v, per in ft.items()}


def _reset():
    import alphagrad.approx.env as E
    for d in (E._INCR_STREAM_CACHE, E._STREAM_STEPS, E._PREFIX_GRAPHS):
        d.clear()
    E._LIVE_CHAINS.clear()
    for d in (E._INCR_STREAM_STATS, E._PREFIX_GRAPH_STATS,
              E._STREAM_STEPS_STATS):
        for k in d:
            d[k] = 0
    E.consume_live_chain_stats()


def walk(env, orders, decisions, shared, seed):
    import alphagrad.approx.env as E
    from alphagrad.approx.common.face_driver import (
        build_live_face_stream, replay_stage1_draw)
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    E._SHARED_PREFIX = bool(shared)
    _reset()
    cfg = env.config
    consts, args = list(env.consts), list(env.args)
    W, S = E.face_wire_faces(), E.wire_slots()
    s = build_live_face_stream(
        cfg.jaxpr, cfg.argnums, consts, args, vocab=incr_token_vocab(),
        max_faces=E.MAX_FACES, max_axes=E.MAX_AXES_PER_VERTEX,
        window=E.MAX_DELTA_TOKENS, cache=64)
    B, N = len(orders), int(orders[0].shape[0])
    specs = -np.ones((N, E.MAX_RULES_PER_VERTEX, 3), np.int32)
    specs[..., 2] = 0
    fh = -np.ones((B, N, W, S, 3), np.int32)
    fh[..., 2] = 0
    kh = np.zeros((B, N, W), np.int32)
    rng = np.random.default_rng(seed)
    rec = {"face": [], "tok": [], "decisions": []}
    t0 = time.perf_counter()
    for n in range(N):
        hks = [s.hist_key(i, fh[i], kh[i], n) for i in range(B)]
        vs = [int(orders[i][n]) for i in range(B)]
        legs = [s.face_slot_legality(orders[i], specs, n, vs[i], fh[i], kh[i],
                                     hist_key=hks[i]) for i in range(B)]
        nfs = [s.n_faces(orders[i], specs, n, vs[i], fh[i], kh[i],
                         hist_key=hks[i]) for i in range(B)]
        step = []
        for i in range(B):
            if decisions is None:
                rows1, skips = draw_stage1(legs[i], nfs[i], W, S, rng)
            else:
                rows1, skips, _rows = decisions[n][i]
            dec = s.vertex_face_decisions(
                orders[i], specs, n, vs[i],
                lambda f, sl, L, _r=rows1: replay_stage1_draw(f, sl, L, _r),
                skips=skips, face_rows_hist=fh[i], face_skips_hist=kh[i],
                hist_key=hks[i])
            rows = (draw_stage2(dec, rows1, skips, nfs[i], rng)
                    if decisions is None else decisions[n][i][2])
            step.append((rows1, skips, rows, _snap(tuple(dec))))
        chunks = [[] for _ in range(B)]
        for f in range(max(nfs) if nfs else 0):
            for i in range(B):
                if f < nfs[i]:
                    chunks[i].append(_snap(s.chunk_ex(
                        orders[i], specs, n, vs[i], specs[n], step[i][2],
                        step[i][1], f, fh[i], kh[i], hist_key=hks[i])))
        for i in range(B):
            rec["face"].append((n, i, _snap(tuple(legs[i])), nfs[i],
                                step[i][3], chunks[i]))
            fh[i, n] = step[i][2]
            kh[i, n] = step[i][1]
        rec["decisions"].append([st[:3] for st in step])
        if n + 1 >= N:
            break
        for i in range(B):
            out = E._callback(cfg, args, consts, orders[i], specs, fh[i],
                              kh[i], n + 1)
            rec["tok"].append((n + 1, i, np.array(out[0], copy=True)))
    ft = []
    for i in range(B):
        o_list = [int(x) for x in orders[i][:N - 1]]
        sl = specs[:N - 1].tolist()
        fsig = E._face_wire_keys(fh[i][:N - 1], kh[i][:N - 1], N - 1)
        if shared:
            got = E._shared_prefix_stream(
                cfg, consts, args, o_list, sl, specs[:N - 1], fh[i][:N - 1],
                kh[i][:N - 1], fsig, True, False)[2]
        else:
            got = E._face_transforms_for_order(
                cfg, consts, args, o_list, sl, fh[i][:N - 1], kh[i][:N - 1],
                wire_sig=fsig)
        ft.append(_ft_sig(got))
    rec["ft"] = ft
    rec["secs"] = time.perf_counter() - t0
    rec["stats"] = {"face": dict(s.stats),
                    "stream": dict(E._INCR_STREAM_STATS),
                    "shared": dict(E._PREFIX_GRAPH_STATS),
                    "chain": E.consume_live_chain_stats()}
    return rec


def _diff(a, b):
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return " length"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y)
            if d is not None:
                return f"[{i}]{d}"
        return None
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
        return f" {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
    return None


def main(target, kind, n_envs, seed):
    env = build(target)
    orders = plan_orders(env, kind, n_envs, seed)
    sep = walk(env, orders, None, shared=False, seed=seed)
    sha = walk(env, orders, sep["decisions"], shared=True, seed=seed)
    bad = []
    for a, b in zip(sep["face"], sha["face"]):
        d = _diff(a, b)
        if d is not None:
            bad.append(f"face step {a[0]} env {a[1]}{d}")
    for a, b in zip(sep["tok"], sha["tok"]):
        d = _diff(a, b)
        if d is not None:
            bad.append(f"tokens step {a[0]} env {a[1]}{d}")
    if len(sep["face"]) != len(sha["face"]) or len(sep["tok"]) != len(
            sha["tok"]):
        bad.append("the two walks made different numbers of calls")
    for i, (a, b) in enumerate(zip(sep["ft"], sha["ft"])):
        if a != b:
            bad.append(f"env {i}: the measured face transforms differ")
    napprox = sum(int((st[2][:, :, 0] != -1).sum()) + int(st[1].sum())
                  for step in sep["decisions"] for st in step)
    print(f"[replay] target={target} order={kind} envs={n_envs} "
          f"vertices={len(orders[0])} decisions={napprox} "
          f"separate_s={sep['secs']:.1f} shared_s={sha['secs']:.1f}",
          flush=True)
    print("STATS " + json.dumps({"separate": sep["stats"],
                                 "shared": sha["stats"],
                                 "vertices": int(len(orders[0])),
                                 "envs": int(n_envs)}), flush=True)
    for b in bad[:30]:
        print(f"[replay] DIFF {b}", flush=True)
    print(f"REPLAY-{'FAIL' if bad else 'OK'} diffs={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                  int(sys.argv[4])))
