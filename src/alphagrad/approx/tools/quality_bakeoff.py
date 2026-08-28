#!/usr/bin/env python3
"""QUALITY-METRIC BAKE-OFF.

Answers the owner's 2026-08-07 question that was never settled: which
GRADIENT-COSINE formulation should reward slot 6 hold --

  (a) gradient cosine at init over K batches, K in {1,4,8}
  (b) cosine of the AGGREGATED (summed) gradients over a short trajectory
  (c) MEAN cosine over all per-step gradients across a short trajectory
  (d) LAST-STEP cosine after a short trajectory
  (e) CLIPPED RELATIVE FROBENIUS, clip(1 - ||Je-Ja||/||Je||, -1, 1)  (A2's
      channel, whose correlation has never been measured)
  (f) loss drop over a 200-step Adam walk (the incumbent reference point)
  (g) the LEGACY Jacobian cosine, for the 0.610 anchor.

Ground truth is the SAME one the owner's table used: the FINAL DOWNSTREAM
TEST ACCURACY of VmappedNeuralNetwork trained on MNIST with the plan's own
gradient, produced by downstream_train.py.  This script does NOT produce the
ground truth; it emits one row of metrics per plan, keyed by the same plan
tag downstream_train.py is run under, and correlate_bakeoff.py joins them.

TARGET MODES
  --target raw   trace get_raw_fn  -- the per-element model, multi-output.
                 jacve builds a real JACOBIAN; the gradient is the contraction
                 sum(class)/mean(batch), exactly downstream_train._contract.
                 This is the regime the owner's 0.610/0.737 table was measured
                 in, so it is the comparable one.
  --target loss  trace get_fn -- the registered scalar loss (post-744fc3d).
                 jacve leaves ARE the gradient already; there is no Jacobian
                 to contract.  This is the CURRENT env.py regime and the one
                 whose cost the stale 9.70 s / 4.24 GB figure misstates.

Every metric is timed and peak-memory'd on its own, warm (one untimed warm-up
execution per compiled callable), so the cost column is per-metric-per-plan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np


# Approximation families used to build the plan population.  Graded from
# "indistinguishable from exact" to "destroys the gradient", so the population
# spans the quality range a correlation needs.
SEVERITIES = {
    "bf16":      [{"op": "Quant", "dtype": "bfloat16"}],
    "f16":       [{"op": "Quant", "dtype": "float16"}],
    "diag2":     [{"op": "Diag", "i": 0, "j": 1, "factor": 2}],
    "diag4":     [{"op": "Diag", "i": 0, "j": 1, "factor": 4}],
    "diag8":     [{"op": "Diag", "i": 0, "j": 1, "factor": 8}],
    "cmax0":     [{"op": "Compress", "axes": [0], "kind": "abs_max"}],
    "cmed0":     [{"op": "Compress", "axes": [0], "kind": "median"}],
    "cmin0":     [{"op": "Compress", "axes": [0], "kind": "abs_min"}],
    "cmax1":     [{"op": "Compress", "axes": [1], "kind": "abs_max"}],
    "cmin1":     [{"op": "Compress", "axes": [1], "kind": "abs_min"}],
}

# Filled in by main() once the target's jaxpr is traced.
_N_VERTICES = [0]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _winsor_mean(xs, frac=0.2):
    a = np.sort(np.asarray(xs, dtype=np.float64))
    if a.size == 0:
        return float("nan")
    k = int(a.size * frac / 2)
    if k:
        a = a[k:a.size - k]
    return float(a.mean())


class Meter:
    """Wall-clock + peak-HBM around a block, warm."""

    def __init__(self, dev):
        self.dev = dev

    def reset(self):
        try:
            self.dev.clear_memory_stats()
        except Exception:
            pass

    def peak(self):
        try:
            return float(self.dev.memory_stats().get("peak_bytes_in_use", 0.0))
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
def build_plan(spec, target_fn, argnums, sample_args, jaxpr_axis_sizes):
    """Return (grad_or_jac_fn, has_aux, label).

    spec is one of:
      exact_rev / exact_fwd        graphax cross-country, no approximation
      jax_grad                     jax.grad on the scalar loss (loss target only)
      jax_jacrev                   jax.jacrev (raw target only)
      seq:<path>[:channel]         replay an archived best_sequences.json
    """
    import jax
    from graphax import jacve

    if spec in ("exact_rev", "exact_fwd"):
        order = "rev" if spec == "exact_rev" else "fwd"
        fn = jacve(target_fn, order=order, argnums=argnums, has_aux=True)
        return fn, True, spec

    if spec == "jax_jacrev":
        return jax.jacrev(target_fn, argnums=argnums), False, spec

    if spec.startswith("seq:"):
        from alphagrad.approx.common.seq_replay import (
            load_best_sequence, parse_recorded_seq,
        )
        rest = spec[len("seq:"):]
        if ":" in rest:
            run_dir, channel = rest.rsplit(":", 1)
        else:
            run_dir, channel = rest, "best_overall"
        seq = load_best_sequence(run_dir, channel)
        order, transforms = parse_recorded_seq(
            seq, axis_sizes=jaxpr_axis_sizes, skip_low_precision_quant=True,
        )
        fn = jacve(target_fn, order=order, transforms=transforms,
                   argnums=argnums, has_aux=True)
        return fn, True, spec

    if spec.startswith("deg:"):
        # deg:<severity>@<vertex|all>  -- a plan built NATIVELY for the graph
        # being traced right now.  The archived best_sequences.json files were
        # recorded against an older, larger graph (4-D edges); replaying them
        # dies in `apply_compress`, so the population has to be generated here.
        # The elimination ORDER is left to graphax ("rev"), which sidesteps the
        # stale-vertex-list problem entirely; only the per-vertex approximation
        # rules are ours.  A single-vertex approximation is also exactly the
        # shape of the dossier's `k24/f0`-style single-face skips.
        from alphagrad.approx.common.seq_replay import parse_recorded_seq
        body = spec[len("deg:"):]
        sev, _, where = body.partition("@")
        ops = SEVERITIES[sev]
        if where in ("", "all"):
            targets = list(range(1, _N_VERTICES[0] + 1))
        else:
            targets = [int(where)]
        seq = [{"vertex": v, "ops": [dict(o) for o in ops]} for v in targets]
        _order, transforms = parse_recorded_seq(
            seq, axis_sizes=jaxpr_axis_sizes, skip_low_precision_quant=True,
        )
        fn = jacve(target_fn, order="rev", transforms=transforms,
                   argnums=argnums, has_aux=True)
        return fn, True, spec

    # ValueError, NOT SystemExit: build_plan is called inside an
    # `except Exception` guard so one unbuildable plan skips its row instead of
    # killing the whole sweep. `jax_grad` legitimately appears in the shared
    # plan list (downstream_train understands it, the metric harness does not).
    raise ValueError(f"unknown plan spec {spec!r}")


def _leaves(out, has_aux):
    import jax
    payload = out[1] if has_aux else out
    return [np.asarray(x) if False else x for x in jax.tree.leaves(payload)]


def _contract_to_grad(leaves, n_out_axes):
    """downstream_train._contract: sum over the class axis, mean over batch."""
    import jax.numpy as jnp
    out = []
    for j in leaves:
        if n_out_axes == 2:            # (batch, class, *wshape)
            out.append(jnp.mean(jnp.sum(j, axis=1), axis=0).astype(jnp.float32))
        else:
            out.append(j.astype(jnp.float32))
    return out


def _cos_and_relfrob(leaves_e, leaves_a):
    """Streamed global cosine + relative Frobenius, env.py._quality_metrics."""
    import jax.numpy as jnp
    if len(leaves_e) != len(leaves_a) or not leaves_e:
        return 0.0, 1.0
    dot = ee = aa = rr = None
    for e, a in zip(leaves_e, leaves_a):
        if e.shape != a.shape:
            if e.shape == a.shape[::-1] and a.ndim == 2:
                a = a.T
            else:
                return 0.0, 1.0
        cdt = jnp.promote_types(jnp.promote_types(e.dtype, a.dtype), jnp.float32)
        ef = jnp.ravel(e).astype(cdt)
        af = jnp.ravel(a).astype(cdt)
        d = jnp.sum(ef * af)
        e2 = jnp.sum(jnp.abs(ef) ** 2)
        a2 = jnp.sum(jnp.abs(af) ** 2)
        r2 = jnp.sum(jnp.abs(ef - af) ** 2)
        dot = d if dot is None else dot + d
        ee = e2 if ee is None else ee + e2
        aa = a2 if aa is None else aa + a2
        rr = r2 if rr is None else rr + r2
    en = jnp.sqrt(ee)
    an = jnp.sqrt(aa)
    eps = jnp.sqrt(1e-7)
    cos = jnp.real(dot / (jnp.maximum(en, eps) * jnp.maximum(an, eps)))
    rel = jnp.sqrt(rr) / jnp.maximum(en, eps)
    return float(cos), float(rel)


def clipped_rel_frob(rel):
    x = float(rel)
    if not np.isfinite(x):
        return -1.0
    return float(min(1.0, max(-1.0, 1.0 - x)))


# ---------------------------------------------------------------------------
def adam_init(w):
    import jax.numpy as jnp
    return [jnp.zeros_like(x) for x in w], [jnp.zeros_like(x) for x in w]


def adam_step(w, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    import jax.numpy as jnp
    nm, nv, nw = [], [], []
    for wi, gi, mi, vi in zip(w, g, m, v):
        mi = b1 * mi + (1 - b1) * gi
        vi = b2 * vi + (1 - b2) * (gi * gi)
        mh = mi / (1 - b1 ** t)
        vh = vi / (1 - b2 ** t)
        nw.append(wi - lr * mh / (jnp.sqrt(vh) + eps))
        nm.append(mi)
        nv.append(vi)
    return nw, nm, nv


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="VmappedNeuralNetwork")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--target", choices=["raw", "loss"], default="raw")
    ap.add_argument("--plans", required=False, default=None,
                    help="comma-separated plan specs, or @file with one per line "
                         "as 'tag=spec'")
    ap.add_argument("--select", type=int, default=0,
                    help="with --enumerate-out: keep this many plans, "
                         "stratified across the cosine range")
    ap.add_argument("--enumerate-out", default=None,
                    help="DISCOVERY MODE: try every (severity, vertex) plan on "
                         "the current graph, keep the ones that build and run, "
                         "and write a plan list here. Does not measure.")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--traj-steps", type=int, default=200)
    ap.add_argument("--walk-steps", type=int, default=200)
    ap.add_argument("--walk-lr", type=float, default=1e-3)
    ap.add_argument("--probe-seed", type=int, default=20260807)
    ap.add_argument("--seed", type=int, default=250197)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import jax.random as jrand

    dev = jax.devices()[0]
    meter = Meter(dev)
    print(f"device={dev} platform={jax.default_backend()} target={args.target}",
          flush=True)

    from alphagrad.approx.common.examples import (
        data_gen, get_args, get_fn, get_raw_fn, infer_argnums,
    )

    target_fn = get_raw_fn(args.example) if args.target == "raw" \
        else get_fn(args.example)
    argnums = tuple(infer_argnums(args.example))

    key = jrand.PRNGKey(args.seed)
    init_args = get_args(args.example, key, dataset=args.dataset)
    weights = list(init_args[2:])
    sampler = data_gen(args.example, dataset=args.dataset)

    # K fixed probe batches, drawn off the walk probe seed so they are the
    # same batches the loss-drop walk uses for its first one.
    pkey = jrand.PRNGKey(args.probe_seed)
    batches = []
    for i in range(args.k_max):
        pkey, sk = jrand.split(pkey)
        batches.append(sampler(jrand.split(sk, 4)))
    x0, y0 = batches[0]

    sample_args = (x0, y0, *weights)
    axis_sizes = []
    try:
        jx = jax.make_jaxpr(target_fn)(*sample_args)
        for eqn in jx.jaxpr.eqns:
            for ov in eqn.outvars:
                sh = getattr(getattr(ov, "aval", None), "shape", ()) or ()
                axis_sizes.extend(int(s) for s in sh)
    except Exception as exc:
        print(f"jaxpr trace failed: {exc}", flush=True)
    axis_sizes = sorted(set(axis_sizes))

    try:
        _N_VERTICES[0] = len(jx.jaxpr.eqns)
    except Exception:
        _N_VERTICES[0] = 0
    print(f"jaxpr eqns={_N_VERTICES[0]} axis_sizes={axis_sizes}", flush=True)

    out0 = target_fn(*sample_args)
    n_out_axes = int(np.ndim(out0))
    print(f"target output ndim={n_out_axes} shape={getattr(out0,'shape',None)}",
          flush=True)

    # ---- plan list -------------------------------------------------------
    plans = []
    if not args.plans:
        pass
    elif args.plans.startswith("@"):
        with open(args.plans[1:]) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tag, _, spec = line.partition("=")
                plans.append((tag.strip(), spec.strip()))
    else:
        for s in args.plans.split(","):
            plans.append((s.strip(), s.strip()))

    # ---- exact reference -------------------------------------------------
    exact_fn, exact_aux, _ = build_plan("exact_rev", target_fn, argnums,
                                        sample_args, axis_sizes)
    exact_jit = jax.jit(exact_fn)

    def exact_leaves(w, xb, yb, as_grad):
        o = exact_jit(xb, yb, *w)
        lv = jax.tree.leaves(o[1] if exact_aux else o)
        return _contract_to_grad(lv, n_out_axes) if as_grad else lv

    # true scalar loss, for the walk
    loss_fn = jax.jit(get_fn(args.example))

    # ---- DISCOVERY MODE --------------------------------------------------
    # Which (severity, vertex) plans are actually legal on THIS graph is not
    # knowable a priori -- a Compress only fits an edge with that axis, a Diag
    # only fits when the primal axis index is in range for EVERY incoming edge.
    # Rather than guess, build each candidate, run it once, and keep the ones
    # that survive and are not bit-identical to exact.
    if args.enumerate_out:
        ge0 = exact_leaves(weights, x0, y0, True)
        keep = []
        for sev in SEVERITIES:
            for v in range(1, _N_VERTICES[0] + 1):
                spec = f"deg:{sev}@{v}"
                try:
                    fn, has_aux, _ = build_plan(spec, target_fn, argnums,
                                                sample_args, axis_sizes)
                    o = jax.jit(fn)(x0, y0, *weights)
                    lv = jax.tree.leaves(o[1] if has_aux else o)
                    ga = _contract_to_grad(lv, n_out_axes)
                    c, r = _cos_and_relfrob(ge0, ga)
                    if not np.isfinite(c):
                        raise ValueError("non-finite cosine")
                except Exception as exc:
                    print(f"  skip {spec}: {type(exc).__name__} "
                          f"{str(exc)[:90]}", flush=True)
                    continue
                keep.append((f"{sev}_v{v}", spec, c, r))
                print(f"  KEEP {spec:<22} cos={c:+.5f} relfrob={r:.5f}",
                      flush=True)
        with open(args.enumerate_out + ".json", "w") as fh:
            json.dump([{"tag": t, "spec": s, "cos": c, "rel_frob": r}
                       for t, s, c, r in keep], fh, indent=2)

        sel = keep
        if args.select and len(keep) > args.select:
            # STRATIFY BY COSINE so the training population spans the quality
            # range instead of piling up at cos=1.  Frozen-gradient plans
            # (cos == 0) are the degenerate end the dossier cares about, so at
            # least a few are forced in by construction.
            frozen = [k for k in keep if abs(k[2]) < 1e-9]
            rest = sorted((k for k in keep if abs(k[2]) >= 1e-9),
                          key=lambda k: k[2])
            n_frozen = min(3, len(frozen))
            n_rest = max(args.select - n_frozen, 1)
            idx = np.linspace(0, len(rest) - 1, num=min(n_rest, len(rest)))
            sel = frozen[:n_frozen] + [rest[int(round(i))] for i in idx]
            # de-duplicate on tag, preserve order
            seen, dedup = set(), []
            for k in sel:
                if k[0] not in seen:
                    seen.add(k[0])
                    dedup.append(k)
            sel = dedup

        with open(args.enumerate_out, "w") as fh:
            fh.write("jax_grad=jax_grad\n")
            fh.write("exact_rev=exact_rev\n")
            for tag, spec, _c, _r in sel:
                fh.write(f"{tag}={spec}\n")
        print(f"\nwrote {args.enumerate_out}: {len(sel)} selected of "
              f"{len(keep)} valid approx plans (+2 exact references)",
              flush=True)
        print("selected cosine spread: " + ", ".join(
            f"{k[2]:.3f}" for k in sorted(sel, key=lambda k: k[2])), flush=True)
        return

    rows = []
    for tag, spec in plans:
        row: dict[str, Any] = {"tag": tag, "spec": spec, "target": args.target}
        print(f"\n=== {tag}  ({spec}) ===", flush=True)
        try:
            fn, has_aux, _ = build_plan(spec, target_fn, argnums,
                                        sample_args, axis_sizes)
            pjit = jax.jit(fn)
        except Exception as exc:
            print(f"  BUILD FAILED {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
            row["error"] = f"build:{type(exc).__name__}"
            rows.append(row)
            continue

        def plan_leaves(w, xb, yb, as_grad):
            o = pjit(xb, yb, *w)
            lv = jax.tree.leaves(o[1] if has_aux else o)
            return _contract_to_grad(lv, n_out_axes) if as_grad else lv

        # warm-up (mandatory: first measurement in a fresh process reads fast)
        try:
            jax.block_until_ready(plan_leaves(weights, x0, y0, True))
            jax.block_until_ready(exact_leaves(weights, x0, y0, True))
        except Exception as exc:
            print(f"  WARMUP FAILED {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
            row["error"] = f"warmup:{type(exc).__name__}"
            rows.append(row)
            continue

        # ---------------- (a) gradient cosine at init, K batches ----------
        cos_k, frob_k = [], []
        meter.reset()
        t0 = time.perf_counter()
        for i in range(args.k_max):
            xb, yb = batches[i]
            ga = plan_leaves(weights, xb, yb, True)
            ge = exact_leaves(weights, xb, yb, True)
            c, r = _cos_and_relfrob(ge, ga)
            cos_k.append(c)
            frob_k.append(r)
        jax.block_until_ready(ga)
        t_k8 = time.perf_counter() - t0
        p_k8 = meter.peak()
        for K in (1, 4, 8):
            if K <= args.k_max:
                row[f"gradcos_init_K{K}"] = float(np.mean(cos_k[:K]))
        row["gradcos_init_K8_s"] = t_k8
        row["gradcos_init_K1_s"] = t_k8 / max(args.k_max, 1)
        row["gradcos_init_peak_mb"] = p_k8 / 2 ** 20

        # ---------------- (e) clipped relative Frobenius ------------------
        for K in (1, 4, 8):
            if K <= args.k_max:
                row[f"clipfrob_K{K}"] = float(
                    np.mean([clipped_rel_frob(r) for r in frob_k[:K]]))
        row["rel_frob_K8"] = float(np.mean(frob_k))

        # ---------------- (g) legacy JACOBIAN cosine ----------------------
        if args.target == "raw":
            meter.reset()
            t0 = time.perf_counter()
            ja = plan_leaves(weights, x0, y0, False)
            je = exact_leaves(weights, x0, y0, False)
            cj, rj = _cos_and_relfrob(je, ja)
            jax.block_until_ready(ja)
            row["jaccos_s"] = time.perf_counter() - t0
            row["jaccos_peak_mb"] = meter.peak() / 2 ** 20
            row["jaccos"] = cj
            ja = je = None

        # ---------------- (b)(c)(d) short-trajectory cosines --------------
        try:
            meter.reset()
            t0 = time.perf_counter()
            w = list(weights)
            m, v = adam_init(w)
            step_cos = []
            sum_a = sum_e = None
            for t in range(1, args.traj_steps + 1):
                xb, yb = batches[(t - 1) % args.k_max]
                ga = plan_leaves(w, xb, yb, True)
                ge = exact_leaves(w, xb, yb, True)
                c, _ = _cos_and_relfrob(ge, ga)
                step_cos.append(c)
                sum_a = ga if sum_a is None else [p + q for p, q in zip(sum_a, ga)]
                sum_e = ge if sum_e is None else [p + q for p, q in zip(sum_e, ge)]
                w, m, v = adam_step(w, ga, m, v, float(t), args.walk_lr)
            agg_cos, _ = _cos_and_relfrob(sum_e, sum_a)
            jax.block_until_ready(w[0])
            row["traj_s"] = time.perf_counter() - t0
            row["traj_peak_mb"] = meter.peak() / 2 ** 20
            row["aggcos_traj"] = float(agg_cos)
            row["meancos_traj"] = float(np.mean(step_cos))
            row["lastcos_traj"] = float(step_cos[-1])
        except Exception as exc:
            print(f"  TRAJ FAILED {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
            row["traj_error"] = f"{type(exc).__name__}"

        # ---------------- (f) loss drop, 200 Adam steps -------------------
        try:
            meter.reset()
            t0 = time.perf_counter()
            w = list(weights)
            m, v = adam_init(w)
            L0 = float(loss_fn(x0, y0, *w))
            ok = np.isfinite(L0) and abs(L0) > 1e-12
            if ok:
                for t in range(1, args.walk_steps + 1):
                    ga = plan_leaves(w, x0, y0, True)
                    ga = [jnp.nan_to_num(g) for g in ga]
                    w, m, v = adam_step(w, ga, m, v, float(t), args.walk_lr)
                L1 = float(loss_fn(x0, y0, *w))
                drop = -1.0 if not np.isfinite(L1) else (L0 - L1) / abs(L0)
                row["loss_drop"] = float(np.clip(drop, -1.0, 1.0))
            else:
                row["loss_drop"] = float("nan")
            row["loss_drop_s"] = time.perf_counter() - t0
            row["loss_drop_peak_mb"] = meter.peak() / 2 ** 20
        except Exception as exc:
            print(f"  WALK FAILED {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
            row["walk_error"] = f"{type(exc).__name__}"

        print("  " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in row.items()}), flush=True)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
