"""Map the (latency, memory, quality) landscape of HAND-SPECIFIED approximation
plans on a FIXED (reverse) elimination order -- no policy, no learning.

WHY THIS EXISTS
---------------
Every conclusion the campaign has drawn about "what the policy could have
found" rests on numbers the POLICY produced: one plan per env per episode,
measured once, on a GPU whose state drifts ~18-20% between windows. Three
things have therefore never been established:

  1. the SHAPE of the objective surface as a function of HOW MANY faces are
     approximated (is there a 1-15-approx basin? does it fall off a cliff?);
  2. the MEASUREMENT SPREAD of a single fixed plan (mean AND sigma, from
     INDEPENDENT measurements -- not repeated reads of one execution);
  3. the QUALITY NOISE FLOOR -- the campaign has never re-measured the same
     plan's 200-step-Adam loss-drop more than once, yet agent 1 found
     episode-to-episode quality autocorrelation ~0 in 11 of 12 runs. Either
     the quality channel is noisy or the plans genuinely differ; nothing in
     the campaign distinguishes those.

PAIRED, ALWAYS. Project memory: an earlier "17.5% beats reverse" was
unpaired-comparison drift (the same plan re-measured back-to-back gave ratio
1.00). Every candidate here is measured BACK-TO-BACK with its own exact
reference inside the same window and reported as a RATIO. The identity plan
is measured against itself the same way, so the identity ratio distribution
IS the drift floor that every other ratio must be read against.

WHAT IT MEASURES ON. The SAME env / target / measurement path the trainer
uses: `alphagrad.approx.env._callback` with `--measure-grad --seed-vertices`,
`--quality-metric loss_drop`, `--cmp-type latency --mem-type peak_memory`.
There is no second measurement implementation here -- that is the whole point.

PLANS
-----
  identity                       0 approximations (the exact-rev reference)
  quant@{1,5,15,50,all}          exactly N live faces carry a QUANT rule
  diag@{1,5,15,50,all}           ... a DIAG rule
  compress@{1,5,15,50,all}       ... a COMPRESS rule
  skip@all                       every live face SKIPped (the destruction floor)
  archive:<label>                a recovered Pareto-archive plan, replayed

"Exactly N faces" is honest: faces are enumerated with graphax's own
`faces_of` on an IncrementalJaxpr that is advanced with THIS plan's own face
transforms, so the count reflects the graph the plan actually produces, not
the exact graph's.

USAGE
-----
CPU validation (do this before ever pointing it at TLM)::

    JAX_PLATFORMS=cpu uv run --no-sync python \
      src/alphagrad/approx/tools/landscape_map.py \
      --example Helmholtz --dataset none --ladder 1,2 --reps 2 \
      --noise-floor-reps 3 --walk-steps 5 --num-data-points 1 \
      --reps-per-point 1 --latency-inner-reps 1 \
      --out-dir /tmp/landscape_smoke

Flagship (as a second job step, after R4's frozen episodes)::

    uv run --no-sync python src/alphagrad/approx/tools/landscape_map.py \
      --example TransformerLM --dataset wikitext2 --hidden-dim 256 \
      --vocab-size 512 --num-layers 3 --exec-on-gpu \
      --out-dir /Users/assmuth/dsnn/run_analysis/landscape

RESTARTABLE. Every measured row is appended to `rows.csv` immediately and
keyed by (plan_id, trial, role); a re-run skips what is already there. Kill
it and restart it; nothing is recomputed.

NO GPU COUNT ASSUMED. Without `--exec-on-gpu` everything runs on whatever
backend JAX picked. With it, the process pins itself as a MEASURE ACTOR
(ALPHAGRAD_MEASURE_ACTOR=1) and uses the single device it can see, so one
visible GPU is enough -- the trainer's ">= 2 GPUs" rule does not apply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback


# ---------------------------------------------------------------------------
# CLI -- parsed BEFORE any heavy import, because several env-var knobs the
# env module reads are IMPORT-TIME constants (_MEASURE_ACTOR, MAX_FACES,
# _SKIP_COUNT_OPS). Nothing here sets JAX_PLATFORMS: an import side-effect
# that pins the platform poisons child processes (project memory), so the
# caller owns that.
# ---------------------------------------------------------------------------
def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # target
    p.add_argument("--example", default="TransformerLM")
    p.add_argument("--dataset", default="wikitext2",
                   help="'none' disables the dataset (Helmholtz smoke).")
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--embd-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=250197)
    # measurement path (defaults = the campaign configuration)
    p.add_argument("--measure-grad", action="store_true", default=True)
    p.add_argument("--no-measure-grad", dest="measure_grad",
                   action="store_false")
    p.add_argument("--seed-vertices", action="store_true", default=True)
    p.add_argument("--no-seed-vertices", dest="seed_vertices",
                   action="store_false")
    p.add_argument("--exec-on-gpu", action="store_true")
    p.add_argument("--cmp-type", default="latency")
    p.add_argument("--mem-type", default="peak_memory")
    p.add_argument("--num-data-points", type=int, default=5)
    p.add_argument("--reps-per-point", type=int, default=4)
    p.add_argument("--latency-inner-reps", type=int, default=5)
    p.add_argument("--num-eval-samples", type=int, default=5)
    p.add_argument("--latency-warmup", type=int, default=1,
                   help="UNTIMED executions before the first timed rep, "
                        "inside env._callback. The smoke run showed the very "
                        "first execution of a plan reading 5-10x the settled "
                        "value; without this the whole trial-0 column is "
                        "first-touch, not latency.")
    p.add_argument("--quality-metric", default="loss_drop",
                   choices=["loss_drop", "cosine", "none"])
    p.add_argument("--walk-steps", type=int, default=200)
    p.add_argument("--walk-lr", type=float, default=1e-3)
    p.add_argument("--walk-probe-seed", type=int, default=0)
    # the experiment
    p.add_argument("--ladder", default="1,5,15,50",
                   help="Comma-separated approximation counts; 'all' is "
                        "appended automatically unless --no-all-rung.")
    p.add_argument("--no-all-rung", action="store_true")
    p.add_argument("--ops", default="quant,diag,compress",
                   help="Which approximation families get a ladder.")
    p.add_argument("--skip-plan", action="store_true", default=True,
                   help="Include the all-SKIP destruction plan.")
    p.add_argument("--no-skip-plan", dest="skip_plan", action="store_false")
    p.add_argument("--quant-dtype", default="bfloat16")
    p.add_argument("--compress-kind", default="mean")
    p.add_argument("--quant-slots", default="0,1,2",
                   help="Face slots (0=lhs/pre, 1=rhs/post, 2=res/new) a "
                        "QUANT rule is written into.")
    p.add_argument("--diag-slots", default="2")
    p.add_argument("--compress-slots", default="2")
    p.add_argument("--reps", type=int, default=5,
                   help="INDEPENDENT paired trials per plan (>=5 required by "
                        "the brief; each trial is a fresh _callback for the "
                        "reference AND for the candidate).")
    p.add_argument("--noise-floor-reps", type=int, default=10,
                   help="Extra repeats of ONE plan to establish the quality "
                        "noise floor (>=10 required by the brief).")
    p.add_argument("--noise-floor-plan", default="identity")
    p.add_argument("--warmup-trials", type=int, default=1,
                   help="Unrecorded-for-the-summary passes over every plan "
                        "before the measured trials. They ARE written to the "
                        "CSV (role 'warmup', trial -1) so nothing is hidden, "
                        "but they are excluded from every aggregate: their "
                        "job is to pay each plan's compile and first-touch "
                        "cost once, where it cannot contaminate a ratio.")
    # archived winners
    p.add_argument("--archive", action="append", default=[],
                   metavar="LABEL=PATH",
                   help="A run's pareto_front.json to recover plans from. "
                        "Repeatable.")
    p.add_argument("--archive-target", action="append", default=[],
                   metavar="LABEL:LATENCY_NS[:QUALITY]",
                   help="Which point to recover from that run's front, "
                        "matched on the recorded objective values. "
                        "Repeatable.")
    p.add_argument("--archive-tol", type=float, default=0.02,
                   help="Relative tolerance when matching a target latency.")
    # plumbing
    p.add_argument("--out-dir",
                   default="/Users/assmuth/dsnn/run_analysis/landscape")
    p.add_argument("--tag", default="",
                   help="Suffix for the output file names.")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Stop cleanly after this wall time (0 = no limit). "
                        "The run is restartable, so a stop is not a loss.")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and describe every plan, measure nothing.")
    return p


ARGS = make_argparser().parse_args()

# --- IMPORT-TIME env knobs -------------------------------------------------
# `_MEASURE_ACTOR` is read at import of alphagrad.approx.env. Setting it here
# is what makes --exec-on-gpu work with ONE visible GPU (the trainer's
# "GPU 0 is the trainer's, so you need >= 2" branch is for the trainer).
if ARGS.exec_on_gpu:
    os.environ["ALPHAGRAD_MEASURE_ACTOR"] = "1"
# The campaign's measurement stack. setdefault, so a launcher that already
# exported these (the sbatch skeleton does) always wins.
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_DIRECT_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_CLEAR_JIT_CACHES_EVERY", "0")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_UNIFIED_FACE_ENUM", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
# The quality channel is configured through the ENVIRONMENT in this codebase
# (one env var, one reader, so two paths cannot disagree) -- mirror ppo.py.
os.environ["ALPHAGRAD_QUALITY_METRIC"] = str(ARGS.quality_metric)
os.environ["ALPHAGRAD_WALK_STEPS"] = str(int(ARGS.walk_steps))
os.environ["ALPHAGRAD_WALK_LR"] = repr(float(ARGS.walk_lr))
os.environ["ALPHAGRAD_WALK_PROBE_SEED"] = str(int(ARGS.walk_probe_seed))
os.environ["ALPHAGRAD_VOCAB_SIZE"] = str(int(ARGS.vocab_size))

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import jax.random as jrand                                    # noqa: E402
import equinox as eqx                                         # noqa: E402

import alphagrad.approx.env as envmod                         # noqa: E402
from alphagrad.approx.env import (                            # noqa: E402
    VertexEliminationEnv,
    REWARD_INDEX,
    MAX_RULES_PER_VERTEX,
    FACE_SLOTS,
    COMPRESS_SENTINEL,
    QUANT_SENTINEL,
    consume_per_face_stats,
)
from alphagrad.approx.common.examples import (                # noqa: E402
    get_fn, get_args, data_gen, infer_argnums, grad_target_setup,
)
from alphagrad.approx.common.eval_samples import (            # noqa: E402
    generate_eval_samples,
)
from alphagrad.approx.common.order_specs import (             # noqa: E402
    build_order_specs,
)
from graphax.sparse.micro_actions import (                    # noqa: E402
    COMPRESS_KINDS, QUANT_DTYPES,
)


# ---------------------------------------------------------------------------
# Target construction -- a LINE-FOR-LINE mirror of ppo.py's, because the whole
# claim of this script is "the same measurement path". Any divergence here
# makes every number below incomparable with the campaign's.
# ---------------------------------------------------------------------------
def _traced_inlined(target_fn, xs):
    """``jax.make_jaxpr(target_fn)(*xs)`` numbered on the form that is
    actually eliminated. Copied from ``cpu_approx_worker._traced_inlined``
    rather than imported, so importing this tool never runs that module's
    import-time side effects."""
    from graphax import inline_call_primitives
    cj = jax.make_jaxpr(target_fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:                                    # jax >= 0.4.31
        from jax.extend.core import ClosedJaxpr
    except ImportError:                     # older / internal layout
        from jax._src.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def build_env(args):
    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and (
        args.example.endswith("NeuralNetwork")
        or args.example.startswith("TransformerLM"))
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call,
                   dataset_size=args.dataset_size)
    target_fn, xs, argnums = grad_target_setup(args, target_fn, xs, args.example)
    closed_jaxpr = _traced_inlined(target_fn, xs)

    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=xs,
        argnums=argnums,
        num_envs=0,
        data_gen=gen,
        target_fun=target_fn,
        cmp_type=args.cmp_type,
        mem_type=args.mem_type,
        exec_on_gpu=bool(args.exec_on_gpu),
        measure_latency=True,
        num_data_points=int(args.num_data_points),
        reps_per_point=int(args.reps_per_point),
        latency_inner_reps=int(args.latency_inner_reps),
        latency_warmup=int(args.latency_warmup),
        # --face-actions implies per-face legality masking (a per-vertex rule
        # that does not fit ONE face's operand would otherwise hit graphax's
        # strict TRANSFORM-DID-NOT-FIT guard and kill the measurement).
        per_face=True,
        measure_grad=bool(args.measure_grad),
        # NOT terminal_rewards_only: we always call at stop == len(order), so
        # every call is terminal anyway, and leaving it off removes one
        # config difference that could silently zero a channel.
        terminal_rewards_only=False,
        delta_obs=True,
    )

    # FACE WIDTH -- the provable per-graph bound, configured BEFORE anything
    # builds a shape from it. ALPHAGRAD_MAX_FACES (the launcher's 2538) wins.
    bound = envmod.derived_max_faces(
        closed_jaxpr.jaxpr, argnums, closed_jaxpr.literals, xs)
    envmod.configure_max_faces(bound)
    print(f"[landscape] face width: derived bound {bound} "
          f"(in force: {envmod.MAX_FACES})", flush=True)

    key, eval_key = jrand.split(key)
    eval_samples = generate_eval_samples(env, eval_key, args.num_eval_samples)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
    return env, eval_samples, closed_jaxpr


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def rev_order(env) -> np.ndarray:
    """The order ALPHAGRAD_FORCE_REV_ORDER pins the policy to.

    `masks.vertex_avail_at_step` keeps only the HIGHEST-indexed still-available
    vertex, and availability is `vertex_valid_static`, so the forced order is
    the valid vertices in descending id -- not `range(n, 0, -1)`."""
    return np.array(sorted((int(v) for v in env.valid_vertices), reverse=True),
                    dtype=np.int32)


def empty_plan(n_steps: int):
    """(sparsity_specs, face_specs, face_skips) for the exact plan.

    The per-vertex `sparsity_specs` stay all-exact for EVERY plan here: under
    the live-face configuration the approximations ride the face wires, and
    the per-vertex rows are the end-sentinel rows the trainer also sends."""
    mf = envmod.MAX_FACES
    specs = np.full((n_steps, MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    face_specs = np.full((n_steps, mf, FACE_SLOTS, 3), -1, dtype=np.int32)
    face_skips = np.zeros((n_steps, mf), dtype=np.int32)
    return specs, face_specs, face_skips


def _rule_row(op: str, args) -> list[int]:
    """One face-slot wire row for `op`, in `decode_vertex_rule_specs`'s
    format (see env.py: QUANT -> [-3, dtype_idx, 0], COMPRESS ->
    [-2, physical_axis, kind_idx], DIAG -> [out_axis, primal_axis, factor]
    with factor == -1 meaning "joint gcd")."""
    if op == "quant":
        try:
            di = QUANT_DTYPES.index(args.quant_dtype)
        except ValueError:
            raise SystemExit(
                f"--quant-dtype {args.quant_dtype!r} is not in QUANT_DTYPES="
                f"{list(QUANT_DTYPES)}")
        return [QUANT_SENTINEL, di, 0]
    if op == "compress":
        try:
            ki = COMPRESS_KINDS.index(args.compress_kind)
        except ValueError:
            raise SystemExit(
                f"--compress-kind {args.compress_kind!r} is not in "
                f"COMPRESS_KINDS={list(COMPRESS_KINDS)}")
        return [COMPRESS_SENTINEL, 0, ki]
    if op == "diag":
        # factor -1 = joint gcd across the out axis and every primal axis --
        # the maximal legal diagonal. Rows that do not fit a given face are
        # dropped by the decoder for that face only.
        return [0, 0, -1]
    raise SystemExit(f"unknown op {op!r}")


def _slots(spec: str) -> tuple[int, ...]:
    out = tuple(int(s) for s in str(spec).split(",") if s.strip() != "")
    for s in out:
        if not (0 <= s < FACE_SLOTS):
            raise SystemExit(f"slot {s} outside 0..{FACE_SLOTS - 1}")
    return out


def build_ladder_plan(env, order, op: str, budget, args):
    """Place `op` on exactly `budget` LIVE faces along `order` ('all' = every
    live face), walking the graph forward so the face counts are the ones the
    plan itself produces.

    Faces are enumerated with graphax's own `faces_of` on an IncrementalJaxpr
    that is advanced with `env._face_dict_for_vertex`'s output -- i.e. the
    SAME enumeration `_callback` will do when it measures this plan, not a
    parallel reimplementation.
    """
    from graphax import faces_of
    from graphax.incremental import IncrementalJaxpr

    cfg = env.config
    specs, face_specs, face_skips = empty_plan(len(order))
    ij = IncrementalJaxpr(cfg.jaxpr, tuple(cfg.argnums), list(env.consts),
                          list(env.args), track_faces=False)
    unlimited = (budget == "all")
    remaining = (1 << 60) if unlimited else int(budget)

    row = None if op == "skip" else _rule_row(op, args)
    slots = () if op == "skip" else {
        "quant": _slots(args.quant_slots),
        "diag": _slots(args.diag_slots),
        "compress": _slots(args.compress_slots),
    }[op]

    used = 0
    n_slot_rows = 0
    per_vertex_faces = []
    for k, v in enumerate(order):
        v = int(v)
        keys = faces_of(ij.graph, ij.tgraph, v, cfg.jaxpr)
        nf = min(len(keys), envmod.MAX_FACES)
        per_vertex_faces.append(nf)
        for f in range(nf):
            if remaining <= 0:
                break
            if op == "skip":
                face_skips[k, f] = 1
            else:
                for s in slots:
                    face_specs[k, f, s] = row
                    n_slot_rows += 1
            used += 1
            remaining -= 1
        # Advance the graph exactly as the measurement will: the k-th vertex's
        # face keys are only valid on the graph the first k-1 TRANSFORMED
        # eliminations produced.
        per_face = envmod._face_dict_for_vertex(
            cfg, ij, v, face_specs[k], face_skips[k])
        ij.eliminate(v, (), per_face or None)

    total_live = int(sum(per_vertex_faces))
    return {
        "specs": specs,
        "face_specs": face_specs,
        "face_skips": face_skips,
        "n_faces_approx": int(used),
        "n_slot_rows": int(n_slot_rows),
        "total_live_faces": total_live,
        "per_vertex_faces": per_vertex_faces,
    }


# ---------------------------------------------------------------------------
# Archived-winner recovery
# ---------------------------------------------------------------------------
def recover_archive_plan(path, env, order, target_latency_ns, target_quality,
                         tol):
    """Rebuild (order, specs, face_specs, face_skips) for one archived Pareto
    point, matched on its RECORDED objective values.

    `pareto_front.json` stores `{"obj": {...}, "seq": ...}` where `seq` is
    either the plain (vertex, calls) list or -- under --face-actions --
    `{"seq": [...], "faces": [{"k","f","rows","skips"}, ...]}` (ppo.py
    `_decode_arch`). It does NOT store the admitting episode, so a point can
    only be addressed by its objectives.

    Returns (plan, note). `plan` is None when the point cannot be recovered;
    `note` always says why, because the brief is explicit that an
    unrecoverable spec must be REPORTED, never approximated.
    """
    if not os.path.exists(path):
        return None, f"no such archive file: {path}"
    with open(path) as fh:
        doc = json.load(fh)
    front = doc.get("front") or []
    if not front:
        return None, f"{path}: front is empty"
    objs = doc.get("objectives") or []
    if "latency" not in objs and "latency_ns" not in objs:
        return None, (f"{path}: objectives {objs} carry no latency channel -- "
                      "cannot address a point by latency")
    lat_name = "latency" if "latency" in objs else "latency_ns"
    q_name = next((n for n in objs if "cos" in n or "quality" in n), None)

    best, best_err = None, None
    for pt in front:
        o = pt.get("obj", {})
        lat = -float(o.get(lat_name, 0.0))          # stored as a reward
        if lat <= 0:
            continue
        err = abs(lat - target_latency_ns) / max(target_latency_ns, 1.0)
        if target_quality is not None and q_name:
            err += abs(float(o.get(q_name, 0.0)) - target_quality)
        if best_err is None or err < best_err:
            best, best_err = pt, err
    if best is None or best_err > tol + (0.0 if target_quality is None else 0.05):
        return None, (f"{path}: no front point within {tol:.1%} of "
                      f"{target_latency_ns:.0f} ns "
                      f"(closest relative error {best_err!r}); the winner "
                      "reported in the analysis is NOT on the final dumped "
                      "front, so its spec is unrecoverable from this file")

    seq_field = best["seq"]
    if isinstance(seq_field, dict):
        seq, faces = seq_field.get("seq"), seq_field.get("faces") or []
    else:
        seq, faces = seq_field, []
    if seq is None:
        return None, f"{path}: matched point carries no 'seq'"

    try:
        a_order, specs, n_rules, v_skips = build_order_specs(
            seq, env, return_skips=True)
    except Exception as exc:
        return None, f"{path}: build_order_specs failed: {exc!r}"

    mf = envmod.MAX_FACES
    face_specs = np.full((len(a_order), mf, FACE_SLOTS, 3), -1, dtype=np.int32)
    face_skips = np.zeros((len(a_order), mf), dtype=np.int32)
    n_faces = 0
    for grp in faces:
        k = int(grp["k"])
        if not (0 <= k < len(a_order)):
            return None, (f"{path}: face group k={k} outside the recorded "
                          f"order of length {len(a_order)}")
        for idx, f in enumerate(grp["f"]):
            f = int(f)
            if f >= mf:
                return None, (f"{path}: face index {f} >= MAX_FACES={mf} -- "
                              "the archive was written with a different face "
                              "width; refusing to truncate the plan")
            face_specs[k, f] = np.asarray(grp["rows"][idx], dtype=np.int32)
            face_skips[k, f] = int(grp["skips"][idx])
            n_faces += 1
    # A vertex-level skip() has no face-level address; broadcasting it would
    # be an approximation of the recorded plan, so refuse instead.
    if bool(np.any(v_skips)) and not faces:
        return None, (f"{path}: the recorded seq carries vertex-level skip() "
                      "but no face wires -- cannot place it on the face "
                      "skip wire without guessing which faces it meant")

    note = (f"recovered from {os.path.basename(path)}: "
            f"{len(a_order)} vertices, {n_rules} per-vertex rules, "
            f"{n_faces} face wires, obj={best['obj']}")
    if list(map(int, a_order)) != list(map(int, order)):
        note += " [NOTE: elimination order differs from rev]"
    return {
        "order": np.asarray(a_order, dtype=np.int32),
        "specs": specs,
        "face_specs": face_specs,
        "face_skips": face_skips,
        "n_faces_approx": n_faces,
        "n_slot_rows": int(np.sum(face_specs[..., 0] != -1)),
        "total_live_faces": -1,
        "per_vertex_faces": [],
    }, note


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(env, eval_samples, order, plan):
    """ONE independent measurement of one plan through the trainer's own
    reward harness. Returns a dict of the channels plus wall time."""
    consume_per_face_stats()          # drop whatever the plan-build replay left
    t0 = time.perf_counter()
    _, _, reward = envmod._callback(
        env.config, env.args, env.consts,
        jnp.asarray(order),
        jnp.asarray(plan["specs"]),
        jnp.asarray(plan["face_specs"]),
        jnp.asarray(plan["face_skips"]),
        int(len(order)),
        *eval_samples,
    )
    wall = time.perf_counter() - t0
    r = np.asarray(reward, dtype=np.float64)
    st = consume_per_face_stats()
    return {
        "latency_ns": float(-r[REWARD_INDEX["latency_ns"]]),
        "peak_memory": float(-r[REWARD_INDEX["peak_memory"]]),
        "quality": float(r[REWARD_INDEX["quality"]]),
        "frob_residual": float(r[REWARD_INDEX["frob_residual"]]),
        "wall_s": wall,
        "applied": int(st.get("applied", 0)),
        "skipped": int(st.get("skipped", 0)),
    }


CSV_FIELDS = [
    "plan_id", "op", "budget", "trial", "role",
    "n_faces_approx", "n_slot_rows", "total_live_faces",
    "latency_ns", "peak_memory", "quality", "frob_residual",
    "applied", "skipped", "wall_s", "timestamp",
]


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            done[(row["plan_id"], int(row["trial"]), row["role"])] = row
    return done


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise(rows):
    """Per-plan aggregates plus the PAIRED ratios.

    A trial contributes a ratio only when BOTH its candidate row and its own
    reference row exist, and the two were measured back-to-back in the same
    window -- that is the whole point of pairing.

    `applied` / `skipped` (how many face rules graphax actually landed vs
    silently dropped as not fitting their operand) are harvested from EVERY
    role including warmup, because the per-face hooks only count inside the
    armed compile scope: once the executable is cached the counters read 0.
    Warmup rows contribute NOTHING else -- their whole purpose is to keep
    first-touch cost out of the ratios."""
    by = {}
    rules = {}
    for r in rows:
        pid = r["plan_id"]
        a, k = int(r.get("applied") or 0), int(r.get("skipped") or 0)
        prev = rules.get(pid, (0, 0))
        if a + k > prev[0] + prev[1]:
            rules[pid] = (a, k)
        if r["role"] == "warmup":
            continue
        by.setdefault(pid, {}).setdefault(int(r["trial"]), {})[r["role"]] = r
    out = {}
    for pid, trials in by.items():
        lat, mem, qual, lat_ratio, mem_ratio, q_delta = [], [], [], [], [], []
        meta = None
        for _t, roles in sorted(trials.items()):
            cand = roles.get("candidate")
            ref = roles.get("reference")
            if cand is None:
                continue
            meta = meta or cand
            lat.append(float(cand["latency_ns"]))
            mem.append(float(cand["peak_memory"]))
            qual.append(float(cand["quality"]))
            if ref is not None:
                rl, rm = float(ref["latency_ns"]), float(ref["peak_memory"])
                if rl > 0:
                    lat_ratio.append(float(cand["latency_ns"]) / rl)
                if rm > 0:
                    mem_ratio.append(float(cand["peak_memory"]) / rm)
                q_delta.append(float(cand["quality"]) - float(ref["quality"]))
        if meta is None:
            continue

        def _s(v):
            a = np.asarray(v, dtype=np.float64)
            if a.size == 0:
                return {"n": 0, "mean": float("nan"), "std": float("nan"),
                        "min": float("nan"), "max": float("nan")}
            return {"n": int(a.size), "mean": float(a.mean()),
                    "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                    "min": float(a.min()), "max": float(a.max())}

        out[pid] = {
            "op": meta["op"], "budget": meta["budget"],
            "n_faces_approx": int(meta["n_faces_approx"]),
            "n_slot_rows": int(meta["n_slot_rows"]),
            "total_live_faces": int(meta["total_live_faces"]),
            "applied": float(rules.get(pid, (0, 0))[0]),
            "skipped": float(rules.get(pid, (0, 0))[1]),
            "latency_ns": _s(lat), "peak_memory": _s(mem), "quality": _s(qual),
            "latency_ratio": _s(lat_ratio), "mem_ratio": _s(mem_ratio),
            "quality_delta": _s(q_delta),
        }
    return out


def write_markdown(path, summ, notes, args, extra):
    L = []
    L.append("# Approximation landscape on the fixed reverse order\n")
    L.append(f"- target: `{args.example}` / `{args.dataset}` "
             f"(hidden {args.hidden_dim}, layers {args.num_layers}, "
             f"vocab {args.vocab_size})")
    L.append(f"- measurement: the trainer's own `env._callback` "
             f"(measure_grad={args.measure_grad}, "
             f"seed_vertices={args.seed_vertices}, "
             f"quality={args.quality_metric}, walk {args.walk_steps} steps)")
    L.append(f"- {args.reps} INDEPENDENT paired trials per plan; every "
             f"candidate measured back-to-back with its own exact reference "
             f"and reported as a RATIO")
    L.append("")
    L.append("## Ratios (candidate / its own paired exact-rev reference)\n")
    L.append("")
    L.append("`rules applied / skipped` is how many face rules graphax "
             "actually landed vs silently dropped as not fitting their "
             "operand, counted at the compile that built the plan. A rung "
             "with skipped > 0 did NOT get the approximation it asked for.")
    L.append("")
    L.append("| plan | approx faces | rules applied / skipped | "
             "latency ratio (mean +/- sd) | mem ratio | "
             "quality (mean +/- sd) | latency ns (mean) |")
    L.append("|---|---:|---:|---|---|---|---:|")

    def _fmt(s):
        if s["n"] == 0:
            return "n/a"
        return f"{s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})"

    ident = summ.get("identity")
    for pid in sorted(summ, key=lambda p: (summ[p]["op"],
                                           summ[p]["n_faces_approx"])):
        s = summ[pid]
        L.append(f"| `{pid}` | {s['n_faces_approx']} | "
                 f"{s['applied']:.0f} / {s['skipped']:.0f} | "
                 f"{_fmt(s['latency_ratio'])} | {_fmt(s['mem_ratio'])} | "
                 f"{_fmt(s['quality'])} | {s['latency_ns']['mean']:.0f} |")
    L.append("")
    if ident is not None and ident["latency_ratio"]["n"]:
        d = ident["latency_ratio"]
        L.append(f"**Drift floor.** The identity plan measured against ITSELF "
                 f"gives latency ratio {d['mean']:.4f} +/- {d['std']:.4f} "
                 f"(range {d['min']:.4f}-{d['max']:.4f}). No candidate ratio "
                 f"inside that band is evidence of anything.\n")
    if extra.get("noise_floor"):
        nf = extra["noise_floor"]
        L.append("## Quality noise floor\n")
        L.append(f"Plan `{nf['plan_id']}` measured {nf['n']} times "
                 f"independently:\n")
        L.append(f"- quality mean {nf['mean']:.6f}, sd {nf['std']:.6g}, "
                 f"range {nf['min']:.6f} - {nf['max']:.6f}")
        L.append(f"- latency mean {nf['lat_mean']:.0f} ns, "
                 f"sd {nf['lat_std']:.0f} ns "
                 f"(CV {nf['lat_cv']:.2%})")
        L.append("")
        L.append(f"The {args.walk_steps}-step Adam walk reads a FIXED probe batch and the "
                 "env's fixed initial weights (`env.args`, never re-drawn), "
                 "so this spread is pure execution nondeterminism, not "
                 "data resampling. Read agent 1's ~0 episode-to-episode "
                 "quality autocorrelation against it: a spread far below the "
                 "between-episode variation means the episodes really were "
                 "measuring DIFFERENT plans.\n")
    if notes:
        L.append("## Archived winners\n")
        for n in notes:
            L.append(f"- {n}")
        L.append("")
    L.append("## Provenance\n")
    L.append("```")
    L.append(" ".join(sys.argv))
    L.append("```")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_figure(path, summ, archive_ids):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[landscape] figure skipped: {exc!r}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = {"identity": "#111111", "quant": "#1f77b4", "diag": "#d62728",
              "compress": "#2ca02c", "skip": "#7f7f7f", "archive": "#ff7f0e"}
    for pid, s in summ.items():
        x, y = s["latency_ns"]["mean"], s["quality"]["mean"]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        op = "archive" if pid in archive_ids else s["op"]
        marker = "*" if op == "archive" else ("D" if op == "identity" else "o")
        size = 260 if op in ("archive", "identity") else 60
        ax.errorbar(x, y,
                    xerr=s["latency_ns"]["std"] or None,
                    yerr=s["quality"]["std"] or None,
                    fmt="none", ecolor=colors.get(op, "#888888"),
                    elinewidth=1, alpha=0.7)
        ax.scatter([x], [y], s=size, marker=marker,
                   color=colors.get(op, "#888888"), zorder=3,
                   label=op if op not in ax.get_legend_handles_labels()[1]
                   else None)
        ax.annotate(pid, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7)
    ax.set_xlabel("measured latency (ns), mean of independent trials")
    ax.set_ylabel("quality (200-step Adam loss drop)")
    ax.set_title("Approximation landscape on the fixed reverse order")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[landscape] figure -> {path}", flush=True)


# ---------------------------------------------------------------------------
def main():
    args = ARGS
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(args.out_dir, f"rows{tag}.csv")
    md_path = os.path.join(args.out_dir, f"summary{tag}.md")
    fig_path = os.path.join(args.out_dir, f"landscape{tag}.png")
    plans_path = os.path.join(args.out_dir, f"plans{tag}.json")

    env, eval_samples, _cj = build_env(args)
    order = rev_order(env)
    print(f"[landscape] {len(order)} valid vertices; rev order "
          f"{order[:6].tolist()}...{order[-3:].tolist()}", flush=True)

    rungs = [int(x) for x in args.ladder.split(",") if x.strip()]
    if not args.no_all_rung:
        rungs = rungs + ["all"]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    plans: dict[str, dict] = {}
    plan_orders: dict[str, np.ndarray] = {}

    print("[landscape] building plans (this walks the graph once per plan)",
          flush=True)
    ident = build_ladder_plan(env, order, "quant", 0, args)
    ident["op"], ident["budget"] = "identity", "0"
    plans["identity"] = ident
    plan_orders["identity"] = order
    total_live = ident["total_live_faces"]
    print(f"[landscape] total LIVE faces on the rev order: {total_live} "
          f"(per-vertex max {max(ident['per_vertex_faces'] or [0])})",
          flush=True)

    for op in ops:
        for b in rungs:
            if b != "all" and int(b) > total_live:
                print(f"[landscape] skip {op}@{b}: only {total_live} live "
                      f"faces exist", flush=True)
                continue
            pid = f"{op}@{b}"
            pl = build_ladder_plan(env, order, op, b, args)
            pl["op"], pl["budget"] = op, str(b)
            plans[pid] = pl
            plan_orders[pid] = order
    if args.skip_plan:
        pl = build_ladder_plan(env, order, "skip", "all", args)
        pl["op"], pl["budget"] = "skip", "all"
        plans["skip@all"] = pl
        plan_orders["skip@all"] = order

    # --- archived winners ---------------------------------------------------
    notes: list[str] = []
    archive_paths = {}
    for spec in args.archive:
        if "=" not in spec:
            raise SystemExit(f"--archive wants LABEL=PATH, got {spec!r}")
        lab, path = spec.split("=", 1)
        archive_paths[lab] = path
    archive_ids = set()
    for spec in args.archive_target:
        parts = spec.split(":")
        lab, lat = parts[0], float(parts[1])
        q = float(parts[2]) if len(parts) > 2 else None
        path = archive_paths.get(lab)
        if path is None:
            notes.append(f"**{lab}: NOT RECOVERED** -- no --archive path given "
                         f"for this label.")
            continue
        pl, note = recover_archive_plan(path, env, order, lat, q,
                                        args.archive_tol)
        if pl is None:
            notes.append(f"**{lab}: NOT RECOVERED** -- {note}")
            print(f"[landscape] archive {lab}: NOT RECOVERED -- {note}",
                  flush=True)
            continue
        pid = f"archive:{lab}"
        pl["op"], pl["budget"] = "archive", lab
        plans[pid] = pl
        plan_orders[pid] = pl["order"]
        archive_ids.add(pid)
        notes.append(f"**{lab}: recovered** -- {note}")
        print(f"[landscape] archive {lab}: {note}", flush=True)

    with open(plans_path, "w") as fh:
        json.dump({pid: {"op": p["op"], "budget": p["budget"],
                         "n_faces_approx": p["n_faces_approx"],
                         "n_slot_rows": p["n_slot_rows"],
                         "total_live_faces": p["total_live_faces"],
                         "per_vertex_faces": p["per_vertex_faces"]}
                   for pid, p in plans.items()}, fh, indent=2)
    print(f"[landscape] {len(plans)} plans; manifest -> {plans_path}",
          flush=True)
    for pid, p in plans.items():
        print(f"    {pid:22s} approx_faces={p['n_faces_approx']:5d} "
              f"slot_rows={p['n_slot_rows']:5d}", flush=True)
    if args.dry_run:
        print("[landscape] --dry-run: nothing measured", flush=True)
        return

    # --- measure ------------------------------------------------------------
    done = load_done(csv_path)
    t_start = time.perf_counter()
    stop = False

    def _budget_left():
        return (args.max_seconds <= 0
                or (time.perf_counter() - t_start) < args.max_seconds)

    ident_plan = plans["identity"]

    # WARMUP. The smoke run made this non-optional: the FIRST execution of a
    # plan read 5-10x its settled latency (cold XLA executable, cold
    # allocator), which is exactly the trial-0 column of every plan. Pay it
    # once, per plan, under a role the aggregates ignore.
    for w in range(args.warmup_trials):
        for pid, plan in plans.items():
            key = (pid, -1 - w, "warmup")
            if key in done or not _budget_left():
                continue
            try:
                m = measure(env, eval_samples, plan_orders[pid], plan)
            except Exception:
                traceback.print_exc()
                print(f"[landscape] WARMUP FAILED {pid}", flush=True)
                continue
            row = {
                "plan_id": pid, "op": plan["op"], "budget": plan["budget"],
                "trial": -1 - w, "role": "warmup",
                "n_faces_approx": plan["n_faces_approx"],
                "n_slot_rows": plan["n_slot_rows"],
                "total_live_faces": plan["total_live_faces"],
                "timestamp": f"{time.time():.3f}",
                **{k: m[k] for k in ("latency_ns", "peak_memory", "quality",
                                     "frob_residual", "applied", "skipped",
                                     "wall_s")},
            }
            append_row(csv_path, row)
            done[key] = row
            print(f"[landscape] warmup {pid:22s} lat={m['latency_ns']:.0f}ns "
                  f"applied={m['applied']} skipped={m['skipped']} "
                  f"({m['wall_s']:.1f}s)", flush=True)

    for trial in range(args.reps):
        for pid, plan in plans.items():
            if stop:
                break
            porder = plan_orders[pid]
            # PAIRED: the reference is re-measured immediately before every
            # candidate, in the same window, on the same device. Unpaired
            # comparison is exactly how the campaign got a phantom 17.5%.
            for role, use_plan, use_order in (
                    ("reference", ident_plan, order),
                    ("candidate", plan, porder)):
                key = (pid, trial, role)
                if key in done:
                    continue
                if not _budget_left():
                    print("[landscape] wall budget reached -- stopping "
                          "cleanly (restart to continue)", flush=True)
                    stop = True
                    break
                try:
                    m = measure(env, eval_samples, use_order, use_plan)
                except Exception:
                    traceback.print_exc()
                    print(f"[landscape] FAILED {pid} trial {trial} {role} "
                          "-- recorded as missing, not as a bad score",
                          flush=True)
                    continue
                row = {
                    "plan_id": pid, "op": plan["op"], "budget": plan["budget"],
                    "trial": trial, "role": role,
                    "n_faces_approx": plan["n_faces_approx"],
                    "n_slot_rows": plan["n_slot_rows"],
                    "total_live_faces": plan["total_live_faces"],
                    "timestamp": f"{time.time():.3f}",
                    **{k: m[k] for k in ("latency_ns", "peak_memory",
                                         "quality", "frob_residual",
                                         "applied", "skipped", "wall_s")},
                }
                append_row(csv_path, row)
                done[key] = row
                print(f"[landscape] {pid:22s} t{trial} {role:9s} "
                      f"lat={m['latency_ns']:.0f}ns "
                      f"mem={m['peak_memory']:.3g} q={m['quality']:.4f} "
                      f"({m['wall_s']:.1f}s)", flush=True)

    # --- quality noise floor ------------------------------------------------
    nf_pid = args.noise_floor_plan
    nf_rows = []
    if nf_pid in plans and args.noise_floor_reps > 0:
        for i in range(args.noise_floor_reps):
            key = (f"noisefloor:{nf_pid}", i, "candidate")
            if key in done:
                nf_rows.append(done[key])
                continue
            if not _budget_left():
                break
            try:
                m = measure(env, eval_samples, plan_orders[nf_pid],
                            plans[nf_pid])
            except Exception:
                traceback.print_exc()
                continue
            row = {
                "plan_id": f"noisefloor:{nf_pid}", "op": "noisefloor",
                "budget": nf_pid, "trial": i, "role": "candidate",
                "n_faces_approx": plans[nf_pid]["n_faces_approx"],
                "n_slot_rows": plans[nf_pid]["n_slot_rows"],
                "total_live_faces": plans[nf_pid]["total_live_faces"],
                "timestamp": f"{time.time():.3f}",
                **{k: m[k] for k in ("latency_ns", "peak_memory", "quality",
                                     "frob_residual", "applied", "skipped",
                                     "wall_s")},
            }
            append_row(csv_path, row)
            done[key] = row
            nf_rows.append(row)
            print(f"[landscape] noise-floor {i}: q={m['quality']:.6f} "
                  f"lat={m['latency_ns']:.0f}ns", flush=True)

    # --- report -------------------------------------------------------------
    all_rows = list(load_done(csv_path).values())
    # Warmup rows ARE passed in -- summarise() uses them only for the
    # applied/skipped rule counters (see its docstring) and excludes them
    # from every measured aggregate.
    summ = summarise([r for r in all_rows if r["op"] != "noisefloor"])
    extra = {}
    nf = [r for r in all_rows if r["op"] == "noisefloor"]
    if len(nf) >= 2:
        q = np.array([float(r["quality"]) for r in nf])
        la = np.array([float(r["latency_ns"]) for r in nf])
        extra["noise_floor"] = {
            "plan_id": nf_pid, "n": int(q.size),
            "mean": float(q.mean()), "std": float(q.std(ddof=1)),
            "min": float(q.min()), "max": float(q.max()),
            "lat_mean": float(la.mean()), "lat_std": float(la.std(ddof=1)),
            "lat_cv": float(la.std(ddof=1) / la.mean()) if la.mean() else 0.0,
        }
    write_markdown(md_path, summ, notes, args, extra)
    print(f"[landscape] summary -> {md_path}", flush=True)
    with open(os.path.join(args.out_dir, f"summary{tag}.json"), "w") as fh:
        json.dump({"summary": summ, "extra": extra, "notes": notes}, fh,
                  indent=2)
    if not args.no_figure:
        write_figure(fig_path, summ, archive_ids)
    print(f"[landscape] rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
