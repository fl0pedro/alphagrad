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
uses: `alphagrad.approx.env._callback` on the registered scalar-loss target
(model + loss -- jacve of it IS the gradient; `--measure-grad` is a
deprecated no-op) and NOT
`--seed-vertices` -- workstream A4 removed it from all 64 launchers because
seeds are NOT vertices; leaving it on here built a 2-vertex-larger graph
than the runs whose landscape this is supposed to be),
`--quality-metric grad_cosine`, `--cmp-type latency --mem-type peak_memory`.

QUALITY CHANNEL (changed 2026-08-30).  This tool now defaults to
`grad_cosine`, the same channel the campaign's training arms were settled on
(949f1af: 0.874 Pearson / 0.805 Spearman against downstream accuracy, at
0.003 s per plan).  It replaces `loss_drop`, which cost a 200-step Adam walk
per plan -- the single largest share of the measurement budget -- for no
better a signal.  THE TWO ARE NOT COMPARABLE NUMBERS: a `loss_drop` quality
column and a `grad_cosine` quality column are different quantities on
different scales, so a `rows_*.csv` written before this change must not be
read against one written after it.  Every archived CSV predating this carries
`quality_metric=loss_drop` in its header note for exactly that reason.
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
import math
import os
import sys
import time
import traceback


class _NoMeasureGradIsGone(argparse.Action):
    """``--no-measure-grad`` asked for the raw per-class Jacobian target.

    That mode no longer exists: every registered target IS model + loss, so
    the traced graph is the scalar-loss graph and jacve of it is the gradient.
    Accepting the switch and ignoring it would hand back a landscape measured
    on a different object than the one asked for, so it is an error.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--no-measure-grad is no longer supported: the per-class-Jacobian "
            "target it selected does not exist any more. Every registered "
            "target is model + loss (common.examples.get_fn), so jacve of the "
            "traced graph is the gradient. Drop the flag.")


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
    # DEPRECATED NO-OP, accepted so an archived command line still runs.
    # The traced target is unconditionally the registered scalar training loss
    # (common.examples.get_fn = model + loss), so the flag selects nothing.
    p.add_argument("--measure-grad", action="store_true", default=True,
                   help="DEPRECATED NO-OP (see common.examples."
                        "warn_measure_grad_deprecated).")
    # ITS NEGATION IS A HARD ERROR. --no-measure-grad asked for the raw
    # per-class JACOBIAN target; that mode is gone, and silently ignoring a
    # request for it is how a run reports a Jacobian as a gradient.
    p.add_argument("--no-measure-grad", dest="measure_grad", nargs=0,
                   action=_NoMeasureGradIsGone)
    # Elimination order (default: markowitz per owner ruling)
    p.add_argument("--order", choices=["markowitz", "reverse"], default="markowitz",
                   help="Elimination order to evaluate approximations on (default: markowitz).")
    p.add_argument("--exec-on-gpu", action="store_true")
    p.add_argument("--cmp-type", default="latency")
    # The launchers (run_campaign_2node.sh:144, run_campaign_gpu2node.sh)
    # measure `xla_peak_memory` -- the deterministic compile-time channel.
    # `peak_memory` is a runtime high-water mark and is not comparable
    # with the campaign rows, which is what this tool exists to explain.
    p.add_argument("--mem-type", default="xla_peak_memory")
    p.add_argument("--num-data-points", type=int, default=5)
    p.add_argument("--reps-per-point", type=int, default=4)
    p.add_argument("--latency-inner-reps", type=int, default=5)
    p.add_argument("--num-eval-samples", type=int, default=5)
    # 2, as the launchers pass (--latency-warmup 2). Warmup is a BIAS
    # knob, not a precision knob: one untimed execution leaves first-touch
    # cost in the first timed one.
    p.add_argument("--latency-warmup", type=int, default=2,
                   help="UNTIMED executions before the first timed rep, "
                        "inside env._callback. The smoke run showed the very "
                        "first execution of a plan reading 5-10x the settled "
                        "value; without this the whole trial-0 column is "
                        "first-touch, not latency.")
    p.add_argument("--grad-oracle", choices=["reference", "off"],
                   default="reference",
                   help="Oracle A (ticket .62): the exact gradient of every "
                        "order vs jax.grad once per process; a disagreement "
                        "aborts. Same reader as ppo.py (ALPHAGRAD_GRAD_ORACLE).")
    p.add_argument("--quality-metric", default="grad_cosine",
                   choices=["loss_drop", "grad_cosine", "jac_cosine",
                            "cosine", "none"],
                   help="Which quantity the `quality` column holds. Default "
                        "grad_cosine (the settled campaign channel; cheap and "
                        "the better predictor). loss_drop is the old default "
                        "and runs a 200-step Adam walk per plan. NOT "
                        "comparable across values -- see the module "
                        "docstring. 'cosine' is deprecated in env.py and "
                        "resolves to grad_cosine with a warning.")
    p.add_argument("--approx-add", default=_APPROX_ADD_DEFAULT,
                   choices=list(_APPROX_ADD_CHOICES),
                   help="How the face ADD's two addends meet (ticket .56, "
                        "finding 73): lossy = force both into the container "
                        "the approximated new slot landed on, projecting the "
                        "old edge onto it; lossless = the sum's support is "
                        "the UNION of the two, dropping no non-zero. NOT "
                        "comparable across values; stamped into every row. "
                        "Mirrors ppo.py --approx-add.")
    p.add_argument("--approx-old", default=None, help=argparse.SUPPRESS)
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
    p.add_argument("--noise-floor-plan", default="identity",
                   help="Comma-separated plan ids. The brief wants the "
                        "quality noise floor on identity AND a mid-ladder "
                        "plan, because a floor measured only on the exact "
                        "plan says nothing about a plan that approximates.")
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
    p.add_argument("--archive-all-points", action="store_true",
                   help="Replay EVERY point of every --archive front, not "
                        "just the --archive-target ones. Recovery is "
                        "attempted for all of them and reported as "
                        "recovered/held per run; MEASUREMENT is capped by "
                        "--archive-max-measure.")
    p.add_argument("--archive-max-measure", type=int, default=10,
                   help="How many recovered points per run actually get "
                        "measured, taken in order of best (lowest) recorded "
                        "latency -- that is the part of the front the "
                        "owner's question is about.")
    p.add_argument("--face-inventory", action="store_true",
                   help="Dump the (step, vertex, face-index, face-key, "
                        "primitive, operand shapes/dtypes) inventory of every "
                        "LIVE face on the order, then continue. This is what "
                        "lets a measured ratio be attributed to a NAMED face "
                        "instead of an anonymous index.")
    p.add_argument("--inventory-only", action="store_true",
                   help="Dump the face inventory and exit without measuring.")
    p.add_argument("--skip-face", action="append", default=[], metavar="K:F",
                   help="Build a plan that skips ONLY face F of elimination "
                        "step K (repeatable; all listed faces go in ONE "
                        "plan). This is how the minimal plan is built.")
    p.add_argument("--singleton-sweep", action="store_true",
                   help="Exhaustive sweep over all singleton approximations "
                        "(class x face x slot x legal sub-arguments).")
    p.add_argument("--singleton-skip-sweep", dest="singleton_sweep",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--f-star-threshold", type=float, default=0.8,
                   help="Quality threshold q* for F* inclusion (default: 0.8).")
    p.add_argument("--stack-ladder", default="",
                   help="Comma-separated stack sizes N to draw from F* (e.g. '2,5,10').")
    p.add_argument("--stack-samples", type=int, default=5,
                   help="Number of random samples M per stack size N (default: 5).")
    p.add_argument("--pair-samples", type=int, default=0,
                   help="Number of random pair samples from F* (0 = disabled).")
    p.add_argument("--shard", default="",
                   help="Shard specification 'I/N' (0-indexed, e.g. '0/4') to evaluate a slice of plans.")
    p.add_argument("--sweep-stride", type=int, default=1,
                   help=argparse.SUPPRESS)
    p.add_argument("--report-only", action="store_true",
                   help="Measure nothing; read every rows_*.csv in --out-dir "
                        "and emit the COMBINED report across configs.")
    p.add_argument("--config-note", default="",
                   help="Free-text note stamped on every row of this phase.")
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


# --- --approx-add choices, RESTATED and CROSS-CHECKED ----------------------
# The canonical list is ``env.APPROX_ADD_CHOICES``, but this module builds its
# argparser BEFORE importing alphagrad on purpose: several env knobs below are
# read at IMPORT of ``alphagrad.approx.env`` (``ALPHAGRAD_MEASURE_ACTOR``), so
# pulling env in up here would read them before they are set. The literal is
# therefore a copy, and ``tests/approx_add_test.py`` asserts it still equals
# ``env.APPROX_ADD_CHOICES`` / ``env.APPROX_ADD_DEFAULT`` -- a drift is a test
# failure, not a silently different choice list.
_APPROX_ADD_CHOICES = ("lossy", "lossless")
_APPROX_ADD_DEFAULT = "lossless"


if __name__ == "__main__":
    ARGS = make_argparser().parse_args()
else:
    ARGS = make_argparser().parse_args([])

# --- IMPORT-TIME env knobs -------------------------------------------------
# `_MEASURE_ACTOR` is read at import of alphagrad.approx.env. Setting it here
# is what makes --exec-on-gpu work with ONE visible GPU (the trainer's
# "GPU 0 is the trainer's, so you need >= 2" branch is for the trainer).
if "--exec-on-gpu" in sys.argv or ARGS.exec_on_gpu:
    os.environ["ALPHAGRAD_MEASURE_ACTOR"] = "1"
# The campaign's measurement stack. setdefault, so a launcher that already
# exported these (the sbatch skeleton does) always wins.
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_DIRECT_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_CLEAR_JIT_CACHES_EVERY", "0")
os.environ.setdefault("ALPHAGRAD_UNIFIED_FACE_ENUM", "1")
# The quality channel is configured through the ENVIRONMENT in this codebase
# (one env var, one reader, so two paths cannot disagree) -- mirror ppo.py.
os.environ["ALPHAGRAD_QUALITY_METRIC"] = str(ARGS.quality_metric)
os.environ["ALPHAGRAD_GRAD_ORACLE"] = str(ARGS.grad_oracle)
if getattr(ARGS, "approx_old", None) is not None:
    raise SystemExit(
        f"--approx-old is RETIRED (you passed {ARGS.approx_old!r}). Use "
        f"--approx-add {{{','.join(_APPROX_ADD_CHOICES)}}}; the old values are "
        f"NOT aliases (see alphagrad.approx.env).")
os.environ.pop("ALPHAGRAD_APPROX_OLD", None)
os.environ["ALPHAGRAD_APPROX_ADD"] = str(ARGS.approx_add)
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
    consume_mem_parity,
)
from alphagrad.approx.common.masks import (                   # noqa: E402
    quant_valid_mask,
    compress_slot_mask,
    slot_legality,
    diag_valid_mask,
    diag_pair_gcd,
)
from alphagrad.approx.common.examples import (
    has_scalar_loss as _has_scalar_loss,                # noqa: E402
    get_fn, get_args, data_gen, infer_argnums, grad_target_setup,
)
from alphagrad.approx.common.eval_samples import (            # noqa: E402
    generate_eval_samples,
)
from alphagrad.approx.common import order as _order            # noqa: E402
from alphagrad.approx.common.order_specs import (             # noqa: E402
    build_order_specs, parse_calls, calls_have_skip,
)
from alphagrad.approx.env import micro_actions_to_rule_specs  # noqa: E402
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
        # THE SCALAR-OUTPUT CONTRACT, armed by a property of the EXAMPLE, not
        # by a flag: the registered target is model + loss for every trainable
        # family, and the analytic AD benchmarks (no training loss) are
        # measured as full Jacobians on purpose.
        scalar_target=_has_scalar_loss(args.example),
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
    """The reverse order: the valid vertices in descending id (the order of
    the paired rev-exact reference). ONE implementation, common/order.py
    (ticket .64); the trainer's --fixed-order reverse pins to the same table."""
    return _order.reverse_order(env.valid_vertices)


def markowitz_order(env) -> np.ndarray:
    """The STATIC minimum Markowitz degree order (finding 59). ONE
    implementation, common/order.py (ticket .64); the trainer's --fixed-order
    markowitz pins to the same table, so the sweep and the trainer agree."""
    return _order.fixed_order_for_env("markowitz", env)


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


def face_inventory(env, order, capture_tensors: bool = False):
    """Every LIVE face on `order`, named.

    A face KEY is graphax's ``(vidx[in_edge], vidx[out_edge])`` under its
    stable var index -- the same key ``_eliminate_vertex`` looks up in
    ``face_transforms``. Keys are graph-state dependent, so they are
    enumerated on the EXACT prefix: valid for attributing a plan whose
    approximations start at or after that step, which is every plan here
    (the archived winners carry a single wire).

    When ``capture_tensors=True``, captures the live operand tensors at
    slots lhs (0), rhs (1), and new (2) for exact legality testing.
    """
    from graphax import faces_of
    from graphax.incremental import IncrementalJaxpr
    cfg = env.config
    ij = IncrementalJaxpr(cfg.jaxpr, tuple(cfg.argnums), list(env.consts),
                          list(env.args), track_faces=False)
    inv = []
    for k, v in enumerate(order):
        v = int(v)
        keys = faces_of(ij.graph, ij.tgraph, v, cfg.jaxpr)
        eqn = cfg.jaxpr.eqns[v - 1]
        outs = [[list(o.aval.shape), str(o.aval.dtype)]
                for o in eqn.outvars if hasattr(o, "aval")]
        ins = [[list(i.aval.shape), str(i.aval.dtype)]
               for i in eqn.invars if hasattr(i, "aval")]
        face_tensors = {}
        if capture_tensors:
            def make_rec(fk, site):
                def hook(t):
                    face_tensors.setdefault(fk, {})[site] = t
                    return t
                return hook
            face_transforms = {
                fk: ((make_rec(fk, 0), make_rec(fk, 1), make_rec(fk, 2)),
                     (None, None, None))
                for fk in keys
            }
            ij.eliminate(v, (), face_transforms=face_transforms)
        else:
            ij.eliminate(v, (), None)
        for f, key in enumerate(keys[: envmod.MAX_FACES]):
            entry = {"k": int(k), "vertex": v, "f": int(f),
                     "key": [int(x) for x in key],
                     "prim": eqn.primitive.name, "out": outs, "in": ins}
            if capture_tensors:
                entry["tensors"] = face_tensors.get(key, {})
            inv.append(entry)
    return inv


def get_plan_arrays(plan, n_steps: int):
    """Return (specs, face_specs, face_skips) for plan.
    If already materialized, returns them directly.
    Otherwise allocates and populates on demand from plan['wires'].
    """
    if plan.get("face_specs") is not None:
        return plan["specs"], plan["face_specs"], plan["face_skips"]
    specs, face_specs, face_skips = empty_plan(n_steps)
    for w in plan.get("wires", []) or []:
        k = int(w["k"])
        f = int(w["f"])
        if w.get("kind") == "SKIP":
            face_skips[k, f] = 1
        else:
            slot = int(w["slot"])
            face_specs[k, f, slot] = w["row"]
    return specs, face_specs, face_skips


def build_singleton_plan(env, order, k: int, f: int, op: str, slot: int = 0,
                         row: list[int] | None = None):
    """Build a plan that applies exactly ONE approximation on face (k, f)."""
    k, f = int(k), int(f)
    if op == "skip":
        wires = [{"k": k, "f": f, "kind": "SKIP"}]
        n_slot_rows = 0
    else:
        wires = [{"k": k, "f": f, "slot": slot, "row": list(row),
                  "kind": f"{op.upper()}@slot{slot}"}]
        n_slot_rows = 1
    return {
        "specs": None,
        "face_specs": None,
        "face_skips": None,
        "n_faces_approx": 1,
        "n_slot_rows": n_slot_rows,
        "total_live_faces": -1,
        "per_vertex_faces": [],
        "wires": wires,
    }


def build_singleton_sweep_plans(env, order, inv):
    """Exhaustive singletons over all live faces on order:
    - SKIP: 1 per face
    - QUANT: bf16 per legal slot
    - REDUCE: mean per legal axis per slot
    - DIAG: explicit gcd per legal out-primal axis pair per slot (never -1)
    """
    plans = {}
    plan_orders = {}

    bf16_idx = QUANT_DTYPES.index("bfloat16")
    mean_idx = COMPRESS_KINDS.index("mean")
    slot_names = ["lhs", "rhs", "new"]

    for entry in inv:
        k = int(entry["k"])
        f = int(entry["f"])
        v = int(entry["vertex"])
        prim = entry["prim"]
        key = entry["key"]
        tensors = entry.get("tensors", {})
        # Face id format per owner ruling: (vertex, predecessor, successor) with primitive name
        face_tag = f"v{v}({key[0]}->{key[1]})/{prim}"
        face_short = f"v{v}/{prim}"

        # 1. SKIP (1 per face)
        pid_skip = f"singleton:skip:k{k}.f{f}:{face_short}"
        pl_skip = build_singleton_plan(env, order, k, f, op="skip")
        pl_skip["op"] = "skip"
        pl_skip["budget"] = face_tag
        plans[pid_skip] = pl_skip
        plan_orders[pid_skip] = order

        # For slots lhs(0), rhs(1), new(2):
        for s in range(3):
            st = tensors.get(s)
            if st is None:
                continue
            sname = slot_names[s]

            # 2. QUANT (bf16 only)
            if quant_valid_mask(st, ("bfloat16",))[0]:
                pid_q = f"singleton:quant:k{k}.f{f}:{sname}:bf16"
                row_q = [QUANT_SENTINEL, bf16_idx, 0]
                pl_q = build_singleton_plan(env, order, k, f, op="quant",
                                            slot=s, row=row_q)
                pl_q["op"] = "quant"
                pl_q["budget"] = f"{face_tag}:{sname}"
                plans[pid_q] = pl_q
                plan_orders[pid_q] = order

            # 3. REDUCE (mean first, every legal axis)
            # Wire row expects a logical dimension index; slot_legality.comp
            # tests logical dimensions, unlike compress_slot_mask which indexes
            # canonical slots (dsnn-3qm.73).
            leg = slot_legality(st, 8)
            for a in range(8):
                if leg.comp[a]:
                    pid_r = f"singleton:reduce:k{k}.f{f}:{sname}:ax{a}"
                    row_r = [COMPRESS_SENTINEL, a, mean_idx]
                    pl_r = build_singleton_plan(env, order, k, f, op="compress",
                                                slot=s, row=row_r)
                    pl_r["op"] = "compress"
                    pl_r["budget"] = f"{face_tag}:{sname}:ax{a}"
                    plans[pid_r] = pl_r
                    plan_orders[pid_r] = order


            # 4. DIAG (explicit gcd > 1 per legal axis pair, never -1)
            d_mask = diag_valid_mask(st, 8)
            n_out = len(getattr(st, "out_dims", ()))
            dims = tuple(getattr(st, "out_dims", ())) + tuple(getattr(st, "primal_dims", ()))
            for i in range(min(n_out, 8)):
                for j in range(n_out, min(len(dims), 8)):
                    if d_mask[i, j]:
                        g = diag_pair_gcd(st, i, j)
                        if g > 1:
                            primal_j = j - n_out
                            pid_d = f"singleton:diag:k{k}.f{f}:{sname}:p{i}.{primal_j}.fac{g}"
                            row_d = [i, primal_j, g]
                            pl_d = build_singleton_plan(env, order, k, f, op="diag",
                                                        slot=s, row=row_d)
                            pl_d["op"] = "diag"
                            pl_d["budget"] = f"{face_tag}:{sname}:p{i}.{primal_j}"
                            plans[pid_d] = pl_d
                            plan_orders[pid_d] = order

    return plans, plan_orders


def compose_stack_plan(env, order, singletons_list, pid: str, op: str, budget: str):
    """Compose multiple compatible singletons into one plan."""
    wires = []
    used_faces = set()
    used_slots = set()
    for s_plan in singletons_list:
        for w in s_plan.get("wires", []):
            k, f = w["k"], w["f"]
            if w.get("kind") == "SKIP":
                if (k, f) in used_faces:
                    continue
                used_faces.add((k, f))
                wires.append(w)
            else:
                slot = w["slot"]
                if (k, f, slot) in used_slots:
                    continue
                used_slots.add((k, f, slot))
                wires.append(w)
    return {
        "specs": None,
        "face_specs": None,
        "face_skips": None,
        "n_faces_approx": len({(w["k"], w["f"]) for w in wires}),
        "n_slot_rows": sum(1 for w in wires if w.get("kind") != "SKIP"),
        "total_live_faces": -1,
        "per_vertex_faces": [],
        "wires": wires,
        "op": op,
        "budget": budget,
    }


def build_f_star_stacks(env, order, f_star_items, stack_ladder, stack_samples, pair_samples, seed=250197):
    """Generate pairs and random stacks @N from F* members only."""
    import random
    rng = random.Random(seed)
    plans = {}
    plan_orders = {}

    if not f_star_items:
        return plans, plan_orders

    # 1. Pairs from F*
    if pair_samples > 0 and len(f_star_items) >= 2:
        n_pairs = min(pair_samples, len(f_star_items) * (len(f_star_items) - 1) // 2)
        pairs_seen = set()
        for idx in range(n_pairs):
            s1, s2 = rng.sample(f_star_items, 2)
            pair_key = tuple(sorted([s1[0], s2[0]]))
            if pair_key in pairs_seen:
                continue
            pairs_seen.add(pair_key)
            pid = f"pair:fstar:{idx}"
            pl = compose_stack_plan(env, order, [s1[1], s2[1]], pid, "pair", "2")
            plans[pid] = pl
            plan_orders[pid] = order

    # 2. Stacks @N from F*
    if stack_ladder:
        rungs = [int(x) for x in str(stack_ladder).split(",") if x.strip()]
        for N in rungs:
            if N > len(f_star_items):
                continue
            for m in range(stack_samples):
                chosen = rng.sample(f_star_items, N)
                pid = f"stack:fstar:@{N}_s{m}"
                pl = compose_stack_plan(env, order, [c[1] for c in chosen], pid, "stack", str(N))
                plans[pid] = pl
                plan_orders[pid] = order

    return plans, plan_orders


def build_skip_only_plan(env, order, targets, inventory=None):
    """A plan whose ONLY approximation is SKIPping the listed (step, face)s.

    `inventory` is the face_inventory() list. It is what makes a bad
    --skip-face LOUD: face_skips is MAX_FACES wide, so setting a bit for a
    face index that no live face occupies raises nothing, changes nothing,
    and yields a plan that measures identical to identity -- reported as a
    real 1.00 ratio for a face that was never skipped. Refuse instead.
    """
    live = None
    if inventory is not None:
        live = {(int(e["k"]), int(e["f"])) for e in inventory}
    specs, face_specs, face_skips = empty_plan(len(order))
    for (k, f) in targets:
        k, f = int(k), int(f)
        if not (0 <= k < len(order)):
            raise SystemExit(
                f"--skip-face {k}:{f}: step {k} outside 0..{len(order) - 1}")
        if not (0 <= f < envmod.MAX_FACES):
            raise SystemExit(
                f"--skip-face {k}:{f}: face {f} outside "
                f"0..{envmod.MAX_FACES - 1}")
        if live is not None and (k, f) not in live:
            at_k = sorted(ff for kk, ff in live if kk == k)
            raise SystemExit(
                f"--skip-face {k}:{f}: step {k} has no LIVE face {f}. "
                f"Live faces at step {k}: {at_k or 'none'}. Skipping a "
                f"non-live face is a silent no-op that would be reported as "
                f"a measured ratio.")
        face_skips[k, f] = 1
    return {
        "specs": specs, "face_specs": face_specs, "face_skips": face_skips,
        "n_faces_approx": len(targets), "n_slot_rows": 0,
        # Known exactly when the inventory was enumerated; -1 means "not
        # computed on this path", never "zero".
        "total_live_faces": len(inventory) if inventory is not None else -1,
        "per_vertex_faces": [],
        "wires": [{"k": int(k), "f": int(f), "kind": "SKIP"}
                  for k, f in targets],
    }


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

    plan, note2 = _point_to_plan(best, env, order, path)
    if plan is None:
        return None, note2
    return plan, note2


class _RecoveryError(Exception):
    pass


def _recover_order_specs(seq, env):
    """Rebuild (order, specs, n_rules, skips, convention) from a recorded seq.

    WHY NOT ``build_order_specs``. That helper documents the recorded vertex
    column as the agent's 0-BASED ACTION INDEX and resolves it through
    ``env.valid_vertices``. For the face-actions/live-faces archives that is
    simply false: on TLM the recorded column runs 1..95 over 95 valid
    vertices, so ``valid[95]`` walks off the end -- which is why all 264
    archived Pareto points failed to replay with
    ``IndexError: index 95 is out of bounds for axis 0 with size 95``.

    Both conventions exist in the archive set, so DETECT rather than assume,
    and refuse rather than guess: a mis-resolved column shifts the whole
    order by one, drops the real last vertex, and measures a garbage
    Jacobian while reporting a healthy number (the exact failure mode
    ``build_order_specs``'s own comment warns about)."""
    valid = [int(v) for v in env.valid_vertices]
    idx = [int(v) for v, _ in seq]
    if not idx:
        raise _RecoveryError("recorded seq is empty")
    if sorted(idx) == sorted(valid):
        resolved, conv = list(idx), "vertex-ids"
    elif sorted(idx) == list(range(len(valid))):
        resolved, conv = [valid[i] for i in idx], "action-indices"
    else:
        raise _RecoveryError(
            f"NOT RECOVERED: the recorded vertex column matches neither the "
            f"1-based vertex ids nor the 0-based action indices "
            f"(n={len(idx)}, min={min(idx)}, max={max(idx)}, "
            f"|valid_vertices|={len(valid)}) -- refusing to guess")

    axis_static = np.asarray(env.axis_state_static)
    specs = np.full((len(resolved), MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    skips = np.zeros(len(resolved), dtype=bool)
    n_rules = 0
    for k, ((_v, calls), vid) in enumerate(zip(seq, resolved)):
        if not calls:
            continue
        if calls_have_skip(calls):
            skips[k] = True
        op, i, j, fac, kind, quant = parse_calls(calls)
        if not op:
            continue
        n_rules += len(op)
        specs[k] = micro_actions_to_rule_specs(
            np.array(op), np.array(i), np.array(j), np.array(fac),
            axis_state_for_vertex=axis_static[vid - 1],
            compress_kinds=np.array(kind), quant_dtypes=np.array(quant))
    return np.array(resolved, dtype=np.int32), specs, n_rules, skips, conv


def _point_to_plan(best, env, order, path):
    """One front entry -> (plan, note). Split out so the targeted matcher and
    the --archive-all-points sweep cannot diverge in how they replay."""
    seq_field = best["seq"]
    if isinstance(seq_field, dict):
        seq, faces = seq_field.get("seq"), seq_field.get("faces") or []
    else:
        seq, faces = seq_field, []
    if seq is None:
        return None, f"{path}: matched point carries no 'seq'"

    try:
        a_order, specs, n_rules, v_skips, conv = _recover_order_specs(seq, env)
    except _RecoveryError as exc:
        return None, f"{path}: {exc}"
    except Exception as exc:
        return None, f"{path}: order/spec reconstruction failed: {exc!r}"

    mf = envmod.MAX_FACES
    face_specs = np.full((len(a_order), mf, FACE_SLOTS, 3), -1, dtype=np.int32)
    face_skips = np.zeros((len(a_order), mf), dtype=np.int32)
    n_faces = 0
    wires = []
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
            _rows = np.asarray(grp["rows"][idx], dtype=np.int32)
            _kinds = []
            if int(grp["skips"][idx]) == 1:
                _kinds.append("SKIP")
            for _sl in range(_rows.shape[0]):
                _b = int(_rows[_sl, 0])
                if _b == QUANT_SENTINEL:
                    _kinds.append(f"QUANT@slot{_sl}")
                elif _b == COMPRESS_SENTINEL:
                    _kinds.append(f"COMPRESS@slot{_sl}")
                elif _b >= 0:
                    _kinds.append(f"DIAG@slot{_sl}")
            wires.append({"k": k, "f": f,
                          "kind": "+".join(_kinds) or "none"})
            n_faces += 1
    # A vertex-level skip() has no face-level address; broadcasting it would
    # be an approximation of the recorded plan, so refuse instead.
    if bool(np.any(v_skips)) and not faces:
        return None, (f"{path}: the recorded seq carries vertex-level skip() "
                      "but no face wires -- cannot place it on the face "
                      "skip wire without guessing which faces it meant")

    note = (f"recovered from {os.path.basename(path)} [{conv}]: "
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
        "wires": wires,
    }, note


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(env, eval_samples, order, plan):
    """ONE independent measurement of one plan through the trainer's own
    reward harness. Returns a dict of the channels plus wall time."""
    consume_per_face_stats()          # drop whatever the plan-build replay left
    consume_mem_parity()
    specs, face_specs, face_skips = get_plan_arrays(plan, len(order))
    t0 = time.perf_counter()
    _, _, reward = envmod._callback(
        env.config, env.args, env.consts,
        jnp.asarray(order),
        jnp.asarray(specs),
        jnp.asarray(face_specs),
        jnp.asarray(face_skips),
        int(len(order)),
        *eval_samples,
    )
    wall = time.perf_counter() - t0
    r = np.asarray(reward, dtype=np.float64)
    st = consume_per_face_stats()
    # Ticket .49 made the drain return {"records", "measured", "dropped"};
    # the plan's own record is the last TERMINAL one (the paired reference
    # writes its own, non-terminal record beside it).
    m_parity = [rec for rec in consume_mem_parity()["records"]
                if rec.get("terminal", True)]

    static_temp = 0.0
    runtime_watermark = float(-r[REWARD_INDEX["peak_memory"]])
    if m_parity:
        last_rec = m_parity[-1]
        if last_rec.get("static_temp_bytes") is not None:
            static_temp = float(last_rec["static_temp_bytes"])
        if last_rec.get("runtime_peak_bytes") is not None:
            runtime_watermark = float(last_rec["runtime_peak_bytes"])

    # PER-KIND, not just the totals. Agent B measured DIAG landing on ~1% of
    # the rules it was asked for on TLM, which makes a diag rung identity in
    # disguise -- invisible in an aggregate `applied` count that a
    # co-requested QUANT is filling up.
    detail = {k: int(v) for k, v in st.items()
              if k.startswith(("applied_", "skipped_"))}
    return {
        "latency_ns": float(-r[REWARD_INDEX["latency_ns"]]),
        "peak_memory": runtime_watermark,
        "static_temp": static_temp,
        "quality": float(r[REWARD_INDEX["quality"]]),
        "frob_residual": float(r[REWARD_INDEX["frob_residual"]]),
        "wall_s": wall,
        "applied": int(st.get("applied", 0)),
        "skipped": int(st.get("skipped", 0)),
        "applied_detail": json.dumps(detail, sort_keys=True),
    }


# EVERY ROW CARRIES ITS CONFIG. The campaign's numbers are not comparable
# across --latency-inner-reps (agent B measured inner=5 carrying a
# systematic -13.1% amortisation bias vs inner=50), so a row without its
# config is a row that will be misread.
#
# GRAPHAX_QUANT_PULLDOWN IS GONE. GX-A deleted the branch it gated
# (graphax 1f3d311: `_compute_dtype` is plain highest-common promotion,
# 3978 dtype combinations identical to the flag-off predecessor), so the
# variable no longer changes anything anywhere. The COLUMN stays -- the
# archived rows_*.csv files have it and `load_done` reads them -- but it
# now records that fact instead of implying a live configuration axis.
CSV_FIELDS = [
    "plan_id", "op", "budget", "trial", "role",
    "n_faces_approx", "n_slot_rows", "total_live_faces",
    "latency_ns", "peak_memory", "static_temp", "quality", "frob_residual",
    "applied", "skipped", "applied_detail", "wall_s", "timestamp",
    "pulldown", "inner_reps", "warmup_src", "config_note", "gpu",
    # WHICH QUANTITY the `quality` column holds. Added 2026-08-30 with the
    # switch of the default from loss_drop to grad_cosine. They are different
    # quantities on different scales; a row that does not say which one it is
    # cannot be safely compared with anything. Every rows_*.csv written before
    # this lacks the column, and a missing value is read back as `loss_drop`,
    # which is what all of them are.
    "quality_metric",
    # HOW THE FACE ADD's TWO ADDENDS MET (ticket .56, finding 73): "lossy" or
    # "lossless". Rows written before 2026-09-10 carry the RETIRED
    # ``approx_old`` column instead ("same" / "exact"), which named a
    # different computation -- `same` installed one wire row on two
    # differently-structured tensors and `exact` approximated the post-join
    # SUM -- so the two columns must NOT be pooled. Rows written before
    # 2026-09-04 lack even that column; every one of them ran under the
    # then-default "same" unless the launcher exported
    # ALPHAGRAD_NEW_SLOT_JOIN=0 (the R1-R3 / face_attrib / forensics launchers
    # did -- see UNBIASED_PARETO_AND_MEASUREMENT.md sec 7(c)).
    "approx_add",
]


def config_stamp(args):
    """The (pulldown, inner, warmup) triple this process is measuring under."""
    return {
        # Dead knob, stamped honestly: a row that says "removed" cannot be
        # mistaken for one measured under a pulldown that still bit.
        "pulldown": ("removed:" + os.environ["GRAPHAX_QUANT_PULLDOWN"]
                     if os.environ.get("GRAPHAX_QUANT_PULLDOWN")
                     else "removed"),
        "inner_reps": int(args.latency_inner_reps),
        # BOTH warmups, named. `script` = this tool's whole-plan pass, which
        # pays the COMPILE. `env` = agent B's _resolve_warmup, untimed
        # executions inside the measure loop, which pays first-touch on an
        # already-compiled executable. They are not the same thing and both
        # are on.
        "warmup_src": (f"script:{int(args.warmup_trials)}+"
                       f"env:{os.environ.get('ALPHAGRAD_MEASURE_WARMUP', '1')}"
                       f"/cfg:{int(args.latency_warmup)}"),
        "config_note": args.config_note,
        "quality_metric": str(args.quality_metric),
        "approx_add": str(args.approx_add),
        # WHICH PHYSICAL DEVICE measured this row. If plan A and plan B are
        # measured by different actors, a systematic per-device offset lands
        # straight in PPO's within-batch advantage comparison.
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    }


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
        lat, mem, stemp, qual = [], [], [], []
        lat_ratio, mem_ratio, stemp_ratio, q_delta = [], [], [], []
        meta = None
        for _t, roles in sorted(trials.items()):
            cand = roles.get("candidate")
            ref = roles.get("reference")
            if cand is None:
                continue
            meta = meta or cand
            lat.append(float(cand["latency_ns"]))
            mem.append(float(cand["peak_memory"]))
            stemp.append(float(cand.get("static_temp", 0.0) or 0.0))
            qual.append(float(cand["quality"]))
            if ref is not None:
                rl = float(ref["latency_ns"])
                rm = float(ref["peak_memory"])
                rst = float(ref.get("static_temp", 0.0) or 0.0)
                if rl > 0:
                    lat_ratio.append(float(cand["latency_ns"]) / rl)
                if rm > 0:
                    mem_ratio.append(float(cand["peak_memory"]) / rm)
                if rst > 0:
                    stemp_ratio.append(float(cand.get("static_temp", 0.0) or 0.0) / rst)
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
            "latency_ns": _s(lat), "peak_memory": _s(mem),
            "static_temp": _s(stemp), "quality": _s(qual),
            "latency_ratio": _s(lat_ratio), "mem_ratio": _s(mem_ratio),
            "static_temp_ratio": _s(stemp_ratio),
            "quality_delta": _s(q_delta),
        }
    return out


def write_markdown(path, summ, notes, args, extra):
    L = []
    L.append(f"# Approximation landscape on the {getattr(args, 'order', 'markowitz')} order\n")
    L.append(f"- target: `{args.example}` / `{args.dataset}` "
             f"(hidden {args.hidden_dim}, layers {args.num_layers}, "
             f"vocab {args.vocab_size})")
    L.append(f"- measurement: the trainer's own `env._callback` "
             f"(scalar-loss target, "
             f"quality={args.quality_metric}, walk {args.walk_steps} steps)")
    L.append(f"- {args.reps} INDEPENDENT paired trials per plan; every "
             f"candidate measured back-to-back with its own exact reference "
             f"and reported as a RATIO")
    L.append("")
    L.append("## Ratios (candidate / its own paired exact reference)\n")
    L.append("")
    L.append("`rules applied / skipped` is how many face rules graphax "
             "actually landed vs silently dropped as not fitting their "
             "operand, counted at the compile that built the plan. A rung "
             "with skipped > 0 did NOT get the approximation it asked for.")
    L.append("")
    L.append("| plan | approx faces | rules applied / skipped | "
             "latency ratio (mean +/- sd) | temp ratio | watermark ratio | "
             "quality (mean +/- sd) | latency ns (mean) |")
    L.append("|---|---:|---:|---|---|---|---|---:|")

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
                 f"{_fmt(s['latency_ratio'])} | {_fmt(s.get('static_temp_ratio', {'n': 0}))} | "
                 f"{_fmt(s['mem_ratio'])} | "
                 f"{_fmt(s['quality'])} | {s['latency_ns']['mean']:.0f} |")
    L.append("")

    # --- F* Table -----------------------------------------------------------
    q_thresh = getattr(args, "f_star_threshold", 0.8)
    f_star = [
        (pid, s) for pid, s in summ.items()
        if pid != "identity" and s["quality"]["n"] and s["quality"]["mean"] >= q_thresh
    ]
    f_star.sort(key=lambda item: -item[1]["quality"]["mean"])
    L.append(f"## F* Candidate Table (quality >= {q_thresh:.2f})\n")
    L.append(f"Found {len(f_star)} candidates meeting quality threshold {q_thresh:.2f}:\n")
    L.append("| plan | op | budget | quality | latency ratio | temp ratio | watermark ratio |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for pid, s in f_star:
        tr = s.get("static_temp_ratio", {}).get("mean", float("nan"))
        mr = s.get("mem_ratio", {}).get("mean", float("nan"))
        lr = s.get("latency_ratio", {}).get("mean", float("nan"))
        L.append(f"| `{pid}` | {s['op']} | {s['budget']} | {s['quality']['mean']:.4f} | "
                 f"{lr:.4f} | {tr:.4f} | {mr:.4f} |")
    L.append("")

    # --- Per-Class Memory Response ------------------------------------------
    L.append("## Per-Class Memory Response\n")
    L.append("| class | count | static temp ratio (mean) | watermark ratio (mean) | quality (mean) |")
    L.append("|---|---:|---:|---:|---:|")
    for op_name in ("skip", "quant", "compress", "diag"):
        op_plans = [s for pid, s in summ.items() if s["op"] == op_name and pid != "identity"]
        if not op_plans:
            continue
        trs = [s.get("static_temp_ratio", {}).get("mean", float("nan")) for s in op_plans]
        mrs = [s.get("mem_ratio", {}).get("mean", float("nan")) for s in op_plans]
        qs = [s["quality"]["mean"] for s in op_plans if np.isfinite(s["quality"]["mean"])]
        trs_valid = [x for x in trs if np.isfinite(x)]
        mrs_valid = [x for x in mrs if np.isfinite(x)]
        mean_tr = np.mean(trs_valid) if trs_valid else float("nan")
        mean_mr = np.mean(mrs_valid) if mrs_valid else float("nan")
        mean_q = np.mean(qs) if qs else float("nan")
        L.append(f"| `{op_name}` | {len(op_plans)} | {mean_tr:.4f} | {mean_mr:.4f} | {mean_q:.4f} |")
    L.append("")

    # --- Top Singletons by Paired Latency Ratio ----------------------------
    singletons = [
        (pid, s) for pid, s in summ.items()
        if (s["op"] in ("skip", "quant", "compress", "diag") or pid.startswith("singleton:"))
        and s["latency_ratio"]["n"]
    ]
    singletons.sort(key=lambda item: item[1]["latency_ratio"]["mean"])
    L.append("## Top Singletons by Paired Latency Ratio\n")
    L.append("| plan | op | quality | latency ratio (mean) | temp ratio |")
    L.append("|---|---|---:|---:|---:|")
    for pid, s in singletons[:20]:
        tr = s.get("static_temp_ratio", {}).get("mean", float("nan"))
        L.append(f"| `{pid}` | {s['op']} | {s['quality']['mean']:.4f} | "
                 f"{s['latency_ratio']['mean']:.4f} | {tr:.4f} |")
    L.append("")

    if ident is not None and ident["latency_ratio"]["n"]:
        d = ident["latency_ratio"]
        L.append(f"**Drift floor.** The identity plan measured against ITSELF "
                 f"gives latency ratio {d['mean']:.4f} +/- {d['std']:.4f} "
                 f"(range {d['min']:.4f}-{d['max']:.4f}). No candidate ratio "
                 f"inside that band is evidence of anything.\n")
    if extra.get("cold_seq"):
        L.append("## Cold-measurement sequence (fresh process, no warmup)\n")
        L.append("The SAME plan measured repeatedly from a cold process. "
                 "This is how many rounds the campaign must discard: the "
                 "first-round error here is the size of the spurious "
                 "'winner' a round-1 measurement can invent.\n")
        for cs in extra["cold_seq"]:
            seq = cs["latencies"]
            settled = cs["settled"]
            L.append(f"**`{cs['plan_id']}`** (gpu {cs['gpu']}) - "
                     f"{len(seq)} consecutive measurements, ns:\n")
            L.append("| # | " + " | ".join(str(i) for i in
                                           range(len(seq))) + " |")
            L.append("|---|" + "---|" * len(seq))
            L.append("| ns | " + " | ".join(f"{v:.0f}" for v in seq) + " |")
            L.append("| vs settled | " + " | ".join(
                f"{v / settled:.3f}" if settled else "n/a"
                for v in seq) + " |")
            L.append("")
            L.append(f"- settled value (median of the last half): "
                     f"{settled:.0f} ns")
            L.append(f"- FIRST measurement error: {cs['first_err']:+.1%}")
            L.append(f"- rounds until within 2% of settled: "
                     f"{cs['rounds_to_settle']}")
            L.append("")
    if extra.get("noise_floor"):
        L.append("## Quality noise floor\n")
        L.append("| plan | n | quality mean | quality sd | quality range | "
                 "latency mean | latency CV |")
        L.append("|---|---:|---:|---:|---|---:|---:|")
        for nf in extra["noise_floor"]:
            L.append(f"| `{nf['plan_id']}` | {nf['n']} | {nf['mean']:.6f} | "
                     f"{nf['std']:.3g} | {nf['min']:.6f} - {nf['max']:.6f} | "
                     f"{nf['lat_mean']:.0f} ns | {nf['lat_cv']:.2%} |")
        L.append("")
        L.append(f"The {args.walk_steps}-step Adam walk reads a FIXED probe batch and the "
                 "env's fixed initial weights (`env.args`, never re-drawn), "
                 "so this spread is pure execution nondeterminism, not "
                 "data resampling. Read agent 1's ~0 episode-to-episode "
                 "quality autocorrelation against it: a spread far below the "
                 "between-episode variation means the episodes really were "
                 "measuring DIFFERENT plans.\n")
    if extra.get("correction"):
        L.append("## inner=5 vs inner=50 correction factor\n")
        L.append("The campaign published every latency at "
                 "`--latency-inner-reps 5`. Agent B measured that setting "
                 "carrying a systematic amortisation bias against inner=50. "
                 "Same plan, same pulldown, both measured in this session:\n")
        L.append("| plan | pulldown | inner=5 (ns) | inner=50 (ns) | "
                 "in5 / in50 |")
        L.append("|---|---|---:|---:|---:|")
        for c in extra["correction"]:
            L.append(f"| `{c['plan']}` | {c['pulldown']} | {c['in5']:.0f} | "
                     f"{c['in50']:.0f} | {c['in5_over_in50']:.4f} |")
        L.append("")
    if notes:
        L.append("## Archive recovery (points held / recovered / measured)\n")
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
              "compress": "#2ca02c", "skip": "#7f7f7f", "archive": "#ff7f0e",
              "pair": "#9467bd", "stack": "#8c564b"}
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
        if len(summ) <= 30 or op in ("identity", "archive", "pair", "stack"):
            ax.annotate(pid, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=7)
    ax.set_xlabel("measured latency (ns), mean of independent trials")
    ax.set_ylabel("quality (grad_cosine)")
    ax.set_title("Approximation landscape")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[landscape] figure -> {path}", flush=True)



def _cold_sequences(rows):
    """Ordered first-N measurements per noise-floor plan (the cold curve)."""
    out = []
    for pid in sorted({r["plan_id"] for r in rows if r["op"] == "noisefloor"}):
        nf = sorted((r for r in rows if r["plan_id"] == pid),
                    key=lambda r: int(r["trial"]))
        lat = [float(r["latency_ns"]) for r in nf]
        if len(lat) < 4:
            continue
        half = lat[len(lat) // 2:]
        settled = float(np.median(half))
        rts = next((i for i, v in enumerate(lat)
                    if settled and abs(v - settled) / settled <= 0.02),
                   len(lat))
        out.append({
            "plan_id": pid.replace("noisefloor:", ""),
            "gpu": nf[0].get("gpu", "?"), "latencies": lat,
            "settled": settled,
            "first_err": (lat[0] - settled) / settled if settled else 0.0,
            "rounds_to_settle": rts,
        })
    return out

def combined_report(args):
    """Merge every rows_*.csv in --out-dir into ONE cross-config report.

    Rows from different (pulldown, inner_reps) settings are NOT comparable,
    so the plan id is suffixed with its config rather than pooled. The
    inner=5 vs inner=50 pairs are then reported side by side as the
    CORRECTION FACTOR between the campaign's published numbers and honest
    ones."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(args.out_dir, "rows_*.csv"))):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                r.setdefault("pulldown", "unset")
                r.setdefault("inner_reps", "?")
                # A CSV written before 2026-08-30 has no quality_metric
                # column; every one of those was measured under loss_drop.
                # Naming it here is what stops a loss_drop quality column
                # being averaged with a grad_cosine one two lines below.
                _qm = (r.get("quality_metric") or "loss_drop").strip()
                r["quality_metric"] = _qm
                r["_base"] = r["plan_id"]
                r["_cfg"] = (f"pd{r['pulldown']}/in{r['inner_reps']}"
                     f"/gpu{r.get('gpu', '?')}/q{_qm}")
                r["plan_id"] = f"{r['_base']} [{r['_cfg']}]"
                rows.append(r)
    if not rows:
        print(f"[landscape] no rows_*.csv under {args.out_dir}", flush=True)
        return
    print(f"[landscape] merging {len(rows)} rows from "
          f"{len({r['_cfg'] for r in rows})} configs", flush=True)

    summ = summarise([r for r in rows if r["op"] != "noisefloor"])
    extra = {}
    floors = []
    for pid in sorted({r["plan_id"] for r in rows if r["op"] == "noisefloor"}):
        nf = [r for r in rows if r["plan_id"] == pid]
        if len(nf) < 2:
            continue
        q = np.array([float(r["quality"]) for r in nf])
        la = np.array([float(r["latency_ns"]) for r in nf])
        floors.append({"plan_id": pid.replace("noisefloor:", ""),
                       "n": int(q.size), "mean": float(q.mean()),
                       "std": float(q.std(ddof=1)), "min": float(q.min()),
                       "max": float(q.max()), "lat_mean": float(la.mean()),
                       "lat_std": float(la.std(ddof=1)),
                       "lat_cv": float(la.std(ddof=1) / la.mean())
                       if la.mean() else 0.0})
    if floors:
        extra["noise_floor"] = floors
    cs = _cold_sequences(rows)
    if cs:
        extra["cold_seq"] = cs

    # inner=5 vs inner=50 correction factor, per base plan, SAME pulldown.
    corr = []
    base_cfg = {}
    for pid, sm in summ.items():
        b, cfg = pid.rsplit(" [", 1)
        base_cfg[(b, cfg.rstrip("]"))] = sm
    for (b, cfg), sm in sorted(base_cfg.items()):
        if not cfg.endswith("/in5"):
            continue
        pd = cfg.split("/")[0]
        other = base_cfg.get((b, f"{pd}/in50"))
        if other is None:
            continue
        a, c = sm["latency_ns"]["mean"], other["latency_ns"]["mean"]
        if a > 0 and c > 0:
            corr.append({"plan": b, "pulldown": pd, "in5": a, "in50": c,
                         "in5_over_in50": a / c})
    if corr:
        extra["correction"] = corr

    notes = []
    cen = sorted(glob.glob(os.path.join(args.out_dir, "archive_census*.json")))
    for f in cen:
        try:
            for c in json.load(open(f)):
                notes.append(
                    f"`{c['run']}`: front holds {c['held']}, recovered "
                    f"{c['recovered']}, measured {c['measured']}"
                    + (f" -- **NOT RECOVERED**: {c['reason']}"
                       if c.get("reason") else ""))
        except Exception:
            pass

    md = os.path.join(args.out_dir, "summary_COMBINED.md")
    write_markdown(md, summ, notes, args, extra)
    with open(os.path.join(args.out_dir, "summary_COMBINED.json"), "w") as fh:
        json.dump({"summary": summ, "extra": extra, "notes": notes}, fh,
                  indent=2)
    print(f"[landscape] combined summary -> {md}", flush=True)
    if not args.no_figure:
        # Figure shows the PRIMARY config only; mixing pulldown settings on
        # one axis is exactly the comparison this report exists to prevent.
        prim = {k: v for k, v in summ.items() if "pd1/in50" in k}
        write_figure(os.path.join(args.out_dir, "landscape_COMBINED.png"),
                     prim or summ,
                     {k for k in (prim or summ) if k.startswith("arch:")})


# ---------------------------------------------------------------------------
def main():
    args = ARGS
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    if args.shard:
        tag += f"_s{args.shard.split('/')[0]}"
    csv_path = os.path.join(args.out_dir, f"rows{tag}.csv")
    md_path = os.path.join(args.out_dir, f"summary{tag}.md")
    fig_path = os.path.join(args.out_dir, f"landscape{tag}.png")
    plans_path = os.path.join(args.out_dir, f"plans{tag}.json")

    if args.report_only:
        combined_report(args)
        return

    env, eval_samples, _cj = build_env(args)
    if args.order == "markowitz":
        order = markowitz_order(env)
    else:
        order = rev_order(env)
    print(f"[landscape] {len(order)} valid vertices; {args.order} order "
          f"{order[:6].tolist()}...{order[-3:].tolist()}", flush=True)

    INV = None
    if args.face_inventory or args.inventory_only or args.singleton_sweep \
            or args.skip_face:
        capture = bool(args.singleton_sweep)
        INV = face_inventory(env, order, capture_tensors=capture)
        ipath = os.path.join(args.out_dir, f"face_inventory{tag}.json")
        inv_clean = [
            {k: v for k, v in entry.items() if k != "tensors"}
            for entry in INV
        ]
        with open(ipath, "w") as fh:
            json.dump(inv_clean, fh, indent=2)
        print(f"[landscape] face inventory: {len(INV)} live faces -> {ipath}",
              flush=True)
        if args.inventory_only:
            return

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
    print(f"[landscape] total LIVE faces on the {args.order} order: {total_live} "
          f"(per-vertex max {max(ident['per_vertex_faces'] or [0])})",
          flush=True)

    if not args.singleton_sweep:
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

    # --- the MINIMAL plan: skip exactly the named face(s), nothing else -----
    if args.skip_face:
        tg = []
        for spec in args.skip_face:
            k, f = spec.split(":")
            tg.append((int(k), int(f)))
        pid = "skiponly:" + ",".join(f"{k}.{f}" for k, f in tg)
        pl = build_skip_only_plan(env, order, tg, INV)
        pl["op"], pl["budget"] = "skiponly", str(len(tg))
        plans[pid] = pl
        plan_orders[pid] = order
        print(f"[landscape] minimal plan {pid}: skips {tg}", flush=True)

    # --- SINGLETON SWEEP: exhaustive singletons -----------------------------
    if args.singleton_sweep:
        stride = max(1, int(getattr(args, "sweep_stride", 1)))
        picked_inv = INV[::stride] if stride > 1 else INV
        if stride > 1:
            print(f"[landscape] singleton sweep: subsampling {len(picked_inv)} of {len(INV)} faces (stride {stride})",
                  flush=True)
        s_plans, s_orders = build_singleton_sweep_plans(env, order, picked_inv)
        print(f"[landscape] singleton sweep: {len(s_plans)} singleton plans generated across live faces",
              flush=True)
        plans.update(s_plans)
        plan_orders.update(s_orders)

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

    # ---- EVERY point of EVERY archive (coordinator item 2) ---------------
    # Recovery is attempted for ALL points and reported as recovered/held;
    # MEASUREMENT is capped at --archive-max-measure per run, taken from the
    # best-latency end -- that is the part of the front the owner's question
    # ("re-measure the Pareto frontier unbiased") is actually about.
    archive_census = []
    if args.archive_all_points:
        for lab, path in sorted(archive_paths.items()):
            if not os.path.exists(path):
                archive_census.append({"run": lab, "held": 0, "recovered": 0,
                                       "measured": 0,
                                       "reason": f"no such file: {path}"})
                continue
            try:
                with open(path) as fh:
                    doc = json.load(fh)
            except Exception as exc:
                archive_census.append({"run": lab, "held": 0, "recovered": 0,
                                       "measured": 0,
                                       "reason": f"unparseable: {exc!r}"})
                continue
            front = doc.get("front") or []
            objs = doc.get("objectives") or []
            lat_name = ("latency" if "latency" in objs
                        else ("latency_ns" if "latency_ns" in objs else None))
            recovered, failures = [], {}
            for i, pt in enumerate(front):
                pl, note = _point_to_plan(pt, env, order, path)
                if pl is None:
                    key = str(note).split(":")[-1].strip()[:80]
                    failures[key] = failures.get(key, 0) + 1
                    continue
                lat = (-float(pt.get("obj", {}).get(lat_name, 0.0))
                       if lat_name else 0.0)
                recovered.append((lat, i, pl, pt.get("obj", {})))
            # best-latency first; points with no latency sort last
            recovered.sort(key=lambda t: (t[0] <= 0, t[0]))
            take = recovered[:max(0, int(args.archive_max_measure))]
            for lat, i, pl, obj in take:
                pid = f"arch:{lab}:{i}"
                pl["op"], pl["budget"] = "archive", f"{lab}#{i}"
                pl["archive_obj"] = obj
                plans[pid] = pl
                plan_orders[pid] = pl["order"]
                archive_ids.add(pid)
            archive_census.append({
                "run": lab, "held": len(front), "recovered": len(recovered),
                "measured": len(take),
                "reason": ("" if not failures else
                           "; ".join(f"{v}x {k}" for k, v in failures.items())),
            })
            print(f"[landscape] archive {lab}: held {len(front)}, "
                  f"recovered {len(recovered)}, measuring {len(take)}"
                  + (f" | NOT RECOVERED: "
                     + "; ".join(f"{v}x {k}" for k, v in failures.items())
                     if failures else ""), flush=True)
        with open(os.path.join(args.out_dir, f"archive_census{tag}.json"),
                  "w") as fh:
            json.dump(archive_census, fh, indent=2)
        for c in archive_census:
            notes.append(
                f"`{c['run']}`: front holds {c['held']}, recovered "
                f"{c['recovered']}, measured {c['measured']}"
                + (f" -- **NOT RECOVERED**: {c['reason']}"
                   if c["reason"] else ""))

    def _dump_manifest(plans_dict, path, inv):
        _bykf = {(e["k"], e["f"]): e for e in (inv or [])}

        def _named(p):
            if "wires" not in p:
                return None
            out = []
            for w in p.get("wires", []) or []:
                e = _bykf.get((w["k"], w["f"]))
                out.append({**w, **({"vertex": e["vertex"], "key": e["key"],
                                     "prim": e["prim"], "out": e["out"],
                                     "in": e["in"]} if e else
                                    {"vertex": None, "note": "not in "
                                     "inventory (enumerated on exact prefix)"})})
            return out

        with open(path, "w") as fh:
            json.dump({pid: {"op": p["op"], "budget": p["budget"],
                             "n_faces_approx": p["n_faces_approx"],
                             "n_slot_rows": p["n_slot_rows"],
                             "total_live_faces": p["total_live_faces"],
                             "per_vertex_faces": p["per_vertex_faces"],
                             "wires": _named(p)}
                       for pid, p in plans_dict.items()}, fh, indent=2)

    if args.shard:
        parts = [int(x) for x in args.shard.split("/")]
        shard_idx, num_shards = parts[0], parts[1]
        ident = plans.get("identity")
        non_ident = [(pid, plans[pid]) for pid in plans if pid != "identity"]
        chunk_size = math.ceil(len(non_ident) / num_shards)
        shard_items = non_ident[shard_idx * chunk_size : (shard_idx + 1) * chunk_size]
        new_plans = {}
        if ident is not None:
            new_plans["identity"] = ident
        new_plans.update(shard_items)
        plans = new_plans
        plan_orders = {pid: plan_orders[pid] for pid in plans}

    _dump_manifest(plans, plans_path, INV)
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
    stamp = config_stamp(args)
    print(f"[landscape] CONFIG {stamp}", flush=True)
    t_start = time.perf_counter()
    stop = False

    def _budget_left():
        return (args.max_seconds <= 0
                or (time.perf_counter() - t_start) < args.max_seconds)

    ident_plan = plans["identity"]

    def _execute_plans(sub_plans, sub_orders):
        nonlocal stop
        # WARMUP
        for w in range(args.warmup_trials):
            for pid, plan in sub_plans.items():
                key = (pid, -1 - w, "warmup")
                if key in done or not _budget_left():
                    continue
                try:
                    m = measure(env, eval_samples, sub_orders[pid], plan)
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
                    **{k: m[k] for k in ("latency_ns", "peak_memory", "static_temp",
                                         "quality", "frob_residual", "applied", "skipped",
                                         "applied_detail", "wall_s")},
                    **stamp,
                }
                append_row(csv_path, row)
                done[key] = row
                print(f"[landscape] warmup {pid:22s} lat={m['latency_ns']:.0f}ns "
                      f"applied={m['applied']} skipped={m['skipped']} "
                      f"({m['wall_s']:.1f}s)", flush=True)

        # Reps (PAIRED candidate vs reference)
        for trial in range(args.reps):
            for pid, plan in sub_plans.items():
                if stop:
                    break
                porder = sub_orders[pid]
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
                        **{k: m[k] for k in ("latency_ns", "peak_memory", "static_temp",
                                             "quality", "frob_residual",
                                             "applied", "skipped",
                                             "applied_detail", "wall_s")},
                        **stamp,
                    }
                    append_row(csv_path, row)
                    done[key] = row
                    print(f"[landscape] {pid:22s} t{trial} {role:9s} "
                          f"lat={m['latency_ns']:.0f}ns "
                          f"mem={m['peak_memory']:.3g} stemp={m['static_temp']:.3g} "
                          f"q={m['quality']:.4f} ({m['wall_s']:.1f}s)", flush=True)

    # 1. Execute initial plans (identity + singletons/ladder/archive)
    _execute_plans(plans, plan_orders)

    # 2. Draw and execute F* stacks/pairs post-singleton
    if (args.stack_ladder or args.pair_samples > 0) and not stop and not args.shard:
        f_star_items = []
        for pid, pl in plans.items():
            if not pid.startswith("singleton:"):
                continue
            q_vals = [
                float(done[(pid, t, "candidate")]["quality"])
                for t in range(args.reps)
                if (pid, t, "candidate") in done
            ]
            if q_vals and np.mean(q_vals) >= args.f_star_threshold:
                f_star_items.append((pid, pl))
        print(f"[landscape] F* set: {len(f_star_items)} singletons meet quality >= {args.f_star_threshold:.2f}",
              flush=True)
        if f_star_items:
            stack_plans, stack_orders = build_f_star_stacks(
                env, order, f_star_items,
                stack_ladder=args.stack_ladder,
                stack_samples=args.stack_samples,
                pair_samples=args.pair_samples,
            )
            print(f"[landscape] generated {len(stack_plans)} stack/pair plans from F*", flush=True)
            if stack_plans:
                plans.update(stack_plans)
                plan_orders.update(stack_orders)
                _dump_manifest(plans, plans_path, INV)
                _execute_plans(stack_plans, stack_orders)

    # --- quality noise floor ------------------------------------------------
    nf_pids = [x.strip() for x in args.noise_floor_plan.split(",")
               if x.strip()]
    for nf_pid in nf_pids:
        if nf_pid not in plans:
            print(f"[landscape] noise-floor plan {nf_pid!r} is not in this "
                  f"phase's plan set -- skipped", flush=True)
            continue
        if args.noise_floor_reps <= 0:
            continue
        for i in range(args.noise_floor_reps):
            key = (f"noisefloor:{nf_pid}", i, "candidate")
            if key in done:
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
                **{k: m[k] for k in ("latency_ns", "peak_memory", "static_temp",
                                     "quality", "frob_residual", "applied", "skipped",
                                     "applied_detail", "wall_s")},
                **stamp,
            }
            append_row(csv_path, row)
            done[key] = row
            print(f"[landscape] noise-floor[{nf_pid}] {i}: "
                  f"q={m['quality']:.6f} lat={m['latency_ns']:.0f}ns",
                  flush=True)

    # --- report -------------------------------------------------------------
    all_rows = list(load_done(csv_path).values())
    # Warmup rows ARE passed in -- summarise() uses them only for the
    # applied/skipped rule counters (see its docstring) and excludes them
    # from every measured aggregate.
    summ = summarise([r for r in all_rows if r["op"] != "noisefloor"])
    extra = {}
    floors = []
    for pid in sorted({r["plan_id"] for r in all_rows
                       if r["op"] == "noisefloor"}):
        nf = [r for r in all_rows if r["plan_id"] == pid]
        if len(nf) < 2:
            continue
        q = np.array([float(r["quality"]) for r in nf])
        la = np.array([float(r["latency_ns"]) for r in nf])
        floors.append({
            "plan_id": pid.replace("noisefloor:", ""), "n": int(q.size),
            "mean": float(q.mean()), "std": float(q.std(ddof=1)),
            "min": float(q.min()), "max": float(q.max()),
            "lat_mean": float(la.mean()), "lat_std": float(la.std(ddof=1)),
            "lat_cv": float(la.std(ddof=1) / la.mean()) if la.mean() else 0.0,
        })
    if floors:
        extra["noise_floor"] = floors
    cs = _cold_sequences(all_rows)
    if cs:
        extra["cold_seq"] = cs
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
