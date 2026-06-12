"""JAX-free CLI argparser for the mu0 / mu0_ray trainers.

Lives outside the JAX import chain so ``mu0_ray.py`` (the Ray driver) can
construct the parser without importing anything that triggers JAX init.

``SCHEDULES`` is duplicated from
:mod:`alphagrad.approx.common.schedules` — the source is a tuple of three
strings; importing it directly would pull the whole ``alphagrad.approx.common``
package and therefore JAX into the driver process.
"""

from __future__ import annotations

import argparse

from alphagrad.approx.variants import VARIANT_PRESETS


# Duplicate of alphagrad.approx.common.schedules.SCHEDULES.
SCHEDULES = ("constant", "linear", "cosine")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MuZero trainer for vertex-elimination (hierarchical MCTS).",
    )
    # Run / logging
    p.add_argument("--name", type=str, default="approx-muzero")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument("--wandb", type=str, default="offline",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument(
        "--exec-on-gpu",
        action="store_true",
        help="Pin training to GPU 0 and the env eval callback to GPU 1.",
    )
    p.add_argument(
        "--best-sequences-json", type=str, default="",
        help="Path to write the running per-channel + overall best "
        "sequences as JSON. Empty (default) → derive from the wandb "
        "run dir as ``<wandb_dir>/best_sequences.json``. The file holds "
        "one entry per reward channel + an ``overall`` entry, each "
        "with the full (a_i, b_i, c_i, r_i) reward tuple and the "
        "action sequence that produced it. Updated atomically every "
        "``--best-sequences-every`` episodes and at run end.",
    )
    p.add_argument(
        "--best-sequences-every", type=int, default=10,
        help="Cadence (in episodes) for writing the best-sequences "
        "JSON + logging the wandb table snapshot. Set to 0 to only "
        "write at run end.",
    )
    p.add_argument(
        "--calibration-statistic", type=str, default="iqr",
        choices=["iqr", "mean_abs", "std"],
        help="Per-channel dispersion statistic for pre-training "
             "calibration. `iqr` (default, recommended): symlog-space "
             "IQR / 1.349, σ-equivalent robust to outliers. `mean_abs` "
             "(legacy): ``|symlog(mean)|`` — bias proxy, kept for A/B. "
             "`std`: symlog-space std via the IQR proxy. MuZero uses "
             "calibration to balance the scalarised reward stream that "
             "feeds the value head; with raw cmp-channel magnitudes ~ "
             "1e10, an uncalibrated value loss can hit 1e20+ on the "
             "first gradient step.",
    )

    # Environment / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument("--disable-sparsification", action="store_true")
    p.add_argument("--cmp-type", type=str, default="flops",
                   choices=["graphax", "flops", "latency"])
    p.add_argument("--mem-type", type=str, default="peak_memory",
                   choices=["graphax", "bytes_accessed", "peak_memory"])
    p.add_argument("--rewards", nargs="+", type=str,
                   default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"])
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    # frob (Jacobian magnitude residual) is now active by default
    # alongside cossim (direction). Bumped 0.0 → 1.0 so both quality
    # channels contribute to the gradient; the Lagrangian range
    # constraint stays on cossim only.
    p.add_argument("--lambda-frob", type=float, default=1.0)
    p.add_argument(
        "--measure-latency", action="store_true",
        help="Time the compiled approx fn N times per env step to "
             "populate the latency reward component (N = --latency-samples).",
    )
    p.add_argument(
        "--latency-samples", type=int, default=1,
        help="Per-step latency sample count when --measure-latency is on. "
        "Default 1 (single noisy point, denoised by per-episode averaging "
        "over ~12*num_envs calls). >=8 enables top-quartile-mean smoothing.",
    )
    p.add_argument(
        "--terminal-rewards-only", action="store_true",
        help="Compute the env's reward vector only at the final "
             "elimination step; intermediate steps return zeros.",
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # Variant / curriculum (mirrors ppo.py)
    p.add_argument(
        "--variant",
        type=str,
        default="custom",
        choices=list(VARIANT_PRESETS.keys()),
        help=(
            "Pre-canned configuration mapping to --factors / --max-rules / "
            "--pin-rules-to-exact. `custom` honours your explicit flags. "
            "`ve_only` freezes the rule head. `diag_gcd` / `diag_factor` / "
            "`compress` / `full` enable progressively richer action sets. "
            "`full_curriculum` is sugar for `full` plus an auto-generated "
            "curriculum diag_gcd → diag_factor → full when --curriculum "
            "is empty."
        ),
    )
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help=(
            "Curriculum of variants to run in sequence. Format: "
            "`stage1:N1,stage2:N2,...`. Empty = single training run on "
            "--variant (or the default 3-stage curriculum when "
            "--variant=full_curriculum)."
        ),
    )
    p.add_argument(
        "--curriculum-warmup-frac",
        type=float,
        default=0.3,
        help="Fraction of each curriculum stage's optimizer steps spent "
             "in the cosine LR warm-up before the exponential-decay phase.",
    )
    p.add_argument(
        "--curriculum-existing-head-mult",
        type=float,
        default=0.3,
        help="LR multiplier applied to heads introduced in an earlier "
             "curriculum stage.",
    )

    # Loss mode (CLI parity with ppo.py — mu0 already collapses rewards
    # to a single scalar per step, so the flag is accepted for compat).
    p.add_argument(
        "--loss-mode",
        type=str,
        default="scalar",
        choices=["multi_head", "scalar"],
        help="``scalar`` (mu0 default): single-channel reward = "
             "``sum_i(reward_weights[i] * symlog(reward_vec[i]))``. "
             "``multi_head`` accepted for ppo.py CLI parity; the "
             "MuZero value head is scalar by construction so the flag "
             "currently has no behavioural effect.",
    )

    # Dynamic-substeps (CLI parity). mu0's unified action space already
    # handles per-decision typed action sequences; the flag is accepted
    # so the same CLI invocation works for both trainers.
    p.add_argument(
        "--dynamic-substeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for ppo.py CLI parity. mu0's hierarchical MCTS "
             "already encodes the (vertex, pair, factor) micro-action "
             "sequence via per-depth masking of the unified action "
             "space; this flag does not alter the action layout.",
    )
    p.add_argument(
        "--max-substeps",
        type=int,
        default=8,
        help="Accepted for ppo.py CLI parity (no effect in mu0).",
    )
    p.add_argument(
        "--max-axis-size",
        type=int,
        default=1024,
        help="Accepted for ppo.py CLI parity (no effect in mu0).",
    )
    p.add_argument(
        "--allow-compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for ppo.py CLI parity (no effect in mu0 today; "
             "the unified action space doesn't yet emit COMPRESS).",
    )

    # Network architecture
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=64)
    p.add_argument("--op-embd-dim", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")
    p.add_argument(
        "--cache-encoding",
        action="store_true",
        help="Encode the initial state once at episode start and reuse the "
             "resulting latent for every MCTS step / loss-unroll within the "
             "episode. The dynamics network alone evolves the latent across "
             "real elimination steps.",
    )
    p.add_argument(
        "--set-transformer-agg",
        action="store_true",
        help="Aggregate per-vertex features across the calibration samples "
             "with a learned Set Transformer instead of a simple mean.",
    )
    p.add_argument(
        "--pin-rules-to-exact",
        action="store_true",
        help="Pin the rule head to exact-AD (every slot = STOP, factor 0). "
             "In mu0 this masks the pair-slot priors so MCTS only ever "
             "selects PAIR_STOP, leaving gradient to the vertex head.",
    )

    # Hierarchical action layout
    p.add_argument("--max-rules", type=int, default=1,
                   help="Rule-slot count per chosen vertex.")
    p.add_argument("--factors", type=str, default="-1,1,2,4",
                   help="Comma-separated factor values for the rule decoder.")

    # MuZero
    p.add_argument("--num-simulations", type=int, default=25)
    p.add_argument("--unroll-steps", type=int, default=2)
    p.add_argument("--dirichlet-fraction", type=float, default=0.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-final", type=float, default=0.1)
    p.add_argument("--temperature-schedule", type=str, default="constant",
                   choices=SCHEDULES)
    p.add_argument("--reward-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)

    # MCTS variant. Three modes, all sharing the same agent / loss / replay:
    #   * ``hierarchical`` (default): two-stage MCTS — a shallow
    #     ``mctx.muzero_policy`` over the vertex choice (PUCT exploration,
    #     ``max_depth=1``), followed by ``mctx.gumbel_muzero_policy`` over
    #     the rule sub-sequence rooted at the post-vertex latent. The
    #     vertex head gets the strategic explore/exploit signal, while the
    #     deeper pair/factor branch uses Gumbel sequential halving (better
    #     under low simulation budgets than UCB on a wide action space).
    #   * ``gumbel``: single ``mctx.gumbel_muzero_policy`` spanning the
    #     full ``1 + 2·max_rules`` hierarchical tree.
    #   * ``sampled``: standard ``mctx.muzero_policy`` with the root
    #     restricted to ``--sampled-k`` actions drawn from the prior
    #     (root-only approximation of Hubert et al. 2021 Sampled MuZero;
    #     the per-node sampling + importance-weighted improvement of the
    #     paper requires modifying mctx's search loop and is left for
    #     follow-up).
    p.add_argument(
        "--mcts-mode",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "gumbel", "sampled"],
        help="Which mctx policy to run inside each real elimination step.",
    )
    p.add_argument(
        "--gumbel-max-considered",
        type=int,
        default=16,
        help="``max_num_considered_actions`` for the Gumbel root selection. "
             "Used by ``--mcts-mode gumbel`` and by the sub-rule call in "
             "``--mcts-mode hierarchical``.",
    )
    p.add_argument(
        "--gumbel-scale",
        type=float,
        default=1.0,
        help="Scale for the Gumbel noise. 0 = deterministic argmax "
             "(useful at eval time on perfect-information games).",
    )
    p.add_argument(
        "--sampled-k",
        type=int,
        default=16,
        help="Number of actions sampled from the root prior under "
             "``--mcts-mode sampled``. Reduces MCTS branching at the cost "
             "of forgoing actions outside the top-K.",
    )

    # Optimisation
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Parallel rollout envs. -1 = os.cpu_count() "
                        "(or 16 for Vmapped examples).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--head-init-scale", type=float, default=0.1)
    p.add_argument(
        "--lr-decay-min-mult",
        type=float,
        default=0.1,
        help="Cosine decay floor as a multiple of the initial learning rate.",
    )

    # Replay buffer
    p.add_argument("--replay-buffer-size", type=int, default=0)
    p.add_argument("--replay-batch-size", type=int, default=0)
    p.add_argument("--replay-warmup", type=int, default=1)
    p.add_argument("--replay-fresh-fraction", type=float, default=0.0)
    p.add_argument("--replay-priority-alpha", type=float, default=0.0)
    p.add_argument("--replay-checkpoint-path", type=str, default="")
    p.add_argument("--replay-checkpoint-every", type=int, default=10)

    # Preference conditioning
    p.add_argument("--preference-conditioned", action="store_true")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0)

    # Stage F Lagrangian
    p.add_argument(
        "--lagrangian-constraint",
        action="append",
        default=[],
        metavar="NAME>=THRESH",
        help="Hard-constraint of the form ``<reward_name>>=<threshold>`` (or "
             "``<=`` for ceilings). May be repeated.",
    )
    p.add_argument(
        "--lagrangian-lr",
        type=float,
        default=1e-2,
        help="Dual-ascent step size on the Lagrangian multipliers.",
    )
    p.add_argument(
        "--cosine-lower-bound",
        type=float,
        default=0.8,
        help="Floor on cosine_sim enforced via a Lagrangian multiplier. "
             "Pass 0.0 to disable.",
    )
    p.add_argument(
        "--cosine-upper-bound",
        type=float,
        default=0.9,
        help="Ceiling on cosine_sim enforced via a Lagrangian multiplier. "
             "Pass 1.0 to disable.",
    )
    p.add_argument(
        "--anti-degeneracy",
        choices=("none", "delta_ceiling", "corridor"),
        default="none",
        help="High-level mechanism to prevent the policy from collapsing "
             "to cossim=1.0. See ppo_args.py for full description.",
    )
    p.add_argument(
        "--anti-degeneracy-delta", type=float, default=0.01,
        help="δ for ``--anti-degeneracy delta_ceiling``.",
    )

    # Stage G calibration
    #
    # NOTE: ``--calibrate-steps`` is now declared in
    # :func:`alphagrad.approx.common.ray_runtime.add_common_ray_args` so
    # both ``mu0_ray.py`` and ``ppo_ray.py`` share the default. The
    # single-process ``mu0.py`` trainer (which doesn't call
    # ``add_common_ray_args``) still needs its own declaration so its
    # CLI surface stays unchanged.
    if not any(a.dest == "calibrate_steps" for a in p._actions):
        p.add_argument(
            "--calibrate-steps",
            type=int,
            default=0,
            help="Pre-training reward-scale calibration: run K rollouts of "
                 "the un-trained agent, measure mean ``|symlog(reward)|`` "
                 "per channel, and rescale ``reward_weights`` by 1/mean_abs "
                 "so the wide-magnitude channels contribute on a comparable "
                 "scale.",
        )
    p.add_argument(
        "--calibrate-lr",
        type=float,
        default=1e-3,
        help="Accepted for ppo.py CLI parity (mu0 calibration is gradient-free).",
    )

    return p
