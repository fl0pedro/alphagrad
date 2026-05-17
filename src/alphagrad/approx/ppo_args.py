"""JAX-free CLI argparser for `ppo_ray.py`.

Subset of `ppo.make_argparser`'s arguments — only the knobs the
first-cut Ray PPO trainer (`ppo_ray_worker.PPORayWorker`) actually
reads. The full single-process trainer in `ppo.py` carries ~80 args;
duplicating all of them here would just be drift bait. Add to this
file as the Ray trainer grows.

Kept JAX-free so the driver in `ppo_ray.py` can construct the parser
without triggering JAX init.
"""

from __future__ import annotations

import argparse


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ray-actor PPO trainer (first-cut) for vertex elimination.",
    )

    # Run / logging
    p.add_argument("--name", type=str, default="approx-ppo-ray")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument(
        "--wandb", type=str, default="offline",
        choices=["disabled", "offline", "online"],
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument(
        "--exec-on-gpu", action="store_true",
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

    # Environment / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument(
        "--cmp-type", type=str, default="flops",
        choices=["graphax", "flops", "latency"],
    )
    p.add_argument(
        "--mem-type", type=str, default="peak_memory",
        choices=["graphax", "bytes_accessed", "peak_memory"],
    )
    p.add_argument(
        "--rewards", nargs="+", type=str,
        default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"],
    )
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    # Both quality channels (cosine_sim direction + frob magnitude) are
    # now active rewards by default. cossim is gated by ``acc`` in
    # ``--rewards`` (weight = 1.0 when present). frob has its own
    # lambda; bumped from the prior 0.0 default so the Jacobian-error
    # magnitude contributes to the gradient alongside cossim. The
    # Lagrangian range constraint stays on cossim only.
    p.add_argument("--lambda-frob", type=float, default=1.0)
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--terminal-rewards-only", action="store_true")
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--dataset", type=str, default="none")
    p.add_argument("--dataset-size", type=int, default=-1)

    # Rollout
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)

    # Advantage normalisation strategy. Default `gdpo` implements the
    # per-channel z-score → priority-weighted sum → batch-norm pipeline
    # from the NVIDIA GDPO paper (arXiv:2601.05242). The legacy `scalar`
    # path keeps the previous behaviour: scalarise per-channel rewards
    # into a single stream, apply symlog inside GAE, then global
    # advantage normalisation. Calibration is auto-skipped under
    # `gdpo` because per-minibatch z-scoring subsumes magnitude
    # rescaling. Keep `scalar` reachable for one cycle so we can A/B
    # against the pre-refactor baseline on the same rollout buffer.
    p.add_argument(
        "--advantage-norm", type=str, default="gdpo",
        choices=["gdpo", "scalar"],
        help="Advantage normalisation pipeline. `gdpo` (default): "
             "per-channel GAE + per-minibatch z-score per channel + "
             "priority-weighted sum + batch-norm. `scalar` (legacy): "
             "scalarise rewards via weights, symlog inside GAE, global "
             "advantage z-score over the whole rollout. Under `gdpo` "
             "the calibration phase is skipped (z-scoring is "
             "self-calibrating).",
    )
    p.add_argument(
        "--calibration-statistic", type=str, default="iqr",
        choices=["iqr", "mean_abs", "std"],
        help="Per-channel dispersion statistic used by the pre-training "
             "calibration phase to rescale reward weights. `iqr` "
             "(default, recommended): symlog-space IQR / 1.349 — robust "
             "to heavy-tail outliers. `mean_abs` (legacy): "
             "``|symlog(mean)|`` — normalises by the channel BIAS rather "
             "than its spread; kept for A/B regression. `std`: "
             "symlog-space std via the IQR proxy. Only relevant when "
             "calibration runs (advantage-norm=scalar AND "
             "calibrate-steps>0).",
    )

    # PPO hyperparameters
    p.add_argument("--ppo-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Entropy bonus coefficient in the PPO loss. Default bumped "
             "from 0.01 to 0.05 because the dynamic-substeps action "
             "space has 6 categorical heads (vertex / op_type / i / j / "
             "factor / quant) with joint entropy averaged across them — "
             "0.01 was insufficient (observed entropy collapsing 1.4 → "
             "0.4 by ep 13 in run-9s554e3x, where the per-head bonus "
             "drowns in value loss ≈ 4 and PPO clip loss ≈ 0.17).",
    )
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--discount", type=float, default=0.99)

    # Lagrangian (Stage F)
    p.add_argument(
        "--lagrangian-constraint", nargs="*", type=str, default=[],
        help="Inequality constraints of the form NAME>=THRESH or "
        "NAME<=THRESH, where NAME is one of the env reward channel "
        "names (muls_adds_fmas / flops / latency_ns / max_io_sum / "
        "bytes_accessed / peak_memory / cosine_sim / frob_residual). "
        "Mean per-step violations push the policy via dual-ascent on "
        "per-constraint multipliers. Symlog-friendly: cost-family "
        "constraints (everything except cosine_sim) are evaluated in "
        "symlog space so multipliers live on a single scale.",
    )
    p.add_argument(
        "--lagrangian-lr", type=float, default=1e-3,
        help="Dual-ascent step size on the Lagrangian multipliers.",
    )

    # Conditioned-reward gating (the GDPO paper's "easy reward on hard
    # reward" trick). Hard alternative to the Lagrangian penalty: when
    # the gate fails the easier channel is zeroed for that transition,
    # so the policy can't hill-climb the easy signal without first
    # satisfying the harder one. Complementary to the Lagrangian, NOT
    # a replacement.
    p.add_argument(
        "--reward-condition", nargs="*", type=str, default=[],
        help="Conditioned-reward gates of the form "
             "``<easier>:<harder>>=<threshold>`` or "
             "``<easier>:<harder><=<threshold>``. Channel names are "
             "those in the env's reward vector (muls_adds_fmas / flops "
             "/ latency_ns / max_io_sum / bytes_accessed / peak_memory "
             "/ cosine_sim / frob_residual). At every transition where "
             "``op(harder, threshold)`` is False, the easier channel's "
             "reward is set to zero before GAE. Used to force the "
             "policy to maximise the harder objective first — e.g. "
             "``--reward-condition 'flops:cosine_sim>=0.8'`` zeros the "
             "FLOPs reward unless the resulting Jacobian agrees with "
             "the exact one. Multiple gates compose AND-wise.",
    )

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-eps", type=float, default=1e-8)

    # Dynamic substeps (Phase C). When --dynamic-substeps is on the agent
    # emits typed (op_type, i, j, factor) micro-actions per vertex which
    # env.micro_actions_to_rule_specs_jax translates into the legacy
    # rule_specs the env consumes. The first cut here is restricted to
    # ``max_substeps == 1`` — a single DIAG-or-END per vertex. Multi-
    # substep (sequence) support is a follow-up.
    p.add_argument(
        "--dynamic-substeps", action="store_true",
        help="Enable the typed micro-action policy (1 substep per vertex).",
    )
    p.add_argument(
        "--factors", type=str, default="-1,2,3,4",
        help="Comma-separated DIAG factor choices the policy picks from. "
        "``-1`` resolves to ``gcd(n_i, n_j)`` per env edge; everything "
        "else is a literal divisor. When --variant is set, this is "
        "OVERRIDDEN by the variant's factor table at init (the agent is "
        "always built with the union factor table so curriculum stage "
        "transitions can mask via the policy logits rather than "
        "rebuilding the agent).",
    )

    # Variant / curriculum (mirrors mu0_args.py). PPO builds the agent
    # with the FULL action footprint (op_type ∈ {DIAG, COMPRESS, QUANT,
    # END}, full factor table) and masks per-stage in the act_step +
    # loss_fn — same trick MuZero uses for its prior-gating curriculum
    # so the policy weights survive stage transitions. ``full_curriculum``
    # is the recommended default: it auto-expands an empty --curriculum
    # to ``diag_gcd:N/3, diag_factor:N/3, full:N/3``.
    p.add_argument(
        "--variant",
        type=str,
        default="custom",
        choices=[
            "custom", "ve_only", "diag_gcd", "diag_factor",
            "compress", "quantize", "full", "full_curriculum",
        ],
        help="Pre-canned action-space restriction. ``custom`` honours "
             "--factors / --dynamic-substeps directly. ``ve_only`` "
             "freezes op_type=END (pure vertex elimination). "
             "``diag_gcd`` allows DIAG with factor=-1 only. "
             "``diag_factor`` adds factors 2,3,4,8,16. ``compress`` / "
             "``quantize`` enable those op_types alongside DIAG. "
             "``full`` opens everything. ``full_curriculum`` is sugar "
             "for ``full`` plus the default 3-stage curriculum.",
    )
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help="Curriculum of variants in sequence. Format: "
             "``stage1:N1,stage2:N2,...``. Empty = single training run "
             "on --variant (or the default 3-stage curriculum when "
             "--variant=full_curriculum and --curriculum is empty). "
             "Total episodes across stages must equal --episodes.",
    )

    # Model
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--embd-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--policy-dims", type=str, default="128",
        help="Comma-separated MLP hidden dims for the vertex policy head.",
    )
    p.add_argument(
        "--value-dims", type=str, default="128",
        help="Comma-separated MLP hidden dims for the value head.",
    )

    return p
