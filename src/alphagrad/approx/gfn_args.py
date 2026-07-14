"""JAX-free CLI argparser for the gfn_ray trainer.

Lives outside the JAX import chain so ``gfn_ray.py`` (the Ray driver) can
construct the parser without importing anything that triggers JAX init.

``SCHEDULES`` is duplicated from
:mod:`alphagrad.approx.common.schedules` — the source is a tuple of three
strings; importing it directly would pull the whole ``alphagrad.approx.common``
package and therefore JAX into the driver process. Mirrors the same trick
used in :mod:`alphagrad.approx.mu0_args`.
"""

from __future__ import annotations

import argparse

from alphagrad.approx.variants import VARIANT_PRESETS


# Duplicate of alphagrad.approx.common.schedules.SCHEDULES.
SCHEDULES = ("constant", "linear", "cosine")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SEED-style off-policy GFlowNet trainer for vertex-elimination.",
    )

    # ------------------------------------------------------------------
    # Run / logging
    # ------------------------------------------------------------------
    p.add_argument("--name", type=str, default="approx-gfn")
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
        "sequences as JSON. Empty (default) → derive from the wandb run "
        "dir. Mirrors mu0_ray.",
    )
    p.add_argument(
        "--best-sequences-every", type=int, default=10,
        help="Cadence (in episodes) for writing the best-sequences "
        "JSON + logging the wandb table snapshot. 0 = only write at run end.",
    )
    p.add_argument(
        "--calibration-statistic", type=str, default="iqr",
        choices=["iqr", "mean_abs", "std"],
        help="Per-channel dispersion statistic for pre-training "
             "calibration. ``iqr`` (default): symlog-space IQR / 1.349. "
             "``mean_abs`` (legacy): |symlog(mean)|. ``std``: symlog-space std "
             "via the IQR proxy. Mirrors mu0_args.py — TB doesn't use the "
             "value head but the same per-channel scaling helps β-sensitivity.",
    )

    # ------------------------------------------------------------------
    # Environment / reward
    # ------------------------------------------------------------------
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
    p.add_argument("--lambda-frob", type=float, default=0.0)
    p.add_argument(
        "--measure-latency", action="store_true",
        help="Run the compiled approx fn 10x per env step to populate "
             "the latency reward component.",
    )
    p.add_argument(
        "--terminal-rewards-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the env's reward vector only at the final "
             "elimination step; intermediate steps return zeros. "
             "TB only reads ``traj.reward[:, -1, :]`` (gfn.py:603), so "
             "this is a pure speed win — cuts the per-step jacve compile "
             "rate by rollout_length×. Default ON for SEED-style runs; "
             "pass --no-terminal-rewards-only to disable.",
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # ------------------------------------------------------------------
    # Variant / curriculum (mirrors mu0_args.py)
    # ------------------------------------------------------------------
    p.add_argument(
        "--variant",
        type=str,
        default="custom",
        choices=list(VARIANT_PRESETS.keys()),
        help=(
            "Pre-canned configuration mapping to --factors / --max-rules / "
            "--pin-rules-to-exact. ``custom`` honours your explicit flags. "
            "``ve_only`` freezes the rule head. ``diag_gcd`` / ``diag_factor`` / "
            "``compress`` / ``full`` enable progressively richer action sets. "
            "``full_curriculum`` is sugar for ``full`` plus an auto-generated "
            "curriculum diag_gcd → diag_factor → full when --curriculum is empty."
        ),
    )
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help=(
            "Curriculum of variants to run in sequence. Format: "
            "``stage1:N1,stage2:N2,...``. Empty = single training run on "
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

    # Loss mode (CLI parity with ppo / mu0 — TB uses a single scalar TB
    # residual, so the flag is accepted but has no behavioural effect).
    p.add_argument(
        "--loss-mode",
        type=str,
        default="scalar",
        choices=["multi_head", "scalar"],
        help="Accepted for ppo.py / mu0_args.py CLI parity. TB has one "
             "loss (the trajectory-balance residual squared); both values "
             "behave the same here.",
    )

    # Dynamic-substeps CLI parity. gfn.py already encodes
    # (vertex, pair, factor) per decision via the agent's pair / factor
    # heads; the flag is accepted so the same CLI invocation works for
    # PPO / mu0 / gfn.
    p.add_argument(
        "--dynamic-substeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for ppo.py CLI parity. GFN's agent already emits "
             "the (vertex, pair, factor) micro-action sequence via per-depth "
             "policy heads; this flag does not alter the action layout.",
    )
    p.add_argument(
        "--max-substeps",
        type=int,
        default=8,
        help="Accepted for ppo.py CLI parity (no effect in gfn).",
    )
    p.add_argument(
        "--max-axis-size",
        type=int,
        default=1024,
        help="Accepted for ppo.py CLI parity (no effect in gfn).",
    )
    p.add_argument(
        "--allow-compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for ppo.py CLI parity (no effect in gfn today).",
    )

    # ------------------------------------------------------------------
    # Agent variant
    # ------------------------------------------------------------------
    # GFN reuses PPO's Agent, so we expose the same four-way variant
    # toggles (pointer/mlp-vertex × autoreg/single-rule).
    p.add_argument("--no-ptr", action="store_true")
    p.add_argument("--not-autoreg", action="store_true")
    p.add_argument("--max-rules", type=int, default=4,
                   help="Rule-slot count per chosen vertex.")
    p.add_argument("--factors", type=str, default="-1,1,2,4",
                   help="Comma-separated factor values for the rule decoder.")
    # Stage E knobs from ppo.py — surface for variant-preset compatibility.
    p.add_argument("--sparsity-ratio", action="store_true")
    p.add_argument("--rho-prior-bias", type=float, default=4.0)
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
             "Used by variant=ve_only.",
    )

    # ------------------------------------------------------------------
    # Network architecture (mirrors gfn.py / mu0_args.py)
    # ------------------------------------------------------------------
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument("--op-embd-dim", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")
    p.add_argument("--head-init-scale", type=float, default=0.1)
    p.add_argument(
        "--cache-encoding",
        action="store_true",
        help="Reuse the encoder output across all T steps of a "
             "rollout via PPO's encode-once + residual_state path. "
             "Survives Ray transparently — the cached encoding is "
             "intra-episode JAX state on the SPMD actor.",
    )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Parallel rollout envs. -1 = os.cpu_count() "
                        "(or 16 for Vmapped examples).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--logZ-lr", type=float, default=1e-2,
                   help="Separate LR for the scalar logZ — TB convergence is "
                        "much faster when logZ has its own larger step size.")
    p.add_argument("--logZ-init", type=float, default=0.0)
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Inner-loop gradient steps per outer rollout. "
                        "Each step re-evaluates log π_F under the current "
                        "agent against the stored actions. Total per-episode "
                        "updates = train_steps (driver) × gradient_steps.")
    p.add_argument("--minibatches", type=int, default=4,
                   help="Reserved for parity with mu0_args / ppo_args. "
                        "GFN's do_train_steps scans --gradient-steps "
                        "directly; this flag participates in the SPMD "
                        "auto-tune divisibility check only.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-b1", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.0,
                   help="Optional entropy bonus added to the TB objective. "
                        "Default 0 — the TB residual already encourages "
                        "diversity through logZ.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="TB inverse temperature. log R(x) = β · symlog("
                        "weighted terminal reward). Larger β → policy "
                        "concentrates on best trajectories.")
    p.add_argument("--beta-final", type=float, default=1.0,
                   help="Final β when --beta-schedule != constant.")
    p.add_argument("--beta-schedule", type=str, default="constant",
                   choices=SCHEDULES)
    p.add_argument("--discount", type=float, default=1.0,
                   help="Accepted for mu0_args parity. TB uses only the "
                        "terminal reward, so this flag is a no-op in GFN.")

    # ------------------------------------------------------------------
    # Replay buffer (SEED-style off-policy)
    # ------------------------------------------------------------------
    # TB is off-policy by construction (the loss re-evaluates log π_F
    # under the *current* agent at gradient-step time), so retaining
    # past trajectories and sampling from them is the natural design.
    p.add_argument("--replay-buffer-size", type=int, default=0,
                   help="Replay buffer capacity (number of stored "
                        "trajectories). 0 disables — the gfn_ray driver "
                        "promotes this to 1024 if left at 0 because the "
                        "SEED-style off-policy regime needs a buffer.")
    p.add_argument("--replay-batch-size", type=int, default=0,
                   help="Trajectories sampled from buffer per gradient "
                        "step. 0 = use --num-envs.")
    p.add_argument("--replay-warmup", type=int, default=4,
                   help="Episodes (= rollouts) to fill before sampling "
                        "from the buffer. Until then we train on fresh data.")
    p.add_argument("--replay-fresh-fraction", type=float, default=0.0,
                   help="Fraction of the train batch drawn from the *fresh* "
                        "rollout instead of the buffer. 0.0 = pure replay "
                        "(SEED-style default); 1.0 disables replay sampling.")
    p.add_argument("--replay-priority-alpha", type=float, default=0.0,
                   help="Power applied to per-slot priorities at sample "
                        "time. 0 = uniform. The new worker stores symlog'd "
                        "terminal-reward priorities so this can be raised "
                        "without the 10-decade mode-collapse the raw-reward "
                        "priorities in gfn.py:864-869 would cause.")
    p.add_argument("--replay-checkpoint-path", type=str, default="",
                   help="If set, the buffer is saved here every "
                        "--replay-checkpoint-every episodes; loaded at "
                        "startup if the file exists.")
    p.add_argument("--replay-checkpoint-every", type=int, default=10)

    # ------------------------------------------------------------------
    # Preference conditioning + MOGFN-PC scalarization
    # ------------------------------------------------------------------
    p.add_argument("--preference-conditioned", action="store_true",
                   help="Sample a per-env Dirichlet preference each "
                        "episode and condition the policy on it. Implied "
                        "ON when --mogfn-pc is set.")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0)
    p.add_argument(
        "--mogfn-pc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MOGFN-PC (Jain et al. 2023). Sample preference w ~ Dirichlet "
             "per trajectory, condition the policy on w, and scalarize the "
             "reward in *log space*: ``log R(x|w) = β · Σ_k w_k · log r̃_k(x)`` "
             "instead of ``β · symlog(Σ_k w_k r_k(x))``. Fixes the "
             "vector→scalar collapse where one channel dominates the "
             "weighted sum after exponentiation (the calibration-time issue "
             "where flops weight 1000× cos_sim made cosine_sim invisible).",
    )
    p.add_argument(
        "--preference-channels",
        type=str,
        default="flops,peak_memory,cosine_sim",
        help="Comma-separated reward channel names that the preference "
             "vector indexes. Must have exactly NUM_VALUE_HEADS=3 entries "
             "(matches the agent's pref_proj input dim). Order matters: "
             "preference[k] applies to reward channel ``channels[k]``. "
             "Default ``flops,peak_memory,cosine_sim`` lets MOGFN-PC explore "
             "the Pareto front over (cost, accuracy) explicitly.",
    )
    p.add_argument(
        "--reward-normalization",
        type=str,
        choices=["none", "zscore"],
        default="zscore",
        help="Per-channel normalization applied to ``symlog(reward)`` "
             "before the MOGFN-PC scalarization. ``zscore`` (default) "
             "maintains a running EMA mean/var per channel and emits "
             "``(symlog(r_k) - μ_k) / σ_k``; ``none`` skips and uses raw "
             "symlog. Without normalization, the largest-scale dim "
             "(symlog(flops)~23 vs symlog(cos_sim)~0.5) still dominates "
             "even in log-space.",
    )
    p.add_argument(
        "--reward-stats-decay",
        type=float,
        default=0.99,
        help="EMA decay for the per-channel running stats. Higher = "
             "slower adaptation (more stable but slower to track shifts).",
    )

    # ------------------------------------------------------------------
    # Stage F Lagrangian (CLI parity — TB has no critic so these are no-ops).
    # ------------------------------------------------------------------
    p.add_argument(
        "--lagrangian-constraint",
        action="append",
        default=[],
        metavar="NAME>=THRESH",
        help="Accepted for ppo/mu0 CLI parity (no effect in gfn).",
    )
    p.add_argument(
        "--lagrangian-lr",
        type=float,
        default=1e-2,
        help="Accepted for ppo/mu0 CLI parity (no effect in gfn).",
    )
    p.add_argument(
        "--cosine-lower-bound",
        type=float,
        default=0.8,
        help="Accepted for ppo/mu0 CLI parity (no effect in gfn).",
    )
    p.add_argument(
        "--cosine-upper-bound",
        type=float,
        default=0.9,
        help="Accepted for ppo/mu0 CLI parity (no effect in gfn).",
    )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--capture-perfect-grads", action="store_true")
    p.add_argument(
        "--print-top-every", type=int, default=0,
        help="Cadence (in episodes) for printing the top-N best trajectories. "
             "0 disables; only the end-of-run dump is emitted.",
    )

    return p
