"""JAX-free CLI arguments specific to `az_gumbel.py`.

The knobs the Sampled + Gumbel AlphaZero trainer actually reads that
PPO has no counterpart for — az's search knobs (`--n-candidates`,
`--rollout-depth`) chief among them. The flags the two trainers DO
share (`--seed`, `--dataset`, `--lr`, `--latency-inner-reps`) live in
`args_common.add_common_args`, with az's values for them in
`COMMON_DEFAULTS` below; `az_args.make_argparser` composes the two
back into the parser `az_gumbel.py` has always had. Add to this file
as the Gumbel trainer grows.

Kept JAX-free (argparse + os only) so the parser can be built — and
`--help` printed — without triggering JAX init, same hygiene as
`ppo_args.py`.

Search behaviour beyond these flags is env-var driven
(`ALPHAGRAD_GAZ_*`, `ALPHAGRAD_POPART_*`); see `az_gumbel.py`.
"""

from __future__ import annotations

import argparse
import os

#: az's values for the flags defined in `args_common.add_common_args`.
#: Splat into that function (``add_common_args(p, **COMMON_DEFAULTS)``)
#: to reproduce az's half of the shared surface exactly. `dataset_type`
#: is None because az declares `--dataset` without a `type=` (raw
#: string), and the help for `--latency-inner-reps` is left unset.
COMMON_DEFAULTS = {
    "seed": 7,
    "dataset": "mnist",
    "dataset_type": None,
    "lr": 3e-4,
    "latency_inner_reps": 50,
    "latency_inner_reps_help": None,
}


def add_az_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the az-only flags to `p` and return `p`."""
    # Run / task
    p.add_argument("--task", default="VmappedNeuralNetwork")   # e.g. VmappedViT
    p.add_argument("--total-measurements", type=int, default=150)

    # Search (Gumbel root + progressive deepening)
    p.add_argument("--n-candidates", type=int, default=8)      # Gumbel top-m at the root
    p.add_argument("--rollout-depth", type=int, default=3)     # greedy descent depth per sim

    # Training
    p.add_argument("--train-epochs", type=int, default=4)
    p.add_argument("--replay-episodes", type=int, default=16)

    # Model
    p.add_argument("--nn-hidden", type=int, default=256)

    # Measurement
    p.add_argument("--ndata", type=int, default=5)

    # PopArt warm-start (parity with ppo's --popart-init-episodes, which every
    # PPO launcher passes as 3). Without it az's normaliser starts at (0, 1)
    # and its first EMA step sees M == 1 sample, whose variance is 0 -- sigma
    # then clips to sigma_min (0.1/0.2) while the raw memory channel is ~1e9.
    p.add_argument(
        "--popart-init-episodes", type=int, default=3,
        help="Warm-start PopArt (mu, sigma) from this many episodes of "
             "UNIFORMLY RANDOM but valid elimination orders, measured through "
             "the real measurement path, before training. 0 disables. These "
             "episodes ARE logged to wandb (the measurement is real and the "
             "step counter must advance, uniform with ppo's warm-up) but they "
             "carry no `loss`, no `n_meas` and no `ep`, and they do NOT count "
             "against --total-measurements.")

    # Logging
    # ---- approximation-head surface (#79) --------------------------------
    # These were literal constants inside apply_policy_arch, so AZ had no
    # exact arm at all and ALPHAGRAD_GAZ_MICRO (set by 8 launchers) gated
    # nothing. Defaults reproduce the previous hardcoded behaviour exactly:
    # face-level approximation ON. Pass --no-approx-head for a genuine exact
    # arm, the counterpart of PPO fq_v47e.
    p.add_argument(
        "--no-approx-head", action="store_true",
        help="EXACT arm: build no approximation head, so every face is "
        "computed exactly and only the elimination ORDER is searched. The "
        "AZ counterpart of PPO --no-approx-head.")
    p.add_argument(
        "--no-face-actions", dest="face_actions", action="store_false",
        help="Disable the per-face action space (default: enabled).")
    p.add_argument(
        "--no-live-faces", dest="live_faces", action="store_false",
        help="Disable the live per-face token stream (default: enabled).")
    p.set_defaults(face_actions=True, live_faces=True)
    p.add_argument("--out", default=os.path.expanduser("~/dsnn/az_gumbel_out"))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default="az_gumbel")
    p.add_argument("--wandb-project", default="dsnn-jac-gpu")
    p.add_argument("--wandb-entity", default="dll-streetview")

    return p
