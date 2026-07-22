"""JAX-free CLI argparser for `az_gumbel.py`.

The knobs the Sampled + Gumbel AlphaZero trainer actually reads.
Deliberately NOT folded into `ppo_args.make_argparser`: the two
trainers share the policy/value components (`policy.build_policy`) but
not their CLI surface, and az's search knobs (`--n-candidates`,
`--rollout-depth`) have no PPO counterpart. Add to this file as the
Gumbel trainer grows.

Kept JAX-free (argparse + os only) so the parser can be built — and
`--help` printed — without triggering JAX init, same hygiene as
`ppo_args.py`.

Search behaviour beyond these flags is env-var driven
(`ALPHAGRAD_GAZ_*`, `ALPHAGRAD_POPART_*`); see `az_gumbel.py`.
"""

from __future__ import annotations

import argparse
import os


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sampled + Gumbel AlphaZero trainer for vertex elimination.",
    )

    # Run / task
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task", default="VmappedNeuralNetwork")   # e.g. VmappedViT
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--total-measurements", type=int, default=150)

    # Search (Gumbel root + progressive deepening)
    p.add_argument("--n-candidates", type=int, default=8)      # Gumbel top-m at the root
    p.add_argument("--rollout-depth", type=int, default=3)     # greedy descent depth per sim

    # Training
    p.add_argument("--train-epochs", type=int, default=4)
    p.add_argument("--replay-episodes", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)

    # Model
    p.add_argument("--nn-hidden", type=int, default=256)

    # Measurement
    p.add_argument("--ndata", type=int, default=5)
    p.add_argument("--latency-inner-reps", type=int, default=50)

    # Logging
    p.add_argument("--out", default=os.path.expanduser("~/dsnn/az_gumbel_out"))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default="az_gumbel")
    p.add_argument("--wandb-project", default="dsnn-jac-gpu")
    p.add_argument("--wandb-entity", default="dll-streetview")

    return p
