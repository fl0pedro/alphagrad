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
    p.add_argument("--lambda-frob", type=float, default=0.0)
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--terminal-rewards-only", action="store_true")
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--dataset", type=str, default="none")
    p.add_argument("--dataset-size", type=int, default=-1)

    # Rollout
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)

    # PPO hyperparameters
    p.add_argument("--ppo-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--discount", type=float, default=0.99)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-eps", type=float, default=1e-8)

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
