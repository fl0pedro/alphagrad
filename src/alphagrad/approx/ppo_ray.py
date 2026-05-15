"""JAX-free driver for the Ray-actor PPO trainer.

First-cut counterpart to `mu0_ray.py`. The driver:

* spawns one ``PPOActor`` (the JAX trainer; single-GPU for now),
* spawns N ``CpuApproximationActor``s (the tokenizer / approximation
  pool),
* hands the CPU pool to the trainer,
* loops ``--episodes`` × ``run_rollout_and_train.remote(...)`` calls,
* logs to wandb and tqdm.

Driver process stays JAX-free so the Ray actors are the only place
JAX initialises a device — same hygiene as `mu0_ray.py`.

Compared to `mu0_ray.py` this driver is intentionally sparse: no
variant sweep, no calibration phase, no per-channel best tracking, no
replay-checkpoint plumbing. Those features can land later; for the
first cut we want a minimal flight check that the rollout + update
loop runs end-to-end against the external tokenizer pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

# Plain threading lock so tqdm doesn't leak a named POSIX semaphore on
# signal-kill — mirrors `mu0_ray.py`.
import threading as _threading
tqdm.set_lock(_threading.RLock())

from alphagrad.approx.ppo_args import make_argparser  # noqa: E402


def _disable_ray_uv_autodetect() -> None:
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


def _assert_jax_free() -> None:
    """The driver must never import JAX (it'd hog GPU memory the actors
    need). Bail loudly if some helper leaked an import."""
    leaked = sorted(m for m in sys.modules if m == "jax" or m.startswith("jax."))
    if leaked:
        raise RuntimeError(f"jax has leaked into the driver process: {leaked[:5]}...")


def _extend_argparser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Driver-only knobs not present in `ppo.make_argparser`."""
    p.add_argument(
        "--num-cpu-workers",
        type=int,
        default=8,
        help="Number of Ray CpuApproximationActor instances for tokenization.",
    )
    p.add_argument(
        "--ray-address",
        type=str,
        default="",
        help="Address of an existing Ray cluster (passed to ray.init). "
        "Empty = local cluster.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="dsnn-vertex",
        help="wandb project name.",
    )
    p.add_argument(
        "--actor-num-gpus",
        type=float,
        default=1.0,
        help="GPUs reserved for the PPO actor (1 for the first cut; raise "
        "when SPMD sharding lands).",
    )
    return p


def _run(args) -> int:
    import ray
    import wandb
    from alphagrad.approx.ppo_ray_actors import PPOActor
    from alphagrad.approx.cpu_approx_actors import CpuApproximationActor

    args_dict = vars(args)
    wandb.init(
        project=args.wandb_project,
        name=args.name,
        config=args_dict,
        mode="disabled" if args.wandb == "disabled" else args.wandb,
        reinit=True,
    )

    # Disable XLA preallocation in actor processes — the GPU actor needs
    # headroom for transient compile/exec buffers, and the CPU actors
    # don't benefit from preallocation at all.
    actor_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    }
    actor_kwargs = {
        "num_gpus": args.actor_num_gpus,
        "runtime_env": {"env_vars": actor_env},
    } if args.actor_num_gpus > 0 else {
        "runtime_env": {"env_vars": actor_env},
    }

    actor = PPOActor.options(**actor_kwargs).remote(
        args_dict, int(args.seed),
    )

    cpu_workers = [
        CpuApproximationActor.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={
                "env_vars": {
                    "JAX_PLATFORMS": "cpu",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                }
            },
        ).remote(args_dict, variant=None, actor_id=i)
        for i in range(args.num_cpu_workers)
    ]

    ray.get(actor.init_worker.remote(cpu_workers))
    ray.get(
        [actor.ready.remote()] + [c.ready.remote() for c in cpu_workers]
    )
    print(f"  actors spawned + JIT-warm in {time.time() - args.t_start:.1f}s")

    seed_counter = int(args.seed) + 100
    best_return = -float("inf")
    best_ep = -1

    pbar = tqdm(
        total=args.episodes,
        desc="ppo_ray",
        disable=False,
        leave=True,
        ncols=160,
    )

    for ep in range(args.episodes):
        seed_counter += 1
        stats = ray.get(actor.run_rollout_and_train.remote(seed_counter))

        ret_mean = stats.get("episode_return_mean", float("nan"))
        ret_max = stats.get("episode_return_max", float("nan"))
        ploss = stats.get("ppo_loss", float("nan"))
        vloss = stats.get("value_loss", float("nan"))
        ent = stats.get("entropy", float("nan"))

        if ret_max > best_return:
            best_return = ret_max
            best_ep = ep

        pbar.update(1)
        pbar.set_description(
            f"ppo_ray "
            f"best:{best_return:+.3g}(ep{best_ep}) "
            f"mean:{ret_mean:+.3g} "
            f"loss(p/v):{ploss:.2g}/{vloss:.2g} "
            f"ent:{ent:.3f}"
        )

        if (ep + 1) % 5 == 0 or ep == args.episodes - 1:
            tqdm.write(
                f"  [ppo_ray] ep={ep + 1:>4}/{args.episodes} "
                f"best={best_return:+.4g}(ep{best_ep}) mean={ret_mean:+.4g} "
                f"ent={ent:.3f} p_loss={ploss:.3g} v_loss={vloss:.3g}"
            )

        log_dict = {
            "episode": ep,
            "best_return": best_return,
            "best_return_this_ep": ret_max,
            "mean_return": ret_mean,
            "ppo_loss": ploss,
            "value_loss": vloss,
            "entropy": ent,
            "total_loss": stats.get("total_loss", float("nan")),
        }
        wandb.log(log_dict)

    pbar.close()

    tqdm.write(f"\n========== FINAL ==========")
    tqdm.write(f"  best return: {best_return:+.6g}  (ep {best_ep})")
    wandb.log({
        "final/best_return": best_return,
        "final/best_ep": best_ep,
    })
    wandb.finish()

    ray.kill(actor, no_restart=True)
    for c in cpu_workers:
        ray.kill(c, no_restart=True)
    return 0


def main() -> int:
    _disable_ray_uv_autodetect()
    p = make_argparser()
    p = _extend_argparser(p)
    args = p.parse_args()
    args.t_start = time.time()
    _assert_jax_free()

    import ray
    init_kwargs = {"address": args.ray_address} if args.ray_address else {}
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    ray.init(**init_kwargs, ignore_reinit_error=True)

    rc = _run(args)
    ray.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
