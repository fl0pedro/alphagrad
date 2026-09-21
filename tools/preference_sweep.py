"""Query a trained preference-conditioned policy at a chosen w (dsnn-dfw.86).

READ-ONLY. The checkpoint is loaded, the preference vector is pinned to each
point of a grid, plans are rolled out through the trainer's own rollout and
its own paired measurement, and the front they describe is written in the
schema `pareto_front.json` uses. No gradient step runs.

It is NOT `--resume`: a resume continues the run the checkpoint is a state of
and is argument-locked for that reason. This rebuilds the run's argument
namespace from the checkpoint itself, overrides only where the output goes,
and hands it to `ppo.main`.

Run it as an sbatch job on the node profile the run used:

    $PY tools/preference_sweep.py \
        --checkpoint .../ppo_ckpt_ep000002000 \
        --weights edge:11 --plans 16 --out .../sweep
"""
from __future__ import annotations

import argparse
import os
import sys


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="the ppo_ckpt_* directory to read")
    p.add_argument("--weights", default="edge:11",
                   help="edge:N, or explicit vectors '1,0,0;0.5,0.5,0'")
    p.add_argument("--plans", type=int, default=16,
                   help="rollouts per preference point; a multiple of the "
                        "checkpoint's --num-envs")
    p.add_argument("--out", required=True,
                   help="directory for the front and the plan records")
    p.add_argument("--name", default="",
                   help="run name; the checkpoint's own with a suffix when "
                        "not given")
    p.add_argument("--wandb", default="offline",
                   choices=["disabled", "offline", "online"])
    return p


def main() -> int:
    a = make_argparser().parse_args()
    from alphagrad.approx import ppo
    from alphagrad.approx.common import checkpoint as ckpt
    from alphagrad.approx.common import preference_sweep as psweep

    meta = ckpt.read_ppo_meta(a.checkpoint)
    parser = ppo.make_argparser()
    if not psweep.sweep_args_round_trip(meta, parser):
        raise ckpt.CheckpointError(
            "the argument namespace rebuilt from this checkpoint does not "
            "write back the saved one. The sweep would run a configuration "
            "the checkpoint was not written under.")
    name = a.name or (str(meta["args"].get("name") or "run")
                      + "_prefsweep_ep%d" % int(meta["episode"]))
    args = psweep.load_sweep_args(meta, parser, overrides={
        "name": name,
        "wandb": a.wandb,
        # A sweep takes no gradient step, so there is no state to
        # checkpoint, no window to stop on and no exact gradient to check.
        "checkpoint_every": 0,
        "resume": "",
        "auto_stop": False,
        "grad_oracle": "off",
        "print_top_every": 0,
        "preference_sweep_checkpoint": a.checkpoint,
        "preference_sweep_weights": a.weights,
        "preference_sweep_plans": int(a.plans),
        "preference_sweep_out": a.out,
    })
    os.makedirs(a.out, exist_ok=True)
    print(f"[pref-sweep] {a.checkpoint} at episode {int(meta['episode'])}, "
          f"seed {meta['args'].get('seed')}, weights {a.weights}, "
          f"{int(a.plans)} plans per point -> {a.out}", flush=True)
    ppo.main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
