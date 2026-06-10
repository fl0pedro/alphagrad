"""JAX-free driver for the Ray-actor C-MORL trainer.

Implements C-MORL (Liu et al., "Multi-Objective Reinforcement Learning through
Efficient Discovery of Pareto Front", ICLR 2025, arXiv:2410.02236) on top of
the PPO Ray stack (``cmorl_ray_worker`` / ``cmorl_ray_actors``). C-MORL has two
stages, both run here from the JAX-free driver:

* Stage 1 — Pareto initialization: a population of preference weight vectors
  ``w`` is swept; each episode sets the worker's scalarizing weights to a
  sampled ``w`` (``set_reward_weights``) so the policy is trained toward diverse
  trade-offs. The achieved terminal objective vectors seed a non-dominated
  Pareto archive.
* Stage 2 — constrained Pareto extension: for each step one objective is pushed
  while the others are *constrained* not to regress below an archive seed point
  (minus a tolerance). The constraint is enforced by the worker's Lagrangian
  primal-dual update (``set_lagrangian_constraints`` + dual ascent) — the "C"
  in C-MORL. This fills sparse regions of the front.

Heterogeneous placement (per the user's setup): the GPU trainer actor is spawned
with ``num_cpus=0`` so a ``--num-cpus=0`` GPU node physically excludes the
``num_cpus=1`` measurement actors — all latency/cost measurement therefore lands
on the single CPU node (``JAX_PLATFORMS=cpu``). Driver stays JAX-free.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Plain threading lock so tqdm doesn't leak a named POSIX semaphore on
# signal-kill — mirrors `mu0_ray.py`.
import threading as _threading
tqdm.set_lock(_threading.RLock())

from alphagrad.approx.ppo_args import make_argparser  # noqa: E402
from alphagrad.approx.common.calibration import run_calibration  # noqa: E402
from alphagrad.approx.common.compile_cache import (  # noqa: E402
    kill_coordinator as _kill_compile_cache,
    spawn_coordinator as _spawn_compile_cache,
)
from alphagrad.approx.common.ray_runtime import (  # noqa: E402
    _assert_jax_free,
    _disable_ray_uv_autodetect,
    add_common_ray_args,
)
from alphagrad.approx.common.reward_scaling import (  # noqa: E402
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    build_best_sequences_wandb_payload,
    build_best_sequences_wandb_table,
    build_wandb_log_dict,
    dump_best_sequences_json,
    format_milestone_line,
    init_running_bests,
    update_running_bests,
)

DEFAULT_OBJECTIVES = "latency_ns,peak_memory,frob_residual"


# ---------------------------------------------------------------------------
# C-MORL host-side helpers (JAX-free): objective selection, preference
# sampling, Pareto-front filtering, hypervolume. Archives are small so the
# O(N^2) routines are fine.
# ---------------------------------------------------------------------------
def _resolve_objectives(spec: str):
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in REWARD_INDEX:
            raise ValueError(
                f"Unknown objective '{n}'. Valid: {', '.join(REWARD_NAMES)}"
            )
    return names, [REWARD_INDEX[n] for n in names]


def _pref_to_full_weights(w, obj_idx):
    """Embed a preference over the active objectives into a full 8-vector."""
    full = np.zeros(NUM_REWARDS, dtype=np.float32)
    for k, idx in enumerate(obj_idx):
        full[idx] = float(w[k])
    return full


def _sample_preference(rng, n_obj, alpha, sampler, offset):
    """Dirichlet (random) or low-discrepancy Kronecker preference on the simplex."""
    if sampler == "kronecker":
        # Plastic-constant additive-recurrence (R_d) sequence — uniform simplex
        # coverage with O(log N / N) discrepancy (cf. common/preferences.py).
        x = 2.0
        for _ in range(50):
            x = (1.0 + x) ** (1.0 / (n_obj + 1))
        g = 1.0 / x
        pts = np.array([((offset + 1) * g ** (k + 1)) % 1.0 for k in range(n_obj)])
        # map the unit cube point to the simplex via the exponential trick
        e = -np.log(np.clip(pts, 1e-9, 1.0))
        return (e / e.sum()).astype(np.float32)
    return rng.dirichlet(np.full(n_obj, alpha)).astype(np.float32)


def _pareto_mask(points):
    """Boolean mask of non-dominated rows (maximization)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)
    ge = np.all(pts[:, None, :] >= pts[None, :, :], axis=-1)
    gt = np.any(pts[:, None, :] > pts[None, :, :], axis=-1)
    return ~np.any(ge & gt, axis=0)


def _hv2d_min(pts, ref):
    pts = pts[np.argsort(pts[:, 0])]
    rx, ry = ref
    hv = (ry - pts[0, 1]) * (rx - pts[0, 0])
    for i in range(1, pts.shape[0]):
        hv += (pts[i - 1, 1] - pts[i, 1]) * (rx - pts[i, 0])
    return float(hv)


def _hv3d_min(pts, ref):
    pts = pts[np.argsort(pts[:, 2])]
    rx, ry, rz = ref
    vol, active, i, n = 0.0, [], 0, pts.shape[0]
    while i < n:
        z = pts[i, 2]
        while i < n and pts[i, 2] == z:
            active.append(pts[i, :2])
            i += 1
        z_next = pts[i, 2] if i < n else rz
        vol += _hv2d_min(np.asarray(active), (rx, ry)) * (z_next - z)
    return float(vol)


def _hypervolume(points, ref):
    """Hypervolume dominated by `points` (maximization) above nadir `ref`."""
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    m = pts.shape[1]
    if m not in (2, 3):
        return float("nan")
    pts = pts[_pareto_mask(pts)]
    pts = pts[np.all(pts > ref, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    if m == 2:
        return _hv2d_min(-pts, (-ref[0], -ref[1]))
    return _hv3d_min(-pts, (-ref[0], -ref[1], -ref[2]))


def _extend_argparser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Driver-only knobs not present in `ppo.make_argparser`."""
    add_common_ray_args(p)
    p.add_argument(
        "--actor-num-gpus",
        type=float,
        default=1.0,
        help="GPUs reserved for the PPO actor (1 for the first cut; raise "
        "when SPMD sharding lands).",
    )
    p.add_argument(
        "--lagrangian-warmup-eps",
        type=int,
        default=20,
        help="Episodes at the start of training where the Lagrangian "
             "violations are computed for logging but the penalty does NOT "
             "flow into the advantage and the multipliers are not updated. "
             "Without this warm-up, a cold cosine_sim constraint (mean ~0.05) "
             "with a 0.5 threshold pushes the multiplier into the tens within "
             "100 episodes, killing exploration via overwhelming penalty.",
    )
    p.add_argument(
        "--lagrangian-multiplier-max",
        type=float,
        default=1.0,
        help="Upper clip on each Lagrangian multiplier. Stops dual ascent "
             "from dominating the PPO objective when a constraint is "
             "structurally hard to satisfy in the early policy.",
    )
    # ---- C-MORL controller knobs ----
    p.add_argument(
        "--objectives", type=str, default=DEFAULT_OBJECTIVES,
        help="Comma-separated reward-channel names that span the objective "
             "space / Pareto front (default: latency_ns,peak_memory,frob_residual).",
    )
    p.add_argument(
        "--num-prefs", type=int, default=8,
        help="Size of the preference population swept in Stage-1 "
             "Pareto-initialization (corners + interior samples).",
    )
    p.add_argument("--pref-alpha", type=float, default=1.0,
                   help="Dirichlet concentration for interior preference samples.")
    p.add_argument("--pref-sampler", type=str, default="dirichlet",
                   choices=["dirichlet", "kronecker"],
                   help="Preference sampler for Stage-1 interior points.")
    p.add_argument(
        "--extension-steps", type=int, default=0,
        help="Number of Stage-2 constrained Pareto-extension steps (0 = skip "
             "Stage 2 and run pure Stage-1 preference sweep for --episodes).",
    )
    p.add_argument("--extension-episodes", type=int, default=10,
                   help="Episodes of IPO extension training per Stage-2 step.")
    p.add_argument(
        "--extension-beta", type=float, default=0.9,
        help="C-MORL constraint slack: during a Stage-2 step the non-target "
             "objectives are kept >= G_i - (1-beta)*|G_i| of the extended "
             "policy's return (paper's G_i >= beta*G_i, sign-robust form).",
    )
    p.add_argument(
        "--ipo-t", type=float, default=1.0,
        help="Interior-point log-barrier temperature for the Pareto extension "
             "(larger => weaker constraint; barrier weight = 1/(t*slack)).",
    )
    return p


def _run(args) -> int:
    import ray
    import wandb
    from alphagrad.approx.cmorl_ray_actors import CMORLActor
    from alphagrad.approx.cpu_approx_actors import CpuApproximationActor

    args_dict = vars(args)
    wandb.init(
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None) or None,
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
    # Propagate any ``ALPHAGRAD_*`` env vars (debug switches like
    # ``ALPHAGRAD_DEBUG_QUALITY``, ``ALPHAGRAD_LEAK_PROFILE``,
    # ``ALPHAGRAD_SKIP_COST_ANALYSIS``) so a single ``sbatch --export``
    # reaches the CpuApproximationActor processes too. Without this
    # the actors run with stock env and the driver-side toggles
    # silently no-op.
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            actor_env[k] = v
    actor_kwargs = {
        # GPU-bound trainer: reserve 0 CPU slots so the GPU node can be
        # declared --num-cpus=0, which physically excludes the num_cpus=1
        # measurement actors from it -> all measurement lands on the single
        # CPU node (consistent cores = clean latency reward, no node-mix).
        "num_cpus": 0,
        "num_gpus": args.actor_num_gpus,
        "runtime_env": {"env_vars": actor_env},
    } if args.actor_num_gpus > 0 else {
        "runtime_env": {"env_vars": actor_env},
    }

    # Cluster-wide shared compile cache. Spawned BEFORE any CPU worker
    # so when those workers' env._callback runs cached_compile() on
    # the very first step, the coordinator is already registered.
    # Named actor, lifetime=detached — the driver kills it on exit.
    _spawn_compile_cache(
        max_size=int(getattr(args, "compile_cache_size", 512)),
    )

    actor = CMORLActor.options(**actor_kwargs).remote(
        args_dict, int(args.seed),
    )

    # Mirror mu0_ray.py: pass the cpu actor's `.options(...)` kwargs to
    # `init_server` so the SPMD/PPO actor can respawn killed workers
    # via the same factory pattern after a timeout. Keep these in sync
    # with the .options(...) below.
    cpu_actor_env = {
        # GPU measurement (--exec-on-gpu) needs the actor to SEE cuda;
        # otherwise pin to CPU JAX so measurement runs on the CPU pool.
        "JAX_PLATFORMS": "cuda" if getattr(args, "exec_on_gpu", False) else "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # NOTE: do NOT force single-thread here. The approx-Jacobian exec
        # scales ~linearly with cores (0.45s@64-core → ~80s@1-core), so
        # single-thread starves it. Each actor is pinned to a disjoint
        # core SLICE (see CpuApproximationActor.__init__) sized to
        # cores//num_workers, giving multi-core execs without cross-actor
        # oversubscription. Profiled 2026-06-06.
    }
    # Pass ALPHAGRAD_* / JAX_COMPILATION_* debug toggles through to the
    # CPU actor processes — same rationale as ``actor_env`` above.
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            cpu_actor_env[k] = v
    cpu_actor_options = {
        "num_cpus": 1,
        "num_gpus": float(getattr(args, "cpu_actor_num_gpus", 0.0) or 0.0),
        "runtime_env": {"env_vars": cpu_actor_env},
    }
    # SPREAD the measurement actors across ALL nodes in the cluster
    # (e.g. the 2-node pooled setup: GPU node's CPUs + cpu1's CPUs)
    # rather than letting Ray bin-pack them onto a single node. Combined
    # with the per-node core allocator (--cpu-cores-shared), this uses
    # every core across both nodes for measurement. Opt-in via
    # ``--spread-cpu-actors`` so single-node runs keep the default
    # (locality-friendly) packing.
    if getattr(args, "spread_cpu_actors", False):
        cpu_actor_options["scheduling_strategy"] = "SPREAD"
    cpu_workers = [
        CpuApproximationActor.options(**cpu_actor_options).remote(
            args_dict, variant=None, actor_id=i,
        )
        for i in range(args.num_cpu_workers)
    ]

    # Construct the timeout-bounded pool. `init_server` also internally
    # calls `init_worker` if the worker hasn't been built yet, so we
    # don't need a separate `init_worker` call. The pool kicks in
    # automatically — `_fan_out_tokenize` routes through it.
    ray.get(actor.init_server.remote(
        cpu_workers,
        callback_timeout_s=float(args.cpu_callback_timeout),
        initial_timeout_s=float(args.cpu_callback_initial_timeout),
        warm_after=int(args.cpu_callback_warm_after),
        recycle_every=int(args.cpu_worker_recycle_every),
        cpu_actor_options=cpu_actor_options,
        starting_actor_id=int(args.num_cpu_workers),
    ))
    ray.get(
        [actor.ready.remote()] + [c.ready.remote() for c in cpu_workers]
    )
    # Phase 4c: warm the CPU pool BEFORE calibration so the zero-pref
    # rollouts don't time out and poison `reward_vec_means`. Same
    # ordering as `mu0_ray.py` post-refactor.
    if cpu_workers:
        ray.get([c.compile_approximations.remote() for c in cpu_workers])
    print(
        f"  actors spawned + JIT-warm in {time.time() - args.t_start:.1f}s"
    )

    # Calibration is the pre-training pass that computed
    # ``1/|symlog(mean)|`` per-channel scaling weights from a random-
    # policy rollout. With ``--advantage-norm gdpo`` (the new default),
    # per-minibatch z-scoring inside the loss subsumes this magnitude
    # rescaling, so calibration is a no-op — skip it to save the
    # warm-up time. The legacy ``--advantage-norm scalar`` path still
    # benefits from calibrated weights (its global advantage z-score
    # operates on the weighted-sum reward, which depends on
    # cross-channel magnitude).
    if (
        getattr(args, "advantage_norm", "gdpo") == "scalar"
        and getattr(args, "calibrate_steps", 0) > 0
    ):
        run_calibration(actor, args, args.calibrate_steps, after_warmup=True)
    elif getattr(args, "calibrate_steps", 0) > 0:
        print(
            "  [calibration] skipped under --advantage-norm gdpo "
            "(per-channel z-scoring is self-calibrating)."
        )

    seed_counter = int(args.seed) + 100
    state = init_running_bests()

    # Resolve the best-sequences JSON path (kept from the PPO driver).
    best_seq_json_path = getattr(args, "best_sequences_json", "") or ""
    if not best_seq_json_path:
        wb_dir = getattr(getattr(wandb, "run", None), "dir", None)
        if wb_dir:
            best_seq_json_path = os.path.join(wb_dir, "best_sequences.json")
        else:
            best_seq_json_path = os.path.abspath("best_sequences.json")
    best_seq_every = int(getattr(args, "best_sequences_every", 0) or 0)

    # ------------------------------------------------------------------
    # C-MORL controller (Liu et al., ICLR'25).
    #   Stage 1 — Pareto initialization: a POPULATION of M independent policies
    #     (one per preference) is trained, each maximizing the weighted-sum
    #     advantage for its preference; every policy's per-env terminal solutions
    #     seed a non-dominated archive and the policy is checkpointed.
    #   Stage 2 — IPO Pareto extension: a selected policy is reloaded and one
    #     objective is pushed while the others are kept above thresholds via the
    #     interior-point log-barrier (worker `set_extension`). Critic untouched.
    # ------------------------------------------------------------------
    obj_names, obj_idx = _resolve_objectives(args.objectives)
    n_obj = len(obj_idx)
    rng = np.random.default_rng(int(args.seed))
    archive_pts: list = []   # per-solution objective vectors (maximize)
    archive_seqs: list = []  # the elimination sequence that produced each point

    def _archive_add(solutions) -> None:
        added = False
        for sol in solutions or []:
            # sol = {"obj": full-reward-vector, "seq": elimination sequence}
            vec = sol["obj"] if isinstance(sol, dict) else sol
            seq = sol.get("seq") if isinstance(sol, dict) else None
            g = np.array([vec[obj_idx[k]] for k in range(n_obj)], dtype=np.float64)
            if not np.all(np.isfinite(g)):
                continue
            if any(np.allclose(g, p) for p in archive_pts):  # skip duplicates
                continue
            archive_pts.append(g)
            archive_seqs.append(seq)
            added = True
        if added:
            keep = np.where(_pareto_mask(np.stack(archive_pts)))[0]
            keep_set = set(int(i) for i in keep)
            archive_pts[:] = [archive_pts[i] for i in range(len(archive_pts)) if i in keep_set]
            archive_seqs[:] = [archive_seqs[i] for i in range(len(archive_seqs)) if i in keep_set]

    def _archive_hv() -> float:
        if not archive_pts:
            return 0.0
        pts = np.stack(archive_pts)
        return _hypervolume(pts, pts.min(axis=0) - 1.0)

    import json as _json
    pareto_path = os.path.join(
        os.path.dirname(best_seq_json_path) or os.getcwd(), "cmorl_pareto_front.json"
    )

    def _dump_pareto() -> None:
        """Persist the full Pareto archive — each point's objective vector AND
        the elimination sequence that produced it — so the frontier is
        recoverable (not just the front extremes in best_sequences.json)."""
        payload = {
            "objectives": list(obj_names),
            "hypervolume": _archive_hv(),
            "num_points": len(archive_pts),
            "dynamic_substeps": bool(getattr(args, "dynamic_substeps", False)),
            "seq_format": (
                "[vertex, [<call>, ...]] per eliminated vertex, where <call> "
                "is one of diag(i, j, factor) / compress('kind', axis) / "
                "quant('dtype') — the full typed micro-action sub-episode"
                if getattr(args, "dynamic_substeps", False)
                else "[vertex] per step"
            ),
            "front": [
                {"obj": {nm: float(v) for nm, v in zip(obj_names, pt)}, "seq": seq}
                for pt, seq in zip(archive_pts, archive_seqs)
            ],
        }
        try:
            with open(pareto_path, "w") as _f:
                _json.dump(payload, _f, indent=2)
        except Exception as _exc:
            tqdm.write(f"  [C-MORL] pareto dump failed: {_exc}")

    # Preference population: objective corners (unit vectors) + interior samples.
    prefs = [np.eye(n_obj, dtype=np.float32)[k] for k in range(n_obj)]
    for j in range(max(int(args.num_prefs) - n_obj, 0)):
        prefs.append(
            _sample_preference(rng, n_obj, args.pref_alpha, args.pref_sampler, j)
        )

    ckpt_root = os.path.join(
        os.path.dirname(best_seq_json_path) or os.getcwd(), "cmorl_policies"
    )
    os.makedirs(ckpt_root, exist_ok=True)

    do_stage2 = int(args.extension_steps) > 0
    eps_per_policy = max(int(args.episodes) // max(len(prefs), 1), 1)
    total_episodes = eps_per_policy * len(prefs) + (
        int(args.extension_steps) * int(args.extension_episodes) if do_stage2 else 0
    )
    pbar = tqdm(total=total_episodes, desc="cmorl_ray", leave=True, ncols=180)
    global_ep = 0

    def _episode(w_full, stage_tag: str, target=None):
        nonlocal seed_counter, global_ep
        seed_counter += 1
        ray.get(actor.set_reward_weights.remote(w_full))
        stats = ray.get(actor.run_rollout_and_train.remote(seed_counter))
        _archive_add(stats.get("terminal_solutions", []))
        update_running_bests(state, stats, global_ep)
        hv = _archive_hv()
        ep_mean = stats.get("mean_return", float("nan"))
        ent = stats.get("entropy", float("nan"))
        pbar.update(1)
        pbar.set_description(
            f"cmorl[{stage_tag}] hv:{hv:.3g} arch:{len(archive_pts)} "
            f"best:{state['best_global_return']:+.3g} mean:{ep_mean:+.3g} "
            f"ent:{ent:.3f} nan:{int(stats.get('nan_skip_count', 0))}"
        )
        if (global_ep + 1) % 5 == 0:
            tqdm.write(
                format_milestone_line("cmorl_ray", global_ep, total_episodes, stats, state)
            )
        log_dict = build_wandb_log_dict(stats, state, global_ep)
        log_dict.setdefault("entropy", ent)
        log_dict["cmorl/stage"] = 1 if stage_tag == "init" else 2
        log_dict["cmorl/hypervolume"] = hv
        log_dict["cmorl/archive_size"] = len(archive_pts)
        for k, nm in enumerate(obj_names):
            log_dict[f"cmorl/pref/{nm}"] = float(w_full[obj_idx[k]])
        if target is not None:
            log_dict["cmorl/extend_objective"] = obj_names[target]
        wandb.log(log_dict)
        if best_seq_every > 0 and (global_ep + 1) % best_seq_every == 0:
            dump_best_sequences_json(state, best_seq_json_path)
            wandb.log(build_best_sequences_wandb_payload(state, ep=global_ep))
        _dump_pareto()  # keep the Pareto front (points+sequences) JSON current
        global_ep += 1
        return stats

    # ---- Stage 1: Pareto initialization — population of M policies ----
    tqdm.write(
        f"  [C-MORL] Stage 1: Pareto-init, {len(prefs)} policies × "
        f"{eps_per_policy} eps; objectives={obj_names}"
    )
    ray.get(actor.set_extension.remote(-1))  # no extension during Stage 1
    policy_records = []  # (ckpt_path, w_np, obj_vec_np)
    for k, w in enumerate(prefs):
        ray.get(actor.reset_agent.remote(int(args.seed) + 1000 + k))
        w_full = _pref_to_full_weights(w, obj_idx)
        last_stats = None
        for _ in range(eps_per_policy):
            last_stats = _episode(w_full, "init")
        tvec = last_stats.get("terminal_reward_vec", {}) if last_stats else {}
        obj_vec = np.array([tvec.get(n, np.nan) for n in obj_names], dtype=np.float64)
        ckpt = os.path.join(ckpt_root, f"policy_{k}")
        ray.get(actor.save_agent.remote(ckpt))
        policy_records.append((ckpt, np.asarray(w, dtype=np.float64), obj_vec))
        tqdm.write(
            f"  [C-MORL]   policy {k + 1}/{len(prefs)} done; "
            f"obj=({', '.join(f'{n}={v:.4g}' for n, v in zip(obj_names, obj_vec))})"
        )

    # ---- Stage 2: IPO Pareto extension ----
    if do_stage2:
        beta = float(args.extension_beta)
        ipo_t = float(args.ipo_t)
        feasible = [r for r in policy_records if np.all(np.isfinite(r[2]))]
        if not feasible:
            tqdm.write("  [C-MORL] Stage 2 skipped: no policy with finite objectives.")
        else:
            tqdm.write(
                f"  [C-MORL] Stage 2: {args.extension_steps} IPO extension steps × "
                f"{args.extension_episodes} eps (beta={beta}, ipo_t={ipo_t})"
            )
            for step in range(int(args.extension_steps)):
                m = step % n_obj
                # extend the front extreme on m: the policy that already does best on m
                ckpt, _w_sel, obj_vec = max(feasible, key=lambda r: r[2][m])
                if not ray.get(actor.load_agent.remote(ckpt)):
                    # load_agent returns False on a deserialise/missing-file
                    # failure. Silently proceeding would run the IPO extension
                    # on whatever policy is currently loaded (the last Stage-1
                    # member), recording the result against the WRONG agent —
                    # skip this extension step instead.
                    tqdm.write(
                        f"  [C-MORL]   step {step + 1}: load_agent({ckpt}) "
                        "FAILED — skipping this extension step"
                    )
                    continue
                # other objectives constrained to not drop > (1-beta) fraction
                thresholds = {
                    nm: float(obj_vec[j]) - (1.0 - beta) * abs(float(obj_vec[j]))
                    for j, nm in enumerate(obj_names)
                    if j != m
                }
                ray.get(actor.set_extension.remote(int(obj_idx[m]), thresholds, ipo_t))
                tqdm.write(
                    f"  [C-MORL]   step {step + 1}/{args.extension_steps}: "
                    f"push '{obj_names[m]}', constrain {thresholds}"
                )
                w_full = _pref_to_full_weights(
                    np.eye(n_obj, dtype=np.float32)[m], obj_idx
                )
                for _ in range(int(args.extension_episodes)):
                    _episode(w_full, "extend", target=m)
            ray.get(actor.set_extension.remote(-1))  # clear extension

    pbar.close()
    tqdm.write(
        f"  [C-MORL] final Pareto archive: {len(archive_pts)} points, "
        f"hypervolume={_archive_hv():.6g}"
    )
    for _pt in archive_pts:
        tqdm.write(
            "    " + ", ".join(f"{nm}={v:.4g}" for nm, v in zip(obj_names, _pt))
        )
    _dump_pareto()
    tqdm.write(f"  [C-MORL] Pareto front (points+sequences) -> {pareto_path}")

    tqdm.write("\n========== FINAL ==========")
    tqdm.write(
        f"  best return: {state['best_global_return']:+.6g}  "
        f"(ep {state['best_global_ep']})"
    )
    if state["best_global_rewards"]:
        tqdm.write("  best-overall-trajectory per-channel rewards (raw):")
        for name in sorted(state["best_global_rewards"]):
            raw = state["best_global_rewards"][name]
            wt = state["best_global_weighted_split"].get(name, 0.0)
            tqdm.write(f"    {name:<18s}  raw={raw:+.4g}   weighted={wt:+.4g}")
    # Loud CPU-pool-sentinel summary. If ``total_timeouts == 0`` across a
    # full training run, the sentinel-callback path in
    # ``ppo_ray_worker._fan_out_tokenize`` is dead code and can be
    # deleted (see CpuApproxPool.fetch_timeout_delta docstring). We log
    # the running total here as the canonical "did the sentinel fire
    # this run" signal so the cleanup decision is one line away.
    try:
        pool_stats = ray.get(actor.pool_stats.remote()) if hasattr(actor, "pool_stats") else None
    except Exception:
        pool_stats = None
    if pool_stats is not None:
        n_t = int(pool_stats.get("timeouts", 0))
        n_c = int(pool_stats.get("calls", 0))
        tag = "(sentinel path is dead code — can be dropped)" if n_t == 0 else ""
        tqdm.write(f"  cpu-pool timeouts total: {n_t} / {n_c} calls  {tag}")
        final_payload_extra = {
            "final/cpu_pool_timeouts_total": n_t,
            "final/cpu_pool_calls_total": n_c,
        }
    else:
        final_payload_extra = {}
    # Final JSON + wandb dump of the running bests. The JSON file is
    # the canonical record (durable, easy to diff across runs); the
    # wandb Table is a UI nicety so the bests show up in the run page
    # without having to download the file.
    final_path = dump_best_sequences_json(state, best_seq_json_path)
    if final_path:
        tqdm.write(f"  best-sequences JSON written: {final_path}")
    final_payload = build_best_sequences_wandb_payload(
        state, ep=args.episodes - 1
    )
    final_payload["final/best_return"] = state["best_global_return"]
    final_payload["final/best_ep"] = state["best_global_ep"]
    final_payload.update(final_payload_extra)
    if final_path:
        try:
            from alphagrad.approx.common.render_sequence import (
                render_best_sequences_json,
            )
            for k, v in render_best_sequences_json(final_path).items():
                final_payload[f"final/{k}/repr"] = v
        except Exception as exc:
            tqdm.write(f"  [render] best-sequence repr failed: {exc}")
    # The Table goes through wandb's artifact upload path which on some
    # wandb configurations (`base_url='redacted'` in
    # ~/.config/wandb/settings) hits a pydantic-v2 URL validation
    # error and crashes the driver AFTER all training has completed
    # (see job 45278 in slurm/logs). The JSON dump above is the
    # canonical record; the Table is decorative. Wrap in try/except so
    # the wandb side-effect can never kill the training run.
    try:
        final_table = build_best_sequences_wandb_table(state)
        if final_table is not None:
            final_payload["final/best_sequences_table"] = final_table
    except Exception as exc:
        tqdm.write(
            f"  [wandb] best-sequences Table construction failed: {exc}; "
            f"falling back to scalar-only payload (JSON dump on disk "
            f"still has the full data)."
        )
    try:
        wandb.log(final_payload)
    except Exception as exc:
        # If wandb.log itself throws (e.g., from the Table → artifact
        # → pydantic URL validation chain), log the error and continue
        # to wandb.finish() so the run still closes cleanly.
        tqdm.write(
            f"  [wandb] final wandb.log failed: {exc}; continuing to "
            f"wandb.finish() to close the run."
        )
        # Retry without the Table (most common failure mode).
        final_payload.pop("final/best_sequences_table", None)
        try:
            wandb.log(final_payload)
        except Exception as exc2:
            tqdm.write(f"  [wandb] retry without Table also failed: {exc2}")
    try:
        wandb.finish()
    except Exception as exc:
        tqdm.write(f"  [wandb] finish failed: {exc}")

    ray.kill(actor, no_restart=True)
    for c in cpu_workers:
        ray.kill(c, no_restart=True)
    # Tear down the shared compile cache last so any in-flight
    # ``cached_compile`` calls finish before the coordinator dies.
    _kill_compile_cache()
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
    # Cap Ray's LOGICAL cpu count for a local cluster. Ray prestarts one
    # python worker per logical CPU; on a high-core node (e.g. the
    # 384-core pgi15-cpu nodes) the default auto-detect spawns hundreds
    # of workers — a fork storm that hangs init (profiled 2026-06-06:
    # do_wait block + num_prestart_python_workers=384). We only need
    # enough logical CPUs to schedule the driver + num_cpu_workers actors
    # (+ headroom). The PHYSICAL cores stay fully available to the actors
    # via sched_setaffinity (set in CpuApproximationActor.__init__) — Ray
    # logical CPUs are a scheduling abstraction, not a core binding. Only
    # applied for a locally-started cluster (not when joining via
    # --ray-address, where the cluster sized itself).
    if not args.ray_address:
        ray_cpus = int(getattr(args, "num_cpu_workers", 4)) + 8
        init_kwargs["num_cpus"] = ray_cpus
    ray.init(**init_kwargs, ignore_reinit_error=True)

    rc = _run(args)
    ray.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
