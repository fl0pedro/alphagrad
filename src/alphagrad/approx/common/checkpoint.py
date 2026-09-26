"""Unified checkpointing for PPO-ray and MuZero-ray trainers.

A checkpoint bundles four files in a directory:

* ``agent.eqx`` — equinox tree-leaves of the agent (params + buffers)
* ``opt_state.pkl`` — optax opt_state, pickled (it's a pytree of arrays)
* ``meta.json`` — small JSON header with the episode counter, RNG seed,
  reward weights, best-so-far state. Human-readable so a
  failed resume is easy to diagnose.
* ``replay.pkl`` (optional, MuZero only) — replay buffer pickled

The format is intentionally simple — we don't need versioning yet
because the checkpoint is only ever read by the same training run
(crash-resume scenario), not migrated across model architectures.

A SIGTERM handler is provided so SLURM job timeouts get one last
checkpoint before the process is killed.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil as _shutil
import signal
import subprocess
from typing import Any, Callable, Optional

import numpy as np


def _atomic_write(path: str, data: bytes) -> None:
    """Write `data` to `path` via a temp file + rename. Crash-safe."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def save_state(
    checkpoint_dir: str,
    *,
    agent,
    opt_state,
    episode_counter: int,
    reward_weights: Any = None,
    best_state: dict | None = None,
    extras: dict | None = None,
    replay_buffer: Any = None,
) -> str:
    """Save trainer state to ``checkpoint_dir``.

    Idempotent: overwrites the existing dir. Writes atomically per
    file so a partial write doesn't corrupt the checkpoint a previous
    iteration produced.

    Returns the directory path. Raises only on filesystem errors —
    serialization errors are caught and logged so a buggy checkpoint
    can't crash training.
    """
    import equinox as eqx

    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, "agent.eqx"), agent,
        )
    except Exception as exc:
        print(f"[checkpoint] WARN agent serialise failed: {exc}")
    try:
        _atomic_write(
            os.path.join(checkpoint_dir, "opt_state.pkl"),
            pickle.dumps(opt_state, protocol=pickle.HIGHEST_PROTOCOL),
        )
    except Exception as exc:
        print(f"[checkpoint] WARN opt_state serialise failed: {exc}")

    meta = {
        "episode_counter": int(episode_counter),
        "reward_weights": (
            np.asarray(reward_weights).tolist() if reward_weights is not None else None
        ),
        "best_state": best_state or {},
        "extras": extras or {},
    }
    try:
        _atomic_write(
            os.path.join(checkpoint_dir, "meta.json"),
            json.dumps(meta, default=str).encode("utf-8"),
        )
    except Exception as exc:
        print(f"[checkpoint] WARN meta write failed: {exc}")

    if replay_buffer is not None:
        try:
            from alphagrad.approx.common.replay import save_replay_buffer
            save_replay_buffer(
                replay_buffer,
                os.path.join(checkpoint_dir, "replay.pkl"),
            )
        except Exception as exc:
            print(f"[checkpoint] WARN replay save failed: {exc}")
    return checkpoint_dir


def load_state(
    checkpoint_dir: str,
    *,
    template_agent,
    template_opt_state=None,
) -> dict | None:
    """Load trainer state from ``checkpoint_dir`` and return a dict
    of restored pieces, or ``None`` if no checkpoint exists.

    Args:
        checkpoint_dir: directory written by :func:`save_state`.
        template_agent: an instance of the same equinox class —
            ``eqx.tree_deserialise_leaves`` needs the shape template.
        template_opt_state: optional template for the opt_state pytree
            (some optax states have non-array leaves that need their
            initial structure).

    Returns:
        dict with keys ``agent``, ``opt_state``, ``episode_counter``,
        ``reward_weights``, ``best_state``,
        ``extras``, ``replay_path``. Missing pieces are ``None``.
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    agent_path = os.path.join(checkpoint_dir, "agent.eqx")
    opt_path = os.path.join(checkpoint_dir, "opt_state.pkl")
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    replay_path = os.path.join(checkpoint_dir, "replay.pkl")
    if not (os.path.exists(agent_path) and os.path.exists(meta_path)):
        return None

    import equinox as eqx

    out: dict[str, Any] = {
        "agent": None,
        "opt_state": None,
        "episode_counter": 0,
        "reward_weights": None,
        "best_state": {},
        "extras": {},
        "replay_path": replay_path if os.path.exists(replay_path) else None,
    }
    try:
        out["agent"] = eqx.tree_deserialise_leaves(agent_path, template_agent)
    except Exception as exc:
        print(f"[checkpoint] WARN agent load failed: {exc}")
        return None
    if template_opt_state is not None and os.path.exists(opt_path):
        try:
            with open(opt_path, "rb") as f:
                out["opt_state"] = pickle.load(f)
        except Exception as exc:
            print(f"[checkpoint] WARN opt_state load failed: {exc}")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        out["episode_counter"] = int(meta.get("episode_counter", 0))
        rw = meta.get("reward_weights")
        if rw is not None:
            out["reward_weights"] = np.asarray(rw, dtype=np.float32)
        out["best_state"] = meta.get("best_state", {}) or {}
        out["extras"] = meta.get("extras", {}) or {}
    except Exception as exc:
        print(f"[checkpoint] WARN meta load failed: {exc}")
    return out


def install_sigterm_handler(save_callable: Callable[[], None]) -> None:
    """Register a SIGTERM handler that calls ``save_callable`` once.

    SLURM sends SIGTERM ``--time``-grace-period seconds before
    SIGKILL. The handler swallows further SIGTERMs after the first
    so a slow checkpoint doesn't loop.

    Idempotent: re-installing replaces the prior handler.
    """
    state = {"saved": False}

    def _handler(signum, frame):  # pragma: no cover (signal path)
        if state["saved"]:
            return
        state["saved"] = True
        try:
            print(f"[checkpoint] SIGTERM received — saving final checkpoint")
            save_callable()
            print(f"[checkpoint] final checkpoint saved.")
        except Exception as exc:
            print(f"[checkpoint] final checkpoint FAILED: {exc}")
        # Re-raise to let SLURM proceed with shutdown.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _handler)


# ===========================================================================
# THE PPO TRAINER'S EXACT CHECKPOINT (owner ruling 2026-09-15)
# ===========================================================================
#
# Everything above this line belongs to the RAY trainers (ppo_ray_worker,
# mu0_ray_worker, gfn_ray_worker). It is a crash-resume convenience: it
# swallows serialisation errors and keeps training. Nothing below shares that
# behaviour. This half is for `src/alphagrad/approx/ppo.py` and its contract
# is EXACT RESUME, so every failure here RAISES. A checkpoint that silently
# lost a field would produce a resumed run that looks right and is not.
#
# THE FORMAT, AND WHY IT IS TWO FILES
# -----------------------------------
# One directory per checkpoint, named by the episode counter so the last two
# are found by sorting. Inside it:
#
#   state.eqx   equinox `tree_serialise_leaves` of ONE dict holding every
#               piece the next episode's ARITHMETIC reads: the policy
#               parameters, the optimiser state, the probe trees and their
#               optimiser states, the three PopArt accumulators, the global
#               step, the RNG key, and the two host-side duals. equinox
#               writes each leaf with `numpy.save`, so a float32 array, a
#               uint32 key and a Python float all come back BIT for BIT, and
#               `tree_deserialise_leaves` checks the type, the shape and the
#               dtype of every leaf against the live template. That check is
#               the reason for the format: an optimiser chain that changed
#               between the save and the load fails loudly instead of
#               restoring a tree that no longer matches the parameters.
#
#   meta.json   everything that is BOOKKEEPING rather than arithmetic: the
#               episode counter, the two bin policies, the Pareto archive,
#               the top-N heaps and the best-so-far record, the wandb run id
#               and the whole argument namespace. These are read by people
#               and by other tools (`pareto_front.json` has the same shape),
#               and none of them is an array, so a binary format would only
#               make them harder to diagnose. JSON round-trips a Python
#               float64 exactly -- `repr` is round-trip exact -- and every
#               other value here is an int, a string or a bool.
#
# WHY NOT ONE FORMAT. A single JSON would have to base64 the parameter
# arrays, which is neither readable nor cheap. A single equinox file would
# have to invent array encodings for a list of elimination sequences and for
# the argument namespace, and would lose the diagnosability that is the whole
# reason the meta half exists. The split is along a real seam: arrays that
# have to come back bit for bit, and records that have to be readable.
#
# WHAT IS NOT IN THE CHECKPOINT, AND WHY
# --------------------------------------
# * The frozen KL reference policy. It is the identity-initialised agent and
#   is rebuilt from `--seed` on every start, before the restore. Saving it
#   would let a resume disagree with the run it continues about what the
#   reference is.
# * The wandb `Table` of elimination orders. wandb owns it, the resume
#   re-attaches to the same run, and the table is append-only output.
# * The measure actors. A checkpoint is taken at a QUIESCENT point of the
#   episode pipeline, where no measurement ticket is in flight, so there is
#   nothing of theirs to save. `save` asserts that.

#: Bumped whenever the on-disk layout changes in a way a reader must notice.
PPO_CKPT_FORMAT = 1

#: Directory prefix. The episode counter is zero-padded so a plain sort of
#: the directory listing is a sort by episode.
PPO_CKPT_PREFIX = "ppo_ckpt_ep"

#: The arguments a resume is allowed to change. `--episodes` because a resume
#: may extend the run, and `--resume` itself because the first leg did not
#: carry it. `--checkpoint-keep-at` (dsnn-dfw.117) only pins which OLD
#: checkpoints the pruning skips deleting; it reads no state, changes no
#: computation and steps no gradient, so it does not touch training. `--gpus`
#: and `--measure-gpus` (owner ruling 2026-09-26, dsnn-dfw.274): a restart
#: times its reference again, so the GPU numbers add nothing, and the GPU
#: MODEL is checked on its own by `check_resume_gpu_model`. EVERY
#: other difference raises: the checkpoint is a state of one configuration
#: and restoring it under another one is not a resume.
PPO_RESUME_EXEMPT_ARGS = frozenset(
    {"episodes", "resume", "checkpoint_keep_at", "gpus", "measure_gpus"})

#: The meta.json field that names the GPU model of the trainer's device.
PPO_GPU_MODEL_FIELD = "trainer_gpu_model"


class CheckpointError(RuntimeError):
    """Any failure of the exact-checkpoint path. Never swallowed."""


def add_checkpoint_args(p) -> None:
    """Install `--checkpoint-every` and `--resume` on the PPO argparser."""
    p.add_argument(
        "--checkpoint-every", type=int, default=50, metavar="N",
        help="Write a checkpoint every N episodes and once at the end of the "
             "run, into the run directory (the wandb run dir when wandb is "
             "on, else the working directory). The last two checkpoints of a "
             "run are kept and older ones are deleted. 0 turns checkpointing "
             "OFF. A checkpoint is taken at a QUIESCENT point of the episode "
             "pipeline, so under --measure-pipeline 1 it first drains the "
             "pending episode, which runs that episode's PPO update one "
             "iteration earlier than the pipeline would have. The drain is "
             "therefore part of the schedule and two runs compare only at "
             "equal --checkpoint-every. Under --measure-pipeline 0 nothing "
             "is ever pending and the drain is a no-op.")
    p.add_argument(
        "--resume", type=str, default="", metavar="PATH",
        help="Continue the run saved in the checkpoint directory PATH. The "
             "argument namespace must match the one the checkpoint was "
             "written with, except --episodes (a resume may extend the run), "
             "--resume itself, --checkpoint-keep-at, --gpus and "
             "--measure-gpus. Any other difference raises, and so does a "
             "trainer GPU of another model than the one the checkpoint "
             "records. When wandb "
             "is on the resumed run attaches to the SAME wandb run by id.")


def run_directory(wandb_on: bool = True) -> str:
    """The run's output directory: the wandb run dir, else the cwd.

    The same rule `_resolve_plan_log_path` and `_dump_pareto` use, so a run's
    plan log, its Pareto dump and its checkpoints land together. `wandb_on`
    is False under `--wandb disabled`, where wandb still holds a run object
    with a directory of its own that nothing else of this run writes to.
    """
    d = "."
    if wandb_on:
        try:
            import wandb as _wandb
            d = _wandb.run.dir if _wandb.run is not None else "."
        except Exception:
            d = "."
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = "."
    return d


def checkpoint_dir_name(episode: int) -> str:
    return f"{PPO_CKPT_PREFIX}{int(episode):09d}"


def _checkpoint_episode(path: str) -> int:
    """The episode number a checkpoint directory's own name encodes."""
    return int(os.path.basename(path)[len(PPO_CKPT_PREFIX):])


def list_checkpoints(run_dir: str) -> list:
    """Every checkpoint directory in `run_dir`, oldest episode first."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    out = []
    for name in os.listdir(run_dir):
        if not name.startswith(PPO_CKPT_PREFIX):
            continue
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and os.path.exists(
                os.path.join(full, "meta.json")):
            out.append(full)
    return sorted(out)


# ---------------------------------------------------------------------------
# The bookkeeping objects, to and from plain JSON values.
# ---------------------------------------------------------------------------

def bin_policy_to_json(bp) -> dict:
    """The MUTABLE state of an `episode_stream.BinPolicy`.

    Only the state, never the configuration: the history window, the margin,
    the cap and the floor come from the environment and the arguments and are
    re-derived on the resumed run, where a difference has to surface as a
    mismatch rather than be overwritten from the file.
    """
    return {
        "log2": int(bp.log2),
        "last_used": (None if bp.last_used is None else int(bp.last_used)),
        "overflowed": bool(bp.overflowed),
        "recent": [int(x) for x in bp.recent],
        # Carried for the check below, not to be restored.
        "initial": int(bp.initial),
        "window": int(bp.window),
        "margin": float(bp.margin),
        "cap": int(bp.cap),
        "floor": int(bp.floor),
    }


def bin_policy_from_json(bp, d: dict) -> None:
    """Put a saved bin state back on a freshly built `BinPolicy`, in place."""
    for field in ("initial", "window", "margin", "cap", "floor"):
        live = getattr(bp, field)
        saved = d[field]
        live = float(live) if field == "margin" else int(live)
        saved = float(saved) if field == "margin" else int(saved)
        if live != saved:
            raise CheckpointError(
                f"the bin policy's {field} is {live} on this run and {saved} "
                f"in the checkpoint. The bin is a compiled SHAPE, so a run "
                f"that resumes under a different bin configuration is not the "
                f"run that was saved.")
    bp.log2 = int(d["log2"])
    bp.last_used = (None if d["last_used"] is None else int(d["last_used"]))
    bp.overflowed = bool(d["overflowed"])
    bp.recent.clear()
    for x in d["recent"]:
        bp.recent.append(int(x))


def pareto_archive_to_json(archive) -> dict:
    """The archive's points, sequences and admission episodes, plus the
    candidate log and the hypervolume reference."""
    # TICKET dsnn-dfw.44: the band archive carries a band, a merge count and
    # the POOLED WINDOWS per point, and has no reward indices at all. The
    # windows are the state a resumed run needs to keep re-fitting a band
    # that is already tighter than one measurement.
    if hasattr(archive, "samples"):
        out = {
            "kind": ("quantile-front" if hasattr(archive, "quantiles")
                     else "ratio-band"),
            "obj_names": [str(n) for n in archive.obj_names],
            "senses": [str(s) for s in archive.senses],
            "cap": None if archive.cap is None else int(archive.cap),
            "pool_cap": int(archive.pool_cap),
            "quality_floor": (None if archive.quality_floor is None
                              else float(archive.quality_floor)),
            "pts": [[float(x) for x in p] for p in archive.pts],
            "lo": [[float(x) for x in p] for p in archive.lo],
            "hi": [[float(x) for x in p] for p in archive.hi],
            "samples": [[[float(x) for x in w] for w in s]
                        for s in archive.samples],
            # The MEMBER PLANS of every point. A resumed run without these
            # would keep the band and forget which plans earned it.
            "members": [[{"key": str(m["key"]), "seq": m["seq"],
                          "windows": int(m["windows"]), "n": int(m["n"]),
                          "first_episode": int(m["first_episode"]),
                          "last_episode": int(m["last_episode"])}
                         for m in ms] for ms in archive.members],
            "counts": [int(c) for c in archive.counts],
            "seqs": list(archive.seqs),
            "eps": [int(e) for e in archive.eps],
            "mem_sources": [{str(k): int(v) for k, v in m.items()}
                            for m in archive.mem_sources],
            "all_candidates": list(archive.all_candidates),
            "seen": sorted(str(s) for s in archive._seen),
            "hv_ref": (None if archive._hv_ref is None
                       else [float(x) for x in archive._hv_ref]),
            "n_merged": int(archive.n_merged),
            "n_dropped_cap": int(archive.n_dropped_cap),
        }
        if hasattr(archive, "quantiles"):
            # dsnn-dfw.293: the quantile front's bands, where they came from, and its repeat count.
            out.update(banded=[str(b) for b in archive.banded],
                       quantiles=list(archive.quantiles),
                       details=list(archive.details),
                       n_repeats=int(archive.n_repeats))
        return out
    return {
        "obj_names": [str(n) for n in archive.obj_names],
        "obj_idx": [int(i) for i in archive.obj_idx],
        "quality_floor": (None if archive.quality_floor is None
                          else float(archive.quality_floor)),
        "pts": [[float(x) for x in p] for p in archive.pts],
        "seqs": list(archive.seqs),
        "eps": [int(e) for e in archive.eps],
        "all_candidates": list(archive.all_candidates),
        "seen": sorted(str(s) for s in archive._seen),
        "hv_ref": (None if archive._hv_ref is None
                   else [float(x) for x in archive._hv_ref]),
    }


def pareto_archive_from_json(archive, d: dict) -> None:
    """Refill a freshly built `ParetoArchive` from a saved one, in place."""
    if [str(n) for n in archive.obj_names] != [str(n) for n in d["obj_names"]]:
        raise CheckpointError(
            f"the Pareto archive's objectives are {archive.obj_names} on this "
            f"run and {d['obj_names']} in the checkpoint.")
    _band = hasattr(archive, "samples")
    _kind = ("quantile-front" if hasattr(archive, "quantiles")
             else "ratio-band" if _band else "reward-vector")
    _saved = str(d.get("kind") or "reward-vector")
    if _kind != _saved:
        raise CheckpointError(
            f"the Pareto archive is a {_kind} archive on this run and a "
            f"{_saved} archive in the checkpoint.")
    if _band:
        _senses = [str(s) for s in (d.get("senses")
                                    or ["min"] * len(d["obj_names"]))]
        if [str(s) for s in archive.senses] != _senses:
            raise CheckpointError(
                f"the Pareto archive's senses are {archive.senses} on this "
                f"run and {_senses} in the checkpoint.")
        if _kind == "quantile-front" and (
                [str(b) for b in archive.banded] != [str(b) for b in d["banded"]]):
            raise CheckpointError(
                f"the Pareto front's banded objectives are {archive.banded} "
                f"on this run and {d['banded']} in the checkpoint.")
        archive.cap = None if d["cap"] is None else int(d["cap"])
        archive.pool_cap = int(d["pool_cap"])
        archive.pts = [np.asarray(p, dtype=np.float64) for p in d["pts"]]
        archive.lo = [np.asarray(p, dtype=np.float64) for p in d["lo"]]
        archive.hi = [np.asarray(p, dtype=np.float64) for p in d["hi"]]
        archive.samples = [[np.asarray(w, dtype=np.float64) for w in s]
                           for s in d["samples"]]
        archive.members = [[{"key": str(m["key"]), "seq": m["seq"],
                             "windows": int(m["windows"]), "n": int(m["n"]),
                             "first_episode": int(m["first_episode"]),
                             "last_episode": int(m["last_episode"])}
                            for m in ms] for ms in d["members"]]
        archive.counts = [int(c) for c in d["counts"]]
        archive.seqs = list(d["seqs"])
        archive.eps = [int(e) for e in d["eps"]]
        archive.mem_sources = [
            {str(k): int(v) for k, v in m.items()}
            for m in (d.get("mem_sources") or [{} for _p in d["pts"]])]
        archive.all_candidates = list(d["all_candidates"])
        archive._seen = set(d["seen"])
        archive._hv_ref = (None if d["hv_ref"] is None
                           else np.asarray(d["hv_ref"], dtype=np.float64))
        archive.n_merged = int(d["n_merged"])
        archive.n_dropped_cap = int(d["n_dropped_cap"])
        if _kind == "quantile-front":
            archive.quantiles = [dict(q) for q in d["quantiles"]]
            archive.details = list(d["details"])
            archive.n_repeats = int(d["n_repeats"])
        return
    if [int(i) for i in archive.obj_idx] != [int(i) for i in d["obj_idx"]]:
        raise CheckpointError(
            f"the Pareto archive's reward indices are {archive.obj_idx} on "
            f"this run and {d['obj_idx']} in the checkpoint.")
    archive.pts = [np.asarray(p, dtype=np.float64) for p in d["pts"]]
    archive.seqs = list(d["seqs"])
    archive.eps = [int(e) for e in d["eps"]]
    archive.all_candidates = list(d["all_candidates"])
    archive._seen = set(d["seen"])
    archive._hv_ref = (None if d["hv_ref"] is None
                       else np.asarray(d["hv_ref"], dtype=np.float64))


def host_state_to_json(host_state: dict) -> dict:
    """The trainer's `host_state`, minus the wall-clock origin.

    `_wall_t0` is deliberately dropped. It is this process's `perf_counter`
    origin and restoring another process's would make `time/wall_seconds`
    meaningless. A resumed run's wall clock starts at its own start.
    """
    out = {}
    for name in ("samplecounts", "collapsed_total"):
        if name in host_state:
            out[name] = int(host_state[name])
    out["best_global_return"] = float(host_state["best_global_return"])
    out["best_global_act_seq"] = host_state["best_global_act_seq"]
    for name in ("top_n_total", "top_n_cmp", "top_n_mem", "top_n_acc"):
        out[name] = [
            [float(p[0]), int(p[1]), [float(x) for x in p[2]], p[3]]
            for p in host_state[name]
        ]
    return out


def host_state_from_json(host_state: dict, d: dict) -> None:
    """Put a saved `host_state` back, in place, keeping this run's `_wall_t0`."""
    for name in ("samplecounts", "collapsed_total"):
        if name in d:
            host_state[name] = int(d[name])
    host_state["best_global_return"] = float(d["best_global_return"])
    host_state["best_global_act_seq"] = d["best_global_act_seq"]
    for name in ("top_n_total", "top_n_cmp", "top_n_mem", "top_n_acc"):
        host_state[name] = [
            (float(p[0]), int(p[1]), [float(x) for x in p[2]], p[3])
            for p in d[name]
        ]


# ---------------------------------------------------------------------------
# The argument namespace.
# ---------------------------------------------------------------------------

def args_to_json(args) -> dict:
    """The whole argument namespace as plain JSON values."""
    out = {}
    for name, value in sorted(vars(args).items()):
        out[name] = _jsonable(value)
    return out


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise CheckpointError(
        f"an argument of type {type(value).__name__} cannot be written to the "
        f"checkpoint's argument namespace: {value!r}. Add an encoding for it "
        f"rather than dropping it -- a dropped argument is an argument a "
        f"resume cannot check.")


def check_resume_args(saved: dict, args) -> None:
    """RAISE unless the live arguments match the saved ones.

    Exempt: `--episodes`, because a resume may extend the run, `--resume`
    itself, and the names in PPO_RESUME_EXEMPT_ARGS. Everything else is part
    of the state the checkpoint is a state OF.
    """
    live = args_to_json(args)
    problems = []
    for name in sorted(set(saved) | set(live)):
        if name in PPO_RESUME_EXEMPT_ARGS:
            continue
        if name not in saved:
            problems.append(f"  {name}: absent from the checkpoint, "
                            f"{live[name]!r} on the command line")
        elif name not in live:
            problems.append(f"  {name}: {saved[name]!r} in the checkpoint, "
                            f"absent on the command line")
        elif saved[name] != live[name]:
            problems.append(f"  {name}: {saved[name]!r} in the checkpoint, "
                            f"{live[name]!r} on the command line")
    if problems:
        raise CheckpointError(
            "the command line does not match the checkpoint's argument "
            "namespace, so this is not a resume of that run:\n"
            + "\n".join(problems)
            + "\nOnly --episodes and --resume may differ.")
    saved_eps = int(saved.get("episodes", 0))
    live_eps = int(live["episodes"])
    if live_eps < saved_eps:
        raise CheckpointError(
            f"--episodes {live_eps} is below the checkpoint's {saved_eps}. A "
            f"resume may EXTEND a run; it cannot shorten one, because the "
            f"learning-rate schedule's horizon is built from --episodes and a "
            f"shorter horizon is a different schedule.")


# ---------------------------------------------------------------------------
# The GPU model of the trainer (owner ruling 2026-09-26, dsnn-dfw.274): a run
# may resume on other GPU numbers, but never on another GPU model.
# ---------------------------------------------------------------------------

def _nvidia_smi_name(gpu_index: str, run) -> str | None:
    why = None
    try:
        out = run(["nvidia-smi", "--query-gpu=index,name",
                   "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        why = repr(exc)
    else:
        if out.returncode != 0:
            why = f"nvidia-smi exited {out.returncode}: {out.stderr.strip()}"
        else:
            for line in out.stdout.splitlines():
                index, _, name = line.partition(",")
                if index.strip() == str(gpu_index).strip():
                    return name.strip()
            why = f"nvidia-smi lists no GPU {gpu_index}"
    print(f"[checkpoint] the nvidia-smi name of the trainer's GPU "
          f"{gpu_index} is not known: {why}", flush=True)
    return None


def trainer_gpu_model(device, gpu_index: str, run=subprocess.run) -> dict:
    return {"device_kind": str(device.device_kind),
            "nvidia_smi_name": (_nvidia_smi_name(gpu_index, run)
                                if device.platform == "gpu" else None)}


def _gpu_model_text(model: dict) -> str:
    name = model.get("nvidia_smi_name")
    return (str(model["device_kind"])
            + ("" if name is None else f" (nvidia-smi: {name})"))


def check_resume_gpu_model(meta: dict, live: dict) -> None:
    saved = meta.get(PPO_GPU_MODEL_FIELD)
    if not isinstance(saved, dict) or "device_kind" not in saved:
        raise CheckpointError(
            f"the checkpoint has no {PPO_GPU_MODEL_FIELD!r} field with a "
            f"device_kind, so the GPU model its run started on is not known. "
            f"A run may resume only on the GPU model it started on (owner "
            f"ruling 2026-09-26, dsnn-dfw.274), and a checkpoint written "
            f"before that field existed cannot say which model that was.")
    names = (saved.get("nvidia_smi_name"), live.get("nvidia_smi_name"))
    if (saved["device_kind"] != live["device_kind"]
            or (None not in names and names[0] != names[1])):
        raise CheckpointError(
            f"the checkpoint was written on a {_gpu_model_text(saved)} and "
            f"this run's trainer is on a {_gpu_model_text(live)}. A run may "
            f"resume on other GPU numbers but never on another GPU model, "
            f"because latency is comparable only within one model (owner "
            f"ruling 2026-09-26, dsnn-dfw.274).")


# ---------------------------------------------------------------------------
# Save and load.
# ---------------------------------------------------------------------------

def _leaf_tag(leaf) -> str:
    """A description of one leaf that is the same in two PROCESSES.

    `repr` is not: a function or a plain object prints its address, and two
    runs of the same program print two different ones. So an address-bearing
    repr is reduced to the type alone, and every other leaf keeps its value.
    """
    t = type(leaf)
    name = f"{t.__module__}.{t.__qualname__}"
    r = repr(leaf)
    if "0x" in r:
        return name
    return f"{name}:{r}"


def _non_array_manifest(tree) -> list:
    """Every leaf equinox will NOT write, as `path -> tag`.

    equinox serialises array-like leaves and passes over the rest, returning
    the TEMPLATE's value for them on load. That is silent state loss, which
    this module does not allow, so the manifest is saved and compared.
    """
    import equinox as eqx
    import jax.tree_util as jtu

    out = []
    for path, leaf in jtu.tree_flatten_with_path(tree)[0]:
        if not eqx.is_array_like(leaf):
            out.append([jtu.keystr(path), _leaf_tag(leaf)])
    return out


def save_ppo_checkpoint(run_dir, *, episode, tree, meta, keep=2,
                         keep_at=None) -> str:
    """Write ONE checkpoint and prune all but the newest `keep`.

    `tree` is the arithmetic half (see the header); `meta` is the bookkeeping
    half and must already be JSON values. `keep_at` is a set of episodes the
    pruning never deletes, on top of the newest `keep`; every other
    checkpoint older than the newest `keep` is still removed. Returns the
    directory written.

    The write is staged in a sibling `.writing` directory and renamed into
    place, so a checkpoint directory that exists is a checkpoint that is
    complete.
    """
    import equinox as eqx

    if not run_dir:
        raise CheckpointError("the checkpoint needs a run directory.")
    os.makedirs(run_dir, exist_ok=True)
    final = os.path.join(run_dir, checkpoint_dir_name(episode))
    staging = final + ".writing"
    if os.path.exists(staging):
        _shutil.rmtree(staging)
    os.makedirs(staging)

    eqx.tree_serialise_leaves(os.path.join(staging, "state.eqx"), tree)

    doc = dict(meta)
    doc["format"] = PPO_CKPT_FORMAT
    doc["episode"] = int(episode)
    doc["non_array_leaves"] = _non_array_manifest(tree)
    _atomic_write(
        os.path.join(staging, "meta.json"),
        json.dumps(doc, indent=1, sort_keys=True).encode("utf-8"),
    )

    if os.path.exists(final):
        _shutil.rmtree(final)
    os.rename(staging, final)

    if keep is not None and int(keep) > 0:
        pinned = frozenset(int(e) for e in (keep_at or ()))
        existing = list_checkpoints(run_dir)
        for stale in existing[:max(0, len(existing) - int(keep))]:
            if _checkpoint_episode(stale) in pinned:
                continue
            _shutil.rmtree(stale)
    return final


def read_ppo_meta(path: str) -> dict:
    """The `meta.json` of a checkpoint directory. No template needed.

    This is what the trainer reads BEFORE it builds anything, so an argument
    mismatch is refused before a single array is allocated.
    """
    if not path:
        raise CheckpointError("--resume needs a checkpoint directory.")
    if not os.path.isdir(path):
        raise CheckpointError(
            f"--resume {path!r} is not a directory. It must be one of the "
            f"'{PPO_CKPT_PREFIX}*' directories a run wrote.")
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        raise CheckpointError(
            f"--resume {path!r} holds no meta.json, so it is not a complete "
            f"checkpoint.")
    with open(meta_path) as fh:
        doc = json.load(fh)
    fmt = int(doc.get("format", -1))
    if fmt != PPO_CKPT_FORMAT:
        raise CheckpointError(
            f"the checkpoint in {path!r} is format {fmt}; this build reads "
            f"format {PPO_CKPT_FORMAT}.")
    return doc


def load_ppo_tree(path: str, template):
    """The arithmetic half of a checkpoint, against a live `template`.

    `template` must be the same dict of live objects `save_ppo_checkpoint`
    was handed, freshly built by this run. equinox checks every leaf's type,
    shape and dtype against it, and the non-array manifest is checked here,
    so a checkpoint that does not fit this build raises instead of restoring
    a tree that silently keeps some of the template's values.
    """
    import equinox as eqx

    state_path = os.path.join(path, "state.eqx")
    if not os.path.exists(state_path):
        raise CheckpointError(
            f"--resume {path!r} holds no state.eqx, so it is not a complete "
            f"checkpoint.")
    saved_manifest = [list(x) for x in read_ppo_meta(path).get(
        "non_array_leaves", [])]
    live_manifest = [list(x) for x in _non_array_manifest(template)]
    if saved_manifest != live_manifest:
        raise CheckpointError(
            "the checkpoint's non-array leaves differ from this run's. "
            "equinox does not write those leaves, so restoring would keep "
            "this run's values for them and call the result a resume.\n"
            f"  in the checkpoint: {saved_manifest}\n"
            f"  on this run:       {live_manifest}")
    return eqx.tree_deserialise_leaves(state_path, template)
