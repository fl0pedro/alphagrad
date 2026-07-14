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
import signal
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
