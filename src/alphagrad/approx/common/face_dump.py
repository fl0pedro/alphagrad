from __future__ import annotations

import json
import os
import threading

import numpy as np

# ALPHAGRAD_FACE_DUMP=<dir> turns this on. Unset = off and every entry point
# returns before it touches jax.
_STATE: dict = {"read": False, "dir": None}
_LOCK = threading.Lock()
_FILES: dict = {}


def dump_dir() -> str | None:
    if not _STATE["read"]:
        d = os.environ.get("ALPHAGRAD_FACE_DUMP") or ""
        _STATE["dir"] = d or None
        _STATE["read"] = True
        if _STATE["dir"]:
            os.makedirs(_STATE["dir"], exist_ok=True)
    return _STATE["dir"]


def on() -> bool:
    return dump_dir() is not None


def _jsonable(v):
    a = np.asarray(v)
    if a.dtype.kind == "f":
        a = np.round(a.astype(np.float64), 6)
    return a.tolist()


def emit(stage: str, **cols) -> None:
    d = dump_dir()
    if d is None:
        return
    rec = {k: _jsonable(v) for k, v in cols.items()}
    with _LOCK:
        f = _FILES.get(stage)
        if f is None:
            f = open(os.path.join(d, f"{stage}.{os.getpid()}.jsonl"), "a")
            _FILES[stage] = f
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()


def record(stage: str, **arrays) -> None:
    if not on():
        return
    import jax

    def _cb(**kw):
        emit(stage, **kw)

    jax.debug.callback(_cb, ordered=False, **arrays)
