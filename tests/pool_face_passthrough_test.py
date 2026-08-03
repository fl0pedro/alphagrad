"""P3: the pool stack must CARRY face actions byte-identically.

`CpuApproximationServer.evaluate(face_specs=..., face_skips=...)` (the
in-process core of every pool actor) must produce exactly the tokens the
inline `_callback` produces for the same face-action prefix. The pool path
used to hard-refuse face actions; the worker built empty wires
unconditionally — this is the fidelity gate for the pass-through.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import alphagrad.approx.env as E
from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
from alphagrad.approx.env import (
    EnvConfig, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX, _callback,
)


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _setup():
    cj = jax.make_jaxpr(_fn)(*ARGS)
    cfg = EnvConfig(jaxpr=cj.jaxpr, argnums=(0, 1), has_aux=False,
                    sparse=True, cmp_type="graphax", mem_type="graphax",
                    per_face=True)
    return cfg, tuple(cj.literals), tuple(ARGS)


def _wires(V, seed=0):
    rng = np.random.default_rng(seed)
    order = np.asarray(rng.permutation(np.arange(1, V + 1)), np.int32)
    specs = np.full((V, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((V, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    faces[..., 2] = 0
    skips = np.zeros((V, MAX_FACES), np.int32)
    for i in range(V):
        if rng.random() < 0.4:
            specs[i, 0] = (0, 0, 2)
        for f in range(3):
            r = rng.random()
            if r < 0.3:
                faces[i, f, 0] = (0, 0, 2)
            elif r < 0.4:
                skips[i, f] = 1
    return order, specs, faces, skips


def _clear():
    E._INCR_STREAM_CACHE.clear()
    E._FACE_ENUM_CACHE.clear()


def test_server_face_tokens_equal_inline_callback():
    cfg, consts, args = _setup()
    V = len(cfg.jaxpr.eqns)
    order, specs, faces, skips = _wires(V)
    srv = CpuApproximationServer(SimpleNamespace(
        config=cfg, args=args, consts=consts, eval_args_samples=None))
    for stop in range(1, V + 1):
        _clear()
        t_ref, e_ref, _ = _callback(
            cfg, args, consts, jnp.asarray(order), jnp.asarray(specs),
            jnp.asarray(faces), jnp.asarray(skips), stop, init=True)
        _clear()
        t_srv, e_srv, _ = srv.evaluate(
            order, specs, stop, face_specs=faces, face_skips=skips,
            init=True)
        assert getattr(srv, "last_eval_error", None) is None, srv.last_eval_error
        assert np.array_equal(np.asarray(t_ref), np.asarray(t_srv)), (
            f"tokens diverged at stop={stop}")
        assert np.array_equal(np.asarray(e_ref), np.asarray(e_srv)), (
            f"eqn_ids diverged at stop={stop}")


def test_server_without_faces_unchanged():
    """No face wires handed over -> the worker builds the empty (-1/0)
    per-vertex wires exactly as before the pass-through existed."""
    cfg, consts, args = _setup()
    V = len(cfg.jaxpr.eqns)
    order, specs, _f, _s = _wires(V, seed=1)
    empty_f = np.full((V, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    empty_k = np.zeros((V, MAX_FACES), np.int32)
    srv = CpuApproximationServer(SimpleNamespace(
        config=cfg, args=args, consts=consts, eval_args_samples=None))
    stop = V
    _clear()
    t_ref, e_ref, _ = _callback(
        cfg, args, consts, jnp.asarray(order), jnp.asarray(specs),
        jnp.asarray(empty_f), jnp.asarray(empty_k), stop, init=True)
    _clear()
    t_srv, e_srv, _ = srv.evaluate(order, specs, stop, init=True)
    assert getattr(srv, "last_eval_error", None) is None, srv.last_eval_error
    assert np.array_equal(np.asarray(t_ref), np.asarray(t_srv))
    assert np.array_equal(np.asarray(e_ref), np.asarray(e_srv))
