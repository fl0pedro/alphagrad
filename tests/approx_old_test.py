"""Ticket .56 -- THE OLD EDGE as a flag: ``--approx-old {same,exact}``.

A face accumulation multiplies lhs (the predecessor edge) by rhs (the
successor edge) into ``new`` and adds ``new`` onto the OLD edge -- the
existing predecessor-to-successor edge -- when that edge exists. What the old
edge gets is one configuration, ``env.approx_old()``:

* ``same``  -- old carries the SAME approximation as new. env emits graphax's
  two-op face form ``((lhs, rhs, new), (None, jr, None))`` with the new-slot
  hook in ``jr``, the hook graphax applies to the existing edge right before
  the add (core.py, ``_unpack_face_slots``). Today's default.
* ``exact`` -- old is left exact. env emits the bare 3-tuple; graphax never
  reaches the join hooks.

Pinned here, on a toy graph (never TLM):

1. The toy face really has an old edge (the join runs), and under ``same``
   the transform log shows the IDENTICAL hook object fired on ``new`` and on
   ``old``; under ``exact`` the old edge carries none and only ``new`` fires.
2. The configuration is an ARGUMENT. The deleted ``ALPHAGRAD_NEW_SLOT_JOIN``
   variable moves nothing any more, and a value outside {same, exact} is a
   loud error, not a fallback.
3. Every plan-log record carries the configuration that measured it.
4. ``ppo.py`` declares the flag with the default ``same``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.pop("ALPHAGRAD_PLAN_LOG", None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from graphax import IncrementalJaxpr                            # noqa: E402
from graphax.core import _stable_var_index                      # noqa: E402
from graphax.sparse.micro_actions import QUANT_DTYPES           # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    QUANT_SENTINEL,
    StepAction,
    VertexEliminationEnv,
)

# Deliberately NOT bf16-representable, so a bf16 Quant on the old edge is a
# real change of the tensor, not a structural no-op.
_B = jnp.asarray(
    (np.arange(16, dtype=np.float64).reshape(4, 4) * 0.1234567 + 0.7654321)
    .astype(np.float32))
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _join_mat(x):
    """Two paths from ``x`` into ``w``: eliminating vertex 2 first CREATES the
    edge ``x -> w``; eliminating vertex 1 afterwards MERGES into it, so the face
    ``x -> u -> w`` has an OLD edge and the join runs."""
    u = jnp.sin(x)                   # vertex 1
    v = _B @ x                       # vertex 2
    w = u + v                        # vertex 3
    return jnp.sum(_W @ w)           # vertices 4, 5 (scalar loss)


def _bf16_index() -> int:
    for i, d in enumerate(QUANT_DTYPES):
        if jnp.dtype(d) == jnp.dtype(jnp.bfloat16):
            return i
    raise AssertionError(f"no bfloat16 in QUANT_DTYPES: {QUANT_DTYPES}")


def _toy():
    closed = jax.make_jaxpr(_join_mat)(_X4)
    cfg = SimpleNamespace(jaxpr=closed.jaxpr, argnums=(0,))
    ij = IncrementalJaxpr(closed.jaxpr, (0,), list(closed.literals), [_X4],
                          track_faces=False)
    vidx = _stable_var_index(closed.jaxpr)
    x_var = closed.jaxpr.invars[0]
    w_var = closed.jaxpr.eqns[2].outvars[0]
    return closed, cfg, ij, (vidx[x_var], vidx[w_var]), (x_var, w_var)


def _quant_new_row():
    """ONE face row: a bf16 QUANT in the ``new`` slot, nothing else."""
    face_row = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    face_row[..., 2] = 0
    face_row[0, 2] = (QUANT_SENTINEL, _bf16_index(), 0)
    face_skip = np.zeros((MAX_FACES,), np.int32)
    return face_row, face_skip


def _recorder(hook, site, log):
    if hook is None:
        return None

    def _h(st):
        out = hook(st)
        log.append((site, id(hook), out is not st))
        return out
    return _h


def _emit_and_eliminate(monkeypatch, approx_old):
    """Eliminate vertex 2, then vertex 1 with env's own face dict for the
    face ``x -> u -> w`` wrapped in site recorders. Returns (per_face entry,
    the transform log of the vertex-1 elimination, the old edge existed)."""
    if approx_old is None:
        monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    else:
        monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", approx_old)
    closed, cfg, ij, key, (x_var, w_var) = _toy()
    ij.eliminate(2)
    old_exists = ij.graph[x_var][w_var] is not None
    face_row, face_skip = _quant_new_row()
    per_face = envmod._face_dict_for_vertex(cfg, ij, 1, face_row, face_skip)
    assert list(per_face) == [key], (list(per_face), key)
    entry = per_face[key]
    log: list = []
    if len(entry) == 2:
        (lhs, rhs, new), (jl, jr, jres) = entry
        wrapped = ((_recorder(lhs, "lhs", log), _recorder(rhs, "rhs", log),
                    _recorder(new, "new", log)),
                   (_recorder(jl, "jl", log), _recorder(jr, "old", log),
                    _recorder(jres, "new_old", log)))
    else:
        lhs, rhs, new = entry
        wrapped = (_recorder(lhs, "lhs", log), _recorder(rhs, "rhs", log),
                   _recorder(new, "new", log))
    ij.eliminate(1, (), {key: wrapped})
    return entry, log, old_exists


# --------------------------------------------------------------------------
# 1. same: identical approximation on new and old; exact: old carries none
# --------------------------------------------------------------------------

def test_same_puts_the_identical_new_slot_hook_on_the_old_edge(monkeypatch):
    entry, log, old_exists = _emit_and_eliminate(monkeypatch, "same")
    assert old_exists, "the toy face has no old edge -- the join never ran"
    # The two-op form, with new's hook in jr and jl / jres untouched.
    assert len(entry) == 2, entry
    (lhs, rhs, new), (jl, jr, jres) = entry
    assert lhs is None and rhs is None and new is not None
    assert jl is None and jres is None
    assert jr is new, "old must carry the SAME hook object as new"
    fired = {site: (hid, changed) for site, hid, changed in log}
    assert set(fired) == {"new", "old"}, log
    assert fired["new"][0] == fired["old"][0] == id(new)
    # A real change on both operands of the add (bf16 on a non-bf16 tensor).
    assert fired["new"][1] and fired["old"][1], log


def test_exact_leaves_the_old_edge_alone(monkeypatch):
    entry, log, old_exists = _emit_and_eliminate(monkeypatch, "exact")
    assert old_exists, "the toy face has no old edge -- the join never ran"
    assert len(entry) == 3, entry            # the bare triple: no join hooks
    lhs, rhs, new = entry
    assert lhs is None and rhs is None and new is not None
    assert [s for s, _, _ in log] == ["new"], log


def test_the_default_is_same(monkeypatch):
    entry, log, _ = _emit_and_eliminate(monkeypatch, None)
    assert envmod.approx_old() == "same"
    assert len(entry) == 2 and entry[1][1] is entry[0][2]
    assert {s for s, _, _ in log} == {"new", "old"}


# --------------------------------------------------------------------------
# 2. an argument, not an env knob
# --------------------------------------------------------------------------

def test_the_deleted_env_var_moves_nothing(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_NEW_SLOT_JOIN", "0")
    entry, log, _ = _emit_and_eliminate(monkeypatch, "same")
    assert len(entry) == 2, "ALPHAGRAD_NEW_SLOT_JOIN=0 must not switch to exact"
    assert {s for s, _, _ in log} == {"new", "old"}


def test_a_value_outside_the_choices_is_an_error(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", "1")
    with pytest.raises(ValueError, match="ALPHAGRAD_APPROX_OLD"):
        envmod.approx_old()


# --------------------------------------------------------------------------
# 3. the plan log records which configuration ran
# --------------------------------------------------------------------------

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)


def _square(x):
    e = _M @ x
    return jnp.sum(_P @ e)


def _one_terminal_record():
    closed = jax.make_jaxpr(_square)(_X4)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square)
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
    face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        state = env.step(
            state, StepAction(jnp.asarray(v, jnp.int32), no_rules,
                              face_rows, face_skip)).state
    recs = envmod.consume_plan_records()["records"]
    assert len(recs) == 1, [r.get("order") for r in recs]
    return recs[0]


@pytest.mark.parametrize("cfg", ["same", "exact"])
def test_every_plan_record_carries_the_configuration(monkeypatch, cfg):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", cfg)
    envmod.consume_plan_records()
    try:
        rec = _one_terminal_record()
        assert rec["approx_old"] == cfg, rec.get("approx_old")
    finally:
        envmod.consume_plan_records()


# --------------------------------------------------------------------------
# 4. the flag on ppo.py
# --------------------------------------------------------------------------

def test_ppo_declares_the_flag_with_default_same():
    ppo = pytest.importorskip("alphagrad.approx.ppo")
    p = ppo.make_argparser()
    assert p.parse_args([]).approx_old == "same"
    assert p.parse_args(["--approx-old", "exact"]).approx_old == "exact"
    with pytest.raises(SystemExit):
        p.parse_args(["--approx-old", "1"])
