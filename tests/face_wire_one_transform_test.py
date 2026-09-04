"""Ticket .17 (D1) -- ONE face wire, ONE transform semantics.

Finding 56 (job 63579) confirmed D1 on TLM: for the same wire row on the same
face at the same prefix, ``env._face_dict_for_vertex`` emitted graphax's
two-op form (slot 2 hooks ``new`` AND the old edge ``jr``) while
``live_faces.LiveFaceStream._decided`` and
``plan_tokens.PlanTokenizer.face_transforms`` emitted the flat triple (slot 2
hooks the post-join sum ``jres``). The head chose one transform and the
measurement scored another.

Ruling: the two-op form is canonical. Every decoder now builds its entry
through ``env.face_entry_from_slots``, so ``env.approx_old()`` is the one
reader of the old-edge configuration and one wire decodes to one entry.

This is the ticket-16 diff probe
(``.scratch/trustworthy-approx-search/probes/t16/probe_d1.py``) run on the
ticket-.56 toy graph, never TLM. Pinned, for BOTH ``--approx-old`` values:

1. The site set graphax's own ``_unpack_face_slots`` derives from each
   decoder's entry is identical across the four decoders (env, live_faces,
   plan_tokens, and the mask oracle's wire replay ``masks._face_ft``), on a
   JOIN face and on a MERGE-FREE face. The diff is EMPTY.
2. Eliminating with each decoder's entry fires the same sites, in the same
   order, with the same dtype transitions, and leaves the same edge dtype.
3. The head's tokens for a wire equal the tokens of an elimination driven by
   env's own dict -- the tokens describe the graph the measurement builds.
   Under ``same`` the join face's tokens differ from ``exact``'s: the head
   now sees the old-edge choice it is scored on.
4. The head-side decoders reach the configuration through ``env.approx_old``
   and nothing else (no second reader of the hand-off variable).
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
from graphax import SKIP_FACE, IncrementalJaxpr                 # noqa: E402
from graphax.core import (                                      # noqa: E402
    _force, _is_two_op_slots, _stable_var_index, _unpack_face_slots)
from graphax.sparse.micro_actions import QUANT_DTYPES           # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common.masks import LiveVertexMaskOracle  # noqa: E402
from alphagrad.approx.common.plan_tokens import PlanTokenizer   # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS, MAX_FACES, QUANT_SENTINEL)
from alphagrad.approx.live_faces import LiveFaceStream          # noqa: E402

# The .56 toy: deliberately NOT bf16-representable, so a bf16 Quant is a real
# change of the tensor and shows up as a dtype transition.
_B = jnp.asarray(
    (np.arange(16, dtype=np.float64).reshape(4, 4) * 0.1234567 + 0.7654321)
    .astype(np.float32))
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _join_mat(x):
    """Two paths from ``x`` into ``w``. Eliminating vertex 2 first CREATES the
    edge ``x -> w`` (its face ``x -> v -> w`` is MERGE-FREE); eliminating
    vertex 1 afterwards MERGES into it (its face ``x -> u -> w`` is a JOIN)."""
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
    vidx = _stable_var_index(closed.jaxpr)
    x_var = closed.jaxpr.invars[0]
    w_var = closed.jaxpr.eqns[2].outvars[0]
    T = SimpleNamespace(jaxpr=closed.jaxpr, argnums=(0,),
                        consts=list(closed.literals), xs=[_X4],
                        key=(vidx[x_var], vidx[w_var]), x_var=x_var,
                        w_var=w_var)
    return T


def _fresh_ij(T):
    return IncrementalJaxpr(T.jaxpr, T.argnums, list(T.consts), list(T.xs))


def _quant_new_rows():
    """The wire: a bf16 QUANT in face 0's ``new`` slot, nothing else."""
    rows = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    rows[..., 2] = 0
    rows[0, 2] = (QUANT_SENTINEL, _bf16_index(), 1)
    skips = np.zeros((MAX_FACES,), np.int32)
    return rows, skips


# The two prefixes of the toy: (vertex to decide, vertices eliminated before).
_MERGE_FREE = (2, ())
_JOIN = (1, (2,))


def _advance(ij, prefix):
    for v in prefix:
        ij.eliminate(v, ())


def _four_dicts(T, v, prefix, rows, skips):
    """The four decoders on the SAME IncrementalJaxpr at the SAME prefix."""
    ij = _fresh_ij(T)
    _advance(ij, prefix)
    cfg = SimpleNamespace(jaxpr=T.jaxpr)
    A = envmod._face_dict_for_vertex(cfg, ij, v, rows, skips)
    stream = LiveFaceStream(T.jaxpr, T.argnums, T.consts, T.xs, vocab=512)
    _keys, B = LiveFaceStream._decided(stream, SimpleNamespace(ij=ij), v,
                                       rows, skips, MAX_FACES)
    C = PlanTokenizer.face_transforms(
        SimpleNamespace(tk=SimpleNamespace(ij=ij), jaxpr=T.jaxpr),
        v, rows, skips) or {}
    D = LiveVertexMaskOracle._face_ft(
        SimpleNamespace(jaxpr=T.jaxpr), ij, v, (rows, skips)) or {}
    return {"env": A, "live_faces": B, "plan_tokens": C, "masks": D}


def _site_set(entry, v):
    """Which of the six sites an entry hooks, decided by graphax's OWN
    unpacker -- the probe's definition of "the transform dict"."""
    if entry is SKIP_FACE:
        return "SKIP", frozenset(["SKIP"]), {}
    lhs, rhs, jres, new, join = _unpack_face_slots(entry, v)
    jl, jr = join if join is not None else (None, None)
    hooks = dict(lhs=lhs, rhs=rhs, new=new, jl=jl, jr=jr, jres=jres)
    form = "two-op" if _is_two_op_slots(entry) else "flat"
    return form, frozenset(s for s, h in hooks.items() if h is not None), hooks


def _instrument(entry, v, log):
    """Wrap every hook with a site-labelled recorder, preserving the form."""
    if entry is SKIP_FACE:
        return entry
    lhs, rhs, jres, new, join = _unpack_face_slots(entry, v)

    def w(h, site):
        if h is None:
            return None

        def g(t):
            din = None if t.val is None else str(t.val.dtype)
            out = h(t)
            dout = None if out.val is None else str(out.val.dtype)
            log.append((site, din, dout))
            return out
        return g

    if _is_two_op_slots(entry):
        jl, jr = join
        return ((w(lhs, "lhs"), w(rhs, "rhs"), w(new, "new")),
                (w(jl, "jl"), w(jr, "jr"), w(jres, "jres")))
    return (w(lhs, "lhs"), w(rhs, "rhs"), w(jres, "res"))


def _run_elim(T, v, prefix, entry):
    """Fresh ij at the prefix, eliminate ``v`` with ONLY this entry
    (instrumented); return the fired-site log and the edge dtype after."""
    ij = _fresh_ij(T)
    _advance(ij, prefix)
    log: list = []
    ij.eliminate(v, (), face_transforms={T.key: _instrument(entry, v, log)})
    edge = _force(ij.graph.get(T.x_var, {}).get(T.w_var))
    edt = None if edge is None or edge.val is None else str(edge.val.dtype)
    return log, edt


def _set(monkeypatch, approx_old):
    monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", approx_old)
    assert envmod.approx_old() == approx_old


# --------------------------------------------------------------------------
# 1 + 2. the ticket-16 diff probe: an EMPTY diff across the four decoders
# --------------------------------------------------------------------------

@pytest.mark.parametrize("approx_old", ["same", "exact"])
@pytest.mark.parametrize("face", [_JOIN, _MERGE_FREE],
                         ids=["join", "merge-free"])
def test_the_diff_probe_is_empty(monkeypatch, approx_old, face):
    _set(monkeypatch, approx_old)
    T = _toy()
    v, prefix = face
    rows, skips = _quant_new_rows()
    dicts = _four_dicts(T, v, prefix, rows, skips)
    for name, d in dicts.items():
        assert list(d) == [T.key], (name, list(d), T.key)

    forms = {n: _site_set(d[T.key], v) for n, d in dicts.items()}
    env_form, env_sites, env_hooks = forms["env"]
    # env is the reference: what the measurement applies.
    if approx_old == "same":
        assert env_form == "two-op" and env_sites == {"new", "jr"}, forms["env"]
        assert env_hooks["new"] is env_hooks["jr"], "old must carry new's hook"
    else:
        assert env_form == "flat" and env_sites == {"jres"}, forms["env"]
    diff = {n: (f, sorted(env_sites - s), sorted(s - env_sites))
            for n, (f, s, _) in forms.items()
            if f != env_form or s != env_sites}
    assert not diff, f"NON-EMPTY diff against env: {diff}"

    # The same sites FIRE, in the same order, with the same dtype
    # transitions, and the edge ends in the same dtype.
    ref_log, ref_dt = _run_elim(T, v, prefix, dicts["env"][T.key])
    if v == _JOIN[0] and approx_old == "same":
        assert [s for s, _, _ in ref_log] == ["new", "jr"], ref_log
        assert all(din == "float32" and dout == "bfloat16"
                   for _, din, dout in ref_log), ref_log
    elif v == _JOIN[0]:
        # the flat triple's slot 2 lands at the post-join site; graphax
        # tags it ``res`` -- the old edge is left alone.
        assert [s for s, _, _ in ref_log] == ["res"], ref_log
    for name in ("live_faces", "plan_tokens", "masks"):
        log, dt = _run_elim(T, v, prefix, dicts[name][T.key])
        assert log == ref_log, (name, log, ref_log)
        assert dt == ref_dt, (name, dt, ref_dt)


# --------------------------------------------------------------------------
# 3. the head's tokens ARE the tokens of the graph env measures
# --------------------------------------------------------------------------

def _tokens_from_wire(T, rows, skips):
    """Tokens of vertex 1 (the join) decided from the WIRE, the way the AZ
    tokenizer and the live face stream do it."""
    pt = PlanTokenizer(T.jaxpr, T.argnums, T.consts, T.xs, vocab=512)
    pt.base()
    pt.eliminate(2)
    toks, _ids = pt.eliminate(1, None, rows, skips)
    return toks


def _tokens_from_env_dict(T, rows, skips):
    """Tokens of vertex 1 driven by env's OWN face dict."""
    pt = PlanTokenizer(T.jaxpr, T.argnums, T.consts, T.xs, vocab=512)
    pt.base()
    pt.eliminate(2)
    ft = envmod._face_dict_for_vertex(SimpleNamespace(jaxpr=T.jaxpr),
                                      pt.tk.ij, 1, rows, skips)
    return [int(t) for t in pt.tk.eliminate(1, (), ft)]


@pytest.mark.parametrize("approx_old", ["same", "exact"])
def test_head_tokens_equal_the_measured_graphs_tokens(monkeypatch, approx_old):
    _set(monkeypatch, approx_old)
    T = _toy()
    rows, skips = _quant_new_rows()
    assert _tokens_from_wire(T, rows, skips) == _tokens_from_env_dict(
        T, rows, skips)


def test_the_head_sees_the_old_edge_choice(monkeypatch):
    T = _toy()
    rows, skips = _quant_new_rows()
    _set(monkeypatch, "same")
    same = _tokens_from_wire(T, rows, skips)
    _set(monkeypatch, "exact")
    exact = _tokens_from_wire(T, rows, skips)
    assert same != exact, "the join face's chunk must carry the old-edge cast"
    # An all-NONE wire is untouched by the configuration: no entry is built.
    blank = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    blank[..., 2] = 0
    _set(monkeypatch, "same")
    a = _tokens_from_wire(T, blank, skips)
    _set(monkeypatch, "exact")
    assert a == _tokens_from_wire(T, blank, skips)


# --------------------------------------------------------------------------
# 4. one reader: the head side goes through env.approx_old, nothing else
# --------------------------------------------------------------------------

def test_the_head_side_reads_through_env_approx_old(monkeypatch):
    # The hand-off variable says "same"; env.approx_old is made to say
    # "exact". Every decoder must follow approx_old, so none of them reads
    # the variable on its own.
    monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", "same")
    monkeypatch.setattr(envmod, "approx_old", lambda: "exact")
    T = _toy()
    v, prefix = _JOIN
    rows, skips = _quant_new_rows()
    for name, d in _four_dicts(T, v, prefix, rows, skips).items():
        form, sites, _ = _site_set(d[T.key], v)
        assert (form, sites) == ("flat", {"jres"}), (name, form, sites)
