"""Live elimination state must be INVISIBLE and STRUCTURALLY O(T).

`_face_transforms_for_order` used to rebuild an `IncrementalJaxpr` and replay
the whole prefix on every call: O(T^2) eliminations per episode. The env now
OWNS the live builder and advances it by ONE elimination per step
(ALPHAGRAD_FACE_LIVE_STATE=1, the default). Two things have to hold:

  * EQUIVALENCE -- the enumeration must equal the stateless rebuild's, for
    every prefix length, with several env chains INTERLEAVED (which is what
    `pure_callback(vmap_method="sequential")` does under --num-envs N), and
    with COMPRESS in the plan.
  * ASYMPTOTICS -- eliminations per episode must be T per chain, not
    T(T+1)/2, and `restart` (a mid-episode rebuild) must be exactly 0.

This is only sound because a vertex's decode no longer depends on where the
prefix ends. Under the old `is_last` gate the last vertex of a prefix decoded
differently one step later, which is what forced the prefix caches' COMPRESS
carve-outs; `test_live_state_compress_is_free` pins that COMPRESS now costs a
live chain exactly nothing.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alphagrad.approx.env as E
from alphagrad.approx.env import (
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
)
from graphax import SKIP_FACE


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
ARGNUMS = (0, 1)


def _mk():
    cj = jax.make_jaxpr(_fn)(*ARGS)
    cfg = SimpleNamespace(jaxpr=cj.jaxpr, argnums=ARGNUMS, per_face=True)
    return cfg, list(cj.literals), list(ARGS)


def _ft_sig(out):
    """Comparable description of ONE enumeration: which faces of which
    vertices came back, and whether each is a SKIP or a slot tuple."""
    return {int(v): {str(k): ("SKIP" if val is SKIP_FACE else
                              "T%d" % len(val) if isinstance(val, tuple)
                              else type(val).__name__)
                     for k, val in per.items()}
            for v, per in out.items()}


def _episode(seed, T, compress_at=None, first=None):
    """One env's wire arrays. Faces are SKIP rows: `_face_dict_for_vertex`
    short-circuits a skipped face BEFORE `decode_vertex_rule_specs`, so a
    synthetic rule can never raise on an operand it does not fit, while the
    enumeration and the replay are exercised identically.

    `first` pins the leading vertex. On a graph this small four random orders
    can share a 1-prefix, and two envs that share a prefix legitimately share
    ONE chain -- whichever advances first keeps it and the other rebuilds
    (correctly: the match is exact, so the shared state is the state both
    asked for). That is a real property, but it is not the property these
    tests are about, so the callers that assert `restart == 0` make the
    chains distinct from step 1."""
    cfg, consts, args = _mk()
    rng = np.random.default_rng(seed)
    nv = len(cfg.jaxpr.eqns)
    order = [int(x) for x in rng.permutation(np.arange(1, nv + 1))[:T]]
    if first is not None:
        order.remove(int(first))
        order = [int(first)] + order[:T - 1]
    specs = np.full((T, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[:, :, 2] = 0
    faces = np.full((T, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((T, MAX_FACES), np.int32)
    for k in range(T):
        for f in range(1 + int(rng.random() < 0.24)):
            skips[k, f] = 1
            if compress_at is not None and k == compress_at and f == 0:
                faces[k, f, 0] = [COMPRESS_SENTINEL, 0, 1]
    return cfg, consts, args, order, specs, faces, skips


def _pass(eps, live):
    """Interleave every episode's prefixes 1..T the way the vmapped host
    callback does; return (per-(env,step) signatures, live-chain stats)."""
    prev_live, prev_cache = E._FACE_LIVE_STATE, os.environ.get(
        "ALPHAGRAD_FACE_ENUM_CACHE", "0")
    E._FACE_LIVE_STATE = live
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0"
    E._LIVE_CHAINS.clear()
    E.consume_live_chain_stats()
    try:
        sigs = []
        T = len(eps[0][3])
        for t in range(1, T + 1):
            for (cfg, consts, args, order, specs, faces, skips) in eps:
                sigs.append(_ft_sig(E._face_transforms_for_order(
                    cfg, consts, args, order[:t], specs[:t],
                    faces[:t], skips[:t])))
        return sigs, E.consume_live_chain_stats()
    finally:
        E._FACE_LIVE_STATE = prev_live
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = prev_cache
        E._LIVE_CHAINS.clear()
        E.consume_live_chain_stats()


@pytest.mark.parametrize("n_envs", [1, 4])
def test_live_state_equals_stateless_rebuild(n_envs):
    eps = [_episode(seed=s, T=8, first=s + 1) for s in range(n_envs)]
    T = len(eps[0][3])          # the toy graph has fewer than 8 vertices
    live, stats = _pass(eps, live=True)
    cold, _ = _pass(eps, live=False)
    assert live == cold
    # ASYMPTOTICS: one elimination per (env, step) and one build per env.
    assert stats["step"] == n_envs * T, stats
    assert stats["elims"] == n_envs * T, stats
    assert stats["cold"] == n_envs and stats["restart"] == 0, stats
    # ... versus the rebuild, whose replay count is quadratic in T.
    assert stats["elims"] < n_envs * T * (T + 1) // 2


def test_live_state_compress_is_free():
    """COMPRESS forced the old prefix caches into a cold replay (v40:
    ext=1/431). A live chain does not notice it at all."""
    plain = [_episode(seed=0, T=8)]
    T = len(plain[0][3])
    comp = [_episode(seed=0, T=8, compress_at=T // 2)]
    _, sa = _pass(plain, live=True)
    b, sb = _pass(comp, live=True)
    assert sa["elims"] == sb["elims"] == T, (sa, sb)
    assert sb["restart"] == 0 and sb["cold"] == 1, sb
    c, _ = _pass(comp, live=False)
    assert b == c


def test_live_state_pool_overflow_is_correct_not_wrong():
    """More concurrent chains than the pool holds must still be CORRECT: it
    degrades to extra rebuilds, never to a wrong enumeration."""
    eps = [_episode(seed=s, T=6, first=s + 1) for s in range(4)]
    old = E._LIVE_CHAIN_CAP
    E._LIVE_CHAIN_CAP = 2
    try:
        live, stats = _pass(eps, live=True)
    finally:
        E._LIVE_CHAIN_CAP = old
    cold, _ = _pass(eps, live=False)
    assert live == cold
    assert stats["evict"] > 0 and stats["restart"] > 0, stats


def test_live_state_revisited_prefix_is_free_and_identical():
    """A branching caller (GAZ) may ask for the SAME prefix twice; the second
    ask must cost no eliminations and return the same enumeration."""
    ep = _episode(seed=3, T=6)
    T = len(ep[3])
    prev = E._FACE_LIVE_STATE
    E._FACE_LIVE_STATE = True
    E._LIVE_CHAINS.clear()
    E.consume_live_chain_stats()
    try:
        cfg, consts, args, order, specs, faces, skips = ep
        a = _ft_sig(E._face_transforms_for_order(
            cfg, consts, args, order[:T], specs[:T], faces[:T], skips[:T]))
        s1 = E.consume_live_chain_stats()
        b = _ft_sig(E._face_transforms_for_order(
            cfg, consts, args, order[:T], specs[:T], faces[:T], skips[:T]))
        s2 = E.consume_live_chain_stats()
    finally:
        E._FACE_LIVE_STATE = prev
        E._LIVE_CHAINS.clear()
        E.consume_live_chain_stats()
    assert a == b
    assert s1["elims"] == T and s2["elims"] == 0, (s1, s2)
