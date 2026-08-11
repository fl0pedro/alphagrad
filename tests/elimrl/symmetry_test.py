"""Tests for alphagrad.elimrl.symmetry (M1 acceptance gate).

Covers:
- static path-independence == literal evolving-graph independence (theorem check)
- swap-closure: independent adjacent swaps preserve trace_key; dependent
  adjacent swaps change it
- identical-graph invariant: equal keys => identical final graphs and
  identical Markowitz cost multisets; verified end-to-end on a real jax
  target via graphax jacve count_ops
- orbits: pynauty and WL+verify agree on synthetic graphs with known groups;
  colouring splits orbits; WL+verify is sound (only verified merges)
- augment: relabelled trajectory is legal and cost-identical
"""
import random

import pytest

from alphagrad.elimrl.symmetry import (ElimGraph, augment, build_elim_graph,
                                       eliminate_vertex, find_automorphism,
                                       foata_normal_form, graph_after,
                                       independent_at, is_automorphism,
                                       markowitz_profile, orbits, trace_key,
                                       wl_refine)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def random_dag(rng: random.Random, n: int = 12, p: float = 0.25) -> ElimGraph:
    """Random DAG on 1..n plus an input 0' (-1) and terminal sink (n+1)."""
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < p:
                edges.append((u, v))
    # input feeds every source; every sink vertex feeds the terminal output
    have_in = {v for _, v in edges}
    have_out = {u for u, _ in edges}
    for v in range(1, n + 1):
        if v not in have_in:
            edges.append((-1, v))
        if v not in have_out:
            edges.append((v, n + 1))
    return ElimGraph.from_edges(edges, eliminable=range(1, n + 1))


def random_order(rng: random.Random, graph: ElimGraph):
    order = list(graph.eliminable)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------
# 1. evolving-graph independence == static incomparability
# ---------------------------------------------------------------------

def test_dynamic_independence_matches_static_closure():
    rng = random.Random(0)
    for trial in range(20):
        g = random_dag(rng, n=10, p=0.3)
        order = random_order(rng, g)
        for i in range(len(order) - 1):
            dyn = independent_at(g, order, i)
            stat = not g.dependent(order[i], order[i + 1])
            assert dyn == stat, (
                f"trial {trial} pos {i}: evolving-graph independence "
                f"{dyn} != static incomparability {stat}")


# ---------------------------------------------------------------------
# 2. swap-closure property tests
# ---------------------------------------------------------------------

def test_independent_adjacent_swap_preserves_key():
    rng = random.Random(1)
    checked = 0
    for _ in range(30):
        g = random_dag(rng, n=12, p=0.25)
        order = random_order(rng, g)
        key = trace_key(order, g)
        for i in range(len(order) - 1):
            if not g.dependent(order[i], order[i + 1]):
                swapped = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
                assert trace_key(swapped, g) == key
                checked += 1
    assert checked > 50  # the property was actually exercised


def test_dependent_adjacent_swap_changes_key():
    rng = random.Random(2)
    checked = 0
    for _ in range(30):
        g = random_dag(rng, n=12, p=0.25)
        order = random_order(rng, g)
        key = trace_key(order, g)
        for i in range(len(order) - 1):
            if g.dependent(order[i], order[i + 1]):
                swapped = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
                assert trace_key(swapped, g) != key
                checked += 1
    assert checked > 50


def test_random_independent_swap_walk_key_invariant():
    rng = random.Random(3)
    g = random_dag(rng, n=14, p=0.2)
    order = random_order(rng, g)
    key = trace_key(order, g)
    cur = list(order)
    for _ in range(200):
        i = rng.randrange(len(cur) - 1)
        if not g.dependent(cur[i], cur[i + 1]):
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
    assert trace_key(cur, g) == key


def test_prefix_orders_supported():
    rng = random.Random(4)
    g = random_dag(rng, n=10, p=0.3)
    order = random_order(rng, g)[:5]
    key = trace_key(order, g)
    assert isinstance(key, tuple)
    for i in range(4):
        if not g.dependent(order[i], order[i + 1]):
            swapped = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
            assert trace_key(swapped, g) == key


# ---------------------------------------------------------------------
# 3. identical-graph invariant + cost invariance
# ---------------------------------------------------------------------

def test_equal_key_implies_identical_final_graph_and_cost():
    rng = random.Random(5)
    for _ in range(10):
        g = random_dag(rng, n=12, p=0.25)
        order = random_order(rng, g)
        cur = list(order)
        for _ in range(100):  # random walk through the trace class
            i = rng.randrange(len(cur) - 1)
            if not g.dependent(cur[i], cur[i + 1]):
                cur[i], cur[i + 1] = cur[i + 1], cur[i]
        assert trace_key(cur, g) == trace_key(order, g)
        assert graph_after(g, cur) == graph_after(g, order)
        # cost: same multiset of Markowitz products (single independent swap
        # leaves both factors untouched -> profile is permuted, never changed)
        assert sorted(markowitz_profile(g, cur)) == sorted(markowitz_profile(g, order))
        # ... and the canonical linearization (concatenated Foata blocks) is
        # itself a member of the class: same key, same final graph, same cost
        blocks = trace_key(order, g)
        canon = [v for blk in blocks for v in blk]
        assert trace_key(canon, g) == blocks
        assert graph_after(g, canon) == graph_after(g, order)
        assert sorted(markowitz_profile(g, canon)) == sorted(markowitz_profile(g, order))


def _tiny_mlp():
    """Two parallel branches merged at the loss -- guarantees the elimination
    graph HAS independent vertex pairs (a pure chain would make the trace
    monoid free and the swap tests vacuous)."""
    import jax.numpy as jnp

    def fn(x, W1, W2):
        h1 = jnp.tanh(W1 @ x)
        h2 = jnp.sin(W2 @ x)
        return jnp.sum(h1 * h1) + jnp.sum(h2 * h2)
    return fn


def test_trace_equivalence_on_real_jax_target():
    """Equal trace keys => identical jacve op counts and identical Jacobian."""
    jax = pytest.importorskip("jax")
    import numpy as np
    from graphax import jacve

    fn = _tiny_mlp()
    key = jax.random.PRNGKey(0)
    ks = jax.random.split(key, 3)
    x = jax.random.normal(ks[0], (4,))
    W1 = jax.random.normal(ks[1], (5, 4))
    W2 = jax.random.normal(ks[2], (3, 4))
    args = (x, W1, W2)
    g = build_elim_graph(fn, args, argnums=(1, 2))
    assert len(g.eliminable) >= 4

    rng = random.Random(7)
    order = list(g.eliminable)
    rng.shuffle(order)
    cur = list(order)
    swapped_any = False
    for _ in range(50):
        i = rng.randrange(len(cur) - 1)
        if not g.dependent(cur[i], cur[i + 1]):
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
            swapped_any = True
    assert swapped_any
    assert cur != order
    assert trace_key(cur, g) == trace_key(order, g)

    ja, aux_a = jacve(fn, order, argnums=(1, 2), count_ops=True)(*args)
    jb, aux_b = jacve(fn, cur, argnums=(1, 2), count_ops=True)(*args)
    assert aux_a["muls"] == aux_b["muls"]
    assert aux_a["adds"] == aux_b["adds"]
    for a, b in zip(jax.tree_util.tree_leaves(ja), jax.tree_util.tree_leaves(jb)):
        assert np.allclose(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------
# 4. Foata normal form basics
# ---------------------------------------------------------------------

def test_foata_blocks_are_pairwise_independent_antichains():
    rng = random.Random(8)
    g = random_dag(rng, n=12, p=0.3)
    order = random_order(rng, g)
    for blk in trace_key(order, g):
        for a in blk:
            for b in blk:
                if a != b:
                    assert not g.dependent(a, b)


def test_trace_key_rejects_bad_orders():
    g = ElimGraph.from_edges([(-1, 1), (1, 2), (2, 3)], eliminable=[1, 2])
    with pytest.raises(ValueError):
        trace_key([1, 1], g)
    with pytest.raises(ValueError):
        trace_key([3], g)


# ---------------------------------------------------------------------
# 5. orbits
# ---------------------------------------------------------------------

def _two_parallel_chains():
    """in -> (a1->a2) -> out ; in -> (b1->b2) -> out. Swap of chains is the
    only non-trivial automorphism: orbits {a1,b1}, {a2,b2}."""
    edges = [(-1, 1), (1, 2), (2, 5), (-1, 3), (3, 4), (4, 5)]
    colors = {-1: ("in",), 5: ("out",),
              1: ("op", "f"), 3: ("op", "f"),
              2: ("op", "g"), 4: ("op", "g")}
    return ElimGraph.from_edges(edges, eliminable=[1, 2, 3, 4], colors=colors)


def test_orbits_two_chains_wl_fallback():
    g = _two_parallel_chains()
    res = orbits(g, colored=True, method="wl")
    assert res.method == "wl+verify"
    assert res.exact
    nontrivial = {o for o in res.orbits if len(o) > 1}
    assert frozenset({1, 3}) in nontrivial
    assert frozenset({2, 4}) in nontrivial
    assert res.automorphisms  # explicit witnesses collected


def test_orbits_colored_split():
    g = _two_parallel_chains()
    # distinguish the two chains by colour -> group collapses to identity
    g2 = ElimGraph.from_edges([(-1, 1), (1, 2), (2, 5), (-1, 3), (3, 4), (4, 5)],
                              eliminable=[1, 2, 3, 4],
                              colors={-1: ("in",), 5: ("out",),
                                      1: ("op", "f"), 3: ("op", "OTHER"),
                                      2: ("op", "g"), 4: ("op", "g")})
    res = orbits(g2, colored=True, method="wl")
    assert all(len(o) == 1 for o in res.orbits)
    # uncoloured on the same digraph: chains merge again
    res_u = orbits(g, colored=False, method="wl")
    assert any(len(o) > 1 for o in res_u.orbits)


def test_orbits_pynauty_matches_wl_if_available():
    pytest.importorskip("pynauty")
    g = _two_parallel_chains()
    res_p = orbits(g, colored=True, method="pynauty")
    res_w = orbits(g, colored=True, method="wl")
    assert sorted(map(sorted, res_p.orbits)) == sorted(map(sorted, res_w.orbits))
    assert res_p.method == "pynauty"


def test_wl_verification_splits_false_candidates():
    """WL can merge nodes that are NOT exchangeable; verification must split.

    Two 'g' nodes with the same colour but asymmetric wiring: g1 sits on a
    chain of length 2, g2 on a chain of length 3. With enough surrounding
    symmetry WL separates them here anyway, so instead pin the check directly:
    find_automorphism must refuse a pin between structurally distinct nodes.
    """
    edges = [(-1, 1), (1, 2), (2, 6), (-1, 3), (3, 4), (4, 5), (5, 6)]
    g = ElimGraph.from_edges(edges, eliminable=[1, 2, 3, 4, 5])
    nodes = list(g.nodes)
    colors = {n: 0 for n in nodes}
    # vertex 1 (indeg from -1, chain len 2) vs vertex 3 (chain len 3)
    perm = find_automorphism(nodes, g.succ, g.pred, colors, pin=(1, 3))
    assert perm is None


def test_wl_refine_distinguishes_direction():
    edges = [(1, 2), (2, 3)]
    g = ElimGraph.from_edges(edges, eliminable=[2])
    ref = wl_refine(g.nodes, g.succ, g.pred, {n: 0 for n in g.nodes})
    assert ref[1] != ref[3]  # source vs sink differ under directed WL


# ---------------------------------------------------------------------
# 6. augment
# ---------------------------------------------------------------------

def test_augment_legal_and_cost_identical():
    g = _two_parallel_chains()
    res = orbits(g, colored=True, method="wl")
    # build the chain-swap automorphism from the collected witnesses
    perm = None
    for a in res.automorphisms:
        if a.get(1) == 3:
            perm = a
            break
    assert perm is not None
    assert is_automorphism(g, perm, colored=True)

    rng = random.Random(11)
    for _ in range(10):
        traj = list(g.eliminable)
        rng.shuffle(traj)
        aug = augment(traj, perm)
        # legal: a permutation of the eliminable set
        assert sorted(aug) == sorted(g.eliminable)
        # identical structural cost, step by step
        assert markowitz_profile(g, aug) == markowitz_profile(g, traj)
        # relabelled trajectory replays without error on the engine
        succ, pred = g.mutable()
        for v in aug:
            eliminate_vertex(succ, pred, v)


def test_augment_identity_on_fixed_points():
    g = _two_parallel_chains()
    ident = {n: n for n in g.nodes}
    traj = list(g.eliminable)
    assert augment(traj, ident) == traj


# ---------------------------------------------------------------------
# 7. builder on a small jax target
# ---------------------------------------------------------------------

def test_build_elim_graph_small_target():
    jax = pytest.importorskip("jax")
    fn = _tiny_mlp()
    key = jax.random.PRNGKey(0)
    ks = jax.random.split(key, 3)
    args = (jax.random.normal(ks[0], (4,)),
            jax.random.normal(ks[1], (5, 4)),
            jax.random.normal(ks[2], (3, 4)))
    g = build_elim_graph(fn, args, argnums=(1, 2))
    # DAG, eliminable non-empty, colours present for every node
    assert g.eliminable
    assert all(n in g.colors for n in g.nodes)
    g._topo_order()  # raises on cycles
    # every eliminable vertex is an eqn (positive id)
    assert all(v > 0 for v in g.eliminable)
    # elimination of the full reverse order runs clean
    succ, pred = g.mutable()
    for v in sorted(g.eliminable, reverse=True):
        eliminate_vertex(succ, pred, v)
