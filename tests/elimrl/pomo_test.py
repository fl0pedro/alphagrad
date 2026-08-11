"""elimrl POMO tests (M3, CPU-only, no GPU, no real measurements).

(a) the POMO loss on a toy 2-step MDP with a STUB measure fn pushes probability
    mass toward the cheaper trajectory;
(b) the shared (critic-free) baseline is exactly zero-mean;
(c) two trace-equivalent plans cost ONE measure call (and two under exact keys);
(d) infeasible measurements are scored finitely and never NaN the update;
plus front/hypervolume and plan-realization sanity, and a small end-to-end
learning check on the tiny target.
"""

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from alphagrad.elimrl.baselines import tiny_target
from alphagrad.elimrl.encoder import ElimGNN
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import build_static
from alphagrad.elimrl import pomo as P
from alphagrad.elimrl.symmetry import build_elim_graph, independent_at


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    fn, args, argnums = tiny_target()
    env = ElimEnv(fn, args, argnums, vertex_only=True, symbolic=True)
    static = build_static(env)
    graph = build_elim_graph(fn, args, argnums)
    return fn, args, argnums, env, static, graph


def _stub_result(latency_ns, mem_bytes=1 << 20, status="ok"):
    return {"status": status, "executed": status == "ok",
            "latency_ns": float(latency_ns), "mem_total_bytes": float(mem_bytes)}


def _make_policy(static, seed=0, hidden=32, width=32):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    gnn = ElimGNN(len(static.prim_vocab), hidden=hidden, width=width, key=k1)
    return P.PomoPolicy(gnn, width=width, key=k2)


# ---------------------------------------------------------------------------
# (a) toy 2-step MDP: the gradient moves mass to the cheaper trajectory
# ---------------------------------------------------------------------------
def test_pomo_loss_pushes_mass_to_cheaper_trajectory():
    """Toy 2-step, 2-action MDP with tied step logits. Trajectory A takes
    action 0 twice, trajectory B action 1 twice; the STUB measure fn makes A
    10x cheaper in latency. One gradient step must raise p(action 0)."""
    stub = {0: _stub_result(1e3), 1: _stub_result(1e4)}
    R, feas = P.score_rewards([stub[0], stub[1]], lam=(1.0, 0.0))
    assert feas.all() and R[0] > R[1]            # A is the cheaper trajectory

    theta = jnp.array([0.0, 0.0])                # tied logits, 2 actions

    def loss(th):
        lp = jax.nn.log_softmax(th)
        traj_logp = jnp.stack([2.0 * lp[0], 2.0 * lp[1]])   # 2 steps each
        return P.pomo_surrogate(traj_logp, R)

    g = jax.grad(loss)(theta)
    assert np.all(np.isfinite(np.asarray(g)))
    p0 = float(jax.nn.softmax(theta)[0])
    p1 = float(jax.nn.softmax(theta - 0.1 * g)[0])
    assert p1 > p0, (p0, p1)
    # ... and the gradient direction is symmetric (the loss is a proper PG)
    assert float(g[0]) < 0.0 < float(g[1])


# ---------------------------------------------------------------------------
# (b) shared baseline is exactly zero-mean, no critic anywhere
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rewards", [
    [1.0, 2.0, 3.0, 4.0],
    [-7.5, -7.5, -7.5, -7.5],
    [0.0, 1e6, -1e6, 3.0, 5.0, -2.0, 8.0, 9.0],
])
def test_shared_baseline_is_zero_mean(rewards):
    adv = np.asarray(P.shared_baseline_advantage(jnp.asarray(rewards)))
    assert abs(float(adv.sum())) < 1e-4 * max(1.0, np.abs(adv).max())
    assert np.allclose(adv, np.asarray(rewards) - np.mean(rewards), atol=1e-5)
    # a constant-reward batch yields an exactly zero gradient signal
    if len(set(rewards)) == 1:
        assert np.allclose(adv, 0.0)


def test_shared_baseline_zero_mean_kills_the_surrogate_on_constant_rewards():
    logp = jnp.array([-1.0, -2.0, -3.0, -4.0])
    assert abs(float(P.pomo_surrogate(logp, [5.0] * 4))) < 1e-6


# ---------------------------------------------------------------------------
# (c) measurement cache: trace-equivalent plans measure ONCE
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def diamond():
    """Two independent branches joining at 3: eliminating 1 and 2 COMMUTES, so
    [1,2,3] and [2,1,3] are the same trace. (The tiny target's elimination
    graph is a total chain -- every pair is dependent -- so a synthetic graph
    is what makes this test non-vacuous.)"""
    g = P.ElimGraph.from_edges(
        [(-1, 1), (1, 3), (-2, 2), (2, 3), (3, 4)], eliminable=[1, 2, 3])
    assert independent_at(g, [1, 2, 3], 0)
    return g


def test_trace_equivalent_plans_cost_one_measurement(diamond):
    graph = diamond
    a, b = [1, 2, 3], [2, 1, 3]
    assert P.trace_key(a, graph) == P.trace_key(b, graph)

    calls = []

    def stub_measure(order):
        calls.append(list(order))
        return _stub_result(1234.0)

    cache = P.MeasureCache(graph, stub_measure)
    r1, hit1 = cache.measure_order(a)
    r2, hit2 = cache.measure_order(b)
    assert len(calls) == 1, calls
    assert (hit1, hit2) == (False, True)
    assert r1 is r2 and cache.unique == 1 and cache.hits == 1
    assert cache.hit_rate == 0.5

    # the exact-key fallback must NOT dedup them -- proving the trace key,
    # not accidental equality, is what saved the measurement
    calls.clear()
    exact = P.MeasureCache(None, stub_measure)
    exact.measure_order(a)
    exact.measure_order(b)
    assert len(calls) == 2 and exact.unique == 2 and exact.hits == 0


def test_dependent_swap_is_a_different_trace_and_measures_twice(diamond):
    """The dedup must be SOUND, not merely cheap: swapping a dependent pair
    (3 depends on 2) is a different trace class and must be measured again."""
    graph = diamond
    assert P.trace_key([1, 2, 3], graph) != P.trace_key([1, 3, 2], graph)
    calls = []

    def stub_measure(order):
        calls.append(list(order))
        return _stub_result(1.0)

    cache = P.MeasureCache(graph, stub_measure)
    cache.measure_order([1, 2, 3])
    cache.measure_order([1, 3, 2])
    assert len(calls) == 2 and cache.unique == 2 and cache.hits == 0


def test_cache_repeated_identical_order_is_free(tiny):
    _fn, _args, _an, env, _static, graph = tiny
    calls = []

    def stub_measure(order):
        calls.append(list(order))
        return _stub_result(999.0)

    cache = P.MeasureCache(graph, stub_measure)
    order = sorted(graph.eliminable, reverse=True)
    for _ in range(5):
        cache.measure_order(order)
    assert len(calls) == 1 and cache.unique == 1 and cache.hits == 4


# ---------------------------------------------------------------------------
# (d) infeasible handling: finite, scale-free, and never NaN
# ---------------------------------------------------------------------------
def test_infeasible_results_score_below_the_batch_min_without_nan():
    results = [_stub_result(1e3), _stub_result(2e3),
               {"status": "infeasible", "executed": False, "reason": "oom"},
               {"status": "error", "reason": "boom"},
               None]
    R, feas = P.score_rewards(results, lam=(0.5, 0.5))
    assert list(feas) == [True, True, False, False, False]
    assert np.all(np.isfinite(R))
    assert R[2] == R[3] == R[4]
    assert R[2] < R[feas].min()
    margin = R[feas].min() - R[2]
    assert margin >= P.INFEASIBLE_MARGIN_FLOOR - 1e-12
    # scale-free: multiplying every latency by 1000 shifts R by a constant and
    # leaves the infeasible margin unchanged
    R2, _ = P.score_rewards(
        [_stub_result(1e6), _stub_result(2e6), results[2], results[3], None],
        lam=(0.5, 0.5))
    assert abs((R2[feas].min() - R2[2]) - margin) < 1e-9


def test_all_infeasible_batch_returns_none_and_is_skippable():
    R, feas = P.score_rewards(
        [{"status": "infeasible", "executed": False}] * 4, lam=(1.0, 0.0))
    assert R is None and not feas.any()


def test_infeasible_reward_does_not_nan_the_gradient(tiny):
    """A full POMO update with an infeasible trajectory in the batch produces
    finite loss AND finite gradients (padded/illegal candidates are -inf
    masked, so this also covers the mask path)."""
    _fn, _args, _an, env, static, _graph = tiny
    n = 4
    envs = [ElimEnv(*tiny_target(), vertex_only=True, symbolic=True)
            for _ in range(n)]
    L = len(env.jacve_vertices)
    runner = P.PomoRunner(static, n, L, edge_hint=len(env.state().edges),
                          cand_hint=len(env.reset().legal_vertices))
    policy = _make_policy(static)
    rng = np.random.default_rng(0)
    batch, step_mask, orders = runner.rollout(envs, policy, (0.5, 0.5), rng)

    results = [_stub_result(1e3 * (i + 1)) for i in range(n - 1)]
    results.append({"status": "infeasible", "executed": False, "reason": "oom"})
    R, feas = P.score_rewards(results, (0.5, 0.5))
    assert np.all(np.isfinite(R)) and not feas[-1]

    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    opt_state = optim.init(eqx.filter(policy, eqx.is_inexact_array))
    policy2, _os, loss, gnorm = runner.update(
        policy, optim, opt_state, batch, (0.5, 0.5), R, step_mask, 0.01)
    assert math.isfinite(loss) and math.isfinite(gnorm) and gnorm > 0.0
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(policy2, eqx.is_inexact_array))
    assert all(bool(np.all(np.isfinite(np.asarray(x)))) for x in leaves)


def test_grad_chunking_matches_the_single_shot_gradient(tiny):
    """Chunked accumulation must be arithmetically identical (it is the
    memory-bounded path the campaign runs)."""
    _fn, _args, _an, env, static, _graph = tiny
    n = 4
    envs = [ElimEnv(*tiny_target(), vertex_only=True, symbolic=True)
            for _ in range(n)]
    runner = P.PomoRunner(static, n, len(env.jacve_vertices),
                          edge_hint=len(env.state().edges),
                          cand_hint=len(env.reset().legal_vertices))
    policy = _make_policy(static, seed=1)
    rng = np.random.default_rng(3)
    batch, step_mask, _o = runner.rollout(envs, policy, (1.0, 0.0), rng)
    R = np.array([-1.0, 0.5, 2.0, -3.0])

    def run(chunk):
        optim = optax.sgd(1e-2)
        st = optim.init(eqx.filter(policy, eqx.is_inexact_array))
        pol, _st, loss, gn = runner.update(policy, optim, st, batch,
                                           (1.0, 0.0), R, step_mask, 0.01,
                                           grad_chunk=chunk)
        return loss, gn, jax.tree_util.tree_leaves(
            eqx.filter(pol, eqx.is_inexact_array))

    l_full, g_full, p_full = run(0)
    l_ch, g_ch, p_ch = run(2)
    assert abs(l_full - l_ch) < 1e-4 * max(1.0, abs(l_full))
    assert abs(g_full - g_ch) < 1e-3 * max(1.0, abs(g_full))
    for x, y in zip(p_full, p_ch):
        assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-5)


# ---------------------------------------------------------------------------
# plan realization + front bookkeeping
# ---------------------------------------------------------------------------
def test_order_to_plan_terminates_and_matches_history(tiny):
    _fn, _args, _an, env, _static, graph = tiny
    rng = np.random.default_rng(1)
    for _ in range(5):
        perm = list(rng.permutation(sorted(env.jacve_vertices)))
        plan, realized = P.order_to_plan(env, perm)
        assert env.done
        assert plan == [["V", j] for j in realized]
        assert [a[1] for a in env.history] == realized
        assert len(set(realized)) == len(realized)


def test_markowitz_order_is_a_permutation(tiny):
    _fn, _args, _an, _env, _static, graph = tiny
    mk = P.markowitz_order(graph)
    assert sorted(mk) == sorted(graph.eliminable)


def test_pareto_front_and_hypervolume():
    pts = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0), (5.0, 5.0)]
    front = P.pareto_front(pts)
    assert sorted(pts[i] for i in front) == [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    hv = P.hypervolume_2d([pts[i] for i in front], (10.0, 10.0))
    assert 0.0 < hv < 1.0
    # a strictly better front dominates a strictly worse one
    hv_better = P.hypervolume_2d([(0.5, 0.5)], (10.0, 10.0))
    hv_worse = P.hypervolume_2d([(9.0, 9.0)], (10.0, 10.0))
    assert hv_better > hv_worse
    assert P.hypervolume_2d([(11.0, 11.0)], (10.0, 10.0)) == 0.0


# ---------------------------------------------------------------------------
# end-to-end: the runner + update actually learn a first-vertex preference
# ---------------------------------------------------------------------------
def test_end_to_end_pomo_loop_learns_on_a_toy_2step_mdp(tiny):
    """Full rollout -> STUB measurement -> POMO update loop, truncated to a
    2-step MDP (L=2) so trajectory credit assignment is unambiguous: latency
    grows with the rank of both chosen vertices, so a working update must
    drive the mean reward up and make the greedy rollout pick low-rank
    vertices."""
    _fn, _args, _an, env, static, graph = tiny
    legal0 = env.reset().legal_vertices
    rank = {int(v): i for i, v in enumerate(sorted(legal0))}
    n = min(len(legal0), 8)
    envs = [ElimEnv(*tiny_target(), vertex_only=True, symbolic=True)
            for _ in range(n)]
    runner = P.PomoRunner(static, n, 2, edge_hint=len(env.state().edges),
                          cand_hint=len(legal0))
    policy = _make_policy(static, seed=2)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
    opt_state = optim.init(eqx.filter(policy, eqx.is_inexact_array))
    rng = np.random.default_rng(0)
    calls = {"n": 0}

    def stub_measure(order):
        calls["n"] += 1
        return _stub_result(1e3 * (1.0 + sum(rank[int(v)] for v in order)))

    cache = P.MeasureCache(graph, stub_measure)
    mean_R = []
    for _u in range(40):
        batch, step_mask, orders = runner.rollout(envs, policy, (1.0, 0.0), rng)
        assert all(len(o) == 2 for o in orders)
        results = [cache.measure_order(o)[0] for o in orders]
        R, _f = P.score_rewards(results, (1.0, 0.0))
        mean_R.append(float(np.mean(R)))
        policy, opt_state, loss, _gn = runner.update(
            policy, optim, opt_state, batch, (1.0, 0.0), R, step_mask, 0.0)
        assert math.isfinite(loss)

    assert np.mean(mean_R[-5:]) > np.mean(mean_R[:5]) + 0.3, mean_R
    _b, _sm, gorders = runner.rollout(envs[:1], policy, (1.0, 0.0), rng,
                                      greedy=True, forced_starts=False)
    ranks = [rank[v] for v in gorders[0]]
    assert max(ranks) < len(legal0) / 2, (ranks, len(legal0))
    assert cache.hits > 0 and calls["n"] == cache.unique


def test_full_episode_rollout_terminates_with_distinct_forced_starts(tiny):
    """The campaign regime: a full-length episode per trajectory, N distinct
    forced first vertices, every env terminated, every order a permutation."""
    _fn, _args, _an, env, static, _graph = tiny
    legal0 = env.reset().legal_vertices
    n = min(len(legal0), 8)
    envs = [ElimEnv(*tiny_target(), vertex_only=True, symbolic=True)
            for _ in range(n)]
    runner = P.PomoRunner(static, n, len(env.jacve_vertices),
                          edge_hint=len(env.state().edges),
                          cand_hint=len(legal0))
    policy = _make_policy(static, seed=4)
    rng = np.random.default_rng(7)
    batch, step_mask, orders = runner.rollout(envs, policy, (0.5, 0.5), rng,
                                              forced_starts=True)
    firsts = [o[0] for o in orders]
    assert len(set(firsts)) == len(firsts), firsts
    assert all(len(set(o)) == len(o) for o in orders)
    assert all(e.done for e in envs)
    assert batch.dyn.shape[0] == n * runner.L
    assert step_mask.shape == (n, runner.L)
    # padded steps contribute nothing; live steps are exactly the order length
    assert [int(m.sum()) for m in step_mask] == [len(o) for o in orders]
