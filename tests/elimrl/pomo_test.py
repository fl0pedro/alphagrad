"""elimrl POMO tests (M3, CPU-only, no GPU, no real measurements).

(a) the POMO loss on a toy 2-step MDP with a STUB measure fn pushes probability
    mass toward the cheaper trajectory;
(b) the shared (critic-free) baseline is exactly zero-mean;
(c) two trace-equivalent plans cost ONE measure call (and two under exact keys);
(d) infeasible measurements are scored finitely and never NaN the update;
plus front/hypervolume and plan-realization sanity, and a small end-to-end
learning check on the tiny target.

M3 measurement-protocol fixes (#119 / #120 / #121), all CPU-only:
(e) the PAIRED reward is invariant to reference drift -- identical rewards,
    loss and gradients -- and the unpaired one is NOT (the control that shows
    the fix matters);
(f) unmeasurable (compile-failure class) trajectories are DROPPED and the
    shared baseline is the mean over survivors only;
(g) a batch with fewer than MIN_TRAJ_FOR_UPDATE survivors is skippable and
    never NaNs;
(h) OOM / timeout / predicted-memory keep the floor score and are NOT dropped;
(i) a plan failing the numeric check is invalidated, excluded and logged;
(j) ReferenceTracker interpolation / drift telemetry.
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


def _paired(latency_ns, ref_latency_ns, mem_bytes=1 << 20,
            ref_mem_bytes=1 << 20, **extra):
    """A feasible result already paired with its own reference measurement."""
    r = _stub_result(latency_ns, mem_bytes)
    r.update(extra)
    return P.attach_ratio(r, ref_latency_ns, ref_mem_bytes)


def _compile_failure():
    """The Blackwell failure mode: no verdict about the plan at all."""
    return {"status": "error", "executed": False,
            "reason": ("JaxRuntimeError: INTERNAL: Failed to compile Triton "
                       "kernel. Context: [...]")}


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


# ===========================================================================
# (e) #121 -- paired measurement is drift-invariant, unpaired is not
# ===========================================================================
def _grad_of_surrogate(R, logp=(-1.0, -2.0, -0.5, -3.0)):
    lp = jnp.asarray(logp, jnp.float32)
    return np.asarray(jax.grad(lambda x: P.pomo_surrogate(x, R))(lp))


def test_paired_reward_is_invariant_to_reference_drift():
    """Simulated clock excursion: candidate n AND its own back-to-back
    reference both read d_n times slower. The ratio is untouched, so the
    rewards, the loss and the gradients are bit-for-bit the same."""
    lat = [100e3, 120e3, 90e3, 150e3]
    ref = 100e3
    drift = [1.0, 1.05, 1.12, 1.2]          # a within-batch clock ramp

    clean = [_paired(l, ref) for l in lat]
    drifted = [_paired(l * d, ref * d, mem_bytes=1 << 20,
                       ref_mem_bytes=1 << 20)
               for l, d in zip(lat, drift)]

    rb_c = P.score_rewards_paired(clean, (1.0, 0.0))
    rb_d = P.score_rewards_paired(drifted, (1.0, 0.0))
    assert rb_c.feasible.all() and rb_d.feasible.all()
    assert np.allclose(rb_c.R, rb_d.R, atol=1e-12)
    assert np.allclose(rb_c.lat_ratio, rb_d.lat_ratio, atol=1e-12)

    loss_c = float(P.pomo_surrogate(jnp.array([-1.0, -2.0, -0.5, -3.0]), rb_c.R))
    loss_d = float(P.pomo_surrogate(jnp.array([-1.0, -2.0, -0.5, -3.0]), rb_d.R))
    assert abs(loss_c - loss_d) < 1e-12
    assert np.allclose(_grad_of_surrogate(rb_c.R),
                       _grad_of_surrogate(rb_d.R), atol=1e-12)

    # a UNIFORM excursion (every measurement 1.2x slower) is likewise a no-op
    uniform = [_paired(l * 1.2, ref * 1.2) for l in lat]
    assert np.allclose(P.score_rewards_paired(uniform, (1.0, 0.0)).R, rb_c.R,
                       atol=1e-12)


def test_unpaired_reward_is_NOT_invariant_to_drift_the_control():
    """The control for the fix. Unpaired rewards are absolute microseconds:
    the same clock ramp changes the advantages (and hence the gradient), and
    even a uniform excursion moves the reward scale that the archive, the
    best-so-far and the 'beats reverse' gate are read off."""
    lat = [100e3, 120e3, 90e3, 150e3]
    drift = [1.0, 1.05, 1.12, 1.2]
    clean = [_stub_result(l) for l in lat]
    drifted = [_stub_result(l * d) for l, d in zip(lat, drift)]

    R_c, _f = P.score_rewards(clean, (1.0, 0.0))
    R_d, _f = P.score_rewards(drifted, (1.0, 0.0))
    adv_c = np.asarray(P.shared_baseline_advantage(jnp.asarray(R_c)))
    adv_d = np.asarray(P.shared_baseline_advantage(jnp.asarray(R_d)))
    assert not np.allclose(adv_c, adv_d, atol=1e-6)
    assert not np.allclose(_grad_of_surrogate(R_c), _grad_of_surrogate(R_d),
                           atol=1e-6)
    # the ranking itself flips: trajectory 2 is cheapest clean, 0 under drift
    assert int(np.argmax(R_c)) == 2 and int(np.argmax(R_d)) == 0

    # a UNIFORM excursion leaves the advantage alone but shifts every reward,
    # which is exactly the channel that produced the bogus 17.5% headline
    # (a plan measured hours after its reference).
    R_u, _f = P.score_rewards([_stub_result(l * 1.2) for l in lat], (1.0, 0.0))
    assert not np.allclose(R_u, R_c, atol=1e-6)
    assert abs((R_u - R_c).std()) < 1e-9        # ... a pure, invisible shift


def test_paired_scoring_refuses_to_score_an_unpaired_measurement():
    """A feasible measurement with no reference is dropped, never silently
    scored on absolutes -- that silent fallback IS the defect."""
    rb = P.score_rewards_paired(
        [_paired(100e3, 100e3), _stub_result(1e3)], (1.0, 0.0))
    assert list(rb.dropped) == [False, True]
    assert rb.labels[1] == "unpaired"


# ===========================================================================
# (f) #119 -- unmeasurable trajectories are DROPPED, baseline over survivors
# ===========================================================================
def test_compile_failures_are_dropped_and_baseline_is_over_survivors_only():
    results = [_paired(100e3, 100e3), _paired(200e3, 100e3),
               _compile_failure(), _paired(400e3, 100e3)]
    rb = P.score_rewards_paired(results, (1.0, 0.0))

    assert list(rb.dropped) == [False, False, True, False]
    assert rb.labels[2] == "compile_error"
    assert rb.drop_counts() == {"compile_error": 1}
    assert rb.n_dropped == 1 and abs(rb.drop_rate - 0.25) < 1e-12
    assert rb.usable(min_traj=2)

    idx, R = rb.survivors()
    assert idx == [0, 1, 3]
    assert np.all(np.isfinite(R))
    adv = np.asarray(P.shared_baseline_advantage(jnp.asarray(R)))
    assert abs(float(adv.sum())) < 1e-9
    assert abs(float(np.mean(R)) - float(np.mean(rb.R[[0, 1, 3]]))) < 1e-12

    # the OLD behaviour scored it worst-in-batch, which moves the baseline
    R_old, feas_old = P.score_rewards(results, (1.0, 0.0))
    assert not feas_old[2] and np.isfinite(R_old[2])
    assert abs(float(np.mean(R_old)) - float(np.mean(R))) > 1e-3

    # unmeasurable is NOT missing-at-random (it tracks graph structure), so
    # the drop rate has to be reported, not just silently applied
    assert rb.drop_counts()


def test_worker_death_and_missing_results_are_unmeasurable_too():
    rb = P.score_rewards_paired(
        [_paired(100e3, 100e3), _paired(200e3, 100e3),
         {"status": "infeasible", "reason": "worker_died", "executed": False},
         None], (1.0, 0.0))
    assert list(rb.dropped) == [False, False, True, True]
    assert rb.labels[2] == "worker_died" and rb.labels[3] == "missing"


# ===========================================================================
# (g) fewer than 2 survivors -> the update is skipped, and nothing NaNs
# ===========================================================================
def test_batch_with_too_few_survivors_is_skipped_without_nan():
    rb = P.score_rewards_paired(
        [_paired(100e3, 100e3), _compile_failure(), _compile_failure(),
         _compile_failure()], (1.0, 0.0))
    assert int(rb.keep.sum()) == 1
    assert not rb.usable(min_traj=P.MIN_TRAJ_FOR_UPDATE)
    idx, R = rb.survivors()
    assert idx == [0] and np.all(np.isfinite(R))
    # ... and the skip is not pedantry: a single survivor has zero advantage
    adv = np.asarray(P.shared_baseline_advantage(jnp.asarray(R)))
    assert np.allclose(adv, 0.0)


def test_all_unmeasurable_batch_is_unusable_and_finite_where_it_matters():
    rb = P.score_rewards_paired([_compile_failure()] * 4, (0.5, 0.5))
    assert rb.dropped.all() and not rb.feasible.any()
    assert not rb.usable() and rb.survivors()[0] == []
    assert np.isnan(rb.R).all()             # never scored, never used
    assert rb.drop_rate == 1.0


def test_update_over_survivors_only_is_finite(tiny):
    """Trainer-shaped: drop one trajectory, sub-slice the step batch and take
    a real gradient step over the survivors."""
    _fn, _args, _an, env, static, _graph = tiny
    n = 4
    envs = [ElimEnv(*tiny_target(), vertex_only=True, symbolic=True)
            for _ in range(n)]
    runner = P.PomoRunner(static, n, len(env.jacve_vertices),
                          edge_hint=len(env.state().edges),
                          cand_hint=len(env.reset().legal_vertices))
    policy = _make_policy(static, seed=5)
    rng = np.random.default_rng(11)
    batch, step_mask, _orders = runner.rollout(envs, policy, (1.0, 0.0), rng)

    results = [_paired(1e5 * (i + 1), 1e5) for i in range(n)]
    results[2] = _compile_failure()
    rb = P.score_rewards_paired(results, (1.0, 0.0))
    assert rb.usable()
    idx, R = rb.survivors()
    sub = batch.rows_of(idx, runner.L)
    assert sub.dyn.shape[0] == len(idx) * runner.L

    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(policy, eqx.is_inexact_array))
    pol2, _os, loss, gnorm = runner.update(
        policy, optim, opt_state, sub, (1.0, 0.0), R,
        np.asarray(step_mask)[idx], 0.01)
    assert math.isfinite(loss) and math.isfinite(gnorm)
    assert all(bool(np.all(np.isfinite(np.asarray(x)))) for x in
               jax.tree_util.tree_leaves(eqx.filter(pol2, eqx.is_inexact_array)))


# ===========================================================================
# (h) OOM / timeout are a VERDICT: floor score, not dropped
# ===========================================================================
@pytest.mark.parametrize("reason", ["oom", "timeout", "predicted_memory"])
def test_measured_as_bad_keeps_the_floor_score_and_is_not_dropped(reason):
    results = [_paired(100e3, 100e3), _paired(200e3, 100e3),
               {"status": "infeasible", "executed": False, "reason": reason}]
    rb = P.score_rewards_paired(results, (1.0, 0.0))
    assert not rb.dropped.any(), rb.labels
    assert rb.labels[2] == reason
    assert not rb.feasible[2]
    assert np.all(np.isfinite(rb.R))
    assert rb.R[2] < rb.R[rb.feasible].min()
    margin = float(rb.R[rb.feasible].min() - rb.R[2])
    assert margin >= P.INFEASIBLE_MARGIN_FLOOR - 1e-12
    assert list(rb.survivors()[0]) == [0, 1, 2]


def test_classification_separates_verdicts_from_toolchain_failures():
    assert P.classify_measurement(_paired(1e3, 1e3))[0] == "feasible"
    assert P.classify_measurement(
        {"status": "infeasible", "reason": "oom"}) == ("bad", "oom")
    assert P.classify_measurement(_compile_failure()) == (
        "unmeasurable", "compile_error")
    assert P.classify_measurement(None) == ("unmeasurable", "missing")
    assert P.classify_measurement({"status": "ok", "executed": False})[0] == (
        "unmeasurable")


# ===========================================================================
# (i) #120 -- a plan failing the numeric check is invalidated and excluded
# ===========================================================================
def test_numeric_check_failure_invalidates_excludes_and_logs():
    good = _paired(100e3, 100e3, check_cos=1.00000012, check_maxdiff=1.49e-8)
    bad = _paired(50e3, 100e3, check_cos=0.42, check_maxdiff=3.1)
    assert P.numeric_check(good) == (True, "ok")
    ok, label = P.numeric_check(bad)
    assert not ok and label == "cos_low"

    P.invalidate(bad, label)
    # the invalidation is the LOG record: greppable status/reason on the row
    assert bad["status"] == "error"
    assert bad["reason"].startswith("numeric_check_failed")
    assert bad["numeric_check_failed"] == "cos_low"
    assert P.reason_label(bad) == "numeric_check_failed"

    rb = P.score_rewards_paired([good, bad], (1.0, 0.0))
    assert list(rb.feasible) == [True, False]
    assert list(rb.dropped) == [False, True]
    assert rb.labels[1] == "numeric_check_failed"
    # ... and it is NEVER the batch maximum, i.e. never rewarded for being fast
    assert np.isnan(rb.R[1])


def test_an_unchecked_plan_is_not_a_passed_plan():
    assert P.numeric_check(_paired(1e3, 1e3)) == (False, "unchecked")
    assert P.numeric_check({"check_error": "leaf-size mismatch 4 vs 6"}) == (
        False, "check_error")
    assert P.numeric_check({"check_cos": float("nan")}) == (False, "cos_low")
    assert P.numeric_check(None) == (False, "missing")
    # the floor is tight: the verified good plan reads cos 1.00000012
    assert P.numeric_check({"check_cos": 0.999}, min_cos=P.NUMERIC_CHECK_MIN_COS)[0] is False


# ===========================================================================
# (j) ReferenceTracker: interpolation, modes, drift telemetry
# ===========================================================================
def test_reference_tracker_interpolates_geometrically_between_samples():
    t = [0.0]
    lat = iter([100e3, 144e3])
    trk = P.ReferenceTracker(
        lambda: {"latency_ns": next(lat), "mem_total_bytes": 2.0 ** 20},
        mode="bracket", clock=lambda: t[0])
    trk.sample()
    t[0] = 10.0
    trk.sample()
    assert trk.n_ref == 2 and trk.n_failed == 0
    # geometric midpoint of 100 and 144 is 120, not the arithmetic 122
    lat_mid, _m = trk.reference_at(5.0)
    assert abs(lat_mid - 120e3) < 1e-6
    assert trk.reference_at(-1.0)[0] == 100e3        # clamp to the first
    assert trk.reference_at(99.0)[0] == 144e3        # clamp to the last

    res = trk.pair(_stub_result(240e3), 5.0)
    assert abs(res["lat_ratio"] - 2.0) < 1e-9
    assert res["ref_latency_ns"] == pytest.approx(120e3)
    assert res["ref_mode"] == "bracket"


def test_reference_tracker_off_mode_measures_nothing():
    calls = []

    def ref():
        calls.append(1)
        return {"latency_ns": 1.0, "mem_total_bytes": 1.0}

    trk = P.ReferenceTracker(ref, mode="off")
    assert trk.sample() is None and calls == [] and trk.n_ref == 0
    assert trk.reference_at(0.0) == (None, None)
    # an unpaired result then carries no ratio and is dropped downstream
    res = trk.pair(_stub_result(1e3), 0.0)
    assert res["lat_ratio"] is None
    assert P.score_rewards_paired([res], (1.0, 0.0)).dropped.all()


def test_reference_tracker_counts_failed_references_and_reports_drift():
    seq = iter([100e3, None, 110e3, 105e3])
    t = [0.0]

    def ref():
        v = next(seq)
        t[0] += 60.0
        return {"latency_ns": v, "mem_total_bytes": 2.0 ** 20}

    trk = P.ReferenceTracker(ref, mode="update", clock=lambda: t[0])
    for _ in range(4):
        trk.sample()
    assert trk.n_ref == 4 and trk.n_failed == 1
    d = trk.drift_stats()
    assert d["n"] == 3
    assert abs(d["span_frac"] - 0.1) < 1e-9
    assert d["max_step_frac"] > 0.0 and d["max_rate_frac_per_min"] > 0.0
    assert P.ReferenceTracker(ref, mode="off").drift_stats() == {"n": 0}


def test_pairing_is_stable_under_a_slow_reference_ramp():
    """End-to-end shape of the fix: the same plan measured at t=0 and at
    t=1h, on a reference that ramped 20%, archives the SAME ratio -- which is
    what makes a cross-time 'beats reverse' claim defensible at all."""
    t = [0.0]
    vals = iter([100e3, 120e3])
    trk = P.ReferenceTracker(
        lambda: {"latency_ns": next(vals), "mem_total_bytes": 2.0 ** 20},
        mode="update", clock=lambda: t[0])
    trk.sample()
    early = trk.pair(_stub_result(95e3), 0.0)
    t[0] = 3600.0
    trk.sample()
    late = trk.pair(_stub_result(114e3), 3600.0)
    assert abs(early["lat_ratio"] - late["lat_ratio"]) < 1e-9
    assert abs(early["lat_ratio"] - 0.95) < 1e-9
    # the raw microseconds say the late plan is 20% WORSE -- the artifact
    assert late["latency_ns"] > early["latency_ns"] * 1.19
