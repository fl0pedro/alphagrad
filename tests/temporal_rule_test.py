"""The three temporal rules on the one-step recurrent SHD target.

OWNER RULING, 2026-09-16. The elimination graph is ONE recurrent step. Its
inputs are the weights, the carried state and the input frame; its outputs are
the next state and the step loss. Temporal credit enters as EDGES WITH GIVEN
VALUES, computed outside the graph over the whole, untouched recording.

    tbptt   no temporal edge. The carried state is a constant.
    bptt    an edge from ``s_t`` to the loss carries ``lambda_(t+1)``, the
            adjoint of the suffix. The gradient is the exact contribution step
            ``t`` makes to backpropagation through time.
    rtrl    an edge from the weights to ``s_(t-1)`` carries ``J_(t-1)``, the
            influence matrix of the prefix. The gradient is exactly
            ``dL_t/dW`` through the whole prefix.

THE E-PROP EQUATION THESE TESTS COMPARE AGAINST. Zenke and Neftci (arXiv
2010.11931) replace the state-to-state Jacobian by its BLOCK DIAGONAL, one
block per neuron, which turns the influence matrix into one eligibility trace
per synapse. For ``graphax.examples.neuromorphic.rsnn_cell`` those traces are
written out in ``common/rsnn_shd.eprop_traces``; the one term they drop is
``V[j,k]`` for ``k != j``, the recurrent coupling. Set ``V`` diagonal and the
trace IS the exact carried Jacobian, which is what
:func:`test_eprop_trace_is_exact_when_the_recurrence_is_diagonal` pins.

FLOAT64. Every claim that says EXACT runs in a SUBPROCESS with
``JAX_ENABLE_X64=1`` (the pattern ``plan_log_dtype_name_test.py`` uses). It is
not set in this process: an import-time environment write leaks into every
later test module of a shared pytest process and moves results that have
nothing to do with this one.
"""

import json
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.core import _inline_call_primitives
from graphax.examples.neuromorphic import (
    RSNN_CARRY_BLOCKS,
    RSNN_GIVEN_LENGTHS,
    RSNN_STATE_NAMES,
    RSNN_SURROGATE_SCALE,
    RSNN_WEIGHT_NAMES,
    RSNN_ZERO_BLOCKS,
    SNN_CARRY_SCOPE,
    attach_rsnn_future,
    attach_rsnn_past,
)

from alphagrad.approx.common import examples as ex
from alphagrad.approx.common import rsnn_shd as R
from alphagrad.approx.common import temporal_order as to
from alphagrad.approx.common.datasets import SHD_CHANNELS, SHD_CLASSES
from alphagrad.approx.common.masks import diag_pair_factor_space, diag_valid_mask

T_PIN = 7
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module", autouse=True)
def _this_modules_graphs_only():
    # carry_plan is process state; the tests below name no config, so the
    # registry must hold this module's graphs and nobody else's.
    from alphagrad.approx.common import carry_plan as CP
    CP.reset()
    yield
    CP.reset()


def _args(rule, **kw):
    return ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                       temporal_rule=rule, step_position=T_PIN, **kw)


def _n_bytes(xs, frm):
    return sum(int(np.asarray(x).nbytes) for x in xs[frm:])


# ---------------------------------------------------------------------------
# 1. The three graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule,n_given", [("tbptt", 0), ("bptt", 5),
                                          ("rtrl", 3 + 11)])
def test_each_rule_builds_and_passes_its_own_given_values(rule, n_given):
    xs = _args(rule)
    assert len(xs) == 16 + n_given
    assert RSNN_GIVEN_LENGTHS[n_given] == rule
    fn = ex.get_fn("RSNN_SHD")
    assert float(fn(*xs)) == float(fn(*xs))        # builds and evaluates


def test_the_rules_are_told_apart_by_the_given_count():
    assert RSNN_GIVEN_LENGTHS == {0: "tbptt", 5: "bptt", 14: "rtrl"}
    xs = list(_args("tbptt"))
    fn = ex.get_fn("RSNN_SHD")
    with pytest.raises(ValueError, match="selected by that count"):
        fn(*(xs + [jnp.zeros((3,))]))


def test_the_weights_are_slots_7_8_9_and_V_is_one_of_them():
    assert ex.infer_argnums("RSNN_SHD") == (7, 8, 9) == R.RSNN_ARGNUMS
    xs = _args("tbptt")
    assert xs[7].shape == (R.RSNN_HIDDEN, SHD_CHANNELS)      # W
    assert xs[8].shape == (R.RSNN_HIDDEN, R.RSNN_HIDDEN)     # V, the recurrent one
    assert xs[9].shape == (SHD_CLASSES, R.RSNN_HIDDEN)       # Wo


def test_the_carried_state_is_five_components():
    xs = _args("tbptt")
    h = R.RSNN_HIDDEN
    assert [tuple(x.shape) for x in xs[2:7]] == [
        (h,), (h,), (h,), (h,), (SHD_CLASSES,)]
    assert len(RSNN_STATE_NAMES) == 5


def test_the_forward_value_does_not_move_between_tbptt_and_rtrl():
    """The rtrl attachment is value-neutral, to the last bit.

    ``W - W_ref`` is exactly zero, so the two rules see the same loss and a
    comparison between them is about credit assignment and nothing else.
    """
    fn = ex.get_fn("RSNN_SHD")
    assert float(fn(*_args("tbptt"))) == float(fn(*_args("rtrl")))


def test_the_bptt_scalar_is_the_step_loss_plus_the_adjoint_term():
    """``bptt`` returns ``L_t + <lambda_(t+1), s_t>``, not ``L_t``.

    That is the point: its GRADIENT is the per-step contribution of full
    backpropagation through time, and the extra term is what carries the
    future into it.
    """
    fn = ex.get_fn("RSNN_SHD")
    base = _args("tbptt")
    xs = _args("bptt")
    lam = xs[16:]
    from graphax.examples.neuromorphic import rsnn_cell
    nxt = rsnn_cell(xs[0], *xs[2:7], *xs[7:10], *xs[10:16])
    extra = float(sum(jnp.sum(l * s) for l, s in zip(lam, nxt)))
    assert float(fn(*xs)) == pytest.approx(float(fn(*base)) + extra, rel=1e-5)


# ---------------------------------------------------------------------------
# 2. The given values
# ---------------------------------------------------------------------------

def test_eleven_carried_blocks_and_four_structural_zeros():
    assert len(RSNN_CARRY_BLOCKS) == 11
    assert len(RSNN_ZERO_BLOCKS) == 4
    assert not set(RSNN_CARRY_BLOCKS) & set(RSNN_ZERO_BLOCKS)
    # every (state, weight) pair is in exactly one of the two
    allp = {(s, w) for s in range(5) for w in range(3)}
    assert set(RSNN_CARRY_BLOCKS) | set(RSNN_ZERO_BLOCKS) == allp
    # the four zeros are exactly the ones the READOUT weight cannot reach
    assert set(RSNN_ZERO_BLOCKS) == {(s, 2) for s in range(4)}


def test_carried_block_shapes_and_bytes():
    xs = _args("rtrl")
    h, n_in, n_out = R.RSNN_HIDDEN, SHD_CHANNELS, SHD_CLASSES
    want = {0: (h,), 1: (h,), 2: (h,), 3: (h,), 4: (n_out,)}
    wshape = {0: (h, n_in), 1: (h, h), 2: (n_out, h)}
    total = 0
    for (s_i, w_i), J in zip(RSNN_CARRY_BLOCKS, xs[19:]):
        assert tuple(J.shape) == want[s_i] + wshape[w_i], (s_i, w_i)
        total += int(np.prod(J.shape))
    # the three reference weights lead the tuple
    for ref, w in zip(xs[16:19], xs[7:10]):
        assert np.array_equal(np.asarray(ref), np.asarray(w))
    # 226 MB: four hidden components against W and V, plus the readout row
    want_total = (4 * (h * h * n_in) + 4 * (h * h * h)
                  + n_out * h * n_in + n_out * h * h + n_out * n_out * h)
    assert total == want_total == 56_434_688
    assert total * 4 == 225_738_752


def test_the_structurally_zero_blocks_really_are_zero():
    """``carried_jacobians`` asserts it; this pins that the assertion runs."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.split(jax.random.PRNGKey(1), 3)[1])
    run = R.prefix_state(seq, 3, W)
    jac = jax.jacrev(run, argnums=(0, 1, 2))(*W)
    for s_i, w_i in RSNN_ZERO_BLOCKS:
        assert float(jnp.max(jnp.abs(jac[s_i][w_i]))) == 0.0


def test_the_adjoints_match_the_state_they_multiply():
    xs = _args("bptt")
    for lam, s in zip(xs[16:], xs[2:7]):
        assert tuple(lam.shape) == tuple(s.shape)


def test_a_wrong_given_shape_raises():
    h = R.RSNN_HIDDEN
    st = R.zero_state()
    W = R.rsnn_weights(jax.random.PRNGKey(0))
    bad = tuple(W) + tuple(jnp.zeros((2, 2, 2)) for _ in RSNN_CARRY_BLOCKS)
    with pytest.raises(ValueError, match="Nothing else is a container"):
        attach_rsnn_past(st, W, bad)
    with pytest.raises(ValueError, match="nor the reduced container"):
        attach_rsnn_future(jnp.array(0.0), st,
                           tuple(jnp.zeros((3,)) for _ in range(5)))
    with pytest.raises(ValueError, match="one adjoint per state"):
        attach_rsnn_future(jnp.array(0.0), st, (jnp.zeros((h,)),))


# ---------------------------------------------------------------------------
# 3. The given edge in the graph
# ---------------------------------------------------------------------------

def _jaxpr(rule):
    fn = ex.get_fn("RSNN_SHD")
    xs = _args(rule)
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = _inline_call_primitives(cj.jaxpr, cj.literals)
    return jx, consts, xs


@pytest.mark.parametrize("rule,n_carry", [("tbptt", 0), ("bptt", 15),
                                          ("rtrl", 25)])
def test_the_given_block_is_its_own_scope_and_carries_step_zero(rule, n_carry):
    jx, _, _ = _jaxpr(rule)
    carry = [i for i, e in enumerate(jx.eqns)
             if SNN_CARRY_SCOPE in str(getattr(e.source_info, "name_stack", ""))]
    assert len(carry) == n_carry
    tags = to.step_tags(jx)
    assert all(int(tags[i]) == 0 for i in carry)


def test_the_carried_jacobian_edge_admits_the_eligibility_trace_diag():
    """Every carried-Jacobian edge admits ``Diag(0, 1, gcd)``.

    Pair ``(0, 1)`` ties the STATE index to the weight's ROW index, which is
    the postsynaptic index of the synapse, and that is exactly the pairing an
    eligibility trace has. Its factor space is ``(1, gcd)``, so every divisor
    of the gcd is a legal block granularity and the finest one is the trace.
    """
    from graphax.core import _build_graph, _force
    jx, consts, xs = _jaxpr("rtrl")
    _, graph, _, _ = _build_graph(jx, xs, consts, (7, 8, 9))
    seen = 0
    for src, inner in graph.items():
        for dst, e in inner.items():
            t = _force(e)
            if t is None or t.val is None or np.ndim(t.val) != 3:
                continue
            if len(t.out_dims) != 1 or len(t.primal_dims) != 2:
                continue
            m = diag_valid_mask(t, 8)
            assert m[0, 1] and m[1, 0], "the (state, weight row) pair"
            base, span = diag_pair_factor_space(t, 0, 1)
            assert base == 1 and span > 1
            assert span == np.gcd(int(t.out_dims[0].logical_size),
                                  int(t.primal_dims[0].logical_size))
            seen += 1
    assert seen == len(RSNN_CARRY_BLOCKS)


# ---------------------------------------------------------------------------
# 4. The flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", ["NeuralNetwork", "TransformerLM",
                                     "ADALIF_SNN_SHD", "LIF_SNN_SHD",
                                     "ADALIF_SNN_SEQ", None])
@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_rule_raises_off_the_recurrent_target(example, rule):
    with pytest.raises(ValueError, match="NO time steps"):
        R.resolve_temporal_rule(example, rule)


def test_the_default_is_tbptt_on_the_target_and_nothing_elsewhere():
    assert R.resolve_temporal_rule("RSNN_SHD", None) == "tbptt"
    assert R.resolve_temporal_rule("NeuralNetwork", None) is None
    assert R.resolve_temporal_rule("ADALIF_SNN_SHD", None) is None


def test_an_unknown_rule_raises():
    with pytest.raises(ValueError, match="not one of"):
        R.resolve_temporal_rule("RSNN_SHD", "banana")


def test_the_gradient_window_raises_on_the_recurrent_target():
    """The graph is one step, so a window would size nothing."""
    with pytest.raises(ValueError, match="NO time steps"):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                    grad_window=2)


# ---------------------------------------------------------------------------
# 5. The step position
# ---------------------------------------------------------------------------

def test_the_step_position_is_sampled_and_recorded():
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(3), dataset=None)
    pos = R.last_step_position()
    assert pos["rule"] == "tbptt"
    assert 1 <= pos["t"] < pos["T"]
    assert pos["T"] == int(len(xs[0]) * 0 + 100)     # the loader's 100 bins


def test_two_keys_give_two_step_positions():
    seen = set()
    for seed in range(12):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(seed), dataset=None)
        seen.add(R.last_step_position()["t"])
    assert len(seen) > 1, "the sampler never moved"


def test_a_pinned_step_position_is_used():
    ex.get_args("RSNN_SHD", jax.random.PRNGKey(3), dataset=None,
                step_position=11)
    assert R.last_step_position()["t"] == 11
    with pytest.raises(ValueError, match="outside"):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(3), dataset=None,
                    step_position=1000)


def test_the_plan_record_carries_the_step_position():
    """The graph is the same at every ``t`` and only the given values move.

    A record that does not say which step it measured cannot be read back
    against another, so the position rides on every record.
    """
    import alphagrad.approx.env as envmod

    ex.get_args("RSNN_SHD", jax.random.PRNGKey(3), dataset=None,
                step_position=5)
    # No probe batch has been drawn here, so the builder's own tuple is what
    # the record reports (see `_record_plan`).
    envmod._PROBE_META.clear()
    envmod._PLAN_RECORDS.clear()
    envmod._record_plan({"order": [1, 2]})
    rec = envmod._PLAN_RECORDS[-1]
    envmod._PLAN_RECORDS.clear()
    assert rec["step_position"]["t"] == 5
    assert rec["step_position"]["rule"] == "tbptt"


# ---------------------------------------------------------------------------
# 6. The model constants, and the gate that chose them
# ---------------------------------------------------------------------------

def test_the_decays_come_from_the_time_constants():
    a_syn, a_mem, a_out, rho = R.decay_constants()
    assert a_syn == pytest.approx(np.exp(-R.DT_MS / R.TAU_SYN_MS))
    assert a_mem == pytest.approx(np.exp(-R.DT_MS / R.TAU_MEM_MS))
    assert a_out == pytest.approx(np.exp(-R.DT_MS / R.TAU_OUT_MS))
    assert rho == pytest.approx(np.exp(-R.DT_MS / R.TAU_A_MS))
    assert R.DT_MS == 10.0, "the loader's bin width; do not rebin"


def test_the_surrogate_is_zenkes():
    """``1 / (scale |x| + 1)^2`` with ``scale = 100``, SpyTorch's own."""
    from graphax.examples.neuromorphic import rsnn_surrogate
    assert RSNN_SURROGATE_SCALE == 100.0
    x = jnp.array([0.0, 0.01, -0.02])
    d = jax.vmap(jax.grad(lambda v: rsnn_surrogate(v)))(x)
    want = 1.0 / (100.0 * jnp.abs(x) + 1.0) ** 2
    assert np.allclose(np.asarray(d), np.asarray(want), rtol=1e-6)
    assert float(rsnn_surrogate(jnp.array(1.0))) == 1.0
    assert float(rsnn_surrogate(jnp.array(-1.0))) == 0.0


def test_the_network_fires_at_its_initial_weights():
    """A silent net makes every recurrent block exactly zero.

    The init scale is not Zenke's 0.2 for this reason and this reason only
    (job 65991). If a future change makes the net silent again, the recurrent
    coupling the whole design is about becomes invisible, so it fails here.
    """
    from graphax.examples.neuromorphic import rsnn_cell
    seq, _, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.split(jax.random.PRNGKey(1), 3)[1])
    c = R._consts()
    st = R.zero_state()
    spikes = 0.0
    for t in range(int(seq.shape[0])):
        st = rsnn_cell(seq[t], *st, *W, *c)
        spikes += float(jnp.sum(st[0]))
    rate = spikes / (int(seq.shape[0]) * R.RSNN_HIDDEN)
    assert rate > 1e-3, f"the network is silent at init (rate {rate})"


# ---------------------------------------------------------------------------
# 7. EXACTNESS, in a float64 subprocess
# ---------------------------------------------------------------------------

_EXACT = r'''
import os, json, sys
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np, jax, jax.numpy as jnp
from alphagrad.approx.common import examples as ex, rsnn_shd as R
from graphax.examples.neuromorphic import RSNN_CARRY_BLOCKS

assert jax.config.jax_enable_x64
R.RSNN_HIDDEN = 6
T, NIN = 9, 700
key = jax.random.split(jax.random.PRNGKey(5), 3)
seq = jax.random.bernoulli(key[0], 0.2, (T, NIN)).astype(jnp.float64)
y = jax.nn.one_hot(3, 20).astype(jnp.float64)
W = R.rsnn_weights(key[1])
fn = ex.get_fn("RSNN_SHD")


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n else float(np.linalg.norm(a - b))


def head(t, weights):
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, t, weights)(*weights))
    return (seq[t], y) + st + tuple(weights) + R._consts(), st


out = {}
t = 5
h, st = head(t, W)
g_tb = jax.grad(fn, argnums=(7, 8, 9))(*h)
ref = jax.grad(lambda a, b, c: R.step_target_loss(seq, y, t, (a, b, c), st)[0],
               argnums=(0, 1, 2))(*W)
out["tbptt_vs_truncated"] = max(rel(a, b) for a, b in zip(g_tb, ref))

g_rt = jax.grad(fn, argnums=(7, 8, 9))(
    *(h + R.carried_jacobians(seq, t, W)))
ref = jax.grad(
    lambda a, b, c: R.step_target_loss(
        seq, y, t, (a, b, c), R.prefix_state(seq, t, (a, b, c))(a, b, c))[0],
    argnums=(0, 1, 2))(*W)
out["rtrl_vs_full_prefix"] = max(rel(a, b) for a, b in zip(g_rt, ref))

full = jax.grad(lambda a, b, c: R.sequence_loss(seq, y, (a, b, c)),
                argnums=(0, 1, 2))(*W)
for rule in ("bptt", "rtrl"):
    acc = [jnp.zeros_like(w) for w in W]
    for u in range(T):
        hu, stu = head(u, W)
        given = (R.carried_jacobians(seq, u, W) if rule == "rtrl"
                 else R.future_adjoints(seq, y, u, W, stu))
        g = jax.grad(fn, argnums=(7, 8, 9))(*(hu + tuple(given)))
        acc = [a + b for a, b in zip(acc, g)]
    out[f"sum_{rule}_vs_sequence"] = max(rel(a, b) for a, b in zip(acc, full))

# the e-prop recursion
Wd = (W[0], jnp.diag(jnp.diag(W[1])), W[2])
ex_d = R.carried_jacobians(seq, t, Wd)
tr_d = R.eprop_traces(seq, t, Wd)
out["eprop_vs_exact_V_diagonal"] = max(
    rel(a, b) for a, b in zip(tr_d[3:], ex_d[3:]))
ex_f = R.carried_jacobians(seq, t, W)
tr_f = R.eprop_traces(seq, t, W)
out["eprop_vs_exact_V_full"] = max(
    rel(a, b) for a, b in zip(tr_f[3:], ex_f[3:]))
hf, _ = head(t, W)
ge = jax.grad(fn, argnums=(7, 8, 9))(*(hf + ex_f))
gt = jax.grad(fn, argnums=(7, 8, 9))(*(hf + tr_f))
a = np.concatenate([np.asarray(x, np.float64).ravel() for x in gt])
b = np.concatenate([np.asarray(x, np.float64).ravel() for x in ge])
out["eprop_gradient_cos"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
out["eprop_gradient_rel"] = rel(a, b)
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def exact_results():
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env.setdefault("JAX_PLATFORMS", env.get("JAX_PLATFORMS", ""))
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(_EXACT)],
                         capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-4000:], out.stderr[-4000:])
    return json.loads(lines[-1][len("RESULT "):])


def test_tbptt_is_the_truncated_gradient(exact_results):
    assert exact_results["tbptt_vs_truncated"] < 1e-12


def test_rtrl_is_exact_through_the_whole_prefix(exact_results):
    assert exact_results["rtrl_vs_full_prefix"] < 1e-11


@pytest.mark.parametrize("rule", ["bptt", "rtrl"])
def test_the_step_gradients_sum_to_the_sequence_gradient(exact_results, rule):
    """Both rules decompose the SAME object, from opposite ends.

    ``sum_t`` of the bptt step gradient and ``sum_t`` of the rtrl step gradient
    are both the gradient of the whole sequence loss. That is what makes them
    exact per-step rules rather than two different approximations.
    """
    assert exact_results[f"sum_{rule}_vs_sequence"] < 1e-10


def test_eprop_trace_is_exact_when_the_recurrence_is_diagonal(exact_results):
    """With a diagonal ``V`` there is no coupling to drop, so e-prop is exact.

    This is what pins the eligibility-trace recursion as the RIGHT one: the
    only thing it leaves out is ``V[j,k]`` for ``k != j``.
    """
    assert exact_results["eprop_vs_exact_V_diagonal"] < 1e-12


def test_eprop_is_a_real_approximation_when_the_recurrence_is_full(
        exact_results):
    assert exact_results["eprop_vs_exact_V_full"] > 1e-4
    assert exact_results["eprop_gradient_rel"] > 1e-5
    assert exact_results["eprop_gradient_cos"] < 1.0


# ---------------------------------------------------------------------------
# 8. THE DATA GENERATOR, and the step position per environment and per episode
#
# The gradient cosine is reward slot 6 and it scores the plan's gradient
# against the rev-exact one ON A PROBE BATCH. The SHD family had no data
# generator, so the channel was undefined and read 0.0 for every plan on every
# SHD target (RSNN_SHD, ADALIF_SNN_SHD and LIF_SNN_SHD alike). On the
# recurrent target everything that changes with the step position IS data --
# the input frame, the carried state and the rule's given values -- so the
# generator that samples a new step position is the same object that arms the
# quality channel, and one mechanism answers both.
# ---------------------------------------------------------------------------

def _gen(rule, **kw):
    return ex.data_gen("RSNN_SHD", dataset=None, key=jax.random.PRNGKey(1),
                       temporal_rule=rule, **kw)


@pytest.mark.parametrize("rule,n_given", [("tbptt", 0), ("bptt", 5),
                                          ("rtrl", 14)])
def test_the_generator_declares_the_slots_it_fills(rule, n_given):
    gen = _gen(rule)
    assert gen is not None, "the SHD family had no data generator at all"
    slots = gen.data_slots
    assert slots == tuple(range(0, 10)) + tuple(range(16, 16 + n_given))
    data = gen(jax.random.split(jax.random.PRNGKey(5), 5))
    assert len(data) == len(slots)


@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_draw_has_the_shape_of_the_slots_it_replaces(rule):
    gen = _gen(rule)
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                     temporal_rule=rule)
    data = gen(jax.random.split(jax.random.PRNGKey(5), 5))
    for slot, d in zip(gen.data_slots, data):
        assert jnp.shape(d) == jnp.shape(xs[slot]), f"slot {slot}"
        assert jnp.asarray(d).dtype == jnp.asarray(xs[slot]).dtype


@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_generator_hands_back_the_runs_own_weights(rule):
    """THE WEIGHTS ARE PART OF THE DRAW ON PURPOSE.

    `generate_eval_samples` redraws every differentiated slot a generator does
    NOT cover. Under `rtrl` the given values lead with three REFERENCE
    weights whose whole job is to equal slots 7 to 9 bit for bit, so that the
    attached ``W - W_ref`` is exactly zero and no forward value moves. A
    generator that left the weights out would have had them redrawn and that
    equality broken in silence.
    """
    gen = _gen(rule)
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                     temporal_rule=rule)
    data = dict(zip(gen.data_slots, gen(jax.random.split(
        jax.random.PRNGKey(5), 5))))
    for slot in (7, 8, 9):
        np.testing.assert_array_equal(np.asarray(data[slot]),
                                      np.asarray(xs[slot]))
    if rule == "rtrl":
        for slot, w in zip((16, 17, 18), (7, 8, 9)):
            np.testing.assert_array_equal(np.asarray(data[slot]),
                                          np.asarray(data[w]))


def test_the_generator_needs_the_runs_key():
    with pytest.raises(ValueError, match="needs the same `key`"):
        ex.data_gen("RSNN_SHD", dataset=None)


@pytest.mark.parametrize("example", ["LIF_SNN_SHD", "ADALIF_SNN_SHD"])
def test_the_older_shd_targets_have_a_generator_too(example):
    """The same defect and the same fix. They have no temporal rule, so what
    a draw moves is WHERE the gradient window sits in the recording."""
    gen = ex.data_gen(example, dataset=None, key=jax.random.PRNGKey(1),
                      grad_window=1)
    assert gen is not None
    assert gen.data_slots == tuple(range(0, 11))
    xs = ex.get_args(example, jax.random.PRNGKey(1), dataset=None,
                     grad_window=1)
    data = gen(jax.random.split(jax.random.PRNGKey(5), 5))
    assert len(data) == 11
    for slot, d in zip(gen.data_slots, data):
        assert jnp.shape(d) == jnp.shape(xs[slot]), f"slot {slot}"


@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_graph_shape_does_not_move_with_the_step_position(rule):
    """THE SAFETY PROPERTY OF A SAMPLED STEP POSITION.

    ``t`` is drawn per environment and per episode, so a graph that changed
    shape with it would change the action space mid-run. The prefix and the
    suffix are masked scans over the whole recording, so it does not.
    """
    seen = set()
    for t in (1, 7, 40, 63, 99):
        xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         temporal_rule=rule, step_position=t)
        jx = _jaxpr_of(xs)
        n_valid = sum(1 for i, e in enumerate(jx.eqns, 1)
                      if e.outvars[0] not in jx.outvars)
        n_faces = _face_count(jx, xs)
        seen.add((len(jx.eqns), n_valid, n_faces))
    assert len(seen) == 1, f"the graph moved with t: {seen}"


def _jaxpr_of(xs, example="RSNN_SHD"):
    fn = ex.get_fn(example)
    cj = jax.make_jaxpr(fn)(*xs)
    jx, _ = _inline_call_primitives(cj.jaxpr, cj.literals)
    return jx


def _face_count(jx, xs):
    """Every live face of the reverse order, counted on one replay."""
    from graphax import faces_of
    from graphax.incremental import IncrementalJaxpr
    fn = ex.get_fn("RSNN_SHD")
    cj = jax.make_jaxpr(fn)(*xs)
    _, consts = _inline_call_primitives(cj.jaxpr, cj.literals)
    argnums = ex.infer_argnums("RSNN_SHD")
    ij = IncrementalJaxpr(jx, tuple(argnums), list(consts), list(xs),
                          track_faces=False)
    valid = [i for i, e in enumerate(jx.eqns, 1)
             if e.outvars[0] not in jx.outvars]
    n = 0
    for v in sorted(valid, reverse=True):
        n += len(faces_of(ij.graph, ij.tgraph, int(v), jx))
        ij.eliminate(v, (), None)
    return n


def _probe_t(envmod, cfg, args, episode, slot):
    envmod._PROBE_BATCH.clear()
    os.environ["ALPHAGRAD_WALK_EPISODE"] = str(int(episode))
    envmod._ENV_SLOT[0] = int(slot)
    try:
        envmod._probe_batch(cfg, list(args), role="train", index=0)
        return int(envmod.probe_meta()["t"])
    finally:
        envmod._ENV_SLOT[0] = -1
        os.environ.pop("ALPHAGRAD_WALK_EPISODE", None)
        envmod._PROBE_BATCH.clear()


class _Cfg:
    """The two fields `_probe_batch` reads."""
    def __init__(self, gen):
        self.data_gen = gen


def test_the_step_position_moves_per_environment_and_per_episode():
    """OWNER RULING 2026-09-16. Sampled uniformly over the recording PER
    ENVIRONMENT AND PER EPISODE. Before this the probe batch was drawn once
    per process, so every environment of every episode measured one step."""
    import alphagrad.approx.env as envmod
    gen = _gen("tbptt")
    cfg = _Cfg(gen)
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None)
    seen = {}
    for ep in range(3):
        for slot in range(4):
            seen[(ep, slot)] = _probe_t(envmod, cfg, xs, ep, slot)
    assert len(set(seen.values())) > 6, (
        f"the draw barely moved over 12 (episode, environment) pairs: {seen}")
    by_env = {ep: {s: t for (e, s), t in seen.items() if e == ep}
              for ep in range(3)}
    for ep, row in by_env.items():
        assert len(set(row.values())) > 1, (
            f"episode {ep} gave one step position to every environment: {row}")
    for slot in range(4):
        col = {ep: seen[(ep, slot)] for ep in range(3)}
        assert len(set(col.values())) > 1, (
            f"environment {slot} kept one step position across episodes: {col}")


def test_the_same_environment_and_episode_give_the_same_step_position():
    """The trainer and every measure actor build their probe from the same
    seed, so an actor measuring a different step than the search acts on would
    be invisible."""
    import alphagrad.approx.env as envmod
    cfg = _Cfg(_gen("tbptt"))
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None)
    a = _probe_t(envmod, cfg, xs, 2, 3)
    b = _probe_t(envmod, cfg, xs, 2, 3)
    assert a == b


def test_a_generator_with_no_per_env_draw_is_unchanged():
    """Every image and token generator keeps the one-batch-per-process
    behaviour: the fold is armed by the generator, not by the env."""
    import alphagrad.approx.env as envmod
    calls = []

    def plain(keys):
        calls.append(1)
        return (jnp.zeros((2,)), jnp.zeros((2,)))

    cfg = _Cfg(plain)
    for slot in range(3):
        envmod._ENV_SLOT[0] = slot
        envmod._probe_batch(cfg, [jnp.zeros((2,)), jnp.zeros((2,))])
    envmod._ENV_SLOT[0] = -1
    envmod._PROBE_BATCH.clear()
    assert len(calls) == 1, "the env slot leaked into a generator that never asked"


def test_the_plan_record_says_which_step_the_probe_measured():
    import alphagrad.approx.env as envmod
    cfg = _Cfg(_gen("bptt"))
    xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                     temporal_rule="bptt", step_position=3)
    t = _probe_t(envmod, cfg, xs, 1, 1)
    envmod._PROBE_META.clear()
    envmod._ENV_SLOT[0] = 1
    os.environ["ALPHAGRAD_WALK_EPISODE"] = "1"
    try:
        envmod._probe_batch(cfg, list(xs), role="train", index=0)
        envmod._PLAN_RECORDS.clear()
        envmod._record_plan({"order": [1, 2]})
        rec = envmod._PLAN_RECORDS[-1]
    finally:
        envmod._PLAN_RECORDS.clear()
        envmod._PROBE_BATCH.clear()
        envmod._PROBE_META.clear()
        envmod._ENV_SLOT[0] = -1
        os.environ.pop("ALPHAGRAD_WALK_EPISODE", None)
    assert rec["step_position"]["t"] == t
    assert rec["step_position"]["rule"] == "bptt"


def test_the_declared_slots_must_match_the_arrays():
    import alphagrad.approx.env as envmod

    def liar(keys):
        return (jnp.zeros((2,)),)
    liar.data_slots = (0, 1)
    with pytest.raises(ValueError, match="declares 2 slots"):
        envmod._grad_cosine_quality(
            _Cfg(liar), lambda *a: a, lambda *a: a, b"slots-liar",
            [jnp.zeros((2,)), jnp.zeros((2,))], None, 1)


# ---------------------------------------------------------------------------
# 9. THE CARRY FOLLOWS THE PLAN (owner ruling 2026-09-16)
#
# The given temporal edge is no longer a snapshot of the exact carry. It is
# the RULE, run over the whole prefix (rtrl) or suffix (bptt), so the value
# arriving at step t carries the error that rule accumulated over the
# recording -- and, for rtrl, it is stored in the CONTAINER that rule implies.
# That container is the point: an approximation applied inside the graph
# cannot shrink an argument, so the eligibility trace is the only thing the
# memory channel can ever see.
# ---------------------------------------------------------------------------

def test_the_containers_are_told_apart_by_shape_and_dtype():
    from graphax.examples.neuromorphic import rsnn_carry_container
    h, n_in = R.RSNN_HIDDEN, SHD_CHANNELS
    f32, bf16 = jnp.float32, jnp.bfloat16

    def name(shape, dtype=f32):
        return rsnn_carry_container((0, 0), (h,), (h, n_in), shape,
                                    dtype).name

    assert name((h, h, n_in)) == "exact"
    assert name((h, n_in)) == "diag"
    assert name((h, h, 1)) == "reduce"
    assert name((h, 1)) == "diag+reduce"
    assert name((h, h, n_in), bf16) == "quant"
    assert name((h, n_in), bf16) == "diag+quant"
    assert name((h, 1), bf16) == "diag+reduce+quant"
    with pytest.raises(ValueError, match="Nothing else is a container"):
        rsnn_carry_container((0, 0), (h,), (h, n_in), (3, 3))


def test_every_container_name_round_trips():
    from graphax.examples.neuromorphic import (RSNN_CARRY_CONTAINERS,
                                               carry_container_from_name)
    assert "exact" in RSNN_CARRY_CONTAINERS
    assert len(RSNN_CARRY_CONTAINERS) == 8
    for n in RSNN_CARRY_CONTAINERS:
        assert carry_container_from_name(n).name == n
    with pytest.raises(ValueError, match="is not one of"):
        carry_container_from_name("banana")


def test_skip_dominates_and_none_is_the_identity():
    assert R.container_from_classes([]) == "exact"
    assert R.container_from_classes(["none"]) == "exact"
    assert R.container_from_classes(["diag"]) == "diag"
    assert R.container_from_classes(["quant", "diag"]) == "diag+quant"
    assert R.container_from_classes(["diag", "skip"]) == "skip"
    with pytest.raises(ValueError, match="are not action classes"):
        R.container_from_classes(["banana"])


def test_the_compact_container_is_the_store_the_approximation_buys():
    """225.74 MB against 2.12 MB. THE MEMORY CHANNEL CANNOT SEE ANY OTHER
    saving: the carry is an ARGUMENT, and nothing a plan does inside the
    graph shrinks an argument."""
    xs_d = _args("rtrl", carry_container="exact")
    xs_e = _args("rtrl", carry_container="diag")
    assert len(xs_d) == len(xs_e) == 16 + 14, "the rule selector must not move"
    dense = sum(int(np.asarray(x).nbytes) for x in xs_d[19:])
    compact = sum(int(np.asarray(x).nbytes) for x in xs_e[19:])
    assert dense == 225_738_752
    assert compact == 2_129_920
    assert dense // compact > 100


def test_the_reference_weights_lead_either_container():
    for cont in ("exact", "diag"):
        xs = _args("rtrl", carry_container=cont)
        for slot, ref in zip((7, 8, 9), (16, 17, 18)):
            np.testing.assert_array_equal(np.asarray(xs[slot]),
                                          np.asarray(xs[ref]))


def test_the_forward_value_does_not_move_between_containers():
    """The attached weight delta is exactly zero in both, so the loss is the
    same to the last bit. A container that moved the loss would be measuring
    a different function, not the same one more cheaply."""
    fn = ex.get_fn("RSNN_SHD")
    a = fn(*_args("rtrl", carry_container="exact"))
    b = fn(*_args("rtrl", carry_container="diag"))
    c = fn(*_args("tbptt"))
    assert float(a) == float(b) == float(c)


def test_the_compact_carry_expands_to_the_recursion():
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    compact = R.carry_traces(seq, 9, W)
    expanded = R.eprop_traces(seq, 9, W)[3:]
    assert len(compact) == len(expanded) == len(RSNN_CARRY_BLOCKS)
    for (s, w), c, e in zip(RSNN_CARRY_BLOCKS, compact, expanded):
        assert tuple(c.shape) == tuple(W[w].shape), (s, w)
        assert e.ndim == c.ndim + 1


def test_the_readout_block_against_the_readout_weight_is_exact():
    """``(Uo, Wo)[m, k, j] = delta(m, k) * g[j]``: the readout feeds nothing
    back, so that block is a leaky filter of the hidden SPIKES and the compact
    form loses nothing at all."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    exact = R.carried_jacobians(seq, 9, W)[3:]
    trace = R.eprop_traces(seq, 9, W)[3:]
    k = RSNN_CARRY_BLOCKS.index((4, 2))
    a = np.asarray(trace[k], np.float64)
    b = np.asarray(exact[k], np.float64)
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-6


def test_the_bptt_adjoint_follows_the_plan_too():
    """The container does not move for bptt -- an adjoint is 532 numbers
    either way -- but the VALUE does: the state-to-state Jacobian is block
    diagonalised at every step of the suffix."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, 40, W)(*W))
    lam_e = R.future_adjoints(seq, y, 40, W, st, "exact")
    lam_p = R.future_adjoints(seq, y, 40, W, st, "diag")
    assert len(lam_e) == len(lam_p) == 5
    for a, b in zip(lam_e, lam_p):
        assert a.shape == b.shape
    ae = np.concatenate([np.asarray(x, np.float64).ravel() for x in lam_e])
    ap = np.concatenate([np.asarray(x, np.float64).ravel() for x in lam_p])
    assert np.linalg.norm(ae) > 0
    assert not np.allclose(ae, ap), "the block diagonal changed nothing"


def test_the_container_raises_off_the_recurrent_target():
    with pytest.raises(ValueError, match="carries no temporal edge"):
        ex.get_args("ADALIF_SNN", jax.random.PRNGKey(1),
                    carry_container="diag")
    with pytest.raises(ValueError, match="not one of"):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                    carry_container="banana")


def test_the_plan_record_says_which_container_the_carry_arrived_in():
    import alphagrad.approx.env as envmod
    for cont in ("exact", "diag"):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(3), dataset=None,
                    temporal_rule="rtrl", step_position=5,
                    carry_container=cont)
        envmod._PROBE_META.clear()
        envmod._PLAN_RECORDS.clear()
        envmod._PLAN_CARRY[0] = None
        envmod._record_plan({"order": [1, 2]})
        rec = envmod._PLAN_RECORDS[-1]
        envmod._PLAN_RECORDS.clear()
        assert rec["carry_container"] == cont
        assert rec["step_position"]["carry"] == cont


@pytest.mark.parametrize("cont", ["exact", "diag"])
def test_the_graph_shape_does_not_move_with_the_step_position_per_container(cont):
    seen = set()
    for t in (1, 40, 99):
        xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         temporal_rule="rtrl", step_position=t,
                         carry_container=cont)
        jx = _jaxpr_of(xs)
        n_valid = sum(1 for i, e in enumerate(jx.eqns, 1)
                      if e.outvars[0] not in jx.outvars)
        seen.add((len(jx.eqns), n_valid))
    assert len(seen) == 1, f"the graph moved with t under {cont}: {seen}"


def test_the_generator_draws_the_container_it_was_asked_for():
    for cont in ("exact", "diag"):
        gen = ex.data_gen("RSNN_SHD", dataset=None, key=jax.random.PRNGKey(1),
                          temporal_rule="rtrl", carry_container=cont)
        data = gen(jax.random.split(jax.random.PRNGKey(5), 5))
        xs = ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         temporal_rule="rtrl", carry_container=cont)
        for slot, d in zip(gen.data_slots, data):
            assert jnp.shape(d) == jnp.shape(xs[slot]), (cont, slot)
        assert gen.meta(jax.random.split(jax.random.PRNGKey(5), 5))["carry"] == cont


def test_the_generator_publishes_an_exact_reference_draw_only_when_it_needs_one():
    """Under ``exact`` the in-band rev-exact reference IS the truth, so there
    is nothing to publish. Under ``eprop`` the drawn carry is approximated and
    the reference has to come from the exact draw at the SAME step position,
    or the quality channel would read 1.0 for a rule that accumulated real
    error over the whole recording."""
    plain = ex.data_gen("RSNN_SHD", dataset=None, key=jax.random.PRNGKey(1),
                        temporal_rule="rtrl", carry_container="exact")
    assert getattr(plain, "reference_draw", None) is None
    approx = ex.data_gen("RSNN_SHD", dataset=None, key=jax.random.PRNGKey(1),
                         temporal_rule="rtrl", carry_container="diag")
    ref = getattr(approx, "reference_draw")
    keys = jax.random.split(jax.random.PRNGKey(5), 5)
    a = approx(keys)
    r = ref(keys)
    assert len(a) == len(r) == len(approx.data_slots)
    # the same step position and the same weights, a different carry
    for i in range(10):
        np.testing.assert_array_equal(np.asarray(a[i]), np.asarray(r[i]))
    assert sum(int(np.asarray(x).nbytes) for x in r[10:]) > \
        80 * sum(int(np.asarray(x).nbytes) for x in a[10:])


def test_the_oracle_reference_makes_the_accumulated_error_visible():
    """WHAT REWARD SLOT 6 HOLDS under an approximated carry.

    The plan handed in is EXACT, so everything left in the number is the
    error the RULE accumulated over the whole recording. The generator is a
    stub with a PINNED step position, so the expected number is a constant of
    this test and not a property of whatever the sampler drew.
    """
    import alphagrad.approx.env as envmod

    fn = ex.get_fn("RSNN_SHD")
    argnums = tuple(ex.infer_argnums("RSNN_SHD"))
    t = 40
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    st = tuple(jax.lax.stop_gradient(x) for x in R.prefix_state(seq, t, W)(*W))
    head = (seq[t], y) + st + tuple(W)
    g_p = R.carry_under_plan(seq, t, W, "diag", check_zeros=False)
    g_e = R.carry_under_plan(seq, t, W, "exact", check_zeros=False)
    ap, exa = head + g_p, head + g_e
    full_p = head + R._consts() + g_p
    full_e = head + R._consts() + g_e
    slots = R.rsnn_data_slots("rtrl")

    def stub(keys):
        return ap
    stub.data_slots = slots
    stub.resample_per_env_episode = False
    stub.meta = lambda keys: {"t": t, "T": int(seq.shape[0]), "carry": "diag"}
    stub.reference_draw = lambda keys: exa

    class _C:
        pass
    cfg = _C()
    # ON THE INSTANCE, not the class: a plain function stored as a CLASS
    # attribute is served as a bound method and would be handed `self`.
    cfg.data_gen = stub
    cfg.target_fun = fn
    cfg.scalar_target = True
    cfg.has_aux = False
    cfg.argnums = argnums
    xs = list(ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                          temporal_rule="rtrl", carry_container="diag"))

    def exact_plan(*a):
        return list(jax.grad(fn, argnums=argnums)(*a))

    envmod._PROBE_BATCH.clear()
    envmod._PROBE_META.clear()
    envmod._COSINE_REF.clear()
    got = envmod._grad_cosine_quality(
        cfg, exact_plan, exact_plan, b"ref-draw-visible", xs, None, 1)
    envmod._PROBE_BATCH.clear()
    envmod._PROBE_META.clear()
    envmod._COSINE_REF.clear()
    assert got is not None
    q = got[0]

    g_ap = jax.grad(fn, argnums=argnums)(*full_p)
    g_ex = jax.grad(fn, argnums=argnums)(*full_e)
    a = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ap])
    b = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ex])
    want = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert want < 1.0 - 1e-6, want
    assert abs(q - want) < 1e-5, (q, want)


def test_without_the_reference_draw_the_same_channel_reads_one():
    """The defect the reference draw exists for: both sides read the same
    approximated carry and the channel cannot see what the rule did."""
    import alphagrad.approx.env as envmod

    fn = ex.get_fn("RSNN_SHD")
    argnums = tuple(ex.infer_argnums("RSNN_SHD"))
    t = 40
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    st = tuple(jax.lax.stop_gradient(x) for x in R.prefix_state(seq, t, W)(*W))
    head = (seq[t], y) + st + tuple(W)
    ap = head + R.carry_under_plan(seq, t, W, "diag", check_zeros=False)

    def stub(keys):
        return ap
    stub.data_slots = R.rsnn_data_slots("rtrl")

    class _C:
        pass
    cfg = _C()
    cfg.data_gen = stub
    cfg.target_fun = fn
    cfg.scalar_target = True
    cfg.has_aux = False
    cfg.argnums = argnums
    xs = list(ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                          temporal_rule="rtrl", carry_container="diag"))

    def exact_plan(*a):
        return list(jax.grad(fn, argnums=argnums)(*a))

    envmod._PROBE_BATCH.clear()
    envmod._PROBE_META.clear()
    envmod._COSINE_REF.clear()
    got = envmod._grad_cosine_quality(
        cfg, exact_plan, exact_plan, b"no-ref-draw", xs, None, 1)
    envmod._PROBE_BATCH.clear()
    envmod._PROBE_META.clear()
    envmod._COSINE_REF.clear()
    assert got is not None
    assert abs(got[0] - 1.0) < 1e-6, got[0]


def test_the_oracle_reference_is_not_the_same_number_as_the_in_band_one():
    """Without it the channel reads 1.0 for a rule that accumulated real
    error: both sides read the same approximated carry."""
    fn = ex.get_fn("RSNN_SHD")
    argnums = tuple(ex.infer_argnums("RSNN_SHD"))
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 40
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, t, W)(*W))
    head = (seq[t], y) + st + tuple(W) + R._consts()
    g_ap = jax.grad(fn, argnums=argnums)(
        *(head + R.carry_under_plan(seq, t, W, "diag", check_zeros=False)))
    g_ex = jax.grad(fn, argnums=argnums)(
        *(head + R.carry_under_plan(seq, t, W, "exact", check_zeros=False)))
    a = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ap])
    b = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ex])
    same_args = float(a @ a / (np.linalg.norm(a) * np.linalg.norm(a)))
    oracle = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(same_args - 1.0) < 1e-9
    assert oracle < 1.0 - 1e-6, oracle


# ---------------------------------------------------------------------------
# 12. THE CONTAINER FOLLOWS THE PLAN, PER CLASS (owner rulings 2026-09-16)
#
# The plan-produced carry is the ONLY mode. The exact carry is the plan with
# no approximation on the carried-Jacobian face, and every other container is
# the plan's own classes on that face, applied at every step of the prefix:
#
#   none    exact real time recurrent learning
#   Diag    e-prop, the block-diagonal trace
#   Reduce  a coarser trace, the presynaptic axis collapsed
#   Quant   a low precision trace
#   Skip    no carry at all, which is truncated backpropagation through time
#
# Realized on the MEASUREMENT side: the policy always sees the step body with
# the dense carry edge, and the measurement compiles the consistent recursion
# with the carry in the container that choice implies.
# ---------------------------------------------------------------------------

CLASS_CONTAINERS = ("exact", "diag", "reduce", "quant", "diag+quant",
                    "diag+reduce")


def _carry_bytes(container):
    xs = _args("rtrl", carry_container=container)
    return sum(int(np.asarray(x).nbytes) for x in xs[19:])


@pytest.mark.parametrize("container", CLASS_CONTAINERS)
def test_the_carry_arrives_in_the_container_the_class_implies(container):
    """Shape AND bytes, per class. The attachment reads the container off the
    block itself, so this is the same question the measured program asks."""
    from graphax.examples.neuromorphic import rsnn_carry_container
    xs = _args("rtrl", carry_container=container)
    assert len(xs) == 16 + 14, "the rule selector must not move"
    states = tuple(xs[2:7])
    weights = tuple(xs[7:10])
    for (s, w), J in zip(RSNN_CARRY_BLOCKS, xs[19:]):
        c = rsnn_carry_container((s, w), states[s].shape, weights[w].shape,
                                 J.shape, J.dtype)
        assert c.name == container, ((s, w), c.name)


def test_every_approximated_container_is_smaller_than_the_exact_one():
    """A Reduce and a Quant plan each produce a carry SMALLER than the exact
    one. That is the only thing the memory channel can ever see: an
    approximation applied inside the graph cannot shrink an argument."""
    exact = _carry_bytes("exact")
    assert exact == 225_738_752
    for container in CLASS_CONTAINERS[1:]:
        assert _carry_bytes(container) < exact, container
    assert _carry_bytes("diag") == 2_129_920
    # Quant halves the dense store; Reduce collapses one axis of it.
    assert _carry_bytes("quant") == exact // 2
    assert _carry_bytes("reduce") < _carry_bytes("quant")


def test_skip_on_the_carried_face_means_no_carry_at_all():
    """Skip is not a storage form. The measured rule is the truncated one and
    the given tuple is empty, so the program IS the t-BPTT program."""
    assert R.container_from_classes(["skip"]) == R.SKIP_CONTAINER
    with pytest.raises(ValueError, match="no storage form"):
        R._container(R.SKIP_CONTAINER)
    tb = _args("tbptt")
    assert len(tb) == 16


def test_the_forward_value_does_not_move_between_any_two_containers():
    """Every container attaches a weight delta that is exactly zero, so the
    loss is the same to the last bit. A container that moved the loss would
    be measuring a different function, not the same one more cheaply."""
    fn = ex.get_fn("RSNN_SHD")
    base = float(fn(*_args("tbptt")))
    for container in CLASS_CONTAINERS:
        assert float(fn(*_args("rtrl", carry_container=container))) == base, \
            container


def test_the_reduce_container_is_the_exact_axis_mean():
    """Making an axis implicit stores ONE value for the whole axis, and the
    projection onto that replicated subspace is the mean. The recursion
    commutes with it, so the stored value is the exact mean of the influence
    matrix and the approximation is entirely in the contraction."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 9
    jac = jax.jacrev(R.prefix_state(seq, t, W), argnums=(0, 1, 2))(*W)
    red = R.carry_under_plan(seq, t, W, "reduce")[3:]
    worst = 0.0
    for i, (s, w) in enumerate(RSNN_CARRY_BLOCKS):
        want = jnp.mean(jac[s][w], axis=-1)[..., None]
        assert tuple(red[i].shape) == tuple(want.shape), (s, w)
        scale = float(jnp.max(jnp.abs(want))) + 1e-30
        worst = max(worst, float(jnp.max(jnp.abs(red[i] - want))) / scale)
    assert worst < 1e-2, worst


def test_the_quant_container_is_the_same_recursion_held_narrow():
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 9
    qnt = R.carry_under_plan(seq, t, W, "quant")[3:]
    exact = R.carry_under_plan(seq, t, W, "exact")[3:]
    for a, b in zip(qnt, exact):
        assert a.dtype == jnp.bfloat16
        assert tuple(a.shape) == tuple(b.shape)
    a = np.concatenate([np.asarray(x, np.float64).ravel() for x in qnt])
    b = np.concatenate([np.asarray(x, np.float64).ravel() for x in exact])
    assert not np.array_equal(a, b), "a narrow store that lost nothing"
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.99, cos


def test_the_diag_container_is_the_eprop_recursion():
    """The e-prop test of agent/rtrl, asked of the container the plan picks."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 9
    got = R.carry_under_plan(seq, t, W, "diag")[3:]
    want = R.carry_traces(seq, t, W)
    assert len(got) == len(want) == len(RSNN_CARRY_BLOCKS)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("container", ["reduce", "quant"])
def test_a_reduce_and_a_quant_plan_score_below_one(container):
    """The gradient the container's carry produces is not the exact one, so
    the quality the record shows is below 1.0. Scored against the ORACLE --
    the exact carry on the same step -- because both sides of the in-band
    cosine would read the same approximated argument."""
    fn = ex.get_fn("RSNN_SHD")
    argnums = tuple(ex.infer_argnums("RSNN_SHD"))
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 40
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, t, W)(*W))
    head = (seq[t], y) + st + tuple(W) + R._consts()
    g_ap = jax.grad(fn, argnums=argnums)(
        *(head + R.carry_under_plan(seq, t, W, container, check_zeros=False)))
    g_ex = jax.grad(fn, argnums=argnums)(
        *(head + R.carry_under_plan(seq, t, W, "exact", check_zeros=False)))
    a = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ap])
    b = np.concatenate([np.asarray(x, np.float64).ravel() for x in g_ex])
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos < 1.0 - 1e-9, (container, cos)


def test_the_adjoint_containers_are_the_two_a_vector_can_express():
    """``diag`` moves the VALUE of an adjoint and not its store; ``reduce``
    stores one number per component; ``quant`` stores it narrow."""
    seq, y, _ = R._draw_recording(jax.random.PRNGKey(1), None, -1)
    W = R.rsnn_weights(jax.random.PRNGKey(1))
    t = 40
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, t, W)(*W))
    lam_e = R.future_adjoints(seq, y, t, W, st, "exact")
    lam_r = R.future_adjoints(seq, y, t, W, st, "reduce")
    lam_q = R.future_adjoints(seq, y, t, W, st, "quant")
    for a in lam_r:
        assert tuple(a.shape) == (1,)
    for a, b in zip(lam_q, lam_e):
        assert a.dtype == jnp.bfloat16 and a.shape == b.shape
    bytes_e = sum(int(np.asarray(x).nbytes) for x in lam_e)
    assert sum(int(np.asarray(x).nbytes) for x in lam_r) < bytes_e
    assert sum(int(np.asarray(x).nbytes) for x in lam_q) < bytes_e
    xs = _args("bptt", carry_container="reduce")
    assert len(xs) == 16 + 5
    assert all(tuple(x.shape) == (1,) for x in xs[16:])


# ---------------------------------------------------------------------------

# THE ENV IS BUILT ONCE PER RULE. Building it draws the carry, which is a
# reverse-mode Jacobian of a hundred-step scan; six tests that each built
# their own would spend minutes on the same object.
_ENVS: dict = {}


def _env_for(rule):
    """A landscape_map env on the synthetic Poisson recording, no dataset."""
    import alphagrad.approx.tools.landscape_map as lm
    from alphagrad.approx.common import carry_plan as CP
    hit = _ENVS.get(rule)
    if hit is not None:
        # carry_plan is module state; re-register so a test that reset it
        # still sees this env.
        CP.register(hit["args_ns"], hit["key"], "RSNN_SHD", rule,
                    hit["env"].config, hit["env"].args, hit["env"].consts,
                    dataset=None, dataset_size=-1, step_position=T_PIN)
        return lm, CP, hit["env"]
    argv = ["--example", "RSNN_SHD", "--dataset", "none",
            "--temporal-rule", rule, "--step-position", str(T_PIN),
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--out-dir", "/tmp/carry_test"]
    args = lm.make_argparser().parse_args(argv)
    import jax.random as jrand
    env, eval_samples, cj = lm.build_env(args)
    key, args_key = jrand.split(jrand.PRNGKey(args.seed))
    _ENVS[rule] = {"env": env, "args_ns": args, "key": args_key}
    return lm, CP, env


def test_the_seam_is_armed_only_on_a_rule_with_a_given_edge():
    """A rule with a given edge holds a carry-plan entry, one without holds
    none, and THE GRAPH ANSWERS FOR ITSELF.

    The question used to be asked of the process, because one process served
    one target. Since the owner's ruling of 2026-09-22 a run may alternate
    between two graphs of one target and the process holds an entry per
    graph, so registering the truncated graph must not disarm the other one.
    """
    from alphagrad.approx.common import carry_plan as CP
    _lm, _cp, rtrl_env = _env_for("rtrl")
    assert CP.armed(rtrl_env.config)
    _lm, _cp, tbptt_env = _env_for("tbptt")
    assert not CP.armed(tbptt_env.config)
    assert CP.armed(rtrl_env.config)


@pytest.mark.parametrize("container",
                         ["diag", "reduce", "quant", "diag+quant", "skip"])
def test_every_container_builds_its_own_program_and_the_plan_transports(
        container):
    lm, CP, env = _env_for("rtrl")
    base_valid = sorted(int(v) for v in env.valid_vertices)
    var = CP.measurement_env(container)
    assert var is not None
    # THE STEP BODY IS THE SAME, EQUATION FOR EQUATION. `_alignment` checks
    # the primitive and the output shape of every one of them and raises
    # otherwise, so a non-empty map is a checked map.
    assert var["vertex_map"]
    mask = CP.carry_scope_mask(env.config.jaxpr)
    body = [i + 1 for i, m in enumerate(mask) if not m]
    assert sorted(var["vertex_map"]) == body
    order = [int(v) for v in sorted(base_valid, reverse=True)]
    moved = CP.transport_order(order, var)
    assert sorted(moved) == sorted(var["valid"])
    assert len(set(moved)) == len(moved)


def test_the_container_a_plan_implies_is_read_off_its_wires():
    import alphagrad.approx.env as envmod
    lm, CP, env = _env_for("rtrl")
    jx = env.config.jaxpr
    valid = sorted(int(v) for v in env.valid_vertices)
    mask = CP.carry_scope_mask(jx)
    carry_v = [v for v in valid if mask[v - 1]
               and jx.eqns[v - 1].primitive.name == "dot_general"]
    assert carry_v, "the dense carry block contracts with a dot_general"
    order = np.asarray([v for v in valid if mask[v - 1]]
                       + sorted((v for v in valid if not mask[v - 1]),
                                reverse=True), dtype=np.int32)
    inv = lm.face_inventory(env, order)
    picked = [e for e in inv if int(e["vertex"]) in set(carry_v)]
    assert picked

    def wires_for(row, kind=None):
        if kind == "SKIP":
            return [{"k": int(e["k"]), "f": int(e["f"]), "kind": "SKIP"}
                    for e in picked]
        return [{"k": int(e["k"]), "f": int(e["f"]), "slot": 0,
                 "row": list(row), "kind": "X"} for e in picked]

    from graphax.sparse.micro_actions import COMPRESS_KINDS, QUANT_DTYPES
    cases = {
        "exact": [],
        "diag": wires_for([0, 0, -1]),
        "reduce": wires_for([envmod.COMPRESS_SENTINEL, 0,
                             COMPRESS_KINDS.index("mean")]),
        "quant": wires_for([envmod.QUANT_SENTINEL,
                            QUANT_DTYPES.index("bfloat16"), 0]),
        "skip": wires_for(None, "SKIP"),
    }
    for want, wires in cases.items():
        plan = {"specs": None, "face_specs": None, "face_skips": None,
                "wires": wires}
        specs, faces, skips = lm.get_plan_arrays(plan, len(order))
        got = CP.container_for_plan(env.config, [int(v) for v in order],
                                    faces, skips, specs)
        assert got == want, (want, got)


def test_a_wire_on_the_step_body_does_not_move_the_container():
    """The container follows the plan's classes on the CARRIED face only. A
    Diag somewhere else in the step body is an ordinary approximation."""
    lm, CP, env = _env_for("rtrl")
    jx = env.config.jaxpr
    valid = sorted(int(v) for v in env.valid_vertices)
    mask = CP.carry_scope_mask(jx)
    order = np.asarray(sorted(valid, reverse=True), dtype=np.int32)
    inv = lm.face_inventory(env, order)
    body = [e for e in inv if not mask[int(e["vertex"]) - 1]]
    assert body
    wires = [{"k": int(body[0]["k"]), "f": int(body[0]["f"]), "slot": 0,
              "row": [0, 0, -1], "kind": "X"}]
    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "wires": wires}
    specs, faces, skips = lm.get_plan_arrays(plan, len(order))
    assert CP.container_for_plan(env.config, [int(v) for v in order],
                                 faces, skips, specs) == "exact"


def test_the_skip_variant_scores_against_the_arms_own_rule():
    """A truncated program's own exact gradient is the truncated gradient, so
    scoring against it would read 1.0 for a plan that threw the whole prefix
    away. The generator publishes the arm's rule as the reference instead."""
    lm, CP, env = _env_for("rtrl")
    var = CP.measurement_env("skip")
    ref = getattr(var["config"].data_gen, "reference_oracle", None)
    assert ref is not None
    assert ref["target"] is env.config.target_fun
    assert tuple(ref["argnums"]) == tuple(env.config.argnums)
    assert len(ref["args"]) == len(env.args)
    # the non-skip containers keep the in-band reference draw, which is the
    # exact carry of the SAME rule and the same target
    diag = CP.measurement_env("diag")
    assert getattr(diag["config"].data_gen, "reference_oracle", None) is None
    assert getattr(diag["config"].data_gen, "reference_draw", None) is not None


def test_a_step_body_that_stops_matching_raises():
    """The transport is checked, not assumed: two programs whose step bodies
    differ cannot carry a plan between them."""
    lm, CP, env = _env_for("rtrl")
    other = _jaxpr_of(ex.get_args(R.RSNN_W2_TARGET, jax.random.PRNGKey(1),
                                  dataset=None, temporal_rule="window2",
                                  step_position=T_PIN),
                      example=R.RSNN_W2_TARGET)
    with pytest.raises(ValueError, match="disagree about the STEP BODY"):
        CP._alignment(env.config.jaxpr, other)



# ---------------------------------------------------------------------------
# 14. THE FOURTH ARM: two step copies, no given edge, free order
# ---------------------------------------------------------------------------

def test_window2_is_the_fourth_rule_and_its_own_target():
    assert R.TEMPORAL_RULES == ("tbptt", "bptt", "rtrl", "window2")
    assert R.target_example("RSNN_SHD", "window2") == R.RSNN_W2_TARGET
    assert R.target_example("RSNN_SHD", "rtrl") == "RSNN_SHD"
    assert R.target_example("NeuralNetwork", None) == "NeuralNetwork"
    assert R.is_rsnn(R.RSNN_W2_TARGET)
    assert ex.infer_argnums(R.RSNN_W2_TARGET) == (8, 9, 10)


def test_the_window_arm_has_no_given_edge():
    xs = ex.get_args(R.RSNN_W2_TARGET, jax.random.PRNGKey(1), dataset=None,
                     temporal_rule="window2", step_position=T_PIN)
    # two input frames, the label, five state components, three weights, six
    # constants: seventeen, and nothing after them
    assert len(xs) == 17
    assert xs[0].shape == xs[1].shape == (SHD_CHANNELS,)
    assert xs[2].shape == (SHD_CLASSES,)
    assert xs[8].shape == (R.RSNN_HIDDEN, SHD_CHANNELS)
    fn = ex.get_fn(R.RSNN_W2_TARGET)
    assert float(fn(*xs)) == float(fn(*xs))


def test_the_window_arm_carries_two_step_scopes():
    xs = ex.get_args(R.RSNN_W2_TARGET, jax.random.PRNGKey(1), dataset=None,
                     temporal_rule="window2", step_position=T_PIN)
    jx = _jaxpr_of(xs, example=R.RSNN_W2_TARGET)
    tags = to.step_tags(jx)
    assert set(int(t) for t in tags) >= {0, 1}


def test_the_window_arm_refuses_a_container_and_a_foreign_rule():
    with pytest.raises(ValueError, match="attaches no given temporal edge"):
        ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                    temporal_rule="window2", carry_container="diag")
    with pytest.raises(ValueError, match="only rule it can run is window2"):
        R.resolve_temporal_rule(R.RSNN_W2_TARGET, "rtrl")
    assert R.resolve_temporal_rule(R.RSNN_W2_TARGET, None) == "window2"


def test_the_window_arms_step_position_leaves_room_for_the_second_copy():
    T = 100
    assert R.step_position_bound(T, "window2") == T - 1
    assert R.step_position_bound(T, "rtrl") == T
    seen = set()
    for i in range(64):
        seen.add(R.sample_step_position(jax.random.PRNGKey(i), T, "window2"))
    assert seen and max(seen) <= T - 2 and min(seen) >= 1


def test_the_window_generator_fills_eleven_slots():
    gen = ex.data_gen(R.RSNN_W2_TARGET, dataset=None,
                      key=jax.random.PRNGKey(1), temporal_rule="window2")
    assert tuple(gen.data_slots) == tuple(range(11))
    data = gen(jax.random.split(jax.random.PRNGKey(5), 5))
    xs = ex.get_args(R.RSNN_W2_TARGET, jax.random.PRNGKey(1), dataset=None,
                     temporal_rule="window2")
    for slot, d in zip(gen.data_slots, data):
        assert jnp.shape(d) == jnp.shape(xs[slot]), slot


# ---------------------------------------------------------------------------
# 15. THE QUALITY FLOOR THAT WAS NOT A QUALITY (defect found 2026-09-16)
#
# `_quality_metrics` guarded its denominator with `max(||.||, sqrt(1e-7))`,
# which is a FLOOR at a gradient norm of 3.16e-4. Below it the cosine of two
# BIT-IDENTICAL Jacobians reads ||g||^2 / 1e-7 instead of 1.0, silently. One
# step of a sparse spiking network lives in exactly that regime -- measured
# 2.19e-4 on the two-copy window arm -- so the floor would have priced the
# sampled step position instead of the plan on every SNN row.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4, 1e-6])
def test_two_identical_gradients_score_one_at_every_scale(scale):
    import alphagrad.approx.env as envmod
    g = tuple(jnp.asarray(np.random.RandomState(0).randn(*s), jnp.float32)
              * scale for s in ((8, 5), (5, 5), (3, 5)))
    cos, rel = envmod._quality_metrics(g, g)
    assert abs(float(cos) - 1.0) < 1e-5, (scale, float(cos))
    assert float(rel) < 1e-5


def test_a_zero_reference_still_scores_zero_and_is_dropped():
    """The cosine is UNDEFINED against a zero reference, and the channel drops
    such a batch rather than counting it. The guard against zero stays; only
    the guard against SMALL is gone."""
    import alphagrad.approx.env as envmod
    z = tuple(jnp.zeros(s, jnp.float32) for s in ((8, 5), (5, 5)))
    a = tuple(jnp.ones(s, jnp.float32) for s in ((8, 5), (5, 5)))
    cos, rel = envmod._quality_metrics(z, a)
    assert float(cos) == 0.0
    assert float(rel) == 1.0


def test_a_small_gradient_scores_the_angle_and_not_its_size():
    """Two gradients at a fixed angle score the same cosine whatever their
    length. That is what a cosine means, and what the floor broke."""
    import alphagrad.approx.env as envmod
    rs = np.random.RandomState(1)
    e0 = tuple(jnp.asarray(rs.randn(*s), jnp.float32) for s in ((8, 5), (5, 5)))
    a0 = tuple(x + 0.25 * jnp.asarray(rs.randn(*x.shape), jnp.float32)
               for x in e0)
    ref = float(envmod._quality_metrics(e0, a0)[0])
    for scale in (1e-3, 1e-5, 1e-7):
        e = tuple(x * scale for x in e0)
        a = tuple(x * scale for x in a0)
        got = float(envmod._quality_metrics(e, a)[0])
        assert abs(got - ref) < 1e-4, (scale, got, ref)


def test_a_variant_draws_its_own_eval_samples_from_the_base_draw():
    """The variant's samples cannot be the base ones -- the shapes move with
    the container -- so they are tied to them by a DIGEST of the base draw.
    Every process that measures the plan then builds the same ones, and they
    move per episode exactly as the base draw does."""
    from alphagrad.approx.common import carry_plan as CP
    lm, _CP, env = _env_for("rtrl")
    base = tuple(env.eval_args_samples)
    a = CP.eval_samples_for("diag", base)
    b = CP.eval_samples_for("diag", base)
    assert a is b, "the same draw must not be rebuilt"
    var = CP.measurement_env("diag")
    assert len(a) == len(var["args"])
    for got, want in zip(a, var["args"]):
        assert tuple(got.shape[1:]) == tuple(want.shape), got.shape
    # a DIFFERENT episode's base draw gives a different variant draw
    moved = list(base)
    moved[0] = moved[0] + 1.0
    c = CP.eval_samples_for("diag", tuple(moved))
    assert c is not a
    assert not np.array_equal(np.asarray(a[0]), np.asarray(c[0]))


def test_the_eval_tag_never_pulls_a_big_slot_off_the_device():
    """The carried Jacobian is 226 MB per sample. Hashing its CONTENT on
    every callback would cost more than the measurement, so a slot at or over
    the cap contributes its shape and its dtype and nothing else."""
    from alphagrad.approx.common import carry_plan as CP

    class _Trap:
        shape = (5, 128, 128, 700)
        dtype = np.dtype("float32")

        def __array__(self, *a, **k):
            raise AssertionError("a big slot was pulled off the device")

    small = jnp.zeros((5, 700), jnp.float32)
    tag = CP._eval_tag((small, _Trap()))
    assert isinstance(tag, bytes) and len(tag) == 16


@pytest.mark.parametrize("container", ["diag", "reduce", "skip"])
def test_the_wires_travel_by_position_and_the_carry_rows_are_exact(container):
    """A PLAN IS INDEXED BY (ELIMINATION STEP, FACE POSITION). A face KEY is a
    pair of stable var indices on the LIVE graph and every elimination
    rewires it, so the keys a body vertex shows depend on what the carry
    block left behind, which is exactly what the container changes. The wires
    therefore move to the transported order's positions and the faces are
    enumerated again on the variant's own replay."""
    import alphagrad.approx.env as envmod
    lm, CP, env = _env_for("rtrl")
    jx = env.config.jaxpr
    valid = sorted(int(v) for v in env.valid_vertices)
    mask = CP.carry_scope_mask(jx)
    order = [int(v) for v in sorted(valid, reverse=True)]
    T = len(order)
    mf = envmod.MAX_FACES
    specs = np.full((T, envmod.MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    faces = np.full((T, mf, envmod.FACE_SLOTS, 3), -1, dtype=np.int32)
    skips = np.zeros((T, mf), dtype=np.int32)
    # a Diag on face 0 of EVERY vertex, body and carry alike
    faces[:, 0, 0] = np.array([0, 0, -1], dtype=np.int32)
    var = CP.measurement_env(container)
    o2, s2, f2, k2, j2 = CP.transport_wires(
        order, var, specs, faces, skips, None)
    assert sorted(o2) == sorted(var["valid"])
    assert f2.shape[1:] == faces.shape[1:]
    vmap = var["vertex_map"]
    pos2 = {int(v): i for i, v in enumerate(o2)}
    for k, v in enumerate(order):
        j = vmap.get(v)
        if j is None:
            continue
        np.testing.assert_array_equal(f2[pos2[j]], faces[k])
    # every CARRY vertex of the variant carries an all-exact row: its
    # approximation is what chose the container
    for v in var["alt_carry"]:
        if int(v) not in pos2:
            continue
        row = f2[pos2[int(v)]]
        assert int(row[..., 0].max()) == -1, v
        assert int(k2[pos2[int(v)]].max()) == 0, v


@pytest.mark.parametrize("container", ["quant", "diag+quant"])
def test_the_carry_faces_keep_the_plans_quant_bit(container):
    """A Quant on the carried face is the narrow container AND the narrow
    contraction (owner ruling 2026-09-23): every carry vertex of the measured
    graph gets the plan's own QUANT row on lhs and rhs of every face, while
    its Diag stays in the value and every body row travels by position."""
    import alphagrad.approx.env as envmod
    from alphagrad.approx.unified_face_head import QUANT_SLOTS
    from alphagrad.approx.unified_face_policy import _NARROW_SLOT
    lm, CP, env = _env_for("rtrl")
    valid = sorted(int(v) for v in env.valid_vertices)
    order = [int(v) for v in sorted(valid, reverse=True)]
    T = len(order)
    mf = envmod.MAX_FACES
    specs = np.full((T, envmod.MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    faces = np.full((T, mf, envmod.FACE_SLOTS, 3), -1, dtype=np.int32)
    skips = np.zeros((T, mf), dtype=np.int32)
    q_row = np.array([envmod.QUANT_SENTINEL, int(_NARROW_SLOT), 0], np.int32)
    for s in QUANT_SLOTS:
        faces[:, 0, s] = q_row                # face 0 of EVERY vertex
    if container.startswith("diag"):
        faces[:, 1, 2] = np.array([0, 0, -1], dtype=np.int32)
    assert CP.container_for_plan(env.config, order, faces, skips) == container
    var = CP.measurement_env(container)
    o2, s2, f2, k2, j2 = CP.transport_wires(order, var, specs, faces, skips,
                                            None)
    vmap = var["vertex_map"]
    pos2 = {int(v): i for i, v in enumerate(o2)}
    for k, v in enumerate(order):
        j = vmap.get(v)
        if j is None:
            continue
        np.testing.assert_array_equal(f2[pos2[j]], faces[k])
    n_carry = 0
    for v in var["alt_carry"]:
        if int(v) not in pos2:
            continue
        n_carry += 1
        rows = f2[pos2[int(v)]]
        for f in range(mf):
            for s in QUANT_SLOTS:
                np.testing.assert_array_equal(rows[f, s], q_row)
            assert rows[f, 2, 0] == -1
            envmod.check_face_quant_rows(rows[f], where=f"carry {v} face {f}")
        assert int(k2[pos2[int(v)]].max()) == 0
    assert n_carry > 0


# ---------------------------------------------------------------------------
# 16. ONE JAXPR FOR BOTH PATHS (ticket dsnn-dfw.24)
# ---------------------------------------------------------------------------
# The environment numbers vertices and face keys on ``config.jaxpr``, the
# INLINED trace of the target. The measurement used to hand ``jacve`` the
# function again and let it trace a fresh one inside ``jax.jit(...).lower()``.
# A fresh trace of the same function is not the same equation list: on the
# two-copy window arm the env's jaxpr has 90 equations and the measurement's
# 72, the difference being every ``convert_element_type`` of the six scalar
# constants, so the plan's last 18 vertices addressed nothing and the face
# keys addressed the wrong edges. The unapplied-face guard caught it as 26
# refused vertices per plan (job 66114). ``jacve(jaxpr=..., consts=...)`` is
# the fix: the measurement walks the jaxpr the env numbered.

def _env_for_example(example, extra=()):
    """A landscape_map env for any registered target.

    TransformerLM reads its token table from the wikitext cache even under
    ``--dataset none``; without that cache the builder goes to the network,
    which a test node does not have. That is an environment condition, not a
    property of the code under test, so it SKIPS with the reason named --
    export ``DSNN_WIKITEXT_DIR`` and it runs.
    """
    import urllib.error

    import alphagrad.approx.tools.landscape_map as lm
    argv = ["--example", example, "--dataset", "none", "--num-eval-samples",
            "1", "--num-data-points", "1", "--reps-per-point", "1",
            "--out-dir", "/tmp/carry_test"] + list(extra)
    try:
        env, _samples, _cj = lm.build_env(lm.make_argparser().parse_args(argv))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        pytest.skip(f"{example}: its data is not on this node ({exc}); "
                    f"export DSNN_WIKITEXT_DIR / DSNN_MNIST_DIR to run it")
    return env


def _jaxpr_the_measurement_walks(env, order):
    """The jaxpr the ENVIRONMENT's own measurement hands the elimination.

    Driven through ``landscape_map.measure``, which IS ``env._callback``, so
    this asks what the measured program is built on, not what a test could
    build. The elimination itself is stubbed out and raises at once: the
    question is which jaxpr it was given, and a refused plan is a path the
    callback already handles.
    """
    import alphagrad.approx.tools.landscape_map as lm
    import graphax.core as gxcore

    seen = {}

    def _stub(jaxpr, *a, **kw):
        seen.setdefault("jaxpr", jaxpr)
        raise ValueError("stubbed elimination (test): jaxpr recorded")

    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "n_faces_approx": 0, "n_slot_rows": 0, "total_live_faces": 0,
            "per_vertex_faces": [], "wires": [], "op": "identity",
            "budget": "identity"}
    orig = gxcore.vertex_elimination_jaxpr
    gxcore.vertex_elimination_jaxpr = _stub
    try:
        try:
            lm.measure(env, env.eval_args_samples, list(order), plan)
        except ValueError:
            pass
    finally:
        gxcore.vertex_elimination_jaxpr = orig
    return seen.get("jaxpr")


def _assert_one_jaxpr(env, name):
    order = [int(v) for v in sorted(env.valid_vertices, reverse=True)]
    walked = _jaxpr_the_measurement_walks(env, order)
    assert walked is not None, f"{name}: the measurement never reached jacve"
    assert walked is env.config.jaxpr, (
        f"{name}: the measurement eliminates a jaxpr of "
        f"{len(walked.eqns)} equations and the environment numbered one of "
        f"{len(env.config.jaxpr.eqns)}. The order and the face keys belong to "
        f"the environment's jaxpr; on another one they address other edges.")
    assert ([str(e.primitive) for e in walked.eqns]
            == [str(e.primitive) for e in env.config.jaxpr.eqns])


@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl", "window2"])
def test_the_eliminator_walks_the_jaxpr_the_env_numbers_snn(rule):
    _lm, _CP, env = _env_for(rule)
    _assert_one_jaxpr(env, f"RSNN_SHD/{rule}")


@pytest.mark.parametrize("example,extra", [
    ("NeuralNetwork", ()),
    ("TransformerLM", ("--hidden-dim", "16", "--vocab-size", "16",
                       "--num-layers", "1")),
])
def test_the_eliminator_walks_the_jaxpr_the_env_numbers(example, extra):
    _assert_one_jaxpr(_env_for_example(example, extra), example)


def test_the_elimination_walks_every_vertex_of_the_window_arms_order():
    """THE REGRESSION ITSELF. On the measurement's own trace the last 18
    vertices of the window arm's order were not in that graph at all, and the
    elimination silently walked 71 of 89."""
    import graphax.core as gxcore
    from graphax import jacve

    _lm, _CP, env = _env_for("window2")
    order = [int(v) for v in sorted(env.valid_vertices, reverse=True)]
    walked = []
    orig = gxcore._eliminate_vertex

    def spy(vertex, *a, **kw):
        walked.append(int(vertex))
        return orig(vertex, *a, **kw)

    fn = jacve(env.config.target_fun, list(order),
               argnums=env.config.argnums, has_aux=env.config.has_aux,
               sparse_representation=env.config.sparse,
               jaxpr=env.config.jaxpr, consts=list(env.consts))
    gxcore._eliminate_vertex = spy
    try:
        jax.eval_shape(fn, *env.args)
    finally:
        gxcore._eliminate_vertex = orig
    assert walked == order


# ---------------------------------------------------------------------------
# 17. THE DIAG PLAN IS THE E-PROP RECURSION (owner ruling 2026-09-18)
# ---------------------------------------------------------------------------

_EPROP_HAND = r'''
import os, json
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np, jax, jax.numpy as jnp
from alphagrad.approx.common import examples as ex, rsnn_shd as R
from graphax.examples.neuromorphic import RSNN_SURROGATE_SCALE, rsnn_cell

assert jax.config.jax_enable_x64
H, NIN, NOUT, T = 6, 700, 20, 9
R.RSNN_HIDDEN = H
key = jax.random.split(jax.random.PRNGKey(5), 3)
seq = jax.random.bernoulli(key[0], 0.2, (T, NIN)).astype(jnp.float64)
y = jax.nn.one_hot(3, NOUT).astype(jnp.float64)
W, V, Wo = R.rsnn_weights(key[1])
weights = (W, V, Wo)
c = R._consts()
a_syn, a_mem, a_out, rho = R.decay_constants()
fn = ex.get_fn("RSNN_SHD")


def states(t):
    st = R.zero_state()
    out = [st]
    for u in range(t):
        st = rsnn_cell(seq[u], *st, W, V, Wo, *c)
        out.append(st)
    return out


def hand_traces(t):
    """THE E-PROP RECURSION, written out here: one eligibility trace per
    synapse, the block diagonal of the state-to-state Jacobian at every step,
    plus the two readout filters. Independent of `common.rsnn_shd`."""
    vd = jnp.diag(V)[:, None]
    zW, zV = jnp.zeros((H, NIN)), jnp.zeros((H, H))
    trW, trV = (zW, zW, zW, zW), (zV, zV, zV, zV)
    fW, fV, g = zW, zV, jnp.zeros(H)
    st = R.zero_state()
    for u in range(t):
        S_prev = st[0]
        a_prev = st[3]
        st = rsnn_cell(seq[u], *st, W, V, Wo, *c)
        psi = (1.0 / (RSNN_SURROGATE_SCALE
                      * jnp.abs(st[2] - (1.0 + 1.0 * a_prev)) + 1.0) ** 2)
        psi = psi[:, None]

        def adv(tr, direct):
            eS, eI, eU, ea = tr
            nI = a_syn * eI + vd * eS + direct
            nU = a_mem * eU + (1.0 - a_mem) * nI - 1.0 * eS
            nS = psi * (nU - 1.0 * ea)
            return (nS, nI, nU, rho * ea + nS)

        trW = adv(trW, jnp.broadcast_to(seq[u][None, :], (H, NIN)))
        trV = adv(trV, jnp.broadcast_to(S_prev[None, :], (H, H)))
        fW = a_out * fW + (1.0 - a_out) * trW[0]
        fV = a_out * fV + (1.0 - a_out) * trV[0]
        g = a_out * g + (1.0 - a_out) * st[0]
    return trW, trV, fW, fV, g


def hand_gradient(t):
    """(eligibility trace) x (learning signal), the e-prop gradient."""
    trW, trV, fW, fV, g = hand_traces(t)
    st = states(t)[t]

    def step(state, ws):
        nxt = rsnn_cell(seq[t], *state, *ws, *c)
        return jnp.sum(-y * jax.nn.log_softmax(nxt[4]))

    lam, direct = jax.grad(step, argnums=(0, 1))(tuple(st), weights)
    lS, lI, lU, la, lUo = lam
    gW = (lS[:, None] * trW[0] + lI[:, None] * trW[1] + lU[:, None] * trW[2]
          + la[:, None] * trW[3] + (lUo @ Wo)[:, None] * fW)
    gV = (lS[:, None] * trV[0] + lI[:, None] * trV[1] + lU[:, None] * trV[2]
          + la[:, None] * trV[3] + (lUo @ Wo)[:, None] * fV)
    gWo = lUo[:, None] * g[None, :]
    return (direct[0] + gW, direct[1] + gV, direct[2] + gWo)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n else float(np.linalg.norm(a - b))


out = {}
for t in (3, 7):
    st = tuple(jax.lax.stop_gradient(x)
               for x in R.prefix_state(seq, t, weights)(*weights))
    head = (seq[t], y) + st + weights + c
    given = R.carry_under_plan(seq, t, weights, "diag")
    plan_g = jax.grad(fn, argnums=(7, 8, 9))(*(head + tuple(given)))
    hand_g = hand_gradient(t)
    out[f"diag_vs_hand_eprop_t{t}"] = max(
        rel(a, b) for a, b in zip(plan_g, hand_g))
    out[f"carry_bytes_t{t}"] = int(sum(np.asarray(x).nbytes
                                       for x in given[3:]))
print("RESULT " + json.dumps(out))
'''


def _run_float64(src):
    """One float64 subprocess, the pattern section 7 uses."""
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(src)],
                         capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-4000:], out.stderr[-4000:])
    return json.loads(lines[-1][len("RESULT "):])


def test_the_diag_plan_is_the_eprop_recursion_in_float64():
    """THE PLAN-PRODUCED CARRY IS E-PROP. A Diag on the carried-Jacobian face
    makes the container the eligibility traces of Zenke and Neftci, and the
    gradient the measured program then computes is the e-prop gradient --
    eligibility trace times learning signal -- and not a per-step
    diagonalisation of an exact carry."""
    out = _run_float64(_EPROP_HAND)
    for t in (3, 7):
        assert out[f"diag_vs_hand_eprop_t{t}"] < 1e-12, out
        assert out[f"carry_bytes_t{t}"] > 0
