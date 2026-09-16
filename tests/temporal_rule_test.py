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


def _args(rule, **kw):
    return ex.get_args("RSNN_SHD", jax.random.PRNGKey(1), dataset=None,
                       temporal_rule=rule, step_position=T_PIN, **kw)


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
    with pytest.raises(ValueError, match="d state / d W is"):
        attach_rsnn_past(st, W, bad)
    with pytest.raises(ValueError, match="does not match the state"):
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
        R.resolve_temporal_rule("RSNN_SHD", "eprop")


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
