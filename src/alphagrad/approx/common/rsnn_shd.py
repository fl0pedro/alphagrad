"""The RECURRENT SHD step target, and the three temporal rules.

OWNER RULING, 2026-09-16. The elimination graph is always ONE recurrent step.
Its inputs are the weights, the carried state ``s_(t-1)`` and the input frame
``x_t``; its outputs are ``s_t`` and the step loss. Temporal credit does not
enter as more step copies. It enters as EDGES WITH GIVEN VALUES, computed
numerically outside the graph over the whole, untouched recording.

    tbptt   no temporal edge. ``s_(t-1)`` is a constant. Truncated, spatial
            credit only. The baseline.
    bptt    the FUTURE feeds in. A given edge from ``s_t`` to the future loss
            carries the adjoint ``lambda_(t+1) = dL_(>t)/ds_t``, produced by a
            detached backward pass over the suffix of the recording. The
            graph's gradient is the exact contribution step ``t`` makes to the
            full backpropagation-through-time gradient.
    rtrl    the PAST feeds in. A given edge ``W -> s_(t-1)`` carries
            ``J_(t-1) = ds_(t-1)/dW`` from a detached pass over the prefix.
            Eliminating ``s_(t-1)`` multiplies ``J_(t-1)`` through
            ``A_t = ds_t/ds_(t-1)``. The gradient is exactly ``dL_t/dW``
            through the whole prefix.

Both ``bptt`` and ``rtrl`` are EXACT per-step rules. Neither approximates
anything. What the policy learns is how to treat the FACE where the given
quantity meets the step, and the e-prop approximation of Zenke and Neftci
(arXiv 2010.11931) lives on exactly one of those faces.

THE MODEL. See ``graphax.examples.neuromorphic.RSNN_SHD``. A recurrently
connected adaptive-threshold LIF hidden layer and a non-spiking leaky readout,
with the architecture, the surrogate and the weight init of Zenke's public
SpyTorch tutorial 4. ``V``, the recurrent weight matrix, is among the
differentiated weights, so ``A_t`` has off-diagonal terms and the
block-diagonal drop is a real approximation.

THE TIME STEP is the SHD loader's 10 ms bin, unchanged. Zenke runs 1 ms, so
the decay constants here are his time constants doubled, which is the setting
the learning gate of 2026-09-16 selected (job 65986: 77.8 percent test accuracy
on SHD after thirty epochs of full backpropagation through time, against a 5
percent chance level).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common.datasets import (
    SHD_CHANNELS,
    SHD_CLASSES,
    SHD_TIME_BINS,
    shd_sample,
    shd_split_size,
)

#: The registered name of the one-step recurrent target.
RSNN_TARGET = "RSNN_SHD"

#: The targets on which ``--temporal-rule`` is defined.
TEMPORAL_RULE_TARGETS: frozenset[str] = frozenset({RSNN_TARGET})

#: The three rules. ``tbptt`` is the baseline and the default.
TEMPORAL_RULES: tuple[str, ...] = ("tbptt", "bptt", "rtrl")

#: Hidden layer size. 128, matching the older SHD targets, so the carried
#: Jacobian is the same order of magnitude as theirs (226 MB against 218 MB).
#: Zenke's tutorial uses 200.
RSNN_HIDDEN = 128

#: The weight argument slots. ``V`` is one of them.
RSNN_ARGNUMS: tuple[int, int, int] = (7, 8, 9)

#: THE CONSTANTS, at the loader's 10 ms bin, chosen by the learning gate.
#:
#: ``tau_syn 10 ms``, ``tau_mem 20 ms``, ``tau_out 20 ms`` are Zenke's
#: ``tau_syn 5 ms`` / ``tau_mem 10 ms`` doubled for a step ten times as long;
#: ``tau_a 200 ms`` and ``thresh 1.0`` are Bellec et al.'s adaptive LIF;
#: ``beta_a 1.0`` is the adaptation strength, and the gate shows it is worth
#: 14 accuracy points (0.778 with it, 0.634 without).
#:
#: MEASURED, job 65986, thirty epochs of full backpropagation through time on
#: the real SHD splits, test accuracy against a 0.05 chance level:
#:
#:   tau 5/10 ms, scale 0.2 (Zenke's literal constants)   0.729
#:   tau 10/20 ms, scale 0.2 (THIS SETTING)               0.778
#:   tau 20/50 ms, scale 0.2                              0.793
#:   tau 10/20 ms, scale 1.0                              0.807
#:   tau 20/50 ms, scale 1.0                              0.819
#:   tau 10/20 ms, scale 0.2, beta_a 0 (no adaptation)    0.634
#:
#: This row is the one that keeps Zenke's own init scale and changes only what
#: the ten times longer step forces. Change these six numbers to move to
#: another row; nothing else reads them.
DT_MS = 10.0
TAU_SYN_MS = 10.0
TAU_MEM_MS = 20.0
TAU_OUT_MS = 20.0
TAU_A_MS = 200.0
BETA_A = 1.0
THRESH = 1.0

#: Zenke's ``weight_scale`` (SpyTorch tutorial 4): ``std = scale / sqrt(fan_in)``.
WEIGHT_SCALE = 0.2


def decay_constants() -> tuple[float, float, float, float]:
    """``(a_syn, a_mem, a_out, rho)`` at the loader's 10 ms step."""
    return (float(np.exp(-DT_MS / TAU_SYN_MS)),
            float(np.exp(-DT_MS / TAU_MEM_MS)),
            float(np.exp(-DT_MS / TAU_OUT_MS)),
            float(np.exp(-DT_MS / TAU_A_MS)))


def is_rsnn(example: str | None) -> bool:
    return bool(example) and str(example) == RSNN_TARGET


def resolve_temporal_rule(example: str | None, rule, *,
                          flag: str = "--temporal-rule") -> str:
    """The temporal rule for ``example``, or raise.

    ``None`` resolves to ``tbptt`` on the recurrent target, which is the
    baseline and the graph a run gets when it says nothing. A VALUE on any
    other target is a hard error: the rule names how the state carried between
    steps enters the gradient, and a target with no carried state has nothing
    for it to name.
    """
    if not is_rsnn(example):
        if rule is not None:
            raise ValueError(
                f"{flag} {rule} was passed with --example {example}, which has "
                f"NO time steps. The temporal rule says how the state carried "
                f"between steps enters the gradient; it is defined only on "
                f"{sorted(TEMPORAL_RULE_TARGETS)}. Drop the flag or change "
                f"the target.")
        return None
    if rule is None:
        return "tbptt"
    r = str(rule)
    if r not in TEMPORAL_RULES:
        raise ValueError(f"{flag} {r!r} is not one of {list(TEMPORAL_RULES)}")
    return r


def rsnn_weights(key):
    """``(W, V, Wo)`` at Zenke's ``std = 0.2 / sqrt(fan_in)`` normal init."""
    h = RSNN_HIDDEN
    k = jax.random.split(key, 3)
    W = jax.random.normal(k[0], (h, SHD_CHANNELS)) * (
        WEIGHT_SCALE / math.sqrt(SHD_CHANNELS))
    V = jax.random.normal(k[1], (h, h)) * (WEIGHT_SCALE / math.sqrt(h))
    Wo = jax.random.normal(k[2], (SHD_CLASSES, h)) * (
        WEIGHT_SCALE / math.sqrt(h))
    return W, V, Wo


def zero_state():
    """The state a recording starts from: five zero vectors."""
    h = RSNN_HIDDEN
    return (jnp.zeros((h,)), jnp.zeros((h,)), jnp.zeros((h,)),
            jnp.zeros((h,)), jnp.zeros((SHD_CLASSES,)))


def _cell():
    from graphax.examples.neuromorphic import rsnn_cell
    return rsnn_cell


def _consts():
    a_syn, a_mem, a_out, rho = decay_constants()
    return (jnp.array(a_syn), jnp.array(a_mem), jnp.array(a_out),
            jnp.array(rho), jnp.array(BETA_A), jnp.array(THRESH))


def _step_loss(Uo, y):
    return jnp.sum(-y * jax.nn.log_softmax(Uo))


def prefix_state(seq, t: int, weights, state0=None):
    """``s_(t-1)``: the state after steps ``0 .. t-1`` of ``seq``.

    Written as a function of the weights so the same code gives the VALUE and,
    under ``jax.jacrev``, the carried Jacobian. One ``lax.scan``, so the traced
    graph does not grow with ``t``.
    """
    cell = _cell()
    c = _consts()
    s0 = zero_state() if state0 is None else state0

    def run(W, V, Wo):
        def body(st, x):
            return cell(x, *st, W, V, Wo, *c), None
        st, _ = jax.lax.scan(body, s0, seq[:int(t)])
        return st
    return run


def suffix_adjoint(seq, t: int, weights, state_t):
    """``lambda_(t+1) = d (sum_(u>t) L_u) / d s_t``, by reverse mode.

    The suffix runs steps ``t+1 .. T-1`` from ``s_t`` and sums their losses.
    The gradient with respect to ``s_t`` is the adjoint the ``bptt`` rule
    attaches. An empty suffix gives five zero vectors, which is the right
    answer at the last step.
    """
    cell = _cell()
    c = _consts()
    W, V, Wo = weights
    T = int(seq.shape[0])
    y_free = seq  # unused, kept for the closure's clarity

    def tail_loss(st, y):
        def body(carry, x):
            st = cell(x, *carry, W, V, Wo, *c)
            return st, _step_loss(st[4], y)
        st_end, losses = jax.lax.scan(body, st, seq[int(t) + 1:T])
        return jnp.sum(losses)
    return tail_loss


def step_target_loss(seq, y, t: int, weights, state_prev):
    """``L_t``: what the traced step returns under ``tbptt``."""
    cell = _cell()
    c = _consts()
    W, V, Wo = weights
    nxt = cell(seq[int(t)], *state_prev, W, V, Wo, *c)
    return _step_loss(nxt[4], y), nxt


def carried_jacobians(seq, t: int, weights, *, check_zeros: bool = True):
    """The RTRL attachment tuple: three reference weights, then ELEVEN blocks.

    Block ``(s, w)`` is ``d s_(t-1)^s / d W_w``, taken by reverse-mode
    differentiation of the prefix. That is the same number the RTRL recursion
    ``G_t = A_t G_(t-1) + F_t`` accumulates over the prefix, and reverse mode
    gets it in ``4 * hidden + classes`` cotangent sweeps where the recursion
    needs one tangent per weight entry.

    ``check_zeros`` asserts the four blocks that CANNOT be non-zero (the
    readout weight feeds nothing back) really are zero, so the eleven carried
    blocks are the whole influence matrix and not a silent truncation.
    """
    from graphax.examples.neuromorphic import (
        RSNN_CARRY_BLOCKS, RSNN_STATE_NAMES, RSNN_WEIGHT_NAMES,
        RSNN_ZERO_BLOCKS)

    run = prefix_state(seq, t, weights)
    jac = jax.jacrev(run, argnums=(0, 1, 2))(*weights)
    if check_zeros:
        for s_i, w_i in RSNN_ZERO_BLOCKS:
            nz = float(jnp.max(jnp.abs(jac[s_i][w_i])))
            if nz != 0.0:
                raise ValueError(
                    f"carried block ({RSNN_STATE_NAMES[s_i]}, "
                    f"{RSNN_WEIGHT_NAMES[w_i]}) is not zero (max |.| = {nz}). "
                    f"The readout weight feeds nothing back, so that block "
                    f"cannot carry signal; a non-zero here means the model "
                    f"changed and RSNN_CARRY_BLOCKS no longer lists every "
                    f"block.")
    sg = jax.lax.stop_gradient
    return (tuple(sg(W) for W in weights)
            + tuple(sg(jac[s_i][w_i]) for s_i, w_i in RSNN_CARRY_BLOCKS))


def eprop_traces(seq, t: int, weights):
    """THE E-PROP INFLUENCE MATRIX: the same eleven blocks, block diagonal.

    Zenke and Neftci (arXiv 2010.11931) approximate real-time recurrent
    learning by replacing the state-to-state Jacobian ``A_t`` with its BLOCK
    DIAGONAL, one block per neuron: a neuron's own synaptic current, membrane
    and adaptation survive and the coupling through OTHER neurons' spikes is
    dropped. Running ``G_t = blockdiag(A_t) G_(t-1) + F_t`` over the prefix
    leaves ONE ELIGIBILITY TRACE PER SYNAPSE. For
    :func:`graphax.examples.neuromorphic.rsnn_cell` the traces are, per
    postsynaptic unit ``j`` and presynaptic index ``i``:

        psi[j]     = 1 / (scale * |U[j] - thresh - beta_a * a_prev[j]| + 1)^2
        eI[j,i]   <- a_syn * eI[j,i] + V[j,j] * eS[j,i] + direct[j,i]
        eU[j,i]   <- a_mem * eU[j,i] + (1 - a_mem) * eI[j,i] - thresh * eS[j,i]
        eS[j,i]   <- psi[j] * (eU[j,i] - beta_a * ea[j,i])
        ea[j,i]   <- rho * ea[j,i] + eS[j,i]

    with ``direct = x[i]`` for the input weight and ``direct = S_prev[i]`` for
    the recurrent weight, and every trace zero at the start of the recording.
    THE ONE TERM THE EXACT RECURSION HAS AND THIS DOES NOT is
    ``V[j,k] * eS[k,i]`` for ``k != j``: the recurrent coupling. Set ``V`` to a
    diagonal matrix and the two agree exactly; that is the test.

    The readout blocks stay EXACT and are a plain leaky filter of the hidden
    traces, because the readout feeds nothing back:

        eUo[m,j,i] <- a_out * eUo[m,j,i] + (1 - a_out) * Wo[m,j] * eS[j,i]

    The returned tuple has the SAME SHAPE as :func:`carried_jacobians`, so the
    two are interchangeable as the ``rtrl`` given values and the difference
    between the gradients they produce IS the e-prop approximation.

    THE STORE IS THE POINT. The exact influence matrix is 226 MB of dense
    blocks; these traces are ``4 * (h * n_in + h * h)`` numbers, 1.7 MB, plus
    whatever the readout filter is expanded to for the drop-in shape.
    """
    from graphax.examples.neuromorphic import (
        RSNN_CARRY_BLOCKS, RSNN_SURROGATE_SCALE)

    cell = _cell()
    a_syn, a_mem, a_out, rho = decay_constants()
    c = _consts()
    W, V, Wo = weights
    h, n_in, n_out = RSNN_HIDDEN, SHD_CHANNELS, SHD_CLASSES
    vd = jnp.diag(V)[:, None]

    st = zero_state()
    zW = jnp.zeros((h, n_in))
    zV = jnp.zeros((h, h))
    tr = {"W": [zW, zW, zW, zW], "V": [zV, zV, zV, zV]}   # eS, eI, eU, ea
    oW = jnp.zeros((n_out, h, n_in))
    oV = jnp.zeros((n_out, h, h))
    oWo = jnp.zeros((n_out, n_out, h))

    for u in range(int(t)):
        S_prev, I_prev, U_prev, a_prev, Uo_prev = st
        x = seq[u]
        nxt = cell(x, *st, W, V, Wo, *c)
        S, I, U, a, Uo = nxt
        psi = 1.0 / (RSNN_SURROGATE_SCALE
                     * jnp.abs(U - (THRESH + BETA_A * a_prev)) + 1.0) ** 2
        psi = psi[:, None]
        for name, direct in (("W", jnp.broadcast_to(x[None, :], (h, n_in))),
                             ("V", jnp.broadcast_to(S_prev[None, :], (h, h)))):
            eS, eI, eU, ea = tr[name]
            nI = a_syn * eI + vd * eS + direct
            nU = a_mem * eU + (1.0 - a_mem) * nI - THRESH * eS
            nS = psi * (nU - BETA_A * ea)
            na = rho * ea + nS
            tr[name] = [nS, nI, nU, na]
        # the readout, exact, filtering the hidden traces
        oW = a_out * oW + (1.0 - a_out) * (Wo[:, :, None] * tr["W"][0][None])
        oV = a_out * oV + (1.0 - a_out) * (Wo[:, :, None] * tr["V"][0][None])
        # d Uo[m] / d Wo[m, k] = (1 - a_out) * S[k], filtered
        oWo = a_out * oWo + (1.0 - a_out) * (
            jnp.eye(n_out)[:, :, None] * S[None, None, :])
        st = nxt

    def expand(mat):
        """``(h, n)`` trace -> the full ``(h, h, n)`` block, diagonal in (j, j)."""
        return jnp.eye(mat.shape[0])[:, :, None] * mat[:, None, :]

    blocks = {
        (0, 0): expand(tr["W"][0]), (0, 1): expand(tr["V"][0]),
        (1, 0): expand(tr["W"][1]), (1, 1): expand(tr["V"][1]),
        (2, 0): expand(tr["W"][2]), (2, 1): expand(tr["V"][2]),
        (3, 0): expand(tr["W"][3]), (3, 1): expand(tr["V"][3]),
        (4, 0): oW, (4, 1): oV, (4, 2): oWo,
    }
    sg = jax.lax.stop_gradient
    return (tuple(sg(x) for x in weights)
            + tuple(sg(blocks[b]) for b in RSNN_CARRY_BLOCKS))


def future_adjoints(seq, y, t: int, weights, state_prev):
    """The BPTT attachment: five adjoints ``lambda_(t+1) = dL_(>t)/ds_t``."""
    _, state_t = step_target_loss(seq, y, t, weights, state_prev)
    tail = suffix_adjoint(seq, t, weights, state_t)
    lam = jax.grad(lambda st: tail(st, y))(tuple(state_t))
    sg = jax.lax.stop_gradient
    return tuple(sg(x) for x in lam)


#: The last sampled step position, per process, for the trainer's `[cfg]` line
#: and the plan record. A module-level record because the argument builder is
#: the only place that knows it and the record is written elsewhere.
_LAST_STEP_POSITION: dict = {}


def last_step_position() -> dict:
    """``{"t": int, "T": int, "recording": int}`` of the last built tuple."""
    return dict(_LAST_STEP_POSITION)


def sample_step_position(key, T: int) -> int:
    """A uniform step position with a non-empty prefix.

    ``t`` is drawn from ``1 .. T-1``. ``t = 0`` is excluded because its carried
    state is the zero state and every given quantity is zero there, which makes
    the three rules identical and hides what the run is measuring.
    """
    return int(jax.random.randint(key, (), 1, int(T)))


def rsnn_args(key=None, *, dataset: str | None = None,
              dataset_size: int | None = -1, temporal_rule: str | None = None,
              step_position: int | None = None):
    """The argument tuple of ``graphax.examples.neuromorphic.RSNN_SHD``.

    Slots: ``x_t`` 0, ``y`` 1, the five carried state components 2 to 6, the
    three weights 7 to 9, the six constants 10 to 15, then the rule's given
    values.

    ``step_position`` pins ``t``; ``None`` draws it uniformly from the key
    (see :func:`sample_step_position`). The recording is drawn from the same
    key, so two processes given the same seed build the same tuple.
    """
    from graphax.examples.neuromorphic import RSNN_GIVEN_LENGTHS

    rule = "tbptt" if temporal_rule is None else str(temporal_rule)
    if rule not in TEMPORAL_RULES:
        raise ValueError(f"temporal rule {rule!r} is not one of "
                         f"{list(TEMPORAL_RULES)}")
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 3)
    seq, y, rec = _draw_recording(k[0], dataset, dataset_size)
    weights = rsnn_weights(k[1])
    T = int(seq.shape[0])
    t = (sample_step_position(k[2], T) if step_position is None
         else int(step_position))
    if not 0 <= t < T:
        raise ValueError(f"step position {t} is outside 0 .. {T - 1}")
    _LAST_STEP_POSITION.clear()
    _LAST_STEP_POSITION.update({"t": t, "T": T, "recording": rec,
                                "rule": rule})

    sg = jax.lax.stop_gradient
    state_prev = tuple(sg(x) for x in prefix_state(seq, t, weights)(*weights))
    head = (seq[t], y) + state_prev + weights + _consts()

    if rule == "tbptt":
        given = ()
    elif rule == "rtrl":
        given = carried_jacobians(seq, t, weights)
    else:
        given = future_adjoints(seq, y, t, weights, state_prev)
    want = {v: n for n, v in RSNN_GIVEN_LENGTHS.items()}[rule]
    if len(given) != want:
        raise ValueError(
            f"rule {rule} must pass {want} given values, built {len(given)}")
    return head + tuple(given)


def _draw_recording(key, dataset: str | None, dataset_size: int | None):
    """``(seq [T, 700] float32, y [20] one-hot, index)`` for ONE recording."""
    if dataset is not None and dataset not in ("shd", "none"):
        raise ValueError(
            f"--dataset {dataset} cannot feed the recurrent SHD target: a "
            f"spike frame is ({SHD_CHANNELS},) and nothing in {dataset} has "
            f"that shape. Use --dataset shd for the real recordings, or "
            f"--dataset none for the synthetic Poisson train.")
    if dataset == "shd":
        n = shd_split_size(dataset_size)
        if n == 0:
            raise ValueError(
                "the SHD subset is empty -- --dataset-size cut every sample")
        idx = int(jax.random.randint(key, (), 0, n))
        seq, tgt = shd_sample(idx)
        return jnp.asarray(seq), jnp.asarray(tgt), idx
    k = jax.random.split(key, 2)
    seq = jax.random.bernoulli(
        k[0], 0.1, (SHD_TIME_BINS, SHD_CHANNELS)).astype(jnp.float32)
    tgt = jax.nn.one_hot(
        jax.random.randint(k[1], (), 0, SHD_CLASSES), SHD_CLASSES
    ).astype(jnp.float32)
    return seq, tgt, -1


def sequence_loss(seq, y, weights):
    """``sum_t L_t`` over the whole recording. The object the rules decompose."""
    cell = _cell()
    c = _consts()
    W, V, Wo = weights

    def body(st, x):
        st = cell(x, *st, W, V, Wo, *c)
        return st, _step_loss(st[4], y)
    _, losses = jax.lax.scan(body, zero_state(), seq)
    return jnp.sum(losses)
