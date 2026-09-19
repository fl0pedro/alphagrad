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
percent chance level; job 65992 gives 77.8 to 80.7 percent across the init
scales, and this module takes the one that also FIRES at its initial
weights).
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

#: The registered name of the TWO-COPY WINDOW target, the fourth SNN arm.
RSNN_W2_TARGET = "RSNN_SHD_W2"

#: The targets on which ``--temporal-rule`` is defined.
TEMPORAL_RULE_TARGETS: frozenset[str] = frozenset({RSNN_TARGET,
                                                   RSNN_W2_TARGET})

#: The four rules, which are the four SNN arms of the thesis matrix (owner
#: ruling 2026-09-16).
#:
#: ``tbptt``    no temporal edge. The truncated baseline, and the default.
#: ``bptt``     one step body plus the given FUTURE adjoint over the suffix.
#: ``rtrl``     one step body plus the given PAST Jacobian over the prefix.
#: ``window2``  TWO step copies joined by the temporal edge and NO given edge.
#:              The one arm where the policy picks the direction of the
#:              temporal credit itself, because the temporal edge is an
#:              ordinary edge of the graph and the order across it is free.
TEMPORAL_RULES: tuple[str, ...] = ("tbptt", "bptt", "rtrl", "window2")

#: The rules that attach a GIVEN temporal edge, and therefore the rules on
#: which the carry container is a question at all.
GIVEN_EDGE_RULES: tuple[str, ...] = ("bptt", "rtrl")

#: The dtype the ``Quant`` class stores a carried value in. One name, read
#: from graphax so the producer and the attachment cannot disagree.
CARRY_QUANT_DTYPE = jnp.bfloat16

#: Hidden layer size. 128, matching the older SHD targets, so the carried
#: Jacobian is the same order of magnitude as theirs (226 MB against 218 MB).
#: Zenke's tutorial uses 200.
RSNN_HIDDEN = 128

#: The weight argument slots. ``V`` is one of them.
RSNN_ARGNUMS: tuple[int, int, int] = (7, 8, 9)

#: The weight argument slots of the TWO-COPY WINDOW target. It carries a
#: second input frame ahead of the label, so every slot after the frames moves
#: one along.
RSNN_W2_ARGNUMS: tuple[int, int, int] = (8, 9, 10)

#: THE CONSTANTS, at the loader's 10 ms bin, chosen by the learning gate.
#:
#: ``tau_syn 10 ms``, ``tau_mem 20 ms``, ``tau_out 20 ms`` are Zenke's
#: ``tau_syn 5 ms`` / ``tau_mem 10 ms`` doubled for a step ten times as long;
#: ``tau_a 200 ms`` and ``thresh 1.0`` are Bellec et al.'s adaptive LIF;
#: ``beta_a 1.0`` is the adaptation strength, and the gate shows it is worth
#: 14 accuracy points (0.778 with it, 0.634 without).
#:
#: MEASURED, jobs 65986 and 65992, thirty epochs of full backpropagation
#: through time on the real SHD splits. ``acc`` is test accuracy against a 0.05
#: chance level; ``rate`` is the mean hidden spike rate per unit per step at
#: the INITIAL weights (job 65991).
#:
#:   tau ms   scale   acc     rate at init
#:   5/10     0.2     0.729   0.00000
#:   10/20    0.2     0.778   0.00000
#:   20/50    0.2     0.793   0.00000
#:   10/20    1.0     0.807   0.00492
#:   20/50    1.0     0.819   0.00578
#:   10/20    2.0     0.795   0.01352   <- THIS SETTING
#:   10/20    4.0     0.689   0.03391
#:   10/20    8.0     0.505   0.06...
#:   10/20    0.2, beta_a 0 (no adaptation)  0.634
#:
#: THE INIT SCALE IS NOT ZENKE'S 0.2, AND THE REASON IS THE TARGET, NOT THE
#: TRAINING. At 0.2 the network is SILENT at its initial weights: the membrane
#: never reaches the threshold, so no hidden unit spikes, ``V S`` is zero, and
#: every carried Jacobian block with respect to ``V`` is EXACTLY ZERO. A PPO
#: target is built at the initial weights, so at 0.2 the recurrent coupling the
#: whole design is about would be invisible. 2.0 fires 1.35 percent of units
#: per step (about 173 hidden spikes per recording), which is a working sparse
#: rate, and costs 1.2 accuracy points against the best row. Change these seven
#: numbers to move to another row; nothing else reads them.
DT_MS = 10.0
TAU_SYN_MS = 10.0
TAU_MEM_MS = 20.0
TAU_OUT_MS = 20.0
TAU_A_MS = 200.0
BETA_A = 1.0
THRESH = 1.0

#: The init scale, in Zenke's form ``std = scale / sqrt(fan_in)`` (SpyTorch
#: tutorial 4, where it is 0.2). See the table above for why it is not 0.2.
WEIGHT_SCALE = 2.0


def decay_constants() -> tuple[float, float, float, float]:
    """``(a_syn, a_mem, a_out, rho)`` at the loader's 10 ms step."""
    return (float(np.exp(-DT_MS / TAU_SYN_MS)),
            float(np.exp(-DT_MS / TAU_MEM_MS)),
            float(np.exp(-DT_MS / TAU_OUT_MS)),
            float(np.exp(-DT_MS / TAU_A_MS)))


def is_rsnn(example: str | None) -> bool:
    """Is ``example`` one of the recurrent SHD targets?

    Both of them: the one-step body and the two-copy window. They share the
    model, the recording, the weights and the argument builder, and the
    temporal rule is what tells them apart.
    """
    return bool(example) and str(example) in TEMPORAL_RULE_TARGETS


def resolve_temporal_rule(example: str | None, rule, *,
                          flag: str = "--temporal-rule") -> str:
    """The temporal rule for ``example``, or raise.

    ``None`` resolves to ``tbptt`` on the one-step target, which is the
    baseline and the graph a run gets when it says nothing, and to
    ``window2`` on the two-copy target, which is the only rule that target
    has. A VALUE on any other target is a hard error: the rule names how the
    state carried between steps enters the gradient, and a target with no
    carried state has nothing for it to name.

    ``window2`` on the ONE-STEP target is legal and is how a run asks for the
    window arm; :func:`target_example` then resolves the target it builds.
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
    if is_window2(example):
        if rule is not None and str(rule) != "window2":
            raise ValueError(
                f"{flag} {rule} was passed with --example {example}, which IS "
                f"the two-copy window. That target has no given edge, so the "
                f"only rule it can run is window2.")
        return "window2"
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


@jax.custom_jvp
def narrow_derivative(x):
    """Identity on the value, :data:`CARRY_QUANT_DTYPE` on the derivative.

    THE ``Quant`` CLASS, APPLIED AT EVERY STEP. Put this on the state carried
    between two steps of the prefix or the suffix and the derivative that
    crosses that step boundary is rounded to the narrow dtype and back, while
    the forward value does not move by one bit. Under forward mode it is the
    tangent -- one column of the influence matrix -- that is rounded; under
    reverse mode it is the cotangent -- the adjoint. Either way the statement
    is the same: the temporal derivative this rule carries is held in low
    precision, and the error it accumulates over the recording is what the
    quality channel then prices.

    The rounding is a pair of ``convert_element_type``, which is linear, so
    the rule transposes and ``jax.jacrev`` may be taken through it.
    """
    return x


@narrow_derivative.defjvp
def _narrow_derivative_jvp(primals, tangents):
    x, = primals
    xd, = tangents
    return x, xd.astype(CARRY_QUANT_DTYPE).astype(xd.dtype)


@jax.custom_jvp
def mean_derivative(x):
    """Identity on the value, the axis MEAN on the derivative.

    THE ``Reduce`` CLASS ON AN ADJOINT, APPLIED AT EVERY STEP. Making an axis
    implicit means one value is stored and read as a replication along the
    axis (CONTEXT.md, Implicit axis). The projection onto that replicated
    subspace is the mean, and it is symmetric, so the same rule serves the
    tangent and, transposed, the cotangent. Put this on the state carried
    between two steps of the SUFFIX and the adjoint is projected onto its
    constant subspace at every step, which is exactly a carried value whose
    only axis is implicit.
    """
    return x


@mean_derivative.defjvp
def _mean_derivative_jvp(primals, tangents):
    x, = primals
    xd, = tangents
    return x, jnp.broadcast_to(jnp.mean(xd), xd.shape)


def prefix_state(seq, t, weights, state0=None, *, narrow: bool = False):
    """``s_(t-1)``: the state after steps ``0 .. t-1`` of ``seq``.

    Written as a function of the weights so the same code gives the VALUE and,
    under ``jax.jacrev``, the carried Jacobian. One ``lax.scan``, so the traced
    graph does not grow with ``t``.

    ``narrow`` puts :func:`narrow_derivative` on the state at every step, so
    the derivative this prefix carries is held in the narrow dtype. The
    forward value is unchanged bit for bit.

    ``t`` MAY BE A TRACED VALUE. The scan walks the WHOLE recording and step
    ``u`` writes its result only while ``u < t``; the steps at and after ``t``
    are computed and thrown away. That costs ``T`` cell evaluations instead of
    ``t`` and buys the one property the per-episode step sampler needs: the
    traced shape does not depend on ``t``, so ONE jit and ONE vmap serve every
    step position, and the elimination graph is the same graph at every one of
    them. The selected values are the values the ``seq[:t]`` scan produced,
    entry for entry -- the cell sees the same inputs in the same order, and a
    ``where`` selects, it does not compute.
    """
    cell = _cell()
    c = _consts()
    s0 = zero_state() if state0 is None else state0
    T = int(seq.shape[0])

    def run(W, V, Wo):
        def body(st, inp):
            u, x = inp
            if narrow:
                st = tuple(narrow_derivative(a) for a in st)
            nxt = cell(x, *st, W, V, Wo, *c)
            keep = u < t
            return tuple(jnp.where(keep, a, b) for a, b in zip(nxt, st)), None
        st, _ = jax.lax.scan(body, s0, (jnp.arange(T), seq))
        return st
    return run


def suffix_adjoint(seq, t, weights, state_t, container: str = "exact"):
    """``lambda_(t+1) = d (sum_(u>t) L_u) / d s_t``, by reverse mode.

    The suffix runs steps ``t+1 .. T-1`` from ``s_t`` and sums their losses.
    The gradient with respect to ``s_t`` is the adjoint the ``bptt`` rule
    attaches. An empty suffix gives five zero vectors, which is the right
    answer at the last step.

    ``t`` MAY BE A TRACED VALUE, by the same construction
    :func:`prefix_state` uses: the scan walks the whole recording, the state
    handed in is INJECTED at ``u == t + 1``, and only the losses of the steps
    after ``t`` are summed. Every earlier step reaches the injection through
    the carry and contributes nothing to the gradient, because ``where``
    selects the injected value there.

    ``container`` says WHICH RULE the suffix runs (owner ruling 2026-09-16):
    the plan's classes on the future adjoint's face, applied at EVERY step of
    the suffix, so the adjoint that arrives at ``t`` carries the error the
    rule accumulated over the whole suffix.

    ``diag``    the state-to-state Jacobian is replaced by its block diagonal
                at every step. The store does not move -- an adjoint is one
                number per state component either way -- only the value does.
    ``reduce``  the adjoint is projected onto its constant subspace at every
                step (:func:`mean_derivative`), so its only axis is implicit
                and the store falls to one number per component.
    ``quant``   the adjoint is rounded to the narrow dtype at every step
                (:func:`narrow_derivative`).
    """
    c_kind = _container(container)
    cell = _blockdiag_cell("eprop") if c_kind.diag else _cell()
    c = _consts()
    W, V, Wo = weights
    T = int(seq.shape[0])

    def tail_loss(st, y):
        def body(carry, inp):
            u, x = inp
            inject = u == t + 1
            st_u = tuple(jnp.where(inject, a, b) for a, b in zip(st, carry))
            if c_kind.reduce:
                st_u = tuple(mean_derivative(a) for a in st_u)
            if c_kind.quant:
                st_u = tuple(narrow_derivative(a) for a in st_u)
            nxt = cell(x, *st_u, W, V, Wo, *c)
            return nxt, jnp.where(u > t, _step_loss(nxt[4], y), 0.0)
        _, losses = jax.lax.scan(
            body, zero_state(), (jnp.arange(T), seq))
        return jnp.sum(losses)
    return tail_loss


def step_target_loss(seq, y, t, weights, state_prev):
    """``L_t``: what the traced step returns under ``tbptt``.

    ``t`` may be traced: ``seq[t]`` is a gather, not a slice."""
    cell = _cell()
    c = _consts()
    W, V, Wo = weights
    nxt = cell(seq[t], *state_prev, W, V, Wo, *c)
    return _step_loss(nxt[4], y), nxt


def carried_jacobians(seq, t, weights, *, check_zeros: bool = True,
                      narrow: bool = False):
    """The RTRL attachment tuple: three reference weights, then ELEVEN blocks.

    Block ``(s, w)`` is ``d s_(t-1)^s / d W_w``, taken by reverse-mode
    differentiation of the prefix. That is the same number the RTRL recursion
    ``G_t = A_t G_(t-1) + F_t`` accumulates over the prefix, and reverse mode
    gets it in ``4 * hidden + classes`` cotangent sweeps where the recursion
    needs one tangent per weight entry.

    ``check_zeros`` asserts the four blocks that CANNOT be non-zero (the
    readout weight feeds nothing back) really are zero, so the eleven carried
    blocks are the whole influence matrix and not a silent truncation.

    ``narrow`` runs the same recursion with the derivative held in the narrow
    dtype at every step -- the ``Quant`` class applied at every step of the
    prefix.
    """
    from graphax.examples.neuromorphic import (
        RSNN_CARRY_BLOCKS, RSNN_STATE_NAMES, RSNN_WEIGHT_NAMES,
        RSNN_ZERO_BLOCKS)

    run = prefix_state(seq, t, weights, narrow=narrow)
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


# ---------------------------------------------------------------------------
# THE CARRY CONTAINER (owner rulings, 2026-09-16)
# ---------------------------------------------------------------------------
# THE PLAN-PRODUCED CARRY IS THE ONLY MODE. There is no run-level flag any
# more. The exact carry is simply the plan with NO approximation on the
# carried-Jacobian face, and every other container is the plan's own classes
# on that face, applied at every step of the prefix (or of the suffix, for
# the future adjoint).
#
#   no class   the exact influence matrix, dense, 225.74 MB. Exact RTRL.
#   Diag       the block-diagonal recursion of Zenke and Neftci at every
#              step: e-prop, one eligibility trace per synapse, 2.12 MB.
#   Reduce     a coarser trace: the presynaptic axis is IMPLICIT, so the
#              carried value is the exact axis mean and the contraction reads
#              it as a replication.
#   Quant      a low-precision trace: the derivative that crosses each step
#              is held in bfloat16 and the blocks are stored there.
#   Skip       NO CARRY AT ALL. The given tuple is empty and the rule the
#              measurement compiles is truncated backpropagation through time.
#
# Combinations are ordinary: ``diag+quant`` is a low-precision eligibility
# trace, ``diag+reduce`` a trace with one axis collapsed, and so on.

#: The canonical name of the container a plan with no class on the carried
#: face implies.
EXACT_CONTAINER = "exact"

#: The canonical name of the container a ``Skip`` on the carried face
#: implies. It is not a storage form: it says there is no carried value, and
#: the measured program is the truncated one.
SKIP_CONTAINER = "skip"


def carry_containers() -> tuple[str, ...]:
    """Every container name a plan can imply, ``skip`` included."""
    from graphax.examples.neuromorphic import RSNN_CARRY_CONTAINERS
    return tuple(RSNN_CARRY_CONTAINERS) + (SKIP_CONTAINER,)


def _container(container):
    """A container name (or :class:`CarryContainer`) as a CarryContainer."""
    from graphax.examples.neuromorphic import (
        CarryContainer, carry_container_from_name)
    if isinstance(container, CarryContainer):
        return container
    if container is None:
        return CarryContainer()
    if str(container) == SKIP_CONTAINER:
        raise ValueError(
            "the skip container has no storage form: a Skip on the carried "
            "face means there is no carried value at all and the measured "
            "rule is tbptt. The caller decides that, not the producer.")
    return carry_container_from_name(container)


def container_name(container) -> str:
    """The canonical name of ``container``."""
    if container is not None and str(container) == SKIP_CONTAINER:
        return SKIP_CONTAINER
    return _container(container).name


def container_from_classes(classes) -> str:
    """The container name the plan's CLASSES on the carried face imply.

    ``classes`` is any iterable of the four class names of CONTEXT.md
    (``diag``, ``reduce``, ``quant``, ``skip``) plus ``none``, which is the
    identity and contributes nothing. ``skip`` DOMINATES: a face whose
    contraction is declined carries no value, so nothing else about the
    container can matter.
    """
    from graphax.examples.neuromorphic import CarryContainer
    seen = {str(c) for c in classes}
    unknown = seen - {"none", "diag", "reduce", "quant", "skip"}
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not action classes. The four classes are "
            f"diag, reduce, quant and skip (CONTEXT.md), and none is the "
            f"identity.")
    if "skip" in seen:
        return SKIP_CONTAINER
    return CarryContainer("diag" in seen, "reduce" in seen,
                          "quant" in seen).name


def _blockdiag_cell(container: str):
    """``rsnn_cell``, with the recurrent coupling the container drops.

    Under ``eprop`` the BACKWARD sees ``diag(V)`` where the forward sees
    ``V``, which is exactly Zenke and Neftci's replacement of the
    state-to-state Jacobian by its block diagonal: the forward states of the
    prefix and the suffix do not move at all, and the credit that flows
    through them does. Written as a straight-through substitution so there is
    one cell and one forward value, not two.
    """
    cell = _cell()
    if container == "exact":
        return cell

    def bd_cell(x, S, I, U, a, Uo, W, V, Wo, *c):
        vd = jnp.diag(V)
        rec = jax.lax.stop_gradient(V @ S - vd * S) + vd * S
        # The recurrent term enters only through ``I``; rebuild the step with
        # it replaced, so nothing else about the cell changes.
        a_syn, a_mem, a_out, rho, beta_a, thresh = c
        I_n = a_syn * I + W @ x + rec
        U_n = a_mem * U + (1.0 - a_mem) * I_n - thresh * S
        from graphax.examples.neuromorphic import rsnn_surrogate
        S_n = rsnn_surrogate(U_n - (thresh + beta_a * a))
        a_n = rho * a + S_n
        Uo_n = a_out * Uo + (1.0 - a_out) * (Wo @ S_n)
        return S_n, I_n, U_n, a_n, Uo_n
    return bd_cell


def carry_traces(seq, t, weights, *, reduce: bool = False,
                 quant: bool = False):
    """THE PLAN, RUN OVER THE PREFIX: the eleven blocks in COMPACT form.

    Zenke and Neftci (arXiv 2010.11931) approximate real-time recurrent
    learning by replacing the state-to-state Jacobian ``A_t`` with its BLOCK
    DIAGONAL, one block per neuron: a neuron's own synaptic current, membrane
    and adaptation survive and the coupling through OTHER neurons' spikes is
    dropped. Running ``G_u = blockdiag(A_u) G_(u-1) + F_u`` over the whole
    prefix leaves ONE ELIGIBILITY TRACE PER SYNAPSE. Per postsynaptic unit
    ``j`` and presynaptic index ``i``:

        psi[j]     = 1 / (scale * |U[j] - thresh - beta_a * a_prev[j]| + 1)^2
        eI[j,i]   <- a_syn * eI[j,i] + V[j,j] * eS[j,i] + direct[j,i]
        eU[j,i]   <- a_mem * eU[j,i] + (1 - a_mem) * eI[j,i] - thresh * eS[j,i]
        eS[j,i]   <- psi[j] * (eU[j,i] - beta_a * ea[j,i])
        ea[j,i]   <- rho * ea[j,i] + eS[j,i]

    with ``direct = x[i]`` for the input weight and ``direct = S_prev[i]`` for
    the recurrent weight, every trace zero at the start of the recording. THE
    ONE TERM THE EXACT RECURSION HAS AND THIS DOES NOT is ``V[j,k] * eS[k,i]``
    for ``k != j``. Set ``V`` diagonal and the two agree exactly; that is the
    test.

    WHY PROJECTING THE PRODUCT IS THE SAME AS BLOCK-DIAGONALISING ``A``. If
    ``G`` is already block diagonal, ``(A G)[j, j, i] = sum_k A[j,k] G[k,j,i]``
    has only the ``k = j`` term left, which is ``blockdiag(A) G``. The traces
    start at zero, which is block diagonal, so applying the projection at
    every step and replacing ``A`` by its block diagonal at every step are the
    same recursion. That is why "the plan applied at every step" IS e-prop
    here, and not merely something like it.

    THE THREE READOUT BLOCKS ARE EXACT AND STILL COMPACT. The readout feeds
    nothing back, so it is a leaky filter of the hidden traces through a
    CONSTANT ``Wo``, and the exact block factorises:

        (Uo, W)[m,j,i]  = Wo[m,j] * f_W[j,i],   f_W  <- a_out f_W + (1-a_out) eS_W
        (Uo, V)[m,j,i]  = Wo[m,j] * f_V[j,i]
        (Uo, Wo)[m,k,j] = delta(m,k) * g[j],    g    <- a_out g + (1-a_out) S

    so the compact container carries ``f_W``, ``f_V`` and ``g`` and
    :func:`graphax.examples.attach_rsnn_past` restores the constant factor
    from the REFERENCE weight, which is outside ``argnums`` and therefore adds
    no edge.

    ``t`` may be TRACED: the recursion is a masked scan over the whole
    recording, like :func:`prefix_state`.

    THE STORE IS THE POINT. 2.12 MB against 225.74 MB dense, a factor of 106.
    """
    from graphax.examples.neuromorphic import RSNN_CARRY_BLOCKS, RSNN_SURROGATE_SCALE

    cell = _cell()
    a_syn, a_mem, a_out, rho = decay_constants()
    c = _consts()
    W, V, Wo = weights
    h, n_in, n_out = RSNN_HIDDEN, SHD_CHANNELS, SHD_CLASSES
    vd = jnp.diag(V)[:, None]
    T = int(seq.shape[0])

    # REDUCE makes the presynaptic axis IMPLICIT, so every trace is stored
    # once for the whole axis. The recursion is linear in the direct term and
    # every other coefficient is independent of ``i``, so the mean of the
    # trace over ``i`` obeys the SAME recursion with the direct term replaced
    # by its own mean -- the stored value is the exact axis mean, and the
    # approximation is entirely in reading it back as a replication.
    wW = 1 if reduce else n_in
    wV = 1 if reduce else h

    zW = jnp.zeros((h, wW))
    zV = jnp.zeros((h, wV))
    init = (zero_state(),
            (zW, zW, zW, zW),          # eS, eI, eU, ea  for W
            (zV, zV, zV, zV),          # eS, eI, eU, ea  for V
            zW, zV, jnp.zeros((h,)))   # f_W, f_V, g

    def _narrow(a):
        return a.astype(CARRY_QUANT_DTYPE).astype(a.dtype)

    def body(carry, inp):
        u, x = inp
        st, trW, trV, fW, fV, g = carry
        S_prev, I_prev, U_prev, a_prev, Uo_prev = st
        nxt = cell(x, *st, W, V, Wo, *c)
        S, I, U, a, Uo = nxt
        psi = 1.0 / (RSNN_SURROGATE_SCALE
                     * jnp.abs(U - (THRESH + BETA_A * a_prev)) + 1.0) ** 2
        psi = psi[:, None]

        def step(tr, direct):
            eS, eI, eU, ea = tr
            nI = a_syn * eI + vd * eS + direct
            nU = a_mem * eU + (1.0 - a_mem) * nI - THRESH * eS
            nS = psi * (nU - BETA_A * ea)
            na = rho * ea + nS
            return (nS, nI, nU, na)

        if reduce:
            dW = jnp.broadcast_to(jnp.mean(x), (h, 1))
            dV = jnp.broadcast_to(jnp.mean(S_prev), (h, 1))
        else:
            dW = jnp.broadcast_to(x[None, :], (h, n_in))
            dV = jnp.broadcast_to(S_prev[None, :], (h, h))
        nW = step(trW, dW)
        nV = step(trV, dV)
        nfW = a_out * fW + (1.0 - a_out) * nW[0]
        nfV = a_out * fV + (1.0 - a_out) * nV[0]
        ng = a_out * g + (1.0 - a_out) * S
        keep = u < t
        new = (nxt, nW, nV, nfW, nfV, ng)
        if quant:
            # THE TRACE IS HELD IN LOW PRECISION AT EVERY STEP, which is what
            # a quantized trace means. The forward state ``nxt`` is not
            # touched: the rule's precision is not the model's.
            new = (nxt,) + jax.tree_util.tree_map(_narrow, new[1:])
        return jax.tree_util.tree_map(
            lambda p, q: jnp.where(keep, p, q), new, carry), None

    (st, trW, trV, fW, fV, g), _ = jax.lax.scan(
        body, init, (jnp.arange(T), seq))

    # (Uo, Wo)[m, k, j] = delta(m, k) * g[j]: every row of the compact
    # (n_out, h) form is the same filter, which is what the block diagonal of
    # a delta is. Under REDUCE the hidden axis of ``Wo`` is the implicit one,
    # so the stored value is that filter's own mean.
    gw = (jnp.broadcast_to(jnp.mean(g), (n_out, 1)) if reduce
          else jnp.broadcast_to(g[None, :], (n_out, h)))

    blocks = {
        (0, 0): trW[0], (0, 1): trV[0],
        (1, 0): trW[1], (1, 1): trV[1],
        (2, 0): trW[2], (2, 1): trV[2],
        (3, 0): trW[3], (3, 1): trV[3],
        # The readout blocks, exact, in the weight's own shape.
        (4, 0): fW, (4, 1): fV,
        (4, 2): gw,
    }
    sg = jax.lax.stop_gradient
    return tuple(sg(blocks[b]) for b in RSNN_CARRY_BLOCKS)


def reduced_columns(seq, t, weights, *, quant: bool = False):
    """THE PLAN, RUN OVER THE PREFIX, with the presynaptic axis IMPLICIT.

    The ``Reduce`` class with no ``Diag`` beside it. The exact recursion is
    ``G_u = A_u G_(u-1) + F_u`` on the full influence matrix; making the
    presynaptic axis implicit stores one value for the whole axis, and because
    ``A_u`` does not depend on that axis the mean over it obeys the same
    recursion:

        r_u[., j] = A_u r_(u-1)[., j] + mean_i F_u[., j, i]

    So the stored value is the EXACT axis mean of the influence matrix and the
    approximation lives entirely in the contraction, which reads it back as a
    replication along the axis. That is what an implicit axis is.

    ONE ``jax.jvp`` OF THE CELL PER STORED COLUMN does both terms at once: a
    tangent ``r_(u-1)[., j]`` on the five state components gives ``A_u r``, and
    a tangent ``e_j (x) (1/n) 1`` on the weight gives the averaged direct term.
    The columns are vmapped, so one step costs three vmapped cell tangents
    over 128, 128 and 20 columns -- against the 89 600 columns the exact
    recursion would need, which is why the exact carry is taken by
    :func:`carried_jacobians` instead.
    """
    from graphax.examples.neuromorphic import RSNN_CARRY_BLOCKS

    cell = _cell()
    c = _consts()
    W, V, Wo = weights
    h, n_in, n_out = RSNN_HIDDEN, SHD_CHANNELS, SHD_CLASSES
    T = int(seq.shape[0])
    zeros_w = (jnp.zeros_like(W), jnp.zeros_like(V), jnp.zeros_like(Wo))
    zeros_c = tuple(jnp.zeros_like(a) for a in c)
    #: The stored columns per weight: one per index of the weight's FIRST
    #: axis, because the LAST axis is the implicit one.
    n_cols = (h, h, n_out)

    def _narrow(a):
        return a.astype(CARRY_QUANT_DTYPE).astype(a.dtype)

    def push(st, x, cols, w_idx):
        """``A_u cols + mean_i F_u`` for every column of weight ``w_idx``."""
        shape = weights[w_idx].shape
        scale = 1.0 / float(shape[1])

        def one(col, j):
            # The direction of column ``j``: the row indicator times the mean
            # over the implicit axis. Built here rather than materialised as
            # a (rows, rows, implicit) table, which would be 46 MB for W.
            direction = jnp.zeros(shape).at[j].set(scale)
            wt = list(zeros_w)
            wt[w_idx] = direction
            primals = (x,) + tuple(st) + tuple(weights) + tuple(c)
            tangents = ((jnp.zeros_like(x),) + tuple(col) + tuple(wt)
                        + zeros_c)
            _, out_t = jax.jvp(cell, primals, tangents)
            return tuple(out_t)
        return jax.vmap(one)(cols, jnp.arange(n_cols[w_idx]))

    def zero_cols(n):
        return tuple(jnp.zeros((n,) + a.shape) for a in zero_state())

    init = (zero_state(), zero_cols(h), zero_cols(h), zero_cols(n_out))

    def body(carry, inp):
        u, x = inp
        st, cW, cV, cWo = carry
        nxt = cell(x, *st, W, V, Wo, *c)
        new = (nxt, push(st, x, cW, 0), push(st, x, cV, 1),
               push(st, x, cWo, 2))
        if quant:
            new = (nxt,) + jax.tree_util.tree_map(_narrow, new[1:])
        keep = u < t
        return jax.tree_util.tree_map(
            lambda p, q: jnp.where(keep, p, q), new, carry), None

    (_st, cW, cV, cWo), _ = jax.lax.scan(
        body, init, (jnp.arange(T), seq))

    cols = (cW, cV, cWo)
    sg = jax.lax.stop_gradient
    out = []
    for s, w in RSNN_CARRY_BLOCKS:
        # ``cols[w][s]`` is (column, state); the block wants (state, column,
        # 1) -- the state axes first and the implicit axis stored once.
        block = jnp.moveaxis(cols[w][s], 0, -1)[..., None]
        out.append(sg(block))
    return tuple(out)


def eprop_traces(seq, t, weights):
    """:func:`carry_traces`, EXPANDED to the dense block shapes.

    The same recursion and the same numbers, written into the container
    :func:`carried_jacobians` uses, so the two are DROP-IN interchangeable as
    the ``rtrl`` given values (three reference weights, then eleven blocks)
    and directly comparable block for block. This is the form the earlier
    probes and tests compare against; :func:`carry_traces` is what a run
    actually carries, and it is 106 times smaller.
    """
    from graphax.examples.neuromorphic import RSNN_CARRY_BLOCKS

    W, V, Wo = weights
    n_out = SHD_CLASSES
    compact = carry_traces(seq, t, weights)
    out = []
    for (s, w), m in zip(RSNN_CARRY_BLOCKS, compact):
        if s == 4 and w != 2:
            # (Uo, W) / (Uo, V): restore the constant readout factor.
            out.append(Wo[:, :, None] * m[None])
        elif s == 4:
            # (Uo, Wo): delta(m, k) * g[j]
            out.append(jnp.eye(n_out)[:, :, None] * m[0][None, None, :])
        else:
            out.append(jnp.eye(m.shape[0])[:, :, None] * m[:, None, :])
    sg = jax.lax.stop_gradient
    return tuple(sg(x) for x in weights) + tuple(sg(x) for x in out)


def carry_under_plan(seq, t, weights, container="exact", *,
                     check_zeros: bool = True):
    """The ``rtrl`` given tuple the PLAN implies (owner ruling 2026-09-16).

    ``container`` is the set of classes the plan put on the carried-Jacobian
    face, applied at EVERY step of the prefix:

    ``exact``          the dense influence matrix, from a detached
                       reverse-mode pass over the prefix. 225.74 MB.
    ``diag``           the block diagonal at every step: the eligibility
                       traces of e-prop, 2.12 MB.
    ``reduce``         the exact recursion with the presynaptic axis
                       implicit; see :func:`reduced_columns`.
    ``quant``          the exact recursion with the derivative held narrow at
                       every step, stored narrow.
    combinations       compose, in that order.

    Every container returns three reference weights and eleven blocks, so the
    varargs COUNT that selects the temporal rule is the same for all of them
    and only the SHAPES and the DTYPE move.
    """
    c = _container(container)
    if c.diag:
        blocks = carry_traces(seq, t, weights, reduce=c.reduce, quant=c.quant)
    elif c.reduce:
        blocks = reduced_columns(seq, t, weights, quant=c.quant)
    else:
        blocks = carried_jacobians(
            seq, t, weights, check_zeros=check_zeros, narrow=c.quant)[3:]
    if c.quant:
        blocks = tuple(b.astype(CARRY_QUANT_DTYPE) for b in blocks)
    sg = jax.lax.stop_gradient
    return tuple(sg(W) for W in weights) + tuple(blocks)


def future_adjoints(seq, y, t, weights, state_prev, container="exact"):
    """The BPTT attachment: five adjoints ``lambda_(t+1) = dL_(>t)/ds_t``.

    ``container`` selects which rule the suffix runs and how the result is
    stored; see :func:`suffix_adjoint`. Under ``reduce`` the state axis is
    implicit and each adjoint is stored as ONE number of extent 1; under
    ``quant`` the five adjoints are stored narrow."""
    c = _container(container)
    _, state_t = step_target_loss(seq, y, t, weights, state_prev)
    tail = suffix_adjoint(seq, t, weights, state_t, c)
    lam = jax.grad(lambda st: tail(st, y))(tuple(state_t))
    if c.reduce:
        # The projection ran at every step, so the adjoint is already constant
        # along its axis; storing the mean stores it once, exactly.
        lam = tuple(jnp.mean(x, keepdims=True) for x in lam)
    if c.quant:
        lam = tuple(x.astype(CARRY_QUANT_DTYPE) for x in lam)
    sg = jax.lax.stop_gradient
    return tuple(sg(x) for x in lam)


#: The last sampled step position, per process, for the trainer's `[cfg]` line
#: and the plan record. A module-level record because the argument builder is
#: the only place that knows it and the record is written elsewhere.
_LAST_STEP_POSITION: dict = {}


def last_step_position() -> dict:
    """``{"t": int, "T": int, "recording": int}`` of the last built tuple."""
    return dict(_LAST_STEP_POSITION)


def step_position_bound(T: int, rule) -> int:
    """One past the largest legal step position for ``rule``.

    ``window2`` needs ``t`` AND ``t + 1`` inside the recording, so its last
    legal position is ``T - 2``. Every other rule may sit at the last step.
    """
    return int(T) - 1 if str(rule) == "window2" else int(T)


def sampled_step_position(key, T: int, rule=None):
    """The drawn step position, as an ARRAY. Traceable and vmappable.

    ``t`` is drawn uniformly from ``1`` up to :func:`step_position_bound`.
    ``t = 0`` is excluded because its carried state is the zero state and
    every given quantity is zero there, which makes the rules identical and
    hides what the run is measuring.
    """
    return jax.random.randint(key, (), 1, step_position_bound(T, rule))


def sample_step_position(key, T: int, rule=None) -> int:
    """:func:`sampled_step_position` as a Python int, for the host paths."""
    return int(sampled_step_position(key, T, rule))


#: The argument slots the DATA GENERATOR fills, by rule. Slots 0 and 1 are the
#: input frame and the label, 2 to 6 the carried state, 7 to 9 the weights and
#: 10 to 15 the constants; the rule's given values follow at 16.
RSNN_HEAD_SLOTS = 16


def rsnn_data_slots(rule: str) -> tuple[int, ...]:
    """The slots :func:`rsnn_data_gen` returns, in order.

    THE WEIGHTS ARE AMONG THEM, and that is deliberate. The ``rtrl`` given
    values LEAD with three REFERENCE WEIGHTS whose whole job is to be bit-for-
    bit equal to the weights in slots 7 to 9, so that the attached
    ``W - W_ref`` is exactly zero and no forward value moves. A refresher that
    replaced the weights and not the reference weights (which is what
    ``generate_eval_samples`` does to every slot a generator does not cover)
    would break that equality in silence. So the generator owns the weights
    too, and hands back the run's initial ones -- which is also the point at
    which this target is defined (see WEIGHT_SCALE).
    """
    from graphax.examples.neuromorphic import RSNN_GIVEN_LENGTHS
    if str(rule) == "window2":
        # Two input frames, the label, the five carried state components and
        # the three weights. There are no given values in this arm.
        return tuple(range(0, 11))
    n_given = {v: n for n, v in RSNN_GIVEN_LENGTHS.items()}[str(rule)]
    return (tuple(range(0, 10))
            + tuple(range(RSNN_HEAD_SLOTS, RSNN_HEAD_SLOTS + n_given)))


def target_example(example: str | None, rule) -> str | None:
    """The EXAMPLE the temporal rule builds, which is not always ``example``.

    ``--temporal-rule window2`` is not another given edge on the one-step
    body; it is a DIFFERENT graph, two step copies wide, with no given edge at
    all. It therefore has its own registered target, and this is the one place
    that says so. Every other rule keeps the target it was asked for.
    """
    if is_rsnn(example) and rule is not None and str(rule) == "window2":
        return RSNN_W2_TARGET
    return example


def is_window2(example: str | None) -> bool:
    return bool(example) and str(example) == RSNN_W2_TARGET


def rsnn_data_gen(key=None, *, dataset: str | None = None,
                  dataset_size: int | None = -1,
                  temporal_rule: str | None = None,
                  carry_container: str | None = None):
    """``keys -> the data-dependent argument slots``, at a SAMPLED ``t``.

    OWNER RULING, 2026-09-16: the step position is sampled uniformly over the
    recording PER ENVIRONMENT AND PER EPISODE. This is the object that does
    it. Everything that changes with ``t`` is here -- the input frame, the
    carried state, and the rule's given values (the carried Jacobian under
    ``rtrl``, the future adjoints under ``bptt``, nothing under ``tbptt``) --
    and everything that does not (the weights, the six decay constants) either
    rides along unchanged or is left alone.

    THE RECORDING IS FIXED, THE STEP POSITION IS NOT. The ruling asks for
    ``t`` uniform OVER THE RECORDING, so one recording is drawn from the run's
    key and every draw walks it. Holding the recording still also keeps this
    function jittable and vmappable with a 280 kB closure instead of the
    571 MB the whole binned split would cost on the device.

    THE GRAPH SHAPE DOES NOT MOVE WITH ``t``. The prefix and the suffix are
    masked scans over the whole recording (see :func:`prefix_state`), so the
    traced step body, its vertex count and its face count are the same at
    every step position. ``tests/temporal_rule_test.py`` asserts that.

    WHAT ONE DRAW COSTS. One prefix pass of ``T`` cell evaluations plus, under
    ``rtrl``, one reverse-mode Jacobian of that pass (``4 * hidden + classes``
    = 532 cotangent sweeps) or, under ``bptt``, one suffix pass and one
    reverse sweep. It is drawn once per (environment, episode), not once per
    measured plan: the probe-batch cache in ``env._probe_batch`` is keyed by
    the environment row and the episode.
    """
    rule = "tbptt" if temporal_rule is None else str(temporal_rule)
    if rule not in TEMPORAL_RULES:
        raise ValueError(f"temporal rule {rule!r} is not one of "
                         f"{list(TEMPORAL_RULES)}")
    cont = container_name(carry_container)
    if rule not in GIVEN_EDGE_RULES and cont != EXACT_CONTAINER:
        raise ValueError(
            f"carry container {cont!r} was asked of the {rule} generator, "
            f"which attaches no given temporal edge. A container is how a "
            f"CARRIED value is stored, so it is a question only on "
            f"{list(GIVEN_EDGE_RULES)}.")
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 3)
    seq, y, rec = _draw_recording(k[0], dataset, dataset_size)
    weights = rsnn_weights(k[1])
    T = int(seq.shape[0])
    slots = rsnn_data_slots(rule)

    def _t(keys):
        return sampled_step_position(keys[0], T, rule)

    def _head(t):
        sg = jax.lax.stop_gradient
        state_prev = tuple(
            sg(x) for x in prefix_state(seq, t, weights)(*weights))
        if rule == "window2":
            return (seq[t], seq[t + 1], y) + state_prev + weights, state_prev
        return (seq[t], y) + state_prev + weights, state_prev

    def _build(t, container):
        head, state_prev = _head(t)
        if rule in ("tbptt", "window2"):
            given = ()
        elif rule == "rtrl":
            given = carry_under_plan(seq, t, weights, container,
                                     check_zeros=False)
        else:
            given = future_adjoints(seq, y, t, weights, state_prev, container)
        return head + tuple(given)

    @jax.jit
    def _draw(keys):
        out = _build(_t(keys), cont)
        if len(out) != len(slots):
            raise ValueError(
                f"the {rule} generator built {len(out)} arrays for "
                f"{len(slots)} declared slots {slots}")
        return out

    # A PLAIN PYTHON WRAPPER around the jitted draw: the attributes below are
    # the generator's contract with the env, and a `PjitFunction` is a C type
    # that does not take them.
    def fn(keys):
        return _draw(keys)

    @jax.jit
    def _draw_exact(keys):
        """The SAME draw with the EXACT carry. The quality reference."""
        return _build(_t(keys), EXACT_CONTAINER)

    def meta(keys):
        """``{t, T, recording, rule, carry}`` of the draw ``keys`` produces."""
        return {"t": int(_t(keys)), "T": T, "recording": int(rec),
                "rule": rule, "carry": cont}

    # THE CONTRACT WITH env._probe_batch AND generate_eval_samples. Both used
    # to assume a generator fills the first one or two argument slots; this
    # one fills ten of them and then a block at 16. `data_slots` is how a
    # generator says so, and a generator without the attribute keeps the old
    # contiguous-from-zero behaviour exactly.
    fn.data_slots = slots
    #: Redraw per (environment, episode) rather than once per process.
    fn.resample_per_env_episode = True
    #: HOW MANY PROBE BATCHES THE GRADIENT COSINE NEEDS (dsnn-dfw.51).
    #:
    #: A probe batch of this generator is a STEP POSITION, and the recording
    #: goes silent well before it ends: on the recording the campaign drew
    #: (931 of the real SHD train split) only 57 of the 100 bins carry any
    #: input spike, and the exact gradient of the step loss is IDENTICALLY
    #: ZERO at 43 of the 99 legal positions -- every t from 57 to 99 (probe
    #: 66655, float64 on the CPU). The cosine is undefined on such a batch and
    #: the measurement is refused, so with one batch 43 percent of every
    #: measurement on this target is missing data.
    #:
    #: FIVE, because 0.4343 ** 5 = 0.015: one measurement in 65 still draws
    #: five silent steps and is refused, which is a rate a run can carry, and
    #: because five is the number the owner's own probe-count sweep compared
    #: against thirty and found no difference in correlation -- so the extra
    #: four draws cost four executions of a 784 us program and buy back 42
    #: percent of the channel.
    fn.probe_batches = 5
    fn.meta = meta
    # THE QUALITY REFERENCE, when the carry itself is approximated (owner
    # ruling 2026-09-16). The in-band gradient cosine scores the plan against
    # the rev-exact plan ON THE SAME ARGUMENTS, so an approximation that lives
    # in an ARGUMENT is invisible to it: both sides read the same approximated
    # carry and the cosine is 1.0 whatever the rule did over the recording.
    # A generator whose draw is itself approximated therefore has to publish
    # the EXACT draw at the SAME step position, and the quality channel takes
    # its reference from `jax.grad` of the target on that. Then the number
    # reward slot 6 holds is the error the rule ACCUMULATED over the whole
    # recording, which is what it has to be. Absent under `exact`, where the
    # in-band reference is already the truth.
    if cont != EXACT_CONTAINER:
        def reference_draw(keys):
            return _draw_exact(keys)
        fn.reference_draw = reference_draw
    #: The container this generator draws, so a measurement can ask.
    fn.carry_container = cont
    fn.temporal_rule = rule
    return fn


def rsnn_args(key=None, *, dataset: str | None = None,
              dataset_size: int | None = -1, temporal_rule: str | None = None,
              step_position: int | None = None,
              carry_container: str | None = None):
    """The argument tuple of ``graphax.examples.neuromorphic.RSNN_SHD``.

    Slots: ``x_t`` 0, ``y`` 1, the five carried state components 2 to 6, the
    three weights 7 to 9, the six constants 10 to 15, then the rule's given
    values.

    Under ``window2`` the target is ``RSNN_SHD_W2`` and the tuple is the two
    input frames, the label, the five carried state components, the three
    weights and the six constants, with NO given values.

    ``step_position`` pins ``t``; ``None`` draws it uniformly from the key
    (see :func:`sample_step_position`). The recording is drawn from the same
    key, so two processes given the same seed build the same tuple.
    """
    from graphax.examples.neuromorphic import RSNN_GIVEN_LENGTHS

    rule = "tbptt" if temporal_rule is None else str(temporal_rule)
    if rule not in TEMPORAL_RULES:
        raise ValueError(f"temporal rule {rule!r} is not one of "
                         f"{list(TEMPORAL_RULES)}")
    cont = container_name(carry_container)
    if rule not in GIVEN_EDGE_RULES and cont != EXACT_CONTAINER:
        raise ValueError(
            f"carry container {cont!r} was asked of rule {rule}, which "
            f"attaches no given temporal edge. A container is how a CARRIED "
            f"value is stored, so it is a question only on "
            f"{list(GIVEN_EDGE_RULES)}.")
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 3)
    seq, y, rec = _draw_recording(k[0], dataset, dataset_size)
    weights = rsnn_weights(k[1])
    T = int(seq.shape[0])
    hi = step_position_bound(T, rule)
    t = (sample_step_position(k[2], T, rule) if step_position is None
         else int(step_position))
    if not 0 <= t < hi:
        raise ValueError(f"step position {t} is outside 0 .. {hi - 1} for "
                         f"rule {rule}")
    _LAST_STEP_POSITION.clear()
    _LAST_STEP_POSITION.update({"t": t, "T": T, "recording": rec,
                                "rule": rule, "carry": cont})

    sg = jax.lax.stop_gradient
    state_prev = tuple(sg(x) for x in prefix_state(seq, t, weights)(*weights))
    if rule == "window2":
        return ((seq[t], seq[t + 1], y) + state_prev + weights + _consts())
    head = (seq[t], y) + state_prev + weights + _consts()

    if rule == "tbptt":
        given = ()
    elif rule == "rtrl":
        given = carry_under_plan(seq, t, weights, cont)
    else:
        given = future_adjoints(seq, y, t, weights, state_prev, cont)
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
