"""--face-skip-bias and --face-init-{approx,skips}-per-plan (dsnn-dfw.74).

Owner ruling 2026-09-20: every NN256 arm with approximations collapses from
a bias-2 start (--face-none-bias couples the skip logit to the none bias).
This adds a SKIP-only override and a closed-form derivation of both biases
from target per-plan counts. Pins here:

1. --face-skip-bias DEFAULTS OFF (None): apply_face_none_bias(agent, B) is
   unchanged bit for bit; the SKIP logit still gets -B.
2. --face-skip-bias Bs, set: the SKIP logit gets -Bs, the OP_NONE logits
   still get +B (independent -- B may be 0 while Bs is not, and vice versa).
3. THE INVERSE. derive_face_none_bias(F, S, k, a) is the bisection inverse of
   expected_face_counts (the quant bit at -B rides beside the slot ops since
   2026-09-23) and derive_face_skip_bias(F, kappa) = ln(F/kappa - 1), for
   F = 11 (the dsnn-dfw.74 NN256 rows) and F = 118 (the TLM face inventory,
   finding 51); expected_face_counts inverts them back to (a, kappa).
4. REFUSALS. --face-init-approx-per-plan / --face-init-skips-per-plan
   conflict with --face-none-bias / --face-skip-bias; an argument that
   makes the log undefined raises; and a caller with NO F still refuses --
   a plan's live-face count is the length of a walk over one elimination
   order, not a jaxpr property (tools/faces_per_vertex.py), so
   resolve_face_init_bias(args) with F=None may not guess one, and neither
   may agent_factory.build_and_init_agent, which is reached by trainers
   that build the agent before any env exists. Which F ppo.main passes --
   the reverse-mode reference order's count, owner ruling 2026-09-20 -- is
   pinned in tests/face_init_reference_order_test.py.
5. FLAGS OFF -> INERT. The plan-init flags at their None default never
   call resolve_face_init_bias's F-dependent branches; build_and_init_agent
   is unchanged bit for bit from before this ticket.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from init_scheme_test import _agent, _assert_same, _ns, _raw_agent  # noqa: E402

from alphagrad.approx.common.agent_factory import (             # noqa: E402
    apply_face_none_bias, derive_face_none_bias, derive_face_skip_bias,
    expected_face_counts, resolve_face_init_bias)
from alphagrad.approx.unified_face_head import (                # noqa: E402
    FACE_SLOTS, NUM_APPROX_OPS, OP_NONE, O_QUANT, O_SKIP, S_OP, slot_base)

S = FACE_SLOTS
K = NUM_APPROX_OPS - 1
assert (S, K) == (3, 2)


# --------------------------------------------------------- 1/2. the flag

def test_parser_defaults_are_off():
    from alphagrad.approx.ppo import make_argparser
    ns = make_argparser().parse_args([])
    assert ns.face_skip_bias is None
    assert ns.face_init_approx_per_plan is None
    assert ns.face_init_skips_per_plan is None
    ns = make_argparser().parse_args(["--face-skip-bias", "3"])
    assert ns.face_skip_bias == 3.0


def test_skip_bias_none_is_bitwise_the_old_coupled_behaviour():
    ns0 = _ns()
    raw, init_key = _raw_agent(ns0)
    from alphagrad.approx.ppo import apply_init_scheme
    base = apply_init_scheme(raw, init_key, ns0)
    old = apply_face_none_bias(base, 4.0)
    new = apply_face_none_bias(base, 4.0, None)
    _assert_same(old, new, "skip_bias=None must match the old 2-arg call")


def test_skip_bias_moves_only_skip_none_moves_only_none_logits():
    ns0 = _ns()
    raw, init_key = _raw_agent(ns0)
    from alphagrad.approx.ppo import apply_init_scheme
    base = apply_init_scheme(raw, init_key, ns0)
    b0 = np.asarray(base.face_path_policy.head.proj.layers[-1].bias)
    out = apply_face_none_bias(base, 2.0, 5.0)
    b1 = np.asarray(out.face_path_policy.head.proj.layers[-1].bias)
    none_idx = {slot_base(s) + S_OP + OP_NONE for s in range(FACE_SLOTS)}
    assert b1[O_SKIP] - b0[O_SKIP] == pytest.approx(-5.0)
    for i in none_idx:
        assert b1[i] - b0[i] == pytest.approx(2.0)
    # the face's quant bit is an approximation like a slot's non-none pick,
    # so the identity init pushes it down by the same B (owner ruling
    # 2026-09-23: one Bernoulli per face beside the skip)
    assert b1[O_QUANT] - b0[O_QUANT] == pytest.approx(-2.0)
    keep = [i for i in range(len(b0))
            if i not in none_idx | {O_SKIP, O_QUANT}]
    np.testing.assert_array_equal(b1[keep], b0[keep])


def test_skip_bias_works_with_zero_none_bias():
    """B = 0 (off), Bs != 0: OP_NONE and QUANT logits untouched, SKIP still
    moves."""
    ns0 = _ns()
    raw, init_key = _raw_agent(ns0)
    from alphagrad.approx.ppo import apply_init_scheme
    base = apply_init_scheme(raw, init_key, ns0)
    b0 = np.asarray(base.face_path_policy.head.proj.layers[-1].bias)
    out = apply_face_none_bias(base, 0.0, 5.0)
    b1 = np.asarray(out.face_path_policy.head.proj.layers[-1].bias)
    assert b1[O_SKIP] - b0[O_SKIP] == pytest.approx(-5.0)
    none_idx = [slot_base(s) + S_OP + OP_NONE for s in range(FACE_SLOTS)]
    np.testing.assert_array_equal(b1[none_idx], b0[none_idx])
    assert b1[O_QUANT] == b0[O_QUANT]


# ------------------------------------------------------ 3. the closed form

@pytest.mark.parametrize("F", [11.0, 118.0])
@pytest.mark.parametrize("a_frac", [0.05, 0.1, 0.3])
def test_derive_face_none_bias_inverts_the_expected_count(F, a_frac):
    # a_frac of the "everything approximated" ceiling F*S*K, comfortably
    # inside the reachable range. E[A] carries the quant bit's term,
    # so B is the bisection inverse of expected_face_counts rather than the
    # slot-only closed form ln(F*S*K/a - K).
    a = a_frac * F * S * K
    B = derive_face_none_bias(F, S, K, a)
    e_a, _ = expected_face_counts(F, S, K, B, B)
    assert e_a == pytest.approx(a, rel=1e-9)
    # and E[A] is what the head's own Bernoullis say: per face, the bit at
    # sigmoid(-B), each of the S slots at k/(e^B + k), the two operand slots
    # forced to none behind the bit
    p_q = 1.0 / (1.0 + math.exp(B))
    p_op = K / (math.exp(B) + K)
    assert e_a == pytest.approx(
        F * ((1 - p_q) * S * p_op + p_q * (1 + (S - 2) * p_op)), rel=1e-9)


@pytest.mark.parametrize("F", [11.0, 118.0])
@pytest.mark.parametrize("kappa_frac", [0.1, 0.3, 0.6])
def test_derive_face_skip_bias_closed_form(F, kappa_frac):
    kappa = kappa_frac * F
    Bs = derive_face_skip_bias(F, kappa)
    assert Bs == pytest.approx(math.log(F / kappa - 1.0))
    _, e_k = expected_face_counts(F, S, K, Bs, Bs)
    assert e_k == pytest.approx(kappa, rel=1e-9)


def test_derive_face_none_bias_f11_and_f118_reference_values():
    # F=11 (dsnn-dfw.74 NN256 rows), a=2, and F=118 (TLM face inventory,
    # finding 51), a=7: the bias lands the requested count exactly.
    for F, a in ((11.0, 2.0), (118.0, 7.0)):
        B = derive_face_none_bias(F, S, K, a)
        assert expected_face_counts(F, S, K, B, B)[0] == pytest.approx(a)


def test_derive_face_none_bias_rejects_undefined_log():
    # a at/beyond the F*S ceiling: unreachable by any finite bias.
    ceiling = 11.0 * S
    with pytest.raises(ValueError, match="unreachable"):
        derive_face_none_bias(11.0, S, K, ceiling)
    with pytest.raises(ValueError, match="unreachable"):
        derive_face_none_bias(11.0, S, K, ceiling + 1.0)
    with pytest.raises(ValueError, match="must be > 0"):
        derive_face_none_bias(11.0, S, K, 0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        derive_face_none_bias(11.0, S, K, -1.0)


def test_derive_face_skip_bias_rejects_undefined_log():
    with pytest.raises(ValueError, match="unreachable"):
        derive_face_skip_bias(11.0, 11.0)      # kappa == F
    with pytest.raises(ValueError, match="unreachable"):
        derive_face_skip_bias(11.0, 20.0)      # kappa > F
    with pytest.raises(ValueError, match="must be > 0"):
        derive_face_skip_bias(11.0, 0.0)


# --------------------------------------------------------- 4. the refusals

def test_resolve_face_init_bias_is_inert_when_unset():
    ns = _ns()
    assert resolve_face_init_bias(ns) == (None, None)
    assert resolve_face_init_bias(ns, F=11.0) == (None, None)


def test_resolve_refuses_when_face_none_bias_also_set():
    ns = _ns(face_init_approx_per_plan=2.0, face_none_bias=4.0)
    with pytest.raises(ValueError, match="also set"):
        resolve_face_init_bias(ns, F=11.0)


def test_resolve_refuses_when_face_skip_bias_also_set():
    ns = _ns(face_init_skips_per_plan=2.0, face_skip_bias=3.0)
    with pytest.raises(ValueError, match="also set"):
        resolve_face_init_bias(ns, F=11.0)


def test_resolve_refuses_undefined_log_even_with_f_given():
    ns = _ns(face_init_approx_per_plan=1000.0)  # way past the ceiling
    with pytest.raises(ValueError, match="unreachable"):
        resolve_face_init_bias(ns, F=11.0)


def test_resolve_refuses_when_f_is_not_known():
    """A caller with no F still gets the refusal, unchanged. ppo.main now
    passes one (the reverse reference order's count, owner ruling
    2026-09-20, pinned in face_init_reference_order_test.py), but
    build_and_init_agent is reached by trainers that build the agent before
    any env exists: the default F=None must raise there rather than
    silently derive from a guessed number."""
    ns = _ns(face_init_approx_per_plan=2.0)
    with pytest.raises(ValueError, match="not known"):
        resolve_face_init_bias(ns)
    ns2 = _ns(face_init_skips_per_plan=2.0)
    with pytest.raises(ValueError, match="not known"):
        resolve_face_init_bias(ns2)


def test_real_agent_build_path_refuses_when_the_flag_is_set():
    """build_and_init_agent (the factory path both trainers share) must
    hit the SAME refusal, not silently ignore the flag."""
    ns = _ns(face_init_approx_per_plan=2.0)
    with pytest.raises(ValueError, match="not known"):
        _agent(ns)


def test_ppo_main_inline_path_calls_resolve_before_building():
    import inspect
    from alphagrad.approx import ppo
    main_src = inspect.getsource(ppo.main)
    assert "resolve_face_init_bias(args, F=_face_ref_F)" in main_src
    assert main_src.index("resolve_face_init_bias(args, F=_face_ref_F)") \
        < main_src.index("apply_face_none_bias(agent")


# ---------------------------------------------------- 5. flags-off inert

def test_plan_init_flags_off_leave_build_and_init_agent_unchanged():
    ns_old = _ns(face_none_bias=4.0)
    ns_new = _ns(face_none_bias=4.0, face_skip_bias=None,
                 face_init_approx_per_plan=None,
                 face_init_skips_per_plan=None)
    _assert_same(_agent(ns_old), _agent(ns_new), "plan-init flags at None")
