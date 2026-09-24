# -*- coding: utf-8 -*-
"""--init-scheme {campaign,classic} and --scale-face-head.

CLEAN_DESIGN_AUDIT a3 / CODE-CHANGE #1: ``_scale_output_heads`` scales the
vertex pointer, zeroes ``pref_proj`` and scales ``micro_action_policy.head``
-- but ``face_path_policy.head`` is absent from it, and under ``--live-faces``
``micro_action_policy`` is ``None``. So the ONLY live approximation head is
the one head that is never rescaled, and its 94 logits keep full
orthogonal(sqrt 2) magnitude from step 0 while the pointer sits at 0.1x.

Pins here:

1. FLAG-OFF BIT-IDENTITY. ``apply_init_scheme`` with the parser defaults is
   leaf-for-leaf the historical ``init_linear_weights`` +
   ``_scale_output_heads(head_init_scale)`` pair.
2. ``--scale-face-head S`` touches EXACTLY the face head's output projection
   and scales it by exactly S; every other leaf is bitwise unchanged.
3. ``--init-scheme classic`` gives Glorot-uniform weights (inside the
   textbook bound, and NOT orthogonal), zero biases everywhere, and no head
   rescaling at all -- pointer and face head on one scale.
4. INIT-LOGIT STATISTICS: max|logit| and std of the face head's 94 logits at
   init, per scheme, at --face-none-bias 0 and 6. Printed as a table
   (run with -s) and asserted where the audit makes a claim.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

TOTAL_V = 6
EMBD = 32


def _ns(**over):
    from alphagrad.approx.common.agent_factory import apply_policy_arch
    from alphagrad.approx import ppo as P
    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False)
    ns.vocab_size = 64
    ns.embd_dim = EMBD
    ns.num_heads = 2
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.face_endpoint_read = False
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _raw_agent(ns, seed=11):
    """`_build_agent` alone (no init pipeline), on PPO's key derivation."""
    from alphagrad.approx.common.agent_factory import derive_agent_keys
    from alphagrad.approx.ppo import _build_agent
    key, init_key = derive_agent_keys(seed)
    return _build_agent(ns, TOTAL_V, 4, 4, key), init_key


def _agent(ns, seed=11):
    from alphagrad.approx.common.agent_factory import build_and_init_agent
    return build_and_init_agent(ns, TOTAL_V, 4, 4, seed=seed)


def _warm_quant_scan():
    from graphax.sparse.micro_actions import report_hardware_scan
    report_hardware_scan()


@pytest.fixture(scope="module", autouse=True)
def _warm():
    _warm_quant_scan()


def _leaves(a):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(
        eqx.filter(a, eqx.is_inexact_array))]


def _assert_same(a, b, msg=""):
    la, lb = _leaves(a), _leaves(b)
    assert len(la) == len(lb), f"{msg}: leaf count {len(la)} != {len(lb)}"
    for i, (x, y) in enumerate(zip(la, lb)):
        np.testing.assert_array_equal(x, y, err_msg=f"{msg}: leaf {i}")


def _n_differing(a, b):
    return sum(1 for x, y in zip(_leaves(a), _leaves(b))
               if x.shape != y.shape or not np.array_equal(x, y))


# ------------------------------------------------------- 1. flag-off identity

def test_default_scheme_is_the_historical_pipeline_bitwise():
    from alphagrad.approx.common.init import init_linear_weights
    from alphagrad.approx.ppo import _scale_output_heads, apply_init_scheme
    ns = _ns()
    assert ns.init_scheme == "campaign" and ns.scale_face_head == 0.0
    raw, init_key = _raw_agent(ns)
    want = _scale_output_heads(init_linear_weights(raw, init_key),
                               float(ns.head_init_scale))
    got = apply_init_scheme(raw, init_key, ns)
    _assert_same(got, want, "campaign default")


def test_scale_output_heads_default_kwarg_is_a_noop():
    """The new `face_head_scale` kwarg defaults to off."""
    from alphagrad.approx.common.init import init_linear_weights
    from alphagrad.approx.ppo import _scale_output_heads
    ns = _ns()
    raw, init_key = _raw_agent(ns)
    base = init_linear_weights(raw, init_key)
    _assert_same(_scale_output_heads(base, 0.1),
                 _scale_output_heads(base, 0.1, face_head_scale=0.0),
                 "face_head_scale=0")


# --------------------------------------------------- 2. --scale-face-head

def test_scale_face_head_touches_exactly_the_face_head():
    from alphagrad.approx.ppo import apply_init_scheme
    ns0, ns1 = _ns(), _ns(scale_face_head=0.1)
    raw, init_key = _raw_agent(ns0)
    a0 = apply_init_scheme(raw, init_key, ns0)
    a1 = apply_init_scheme(raw, init_key, ns1)
    assert a0.face_path_policy is not None
    w0 = np.asarray(a0.face_path_policy.head.proj.layers[-1].weight)
    w1 = np.asarray(a1.face_path_policy.head.proj.layers[-1].weight)
    np.testing.assert_allclose(w1, w0 * 0.1, rtol=0, atol=0)
    assert _n_differing(a0, a1) == 1, "more than the face head moved"


# --------------------------------------------------------- 3. classic scheme

def test_classic_is_glorot_zero_bias_and_unscaled():
    from alphagrad.approx.ppo import apply_init_scheme
    ns_c, ns_k = _ns(), _ns(init_scheme="classic")
    raw, init_key = _raw_agent(ns_c)
    camp = apply_init_scheme(raw, init_key, ns_c)
    clas = apply_init_scheme(raw, init_key, ns_k)

    # every Linear bias is zero under BOTH schemes (a2 is kept).
    for a, name in ((camp, "campaign"), (clas, "classic")):
        for m in jax.tree_util.tree_leaves(
                a, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)):
            if isinstance(m, eqx.nn.Linear) and m.bias is not None:
                np.testing.assert_array_equal(
                    np.asarray(m.bias), 0.0, err_msg=f"{name} bias")

    # Glorot uniform: bounded by sqrt(6/(fan_in+fan_out)), and NOT orthogonal.
    fh = clas.face_path_policy.head.proj.layers[-1]
    w = np.asarray(fh.weight)
    fo, fi = w.shape
    bound = np.sqrt(6.0 / (fi + fo))
    assert np.abs(w).max() <= bound + 1e-6
    assert np.abs(w).max() > 0.3 * bound         # actually uniform, not tiny
    wc = np.asarray(camp.face_path_policy.head.proj.layers[-1].weight)
    assert np.abs(wc).max() > np.abs(w).max(), (
        "the campaign face head should carry the larger magnitude")

    # NO head rescaling under classic: the pointer projection is untouched
    # and pref_proj is NOT zeroed.
    _kp = (clas.vertex_policy.k_proj if hasattr(clas.vertex_policy, "k_proj")
           else clas.vertex_policy.pointer_proj)
    _kp0 = (camp.vertex_policy.k_proj if hasattr(camp.vertex_policy, "k_proj")
            else camp.vertex_policy.pointer_proj)
    assert np.abs(np.asarray(_kp.weight)).max() > \
        np.abs(np.asarray(_kp0.weight)).max()
    assert np.abs(np.asarray(clas.pref_proj.weight)).max() > 0.0
    np.testing.assert_array_equal(np.asarray(camp.pref_proj.weight), 0.0)


def test_classic_rejects_unknown_scheme():
    from alphagrad.approx.ppo import apply_init_scheme
    ns = _ns(init_scheme="nonsense")
    raw, init_key = _raw_agent(ns)
    with pytest.raises(ValueError):
        apply_init_scheme(raw, init_key, ns)


# ------------------------------------------- 4. init-logit statistics table

def _face_logit_stats(scheme, none_bias, seed=11, n=256):
    a = _agent(_ns(init_scheme=scheme, face_none_bias=float(none_bias)),
               seed=seed)
    head = a.face_path_policy.head
    ctx = jrand.normal(jrand.PRNGKey(7), (n, EMBD))
    z = np.asarray(jax.vmap(head.logits)(ctx))               # (n, 94)
    return {"maxabs": float(np.abs(z).max()),
            "std": float(z.std()),
            "mean": float(z.mean()),
            "logits": z, "agent": a}


def test_init_logit_statistics_table():
    rows = []
    stats = {}
    for scheme in ("campaign", "classic"):
        for nb in (0, 6):
            s = _face_logit_stats(scheme, nb)
            stats[(scheme, nb)] = s
            rows.append((scheme, nb, s["maxabs"], s["std"], s["mean"]))
    print("\n  face-head INIT LOGITS (94 outputs, 256 N(0,1) contexts, E=32)")
    print(f"  {'scheme':<10}{'NONE_BIAS':>10}{'max|logit|':>13}"
          f"{'std':>10}{'mean':>10}")
    for r in rows:
        print(f"  {r[0]:<10}{r[1]:>10}{r[2]:>13.4f}{r[3]:>10.4f}{r[4]:>10.4f}")

    # The audit's claim: the campaign face head starts with materially
    # LARGER logits than a textbook init would give it.
    assert stats[("campaign", 0)]["maxabs"] > stats[("classic", 0)]["maxabs"]
    assert stats[("campaign", 0)]["std"] > stats[("classic", 0)]["std"]

    # The none-bias is exactly +B on each slot's OP_NONE logit, -B on
    # SKIP and -B on the quant bit, and nothing else -- under BOTH schemes.
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS, OP_NONE, O_QUANT, O_SKIP, S_OP, slot_base)
    for scheme in ("campaign", "classic"):
        d = (stats[(scheme, 6)]["logits"] - stats[(scheme, 0)]["logits"])
        want = np.zeros(d.shape[-1], np.float32)
        for s in range(FACE_SLOTS):
            want[slot_base(s) + S_OP + OP_NONE] = 6.0
        want[O_SKIP] = -6.0
        want[O_QUANT] = -6.0
        np.testing.assert_allclose(d, np.broadcast_to(want, d.shape),
                                   rtol=0, atol=2e-5)
