"""--face-none-bias (ticket dsnn-3qm.44): the face-head init prior as a FLAG.

ALPHAGRAD_FACE_NONE_BIAS was an env var read inside ``apply_face_none_bias``.
Owner ruling: args only, no env vars for knobs, no fallback period. Pinned:

1. DEFAULTS OFF. ``--face-none-bias 0.0`` and ``--face-logit-clamp 0.0``.
   The clamp default moved from 15 to 0 ON PURPOSE (owner Q22/Q30); the
   campaign values are the launcher generator's business (ticket .43).
2. THE ANALYTIC INIT. Every Linear bias is 0 after ``init_linear_weights``,
   so on a ZERO context the head's 94 logits ARE its output bias, i.e.
   exactly the +B / -B the flag writes. Hence at bias B, per face
   ``p_skip = sigmoid(-B)`` and, with the three approximation ops legal,
   per slot ``p_none = e^B / (e^B + 3)``:
       B = 0 -> 0.5    / 0.25      B = 4 -> 0.018  / 0.948
       B = 6 -> 0.0025 / 0.993
   ``--scale-face-head`` is weight-only, so it leaves these untouched.
3. FLAG-OFF BIT-IDENTITY. ``build_and_init_agent`` at ``--face-none-bias 0``
   is leaf-for-leaf the pre-change init at env var unset: ``_build_agent``
   + ``apply_init_scheme`` with the bias step a no-op (it returned the agent
   unchanged at B = 0 before and does now). The env var, if set, no longer
   reaches the bias step at all. The two init pipelines (factory and
   ``ppo.main``'s inline copy) agree at B > 0 as well.
4. A SET ENV VAR IS REFUSED at startup with a message naming the flag, in
   both trainers that build agents through the factory.
5. THE GENERATOR passes ``--face-none-bias`` and never exports the env var.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import pathlib

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

# The smallest agent the suite already builds (TOTAL_V = 6, EMBD = 32).
from init_scheme_test import (                                  # noqa: E402
    EMBD, _agent, _assert_same, _ns, _raw_agent)

from alphagrad.approx.common.agent_factory import (             # noqa: E402
    REMOVED_ENV_KNOBS, apply_face_none_bias, refuse_removed_env_knobs)
from alphagrad.approx.unified_face_head import (                # noqa: E402
    FACE_SLOTS, NUM_APPROX_OPS, OP_NONE, O_SKIP, S_OP, _cat_logp_ent,
    slot_base)

_ALPHAGRAD = pathlib.Path(__file__).resolve().parents[1]
_ENV = "ALPHAGRAD_FACE_NONE_BIAS"


# ------------------------------------------------------------ 1. defaults

def test_parser_defaults_are_off():
    from alphagrad.approx.ppo import make_argparser
    ns = make_argparser().parse_args([])
    assert ns.face_none_bias == 0.0
    assert ns.face_logit_clamp == 0.0          # deliberate: was 15 (Q22/Q30)
    assert ns.scale_face_head == 0.0
    ns = make_argparser().parse_args(["--face-none-bias", "4"])
    assert ns.face_none_bias == 4.0


def test_az_parser_has_the_flag_and_the_default_is_off():
    from alphagrad.approx.az_args import make_argparser
    ns = make_argparser().parse_args([])
    assert ns.face_none_bias == 0.0
    assert make_argparser().parse_args(
        ["--face-none-bias", "6"]).face_none_bias == 6.0


# ------------------------------------------------------ 2. the analytic init

def _init_probs(bias, scale_face_head=0.0, seed=11):
    """(p_skip, [p_none per slot]) of the freshly built head on ctx = 0,
    through the head's OWN Bernoulli / masked-categorical arithmetic."""
    a = _agent(_ns(face_none_bias=float(bias),
                   scale_face_head=float(scale_face_head)), seed=seed)
    z = a.face_path_policy.head.logits(jnp.zeros((EMBD,)))
    assert z.shape == (1 + FACE_SLOTS * 31,)
    p_skip = float(jax.nn.sigmoid(z[O_SKIP]))
    all_ops_legal = jnp.ones((NUM_APPROX_OPS,))
    p_none = []
    for s in range(FACE_SLOTS):
        op = z[slot_base(s) + S_OP: slot_base(s) + S_OP + NUM_APPROX_OPS]
        logp, _ = _cat_logp_ent(op, all_ops_legal, OP_NONE)
        p_none.append(float(jnp.exp(logp)))
    return p_skip, p_none


@pytest.mark.parametrize("bias,skip_expected,none_expected", [
    (0.0, 0.5, 0.25),
    (4.0, 0.018, 0.948),
    (6.0, 0.0025, 0.993),
])
@pytest.mark.parametrize("scale_face_head", [0.0, 0.1])
def test_init_skip_and_none_probabilities_are_analytic(
        bias, skip_expected, none_expected, scale_face_head):
    p_skip, p_none = _init_probs(bias, scale_face_head)
    want_skip = 1.0 / (1.0 + np.exp(bias))                 # sigmoid(-B)
    want_none = np.exp(bias) / (np.exp(bias) + 3.0)        # e^B / (e^B + 3)
    assert p_skip == pytest.approx(want_skip, rel=1e-5), (bias, p_skip)
    assert p_skip == pytest.approx(skip_expected, abs=1e-4)
    for s, p in enumerate(p_none):
        assert p == pytest.approx(want_none, rel=1e-5), (bias, s, p)
        assert p == pytest.approx(none_expected, abs=1e-3)


def test_bias_moves_exactly_four_logits():
    ns0 = _ns()
    raw, init_key = _raw_agent(ns0)
    from alphagrad.approx.ppo import apply_init_scheme
    base = apply_init_scheme(raw, init_key, ns0)
    b0 = np.asarray(base.face_path_policy.head.proj.layers[-1].bias)
    b4 = np.asarray(apply_face_none_bias(base, 4.0)
                    .face_path_policy.head.proj.layers[-1].bias)
    touched = {O_SKIP} | {slot_base(s) + S_OP + OP_NONE
                          for s in range(FACE_SLOTS)}
    assert b4[O_SKIP] - b0[O_SKIP] == pytest.approx(-4.0)
    for s in range(FACE_SLOTS):
        i = slot_base(s) + S_OP + OP_NONE
        assert b4[i] - b0[i] == pytest.approx(4.0)
    keep = [i for i in range(len(b0)) if i not in touched]
    np.testing.assert_array_equal(b4[keep], b0[keep])


# ------------------------------------------------- 3. flag-off bit-identity

def test_flag_off_is_bitwise_the_pre_change_init(monkeypatch):
    """Pre-change, env var unset: _build_agent -> apply_init_scheme ->
    apply_face_none_bias(no-op). That pipeline is untouched; the flag at 0
    must reproduce it leaf for leaf, whatever the environment says."""
    from alphagrad.approx.ppo import apply_init_scheme
    ns = _ns()
    assert ns.face_none_bias == 0.0
    raw, init_key = _raw_agent(ns)
    want = apply_init_scheme(raw, init_key, ns)
    _assert_same(_agent(ns), want, "factory at --face-none-bias 0")
    # B = 0 returns the SAME object, as the env-unset path always did.
    assert apply_face_none_bias(want, 0.0) is want
    # The env var no longer reaches the bias step.
    monkeypatch.setenv(_ENV, "6")
    assert apply_face_none_bias(want, 0.0) is want
    _assert_same(_agent(ns), want, "env var set but not read")


def test_factory_and_inline_pipelines_agree_at_positive_bias():
    """ppo.main's inline copy is _build_agent + apply_init_scheme +
    apply_face_none_bias(B); the factory must produce the same leaves."""
    from alphagrad.approx.ppo import apply_init_scheme
    ns = _ns(face_none_bias=4.0)
    raw, init_key = _raw_agent(ns)
    inline = apply_face_none_bias(apply_init_scheme(raw, init_key, ns), 4.0)
    _assert_same(_agent(ns), inline, "factory vs inline at B=4")


# ------------------------------------------------- 4. the env var is refused

def test_set_env_var_is_refused_with_the_flag_named(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert refuse_removed_env_knobs() is None
    monkeypatch.setenv(_ENV, "6")
    with pytest.raises(SystemExit) as e:
        refuse_removed_env_knobs()
    msg = str(e.value)
    assert _ENV in msg and "--face-none-bias" in msg
    # an EMPTY export is still an export
    monkeypatch.setenv(_ENV, "")
    with pytest.raises(SystemExit):
        refuse_removed_env_knobs()
    assert REMOVED_ENV_KNOBS[_ENV] == "--face-none-bias"


def test_both_trainers_call_the_refusal_before_building():
    from alphagrad.approx import ppo
    main_src = inspect.getsource(ppo.main)
    assert "refuse_removed_env_knobs()" in main_src
    assert main_src.index("refuse_removed_env_knobs()") \
        < main_src.index("_build_agent(")
    # az_gumbel runs at import, so read its source as text.
    az = (_ALPHAGRAD / "src/alphagrad/approx/az_gumbel.py").read_text()
    assert "refuse_removed_env_knobs()" in az
    assert az.index("refuse_removed_env_knobs()") \
        < az.index("agent = build_and_init_agent(")
    assert "_ns.face_none_bias = float(A.face_none_bias)" in az


def test_nothing_under_src_reads_the_env_var():
    hits = []
    for f in (_ALPHAGRAD / "src").rglob("*.py"):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if _ENV in line and "environ" in line:
                hits.append(f"{f}:{n}")
    assert not hits, hits


# ------------------------------------------------------- 5. the generator

def _gen():
    path = _ALPHAGRAD / "tools/gen_fq_launchers.py"
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_passes_the_flag_and_never_the_env_var():
    g = _gen()
    assert "--face-none-bias" in g.REQUIRED_FLAGS
    with_flag = []
    for a in g.ARMS:
        assert _ENV not in a.get("env", {}), a["name"]
        text = g.render(a)
        assert f"export {_ENV}" not in text, a["name"]
        if "--face-none-bias" in a.get("cli", {}):
            with_flag.append(a["name"])
            assert f"  --face-none-bias {a['cli']['--face-none-bias']}" \
                in text, a["name"]
    # the four wave-1 arms carry literal biases; waves 2-4 inherit $W1_BIAS
    assert {"w1a_bias6_lam170", "w1b_bias5_lam170", "w1c_bias4_lam170",
            "w1d_bias5_lam16"} <= set(with_flag)
    assert len(with_flag) >= 16, with_flag
