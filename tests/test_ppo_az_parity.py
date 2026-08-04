"""PPO vs Gumbel-AZ parity: same backbone, same init, same objective.

Regression guard for the 2026-08-04 audit, which found PPO at 161,916 params and
AZ at 1,171,693 (7.2x) because each trainer fed ``_build_agent`` a different args
namespace, and AZ additionally skipped ``init_linear_weights`` /
``_scale_output_heads`` — leaving its initial vertex logits at full scale with
non-zero biases, i.e. a biased Gumbel root prior.

These tests assert the three properties that make a PPO-vs-AZ result mean
"the search/update differs" rather than "the networks differ".
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common.agent_factory import (  # noqa: E402
    ALGO_HEAD_FIELDS, POLICY_ARCH_DEFAULTS, apply_policy_arch, az_w4,
    build_and_init_agent, channel_weights, derive_agent_keys)
from alphagrad.approx import ppo as P  # noqa: E402

TOTAL_V = 28
SEED = 7


def _ns(**algo_heads):
    ns = P.make_argparser().parse_args([])
    apply_policy_arch(ns, **algo_heads)
    return ns


PPO_HEADS = dict(
    dynamic_substeps=True, unified_head=False, no_approx_head=False,
    face_actions=True, unified_face_head=True, live_faces=True,
    max_substeps=16, axis_group_embedding=False,
)
AZ_HEADS = dict(
    dynamic_substeps=True, unified_head=False, no_approx_head=False,
    face_actions=False, unified_face_head=False, live_faces=False,
    max_substeps=1, axis_group_embedding=False,
)


def _build(heads):
    ns = _ns(**heads)
    tbl, tpy, nfac, mrules = P._build_factor_table(ns)
    return build_and_init_agent(ns, TOTAL_V, nfac, mrules, seed=SEED)


@pytest.fixture(scope="module")
def agents():
    return _build(PPO_HEADS), _build(AZ_HEADS)


def _shared_modules(agent):
    """The parts that MUST be identical: backbone, vertex head, value heads."""
    return {
        "embedding": agent.embedding,
        "encoder": agent.encoder,
        "vertex_policy": agent.vertex_policy,
        "value_head_flops": agent.value_head_flops,
        "value_head_mem": agent.value_head_mem,
        "value_head_cos": agent.value_head_cos,
    }


def _nparams(tree):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(
        eqx.filter(tree, eqx.is_array)))


def test_shared_modules_have_identical_structure(agents):
    ppo_a, az_a = agents
    for name, (m_p, m_a) in {
        k: (v, _shared_modules(az_a)[k])
        for k, v in _shared_modules(ppo_a).items()
    }.items():
        sp = jax.tree_util.tree_structure(eqx.filter(m_p, eqx.is_array))
        sa = jax.tree_util.tree_structure(eqx.filter(m_a, eqx.is_array))
        assert sp == sa, f"{name}: pytree structure differs"
        shapes_p = [x.shape for x in jax.tree_util.tree_leaves(
            eqx.filter(m_p, eqx.is_array))]
        shapes_a = [x.shape for x in jax.tree_util.tree_leaves(
            eqx.filter(m_a, eqx.is_array))]
        assert shapes_p == shapes_a, f"{name}: leaf shapes differ"


def test_same_config_same_seed_is_bitwise_identical():
    """Determinism: identical config + identical seed => identical weights.

    NOTE the limit of what is achievable today. init_linear_weights does
    jax.random.split(key, len(weights)) over the Linear modules in TREE
    ORDER (common/init.py:41), so the number of Linears decides every module's
    subkey. While PPO carries a face head and AZ a micro head, the SHARED
    modules necessarily draw different subkeys — bitwise cross-arm identity is
    impossible without changing PPO's init (and thus every existing PPO
    baseline). What matters for the comparison is asserted by the two tests
    below: identical shapes/structure, and the same init DISTRIBUTION. Once W5
    gives both arms the same head inventory, tighten this to a cross-arm
    bitwise check.
    """
    a1 = _build(AZ_HEADS)
    a2 = _build(AZ_HEADS)
    for x, y in zip(jax.tree_util.tree_leaves(eqx.filter(a1, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(a2, eqx.is_array))):
        assert jnp.array_equal(x, y)


def test_shared_modules_share_the_init_distribution(agents):
    """Same init scheme (orthogonal gain sqrt2, zero bias) on both arms."""
    ppo_a, az_a = agents
    for name, m_p in _shared_modules(ppo_a).items():
        m_a = _shared_modules(az_a)[name]
        lp = jax.tree_util.tree_leaves(eqx.filter(m_p, eqx.is_array))
        la = jax.tree_util.tree_leaves(eqx.filter(m_a, eqx.is_array))
        for a, b in zip(lp, la):
            if a.size < 16:
                continue
            sa, sb = float(jnp.std(a)), float(jnp.std(b))
            assert abs(sa - sb) <= 0.25 * max(sa, sb, 1e-6), (
                f"{name}: init spread differs ({sa:.4f} vs {sb:.4f})")


def test_only_approximation_heads_differ(agents):
    """Total params may differ ONLY by the approximation heads."""
    ppo_a, az_a = agents
    shared_p = _nparams(_shared_modules(ppo_a))
    shared_a = _nparams(_shared_modules(az_a))
    assert shared_p == shared_a, (
        f"shared backbone/vertex/value params differ: {shared_p} vs {shared_a}")
    # PPO carries the face head and no micro head; AZ the reverse (until W5).
    assert ppo_a.face_path_policy is not None and ppo_a.micro_action_policy is None
    assert az_a.face_path_policy is None and az_a.micro_action_policy is not None


def test_initial_vertex_logits_are_near_uniform(agents):
    """The Gumbel root prior must be ~unbiased at init.

    Produced by three things together: zeroed Linear biases (init_linear_weights),
    x0.1 on the pointer scoring projection, and the five zeroed additive context
    paths (_scale_output_heads). AZ ran none of them before this fix.
    """
    for agent in agents:
        toks = jnp.arange(1, 65, dtype=jnp.int32)
        eqn_ids = jnp.zeros_like(toks)
        logits, _ctx, _v = agent.encode(
            toks, eqn_ids=eqn_ids, key=jax.random.PRNGKey(0))
        finite = logits[jnp.isfinite(logits)]
        assert float(jnp.max(jnp.abs(finite))) < 0.5, (
            f"initial |logit| too large: {float(jnp.max(jnp.abs(finite)))}")
        assert float(jnp.std(finite)) < 0.2, (
            f"initial logit std too large: {float(jnp.std(finite))}")


def test_objective_weights_match(agents):
    """Both arms must optimize the same scalarization (audit: PPO cos x2, AZ x1)."""
    ns = _ns(**PPO_HEADS)
    ns.lambda_cmp, ns.lambda_mem, ns.lambda_acc = 1.0, 1.0, 1.0
    ns.rewards = ["cmp", "mem", "acc"]
    w = channel_weights(ns)
    assert w == (1.0, 1.0, 1.0)
    # PPO's own builder must agree with the shared helper.
    ppo_w = np.asarray(P._build_head_weights(ns), dtype=np.float64)
    assert np.allclose(ppo_w, np.asarray(w)), (ppo_w, w)
    # AZ's 4-vector is the same weights with the cost channels negated.
    assert np.allclose(np.abs(az_w4(ns)), np.array([1.0, 1.0, 0.0, 1.0]))


def test_arch_defaults_cover_every_shape_bearing_field():
    """A field that changes parameter shapes must live in POLICY_ARCH_DEFAULTS,
    not be inherited from an argparser default — that inheritance is exactly how
    the two arms drifted apart."""
    for f in ("vocab_size", "embd_dim", "num_heads", "num_layers", "hidden_dim",
              "op_embd_dim", "value_dims", "set_pointer", "set_pointer_blocks"):
        assert f in POLICY_ARCH_DEFAULTS, f
    for f in ALGO_HEAD_FIELDS:
        assert f not in POLICY_ARCH_DEFAULTS, (
            f"{f} is algorithm-specific and must be set explicitly per trainer")


def test_key_derivation_is_ppo_s():
    a1, i1 = derive_agent_keys(SEED)
    key = jax.random.PRNGKey(SEED)
    key, _ = jax.random.split(key)
    a2, i2, _ = jax.random.split(key, 3)
    assert jnp.array_equal(a1, a2) and jnp.array_equal(i1, i2)
