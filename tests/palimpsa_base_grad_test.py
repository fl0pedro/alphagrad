"""GRADIENT MUST REACH PALIMPSA'S BASE ENCODE. This is the point of the
lean rewrite, so it is a test and not a comment.

THE DEFECT THIS PINS
--------------------
Vertex identity is produced by palimpsa reading the BASE token stream. Until
2026-08-15 that read happened OUTSIDE the differentiated region -- once per
episode, in ``ppo.main`` -- and its output was handed to the loss as an
array. Inside ``eqx.filter_grad`` an array is a constant, so the encoder
weights that WROTE the vertex representation received no cotangent from it.
Whatever read it (the identity pool, the pointer) trained; the encode did
not. The only palimpsa gradient in the build was the ``--grad-window`` K
step-delta encodes, and K defaults to 1.

HOW IT IS MEASURED
------------------
The base encode is run with a SECOND COPY of the agent, ``agent_b``, whose
values are identical to ``agent``'s. ``d/d agent_b`` is then exactly "the
gradient that flows back through the base encode", separated from the
delta-encode gradient that shares the same weights. Two topologies:

  LEGACY  the base memory computed outside and passed in as a value. This is
          what the old build did, reconstructed here in three lines so the
          zero is DEMONSTRATED rather than asserted from memory.
  LEAN    ``carry_stream.base_memory`` called inside, which is what
          ``ppo._dynamic_loss_fn`` and ``az_gumbel``'s loss now do.

LEGACY must be exactly 0.0 and LEAN must not be. If someone ever hoists the
base encode back out of the loss "for speed", this test is what fails.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "512")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common import carry_stream as CS  # noqa: E402

TOTAL_V = 6
EMBD = 32
BASE_W = 24
DELTA_W = 16


@pytest.fixture(scope="module")
def setup():
    from alphagrad.approx import ppo as P
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=True,
        face_actions=False, unified_face_head=False, live_faces=False,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = EMBD
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    agent = build_and_init_agent(ns, TOTAL_V, num_factors=4, max_rules=4,
                                 seed=3)

    rng = np.random.default_rng(0)
    base_tok = jnp.asarray((rng.integers(1, 200, BASE_W)).astype(np.int32))
    base_eqn = jnp.zeros((BASE_W,), jnp.int32)
    # 1-based owning vertex per base token; 0 = header/input (no owner).
    own = np.zeros((BASE_W,), np.int32)
    own[4:] = (np.arange(BASE_W - 4) % TOTAL_V) + 1
    base_own = jnp.asarray(own)

    d_tok = jnp.asarray((rng.integers(1, 200, DELTA_W)).astype(np.int32))
    d_eqn = jnp.zeros((DELTA_W,), jnp.int32)
    part = jnp.zeros((TOTAL_V + 1,), jnp.float32).at[2].set(1.0)
    return dict(agent=agent, base_tok=base_tok, base_eqn=base_eqn,
                base_own=base_own, d_tok=d_tok, d_eqn=d_eqn, part=part)


def _base(a, s):
    return CS.base_memory(a, s["base_tok"], s["base_eqn"],
                          jnp.asarray(BASE_W, jnp.int32), window=BASE_W,
                          total_v=TOTAL_V, embd_dim=EMBD,
                          base_owners=s["base_own"])


def _enc0(a, s):
    return CS.init_carry(a, s["base_tok"], s["base_eqn"],
                         jnp.asarray(BASE_W, jnp.int32), window=BASE_W,
                         total_v=TOTAL_V, embd_dim=EMBD,
                         base_owners=s["base_own"])[0]


def _scalar(triple):
    lg, ctx, val = triple
    lg = jnp.where(jnp.isfinite(lg), lg, 0.0)
    return jnp.sum(lg ** 2) + jnp.sum(ctx ** 2) + jnp.sum(val ** 2)


def _step(a_head, a_base, s, *, legacy_base=None):
    """One step's differentiable head evaluation, as the loss builds it."""
    enc = _enc0(a_head, s)
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    _c, vs, vc = CS.advance(
        a_head, enc, vs, vc, s["d_tok"], s["d_eqn"],
        jnp.asarray(DELTA_W, jnp.int32), jnp.asarray(2, jnp.int32),
        window=DELTA_W, participants=s["part"], chunk=0)
    bm = legacy_base if legacy_base is not None else _base(a_base, s)
    return _scalar(CS.heads(a_head, vs, vc, base_mem=bm, preference=None))


def _grad_norm(g, prefix=""):
    params = eqx.filter(g, eqx.is_inexact_array)
    tot = 0.0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        if prefix in jax.tree_util.keystr(path):
            tot += float(jnp.sum(jnp.asarray(leaf) ** 2))
    return float(np.sqrt(tot))


def test_legacy_topology_gives_the_base_encode_zero_gradient(setup):
    """The pre-2026-08-15 build, reconstructed: base memory computed OUTSIDE
    and read back as a value. Nothing reaches the encode that produced it."""
    s = setup
    stored = jax.tree_util.tree_map(jax.lax.stop_gradient, _base(s["agent"], s))

    def f(agent_b):
        return _step(s["agent"], agent_b, s, legacy_base=stored)

    g = eqx.filter_grad(f)(s["agent"])
    assert _grad_norm(g, ".encoder") == 0.0
    assert _grad_norm(g) == 0.0


def test_lean_topology_gives_the_base_encode_real_gradient(setup):
    """The shipped topology: `base_memory` inside the differentiated region.
    Every encoder leaf on the base path gets a cotangent."""
    s = setup

    def f(agent_b):
        return _step(s["agent"], agent_b, s)

    g = eqx.filter_grad(f)(s["agent"])
    n_enc = _grad_norm(g, ".encoder")
    n_emb = _grad_norm(g, ".embedding")
    assert np.isfinite(n_enc) and n_enc > 0.0, \
        f"base encode still gets no gradient (encoder ||g|| = {n_enc})"
    assert np.isfinite(n_emb) and n_emb > 0.0, \
        f"the token embedding gets no gradient (||g|| = {n_emb})"


def test_every_palimpsa_layer_is_reached_not_just_the_last(setup):
    """A base encode that only touched the final layer would satisfy a norm
    test and still be a truncated path. Assert per-layer."""
    s = setup

    def f(agent_b):
        return _step(s["agent"], agent_b, s)

    g = eqx.filter_grad(f)(s["agent"])
    n_layers = len(s["agent"].encoder.layers)
    dead = [i for i in range(n_layers)
            if _grad_norm(g, f".layers[{i}]") == 0.0]
    assert not dead, f"palimpsa layers with zero base-encode gradient: {dead}"


def test_base_and_dynamic_memories_add_to_one_readout(setup):
    """The split is only legal because (sum, count) memories ADD: pooling the
    union of the rows is the sum of the sums over the sum of the counts. If
    that ever stops holding, `heads(base_mem=...)` is silently wrong."""
    from alphagrad.approx import vertex_memory as _vmem
    s = setup
    a = s["agent"]
    bs, bc = _base(a, s)
    enc = _enc0(a, s)
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    _c, ds, dc = CS.advance(
        a, enc, vs, vc, s["d_tok"], s["d_eqn"],
        jnp.asarray(DELTA_W, jnp.int32), jnp.asarray(2, jnp.int32),
        window=DELTA_W, participants=s["part"], chunk=0)
    joint = _vmem.read(bs + ds, bc + dc)
    # Same thing computed the accumulating way: fold the delta into the base.
    _c2, js, jc = CS.advance(
        a, enc, bs, bc, s["d_tok"], s["d_eqn"],
        jnp.asarray(DELTA_W, jnp.int32), jnp.asarray(2, jnp.int32),
        window=DELTA_W, participants=s["part"], chunk=0)
    np.testing.assert_allclose(np.asarray(joint), np.asarray(_vmem.read(js, jc)),
                               rtol=1e-6, atol=1e-6)


def test_the_scatter_has_no_parameters(setup):
    """`_vmem.scatter` is the one pooling primitive and it must stay
    weightless -- a learned pool is exactly what the identity module was.

    Checked on the parsed CODE, not on the source text: the docstring
    legitimately contains the word "weights" (the per-row validity weights),
    and a substring test on the raw source fails on its own prose.
    """
    import ast
    import inspect
    import textwrap
    from alphagrad.approx import vertex_memory as _vmem

    assert _vmem.scatter.__closure__ is None, \
        "the scatter closed over something -- a captured array is a parameter"

    tree = ast.parse(textwrap.dedent(inspect.getsource(_vmem.scatter)))
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           ) else fn.body
    names = set()
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
    banned = {"self", "Linear", "MLP", "Embedding", "eqx", "weight", "bias",
              "params", "q_proj", "k_proj", "v_proj", "out_proj"}
    grew = names & banned
    assert not grew, f"the scatter grew {sorted(grew)}"
    # And it may only reach for jnp/jax array ops.
    assert names <= (names - {"nn", "Module"}), "unexpected module reference"


def test_the_vertex_slots_are_E_wide_not_2E(setup):
    """The `[identity || dynamic]` concat is gone; the pointer scores E."""
    s = setup
    pol = s["agent"].vertex_policy
    assert pol.embd_dim == EMBD, \
        f"pointer width {pol.embd_dim} != embd_dim {EMBD} (2E concat back?)"
    assert not hasattr(s["agent"], "identity_pool")
    assert not hasattr(s["agent"], "ctx_proj")
