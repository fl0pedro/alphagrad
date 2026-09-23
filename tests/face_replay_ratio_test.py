"""Sampler and replay must score the SAME face decision (dsnn-dfw.95).

The face head's log-prob is stored at rollout time by
``UnifiedFacePolicy.sample_face`` and rebuilt at loss time by
``UnifiedFacePolicy.evaluate_face``. The PPO ratio is 1 at epoch 0 only when
the two agree on every live slot of every live face.

They stopped agreeing on the recurrent target under ``--face-logit-clamp``.
Measured on RSNN_SHD / bptt, seed 250197, 3 episodes: the face log ratio of
every sample that requested an approximation read -9.2 to -9.4 (a ratio of
1e-4), while samples whose slots all held OP_NONE read -0.05. The head's
bound is ``C*tanh(z/C)``; on the replay side the projection and the bound
fused and the division by C was lost, so the replay scored ``C*tanh(z)`` --
the OP_NONE bias of +5.93 became the rail +15.0, every approximation logit
sat 15 nats below it, and the plans that actually approximated left the
policy gradient.

Two things are pinned here:

  * the LAYOUT case -- the face layout of both targets, every op type and a
    skip, sampler against replay to 1e-5. This is the codec and the masks;
  * the BOUND case -- the same head logits read directly and read through
    the ``lax.scan`` shape ``Agent._face_replay`` scores faces in. This is
    the shape the defect lived in.
"""
import os

os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import equinox as eqx                                           # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as E                                # noqa: E402
from alphagrad.approx.common.masks import (                     # noqa: E402
    set_diag_per_face, set_per_face_masks, set_reduce_axis_space)
from alphagrad.approx.cpu_approx_worker import (                # noqa: E402
    _build_env_from_args)
from alphagrad.approx.face_action import FaceAction             # noqa: E402
from alphagrad.approx.heads import (                            # noqa: E402
    AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
from alphagrad.approx.live_faces import LiveFaceStream          # noqa: E402
from alphagrad.approx.unified_face_head import (                # noqa: E402
    LOGIT_CLAMP, OP_BLOCKDIAG, OP_NONE, OP_REDUCE, O_QUANT, O_SKIP,
    S_OP, set_logit_clamp, slot_base)
from alphagrad.approx.unified_face_policy import (              # noqa: E402
    UnifiedFacePolicy)

CLAMP = 15.0
N_AX = E.MAX_AXES_PER_VERTEX
OPS = (OP_BLOCKDIAG, OP_REDUCE)

# The two targets the thesis matrix runs the face head on: the feed-forward
# one it always agreed on and the recurrent one it did not.
TARGETS = (
    ("NN256", dict(example="NeuralNetwork", dataset="none",
                   hidden_dim=256, vocab_size=256, num_layers=3)),
    ("RSNN_SHD_bptt", dict(example="RSNN_SHD", dataset="none",
                           temporal_rule="bptt", hidden_dim=256,
                           vocab_size=256, num_layers=3)),
)


@pytest.fixture(autouse=True)
def _row_settings():
    """The thesis row's mask settings and logit bound, restored after."""
    from alphagrad.approx.common import masks as _M
    keep = LOGIT_CLAMP[0]
    masks = (_M._PER_FACE_MASKS[0], _M._PER_FACE_REPAIR_AXIS[0])
    diag = (_M._DIAG_PER_FACE[0], _M._DIAG_PER_FACE_RULE[0],
            _M._DIAG_PER_FACE_REPAIR_PAIR[0])
    space = _M._REDUCE_AXIS_SPACE[0]
    set_logit_clamp(CLAMP)
    set_diag_per_face(False)
    set_per_face_masks(True)
    set_reduce_axis_space("physical")
    try:
        yield
    finally:
        set_logit_clamp(keep)
        set_diag_per_face(*diag)
        set_per_face_masks(*masks)
        set_reduce_axis_space(space)


def _neutral_features(n=N_AX):
    """The per-vertex features the per-face masks override every size in."""
    sz = jnp.ones((n,), jnp.int32)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.zeros((n,), jnp.float32),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _live_face_layout(cfg):
    """One live face of a real target: ``(sizes, quant, pair, comp)`` per
    slot, exactly the arrays ``face_slot_legality`` hands the trainer."""
    d = dict(dataset_size=1, cmp_type="latency", mem_type="peak_memory",
             seed=250197, face_actions=True, per_face=True, rewards=["cmp"],
             intermediate_rewards=False)
    d.update(cfg)
    env = _build_env_from_args(d, None)
    cj = env.config.jaxpr
    jx = cj.jaxpr if hasattr(cj, "jaxpr") else cj
    lits = list(cj.literals) if hasattr(cj, "literals") else list(env.consts)
    total_v = len(jx.eqns)
    bound = int(E.derived_max_faces(env.config.jaxpr, env.config.argnums,
                                    env.consts, env.args))
    lf = LiveFaceStream(jx, tuple(env.config.argnums), lits, list(env.args),
                        vocab=256, max_faces=bound, max_axes=N_AX)
    rev = list(range(total_v, 0, -1))
    specs = -np.ones((total_v, 3, 3), np.int32)
    for k in (total_v - 1, total_v // 2, total_v // 3, 2 * total_v // 3):
        order = np.zeros((total_v,), np.int32)
        order[:k] = np.asarray(rev[:k], np.int32)
        sizes, quant, pair, comp, _nout, nf = lf.face_slot_legality(
            order, specs, k, rev[k])
        if int(nf) > 0:
            return bound, (jnp.asarray(sizes[0]), jnp.asarray(quant[0]),
                           jnp.asarray(pair[0]), jnp.asarray(comp[0]))
    raise AssertionError(f"no live face found on {cfg['example']}")


def _policy(bound, embd=32):
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(embd, num_heads=2, max_faces=bound,
                            key=jrand.PRNGKey(0), approx_add="lossless")
    return pol, tables


def _biased(pol, op=None, skip=False, quant=False):
    """The same head with one op (and optionally SKIP) pushed up, so a
    handful of draws covers every class instead of waiting for a 1-in-378
    event at the thesis row's identity init."""
    bias = pol.head.proj.layers[-1].bias
    if op is not None:
        for s in range(pol.n_slots):
            bias = bias.at[slot_base(s, pol.layout) + S_OP + op].add(6.0)
            bias = bias.at[slot_base(s, pol.layout) + S_OP + OP_NONE].add(-6.0)
    if skip:
        bias = bias.at[O_SKIP].add(6.0)
    if quant:
        bias = bias.at[O_QUANT].add(6.0)
    return eqx.tree_at(
        lambda p: p.head.proj.layers[-1].bias, pol, bias)


def _face_action(row, skip, F):
    def _pad(v):
        v = jnp.asarray(v)
        return jnp.zeros((F,) + tuple(v.shape), v.dtype).at[0].set(v)
    return FaceAction(
        skip=jnp.zeros((F,), jnp.int32).at[0].set(jnp.asarray(skip)),
        quant=_pad(row["quant"]),
        op_type=_pad(row["op_type"]), i=_pad(row["i"]), j=_pad(row["j"]),
        exponents=_pad(row["exponents"]), factor=_pad(row["factor"]),
        compress_kind=_pad(row["compress_kind"]),
        quant_dtype=_pad(row["quant_dtype"]),
        quant_scale_sign=_pad(row["quant_scale_sign"]),
        quant_scale_frac=_pad(row["quant_scale_frac"]))


@pytest.mark.parametrize("name,cfg", TARGETS, ids=[t[0] for t in TARGETS])
def test_the_replay_scores_the_sampled_face(name, cfg):
    """Every live slot of a real face layout: replay == sampler to 1e-5,
    with every op class and a skip among the draws."""
    bound, (sizes, quant, pair, comp) = _live_face_layout(cfg)
    pol, tables = _policy(bound)
    feats = _neutral_features()
    ctx = jnp.asarray(np.linspace(-0.5, 0.5, pol.embd_dim, dtype=np.float32))
    seen = {op: 0 for op in OPS}
    seen_quant = 0
    skips = 0
    worst = 0.0
    cases = ([(op, False, False) for op in OPS]
             + [(None, True, False), (None, False, True)])
    for want_op, want_skip, want_quant in cases:
        p = _biased(pol, op=want_op, skip=want_skip, quant=want_quant)
        for k in range(24):
            sk, row, lp, ent, _ar, _sp, _od = p.sample_face(
                feats, tables,
                jrand.PRNGKey(7919 * (want_op or (9 + 4 * want_quant)) + k),
                0, pair, comp, jnp.asarray(1.0), face_context=ctx,
                face_sizes_f=sizes, face_quant_f=quant)
            lp2, ent2 = p.evaluate_face(
                feats, tables, _face_action(row, sk, bound), 0, pair, comp,
                jnp.asarray(1.0), face_context=ctx, face_sizes_f=sizes,
                face_quant_f=quant)[:2]
            worst = max(worst, abs(float(lp) - float(lp2)),
                        abs(float(ent) - float(ent2)))
            skips += int(sk)
            seen_quant += int(row["quant"])
            if int(sk) == 0:
                # the wire's op codes for DIAG and COMPRESS coincide with the
                # head's blockdiag and reduce
                for o in np.asarray(row["op_type"]).tolist():
                    if int(o) in seen:
                        seen[int(o)] += 1
    assert worst < 1e-5, (name, worst)
    assert skips > 0, name
    om, qm = _op_legal(pol, feats, tables, sizes, quant, pair, comp)
    for op in OPS:
        # A class the target's own legality forbids cannot be drawn, and a
        # forced draw would score a rule the engine refuses. Coverage is
        # therefore "every class this layout admits".
        assert seen[op] > 0 or float(om[op]) == 0.0, (name, op, seen)
    assert seen_quant > 0 or float(qm) == 0.0, (name, seen_quant, float(qm))


def _op_legal(pol, feats, tables, sizes, quant, pair, comp):
    out = pol._face_masks(
        [pol._face_feats_1(feats, sizes[s]) for s in range(pol.n_slots)],
        pair, comp, quant, None, tables)
    return jnp.max(out[0], axis=0), out[5]


@pytest.mark.parametrize("name,cfg", TARGETS, ids=[t[0] for t in TARGETS])
def test_the_bound_is_the_same_through_the_replays_scan(name, cfg):
    """``C*tanh(z/C)`` must not change when the head is read face by face
    inside the blocked ``lax.scan`` ``Agent._face_replay`` scores faces in.

    The bound's own arithmetic is the whole point: at the thesis row's
    identity init the OP_NONE logit is +5.93 raw and +5.64 bounded, and a
    replay that read +15.0 instead put every approximation 15 nats down.
    The latents arrive as an ARGUMENT, not a constant, so the projection
    that feeds the bound is a real dot and not a folded literal.
    """
    bound, (sizes, quant, pair, comp) = _live_face_layout(cfg)
    pol, _tables = _policy(bound)
    F, B, K = 128, 8, 64
    key = jrand.PRNGKey(11)
    lat = jrand.normal(key, (B, F, pol.embd_dim), jnp.float32) * 1.5

    def _sum_scanned(rows):
        """The replay's shape: a blocked scan with a per-block cond."""
        def _blk(acc, b):
            def _run(a):
                def _inner(x, k):
                    f = jnp.minimum(b * K + k, F - 1)
                    return x + pol.head.logits(pol._repr(rows[f])), None
                return jax.lax.scan(
                    _inner, a, jnp.arange(K, dtype=jnp.int32))[0]
            return jax.lax.cond(b < 2, _run, lambda a: a, acc), None
        out, _ = jax.lax.scan(
            _blk, jnp.zeros((pol.layout.width,), jnp.float32),
            jnp.arange(-(-F // K), dtype=jnp.int32))
        return out

    def _sum_direct(rows):
        z = jax.vmap(lambda r: pol.head.logits(pol._repr(r)))(rows)
        return jnp.sum(z, axis=0)

    scanned = jax.jit(jax.vmap(_sum_scanned))(lat)
    direct = jax.jit(jax.vmap(_sum_direct))(lat)
    assert float(jnp.max(jnp.abs(direct - scanned))) < 1e-3, (
        name, float(jnp.max(jnp.abs(direct - scanned))))
    # And the bound really is C*tanh(z/C), not C*tanh(z): the raw OP_NONE
    # bias of the thesis init must land at 5.64, not at the rail.
    z = pol.head.logits(jnp.zeros((pol.embd_dim,), jnp.float32))
    raw = pol.head.proj(jnp.zeros((pol.embd_dim,), jnp.float32))
    want = CLAMP * np.tanh(np.asarray(raw, np.float64) / CLAMP)
    assert float(jnp.max(jnp.abs(jnp.asarray(want, jnp.float32) - z))) < 1e-4
