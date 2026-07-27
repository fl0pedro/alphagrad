"""P1b: FacePathPolicy — one skip gate per face + one micro decision per
(face, slot), sample/evaluate mirrored for ratio-1."""
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from alphagrad.approx.heads import (
    AXIS_TAG_BITS,
    AxisTokenFeatures,
    FacePathPolicy,
    OP_END,
    precompute_factor_tables,
)

_N = 4
_F = 4
_S = 3


def _feats():
    tb = jnp.zeros((_N, AXIS_TAG_BITS)).at[:, 0].set(1.0)
    return AxisTokenFeatures(
        size=jnp.full((_N,), 4, dtype=jnp.int32),
        log_size=jnp.log(jnp.ones(_N) * 4),
        tag_bits=tb,
        group_id=jnp.zeros(_N, dtype=jnp.int32),
        valid_mask=jnp.array([1.0, 1.0, 1.0, 1.0]),
    )


def _policy(key):
    return FacePathPolicy(32, 4, max_faces=_F, num_slots=_S, key=key)


def _masks(n_valid_faces):
    pair = jnp.zeros((_F, _N, _N)).at[:, 0, 1].set(1.0).at[:, 1, 0].set(1.0)
    comp = jnp.zeros((_F, _N)).at[:, 0].set(1.0)
    valid = jnp.array(
        [1.0] * n_valid_faces + [0.0] * (_F - n_valid_faces)
    )
    return pair, comp, valid


def test_sample_evaluate_logp_match():
    """ratio-1: evaluate(sampled actions, same masks) == sample's log-prob."""
    pol = _policy(jrand.PRNGKey(0))
    tables = precompute_factor_tables(8)
    pair, comp, valid = _masks(2)
    ctx = jnp.ones((32,)) * 0.1
    fa, logp_s, ent_s, ar_s, _, _, _ = pol.sample(
        ctx, _feats(), tables, jrand.PRNGKey(1), pair, comp, valid
    )
    logp_e, ent_e, ar_e, _, _, _ = pol.evaluate(
        ctx, _feats(), tables, fa, pair, comp, valid
    )
    assert np.allclose(float(logp_s), float(logp_e), atol=1e-5)
    assert np.allclose(float(ent_s), float(ent_e), atol=1e-5)
    assert float(ar_s) == float(ar_e)


def test_padding_faces_contribute_nothing_and_are_canonical():
    pol = _policy(jrand.PRNGKey(2))
    tables = precompute_factor_tables(8)
    pair, comp, valid = _masks(1)
    ctx = jnp.zeros((32,))
    fa, logp1, _, ar1, _, _, _ = pol.sample(
        ctx, _feats(), tables, jrand.PRNGKey(3), pair, comp, valid
    )
    # padding faces: no skip, END everywhere
    assert np.all(np.asarray(fa.skip)[1:] == 0)
    assert np.all(np.asarray(fa.op_type)[1:] == OP_END)
    # a single valid face bounds the arity: 1 skip gate + at most 3 slots'
    # heads; with zero valid faces everything must be zero
    _, logp0, ent0, ar0, _, _, _ = pol.sample(
        ctx, _feats(), tables, jrand.PRNGKey(4), pair, comp, jnp.zeros(_F)
    )
    assert float(logp0) == 0.0 and float(ent0) == 0.0 and float(ar0) == 0.0
    assert float(ar1) > float(ar0)


def test_skipped_face_slots_are_forced_end():
    """Wherever the policy sampled skip=1, that face's slots must be END."""
    pol = _policy(jrand.PRNGKey(5))
    tables = precompute_factor_tables(8)
    pair, comp, valid = _masks(_F)
    ctx = jnp.zeros((32,))
    found_skip = False
    for seed in range(12):
        fa, *_ = pol.sample(
            ctx, _feats(), tables, jrand.PRNGKey(seed), pair, comp, valid
        )
        skip = np.asarray(fa.skip)
        ops = np.asarray(fa.op_type)
        for f in range(_F):
            if skip[f] == 1:
                found_skip = True
                assert np.all(ops[f] == OP_END), (
                    f"skip face {f} has non-END slots: {ops[f]}"
                )
    assert found_skip, "no skip sampled across 12 seeds (untrained ~50%)"
