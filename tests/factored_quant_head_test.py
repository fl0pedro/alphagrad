"""The FactoredQuantHead samples a real dtype by parts, scores it consistently,
and skips forced factors (the ≥2-options rule)."""
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from alphagrad.approx.heads import FactoredQuantHead
from alphagrad.approx.quant_factoring import quant_factor_tables
from graphax.sparse.micro_actions import QUANT_DTYPES

E = 16


def _head():
    return FactoredQuantHead(E, key=jrand.PRNGKey(0))


def test_sample_always_yields_a_usable_dtype_and_valid_sign():
    head = _head()
    usable = set(np.asarray(quant_factor_tables().to_global).tolist())
    s = jnp.zeros((E,))
    for seed in range(60):
        di, sgn = head.sample(s, jrand.PRNGKey(seed))
        assert int(di) in usable, QUANT_DTYPES[int(di)]
        assert int(sgn) in (1, -1)


def test_log_prob_is_finite_and_arity_bounded():
    head, s = _head(), jnp.zeros((E,))
    di, sgn = head.sample(s, jrand.PRNGKey(1))
    lp, ent, ar = head.log_prob(s, di, sgn)
    assert np.isfinite(float(lp)) and np.isfinite(float(ent))
    assert 1.0 <= float(ar) <= float(1 + 7)  # sign always + up to 7 factors


def test_integer_dtype_skips_more_factors_than_a_contested_float():
    """int8 forces exp/mant/bias/finite/uz (skipped); e4m3fn keeps bias+finite
    live (its 4e3m split has sibling variants) -> higher arity."""
    head, s = _head(), jnp.zeros((E,))
    idx = {n: i for i, n in enumerate(QUANT_DTYPES)}
    if "int8" in idx and "float8_e4m3fn" in idx:
        a_int = float(head.log_prob(s, jnp.int32(idx["int8"]), jnp.int32(1))[2])
        a_flt = float(head.log_prob(s, jnp.int32(idx["float8_e4m3fn"]), jnp.int32(1))[2])
        assert a_int < a_flt, (a_int, a_flt)


def test_sign_choice_does_not_change_which_factors_are_live():
    head, s = _head(), jnp.zeros((E,))
    gi = jnp.int32(QUANT_DTYPES.index("int8"))
    a_plus = float(head.log_prob(s, gi, jnp.int32(1))[2])
    a_minus = float(head.log_prob(s, gi, jnp.int32(-1))[2])
    assert a_plus == a_minus and a_plus >= 1.0
