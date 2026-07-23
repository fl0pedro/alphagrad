"""The incremental palimpsa encode must equal the full re-encode.

This is what makes the whole pipeline autoregressive: the policy encodes the
base jaxpr once and then extends by each delta, instead of re-reading thousands
of tokens per action. If the recurrence drifted from the batch path, the policy
would condition on a different representation than the one the reward came from.

incremental_encoder.py carried a "PROVEN recurrence" header and three test
files containing ZERO test functions between them, so nothing actually checked
this.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alphagrad.approx.incremental_encoder as IE
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent

TOKENS = np.array([7, 11, 23, 5, 42, 13, 99, 31, 60, 8, 17, 25], dtype=np.int32)


def _agent(seed=0):
    return MicroPPOAgent(vocab_size=512, embd_dim=32, num_layers=2, num_heads=2,
                         hidden_dim=64, num_vertices=16, value_dims=[32],
                         key=jr.PRNGKey(seed), max_substeps=1, policy="palimpsa")


def _full(agent, toks):
    out = agent.encode_tokens(jnp.asarray(toks), jr.PRNGKey(1))
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _incremental(agent, toks, split):
    st = IE.init_state(agent)
    i = 0
    for n in split:
        IE.extend(agent, st, toks[i:i + n].tolist())
        i += n
    return np.asarray(IE.enc_x(st))


@pytest.mark.parametrize("split", [[12], [5, 7], [1, 11], [4, 4, 4], [1] * 12],
                         ids=["one", "two", "uneven", "three", "per-token"])
def test_incremental_matches_full_encode(split):
    agent = _agent()
    full, inc = _full(agent, TOKENS), _incremental(agent, TOKENS, split)
    assert inc.shape == full.shape
    assert np.max(np.abs(full - inc)) < 1e-4, f"split {split} drifted"


def test_empty_delta_is_a_no_op():
    """Real streams contain empty deltas -- eliminations that emit nothing --
    so the recurrence must tolerate them rather than assume every step adds."""
    agent = _agent()
    st = IE.init_state(agent)
    IE.extend(agent, st, TOKENS[:6].tolist())
    before = np.asarray(IE.enc_x(st))
    IE.extend(agent, st, [])
    after = np.asarray(IE.enc_x(st))
    assert np.array_equal(before, after)
    IE.extend(agent, st, TOKENS[6:].tolist())
    assert np.max(np.abs(_full(agent, TOKENS) - np.asarray(IE.enc_x(st)))) < 1e-4


def test_state_copy_allows_branching():
    """A search / policy that branches must be able to fork the encode state."""
    agent = _agent()
    st = IE.init_state(agent)
    IE.extend(agent, st, TOKENS[:6].tolist())
    a, b = st.copy(), st.copy()
    IE.extend(agent, a, TOKENS[6:].tolist())
    IE.extend(agent, b, [3, 3, 3])
    assert np.asarray(IE.enc_x(a)).shape[0] == 12
    assert np.asarray(IE.enc_x(b)).shape[0] == 9
    assert np.max(np.abs(_full(agent, TOKENS) - np.asarray(IE.enc_x(a)))) < 1e-4
