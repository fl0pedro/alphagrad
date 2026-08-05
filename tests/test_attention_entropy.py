"""entropy/palimpsa: encoder attention-row entropy diagnostic.

Pins that the two attention blocks expose a finite, correctly-bounded mean
row entropy, and that the mask is respected. This is a REPRESENTATION
diagnostic, not a policy entropy -- see the docstrings on the methods.
"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from alphagrad.approx.set_pointer import SetBlock, SetPointerVertexPolicy

E, H, V = 16, 2, 7


def test_set_block_entropy_is_finite_and_bounded():
    blk = SetBlock(E, H, key=jrand.PRNGKey(0))
    h = jrand.normal(jrand.PRNGKey(1), (V, E))
    mask = jnp.ones((V,), jnp.float32)
    ent = float(blk.attention_entropy(h, mask))
    assert np.isfinite(ent)
    # 0 <= H <= ln(number of attendable keys)
    assert -1e-6 <= ent <= np.log(V) + 1e-4


def test_masked_slots_shrink_the_bound():
    """Masking keys reduces the attendable alphabet, so the entropy must fall
    below ln(n_valid) -- proof the mask actually reaches the softmax."""
    blk = SetBlock(E, H, key=jrand.PRNGKey(0))
    h = jrand.normal(jrand.PRNGKey(1), (V, E))
    mask = jnp.array([1, 1, 0, 0, 0, 0, 0], jnp.float32)
    ent = float(blk.attention_entropy(h, mask))
    assert np.isfinite(ent)
    assert ent <= np.log(2) + 1e-4


def test_uniform_attention_saturates_the_bound():
    """Zero q/k projections => uniform attention => entropy exactly ln(N)."""
    import equinox as eqx
    blk = SetBlock(E, H, key=jrand.PRNGKey(0))
    a = blk.attn
    blk = eqx.tree_at(
        lambda b: (b.attn.query_proj.weight, b.attn.key_proj.weight),
        blk,
        (jnp.zeros_like(a.query_proj.weight),
         jnp.zeros_like(a.key_proj.weight)),
    )
    h = jrand.normal(jrand.PRNGKey(1), (V, E))
    ent = float(blk.attention_entropy(h, jnp.ones((V,), jnp.float32)))
    assert abs(ent - np.log(V)) < 1e-3


def test_policy_level_entropy_runs():
    pol = SetPointerVertexPolicy(
        num_vertices=V, embd_dim=E, num_heads=H, num_blocks=2,
        key=jrand.PRNGKey(2),
    )
    vmem = jrand.normal(jrand.PRNGKey(3), (V + 1, E))
    vmask = jnp.ones((V + 1,), jnp.float32)
    ent = float(pol.attention_entropy(vmem, vmask))
    assert np.isfinite(ent)
    assert -1e-6 <= ent <= np.log(V + 1) + 1e-4


def test_axis_set_encoder_entropy_runs():
    from alphagrad.approx.heads import AxisSetEncoder, AxisTokenFeatures
    from alphagrad.approx.heads import AXIS_TAG_BITS
    N = 5
    enc = AxisSetEncoder(E, H, num_layers=2, key=jrand.PRNGKey(4))
    feats = AxisTokenFeatures(
        size=jnp.arange(1, N + 1, dtype=jnp.int32),
        log_size=jnp.log(jnp.arange(1, N + 1, dtype=jnp.float32)),
        tag_bits=jnp.zeros((N, AXIS_TAG_BITS), jnp.float32),
        group_id=jnp.full((N,), -1, jnp.int32),
        valid_mask=jnp.array([1, 1, 1, 0, 0], jnp.float32),
    )
    ent = float(enc.attention_entropy(feats, jnp.zeros((E,), jnp.float32)))
    assert np.isfinite(ent)
    assert -1e-6 <= ent <= np.log(3) + 1e-4


if __name__ == "__main__":
    test_set_block_entropy_is_finite_and_bounded()
    test_masked_slots_shrink_the_bound()
    test_uniform_attention_saturates_the_bound()
    test_policy_level_entropy_runs()
    test_axis_set_encoder_entropy_runs()
    print("ALL PASS")
