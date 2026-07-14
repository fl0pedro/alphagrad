from .encoder import Encoder, EncoderLayer, RelationalMultiheadAttention, SwiGLU
from .palimpsa_encoder import PalimpsaEncoder, PalimpsaEncoderLayer
from .utils import PositionalEncoder, MLP

POLICIES = ("transformer", "palimpsa", "palimpsa_bi")


def make_encoder(policy, num_layers, num_heads, embd_dim, hidden_dim, *, key, **kwargs):
    """Pick the token mixer by class, not by a string threaded through every layer.

    ``policy`` is one of :data:`POLICIES`. ``transformer`` is the O(seq^2)
    relational-bias encoder; the palimpsa variants route through the Pallas
    kernel in :mod:`alphagrad.transformer.palimpsa_pallas`.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if policy == "transformer":
        return Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=key, **kwargs)
    return PalimpsaEncoder(
        num_layers, num_heads, embd_dim, hidden_dim, key=key,
        bidirectional=(policy == "palimpsa_bi"), **kwargs,
    )
