"""THE tokenizer id space: one default, one resolver, one raise.

WHY THIS MODULE EXISTS. ``ALPHAGRAD_INCR_TOKEN_VOCAB`` used to be read with a
literal default in four unrelated places (``env._incremental_stream_tokens``,
``env.base_observation``, ``env.base_owners``, ``ppo.main``'s live-face stream,
``common.plan_tokens.PlanTokenStream`` and ``common.face_driver
.build_live_face_stream``). The base stream and the per-step deltas MUST be
tokenized at the same vocabulary or they do not concatenate, and a default that
is spelled six times is a default that can disagree with itself. Everything now
calls :func:`incr_token_vocab`.

THE VALUE IS 256, AND THE IDS RIDE AS ``uint8`` (owner's decision, 2026-09-13):
"i would still prefer total vocab size of 256 in int8 i think the complexity of
sequence length is better than complexity of IDs, as it is a more compact
representation overall!". graphax reserves 223 structural slots plus 10 digits,
so 256 leaves 23 name symbols. Names are POSITIONAL sequences over that
alphabet, so a small alphabet still spells unlimited names -- it just spends
several atoms on the later ones, and the token stream gets longer. That is the
trade the owner accepted: a longer stream of byte-wide ids instead of a shorter
stream of int32 ids.

TWO WAYS TO GET THIS WRONG, BOTH OF WHICH RAISE HERE.

* TOO SMALL. graphax's ``IncrementalPathTokenizer.__init__`` raises when fewer
  than 2 name symbols are left, but it raises from inside the per-step host
  callback, where ``live_faces`` and ``env`` wrap tokenizer construction in
  ``except Exception`` soft-failure paths that turn it into an empty chunk and
  a bumped ``failures`` counter. Resolving the vocabulary THROUGH this function
  moves the check to the caller, before any tokenizer is built, so the error
  reaches the user instead of a statistics dictionary.
* TOO LARGE. Token ids ride the wire and the trajectory as ``uint8``
  (``env.DELTA_TOKEN_DTYPE``), so an id space wider than 256 wraps silently at
  the cast. Refused here rather than discovered as a policy that learned from
  the wrong embedding rows.
"""
from __future__ import annotations

import os

import numpy as np

# graphax's own vocabulary builder. Imported at module scope on purpose: this
# module must answer without a tokenizer, and `get_vocab` is lru_cached.
from graphax.jaxpr import get_vocab as _graphax_get_vocab

__all__ = [
    "INCR_TOKEN_VOCAB_DEFAULT",
    "INCR_TOKEN_VOCAB_ENV",
    "TOKEN_ID_LIMIT",
    "DIGIT_BASE",
    "reserved_token_slots",
    "incr_token_vocab",
    "DELTA_TOKEN_DTYPE",
    "DELTA_TOKEN_MAX",
    "DELTA_TOKEN_PAD",
    "DELTA_HEADER_SLOTS",
    "encode_delta_header",
    "check_delta_ids",
]

# The owner's decision. Also the largest value `uint8` token ids can carry.
INCR_TOKEN_VOCAB_DEFAULT = 256

# The environment variable that overrides it, named once.
INCR_TOKEN_VOCAB_ENV = "ALPHAGRAD_INCR_TOKEN_VOCAB"

# Token ids are stored as `uint8`, so this is the hard ceiling on the id space.
TOKEN_ID_LIMIT = 256

# graphax's digit alphabet size. `IncrementalPathTokenizer` hardcodes the same
# default; it is spelled here so `reserved_token_slots` can be checked.
DIGIT_BASE = 10


def reserved_token_slots(digit_base: int = DIGIT_BASE) -> int:
    """Ids graphax spends before the first NAME symbol.

    ``len(get_vocab())`` structural/primitive slots plus ``digit_base`` digits.
    Measured, never assumed: graphax appends vocabulary tokens by design ("New
    markers MUST be appended"), so this count moves between graphax revisions
    and a hardcoded copy of it would silently mis-size the name alphabet.
    """
    vocab, _, _ = _graphax_get_vocab(int(digit_base))
    return len(vocab) + int(digit_base)


def incr_token_vocab(vocab=None, *, digit_base: int = DIGIT_BASE) -> int:
    """The tokenizer id space to build every ``IncrementalPathTokenizer`` at.

    ``vocab`` ``None`` reads ``ALPHAGRAD_INCR_TOKEN_VOCAB``, defaulting to
    :data:`INCR_TOKEN_VOCAB_DEFAULT`. An explicit value is checked the same
    way, so a caller that passes its own number cannot bypass the bounds.

    Raises ``ValueError`` on a vocabulary the tokenizer cannot fit (fewer than
    2 name symbols left after the reserved slots) and on one wider than
    :data:`TOKEN_ID_LIMIT` (the ids would not survive the ``uint8`` wire).
    """
    if vocab is None:
        raw = os.environ.get(INCR_TOKEN_VOCAB_ENV)
        v = INCR_TOKEN_VOCAB_DEFAULT if raw is None else int(raw)
    else:
        v = int(vocab)
    reserved = reserved_token_slots(digit_base)
    names = v - reserved
    if names < 2:
        raise ValueError(
            f"{INCR_TOKEN_VOCAB_ENV}={v} leaves {names} name symbols after "
            f"{reserved} reserved graphax slots (structural vocabulary plus "
            f"{digit_base} digits); the tokenizer needs at least 2 to spell "
            f"multi-token names. Raise the vocabulary."
        )
    if v > TOKEN_ID_LIMIT:
        raise ValueError(
            f"{INCR_TOKEN_VOCAB_ENV}={v} is wider than the {TOKEN_ID_LIMIT} "
            f"ids a uint8 token buffer can carry, and the cast into "
            f"env.DELTA_TOKEN_DTYPE would WRAP silently -- the policy would "
            f"embed the wrong rows. Lower the vocabulary, or widen "
            f"env.DELTA_TOKEN_DTYPE and every buffer that names it."
        )
    return v


# ---------------------------------------------------------------------------
# HOW THE TOKENS ARE STORED. Here rather than in ``env`` because the OTHER
# producer of these tokens, ``live_faces.LiveFaceStream``, is deliberately
# JAX-free at import time and must not pull ``env`` in to learn its own wire
# dtype. ``env`` re-exports every name below, so ``env.DELTA_TOKEN_DTYPE``
# keeps meaning what it says.
#
# The vocabulary above is what makes uint8 possible: 256 ids, so every token
# is a byte.
#
# THERE IS NO EQUATION-ID BUFFER. A parallel ``(MAX_DELTA_TOKENS,)`` int32
# buffer of graphax's stream-global segment ids used to ride beside the
# tokens, everywhere the tokens ride. It fed exactly one consumer -- the
# relational forget-gate modulation in the palimpsa mixers -- which was one
# zero-initialised scalar per token per head with no ablation behind it.
# Both were removed on 2026-09-13. The pad sentinel -1 belonged to that
# buffer and is gone with it; token padding is 0, and 0 is a REAL token.
# ---------------------------------------------------------------------------

DELTA_TOKEN_DTYPE = np.uint8
DELTA_TOKEN_MAX = int(np.iinfo(DELTA_TOKEN_DTYPE).max)      # 255

# Token padding. 0 is a REAL token id (graphax's literal '-', which occurs
# INTERIOR to real streams), so no reader may recover a length by scanning for
# the pad. Every reader keys on the stored count.
DELTA_TOKEN_PAD = 0

# The count header: four byte slots = one little-endian int32. See env.py's
# "THE NARROW ID WIRE" note for why the count is not a token slot.
DELTA_HEADER_SLOTS = 4


def encode_delta_header(n: int) -> np.ndarray:
    """``(DELTA_HEADER_SLOTS,) uint8``: ``n`` as a little-endian uint32.

    Spelled with shifts rather than ``np.int32(n).tobytes()`` so the layout is
    the same on a big-endian host as on a little-endian one.
    """
    n = int(n)
    if n < 0 or n > 0xFFFFFFFF:
        # UNSIGNED 32-bit (owner ruling 2026-09-14): `decode_delta_header`
        # reassembles the four bytes with uint32 arithmetic on device, so the
        # whole 32-bit range is a count. Refused here rather than discovered
        # as a wrapped length.
        raise ValueError(
            f"delta count {n} does not fit the uint32 header "
            f"(max {0xFFFFFFFF})")
    return np.asarray([(n >> (8 * i)) & 0xFF
                       for i in range(DELTA_HEADER_SLOTS)], DELTA_TOKEN_DTYPE)


def check_delta_ids(toks, where: str = "_delta_observation") -> None:
    """Raise unless every token fits the byte-wide wire. Host side, exact.

    ``toks`` is a numpy integer array WIDER than the wire dtype (int64), so
    the comparison happens before any cast can hide the overflow. Shared by
    ``env._delta_observation`` and ``live_faces.chunk``, because a face chunk
    is a slice of the same emission and carries the same tokens.
    """
    toks = np.asarray(toks)
    if not toks.size:
        return
    lo, hi = int(toks.min()), int(toks.max())
    if lo < 0 or hi > DELTA_TOKEN_MAX:
        raise ValueError(
            f"[{where}] token id {hi if hi > DELTA_TOKEN_MAX else lo} is "
            f"outside [0, {DELTA_TOKEN_MAX}] and would WRAP in the uint8 "
            f"token buffer. The tokenizer id space is "
            f"{INCR_TOKEN_VOCAB_DEFAULT} by default -- lower "
            f"{INCR_TOKEN_VOCAB_ENV}, or widen DELTA_TOKEN_DTYPE and every "
            f"buffer that names it."
        )
