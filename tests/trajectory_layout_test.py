#!/usr/bin/env python3
"""What the rollout STORES per step, and what it may not store.

Three storage changes were proposed for the per-step `ppo.Trajectory`
(handoff 2026-09-13, section 6). This module pins the facts each one stands
or falls on.

1. BIT-PACKED MASKS (landed). `face_pair_valid`, `face_comp_valid` and
   `face_quant` are strictly 0/1, so they ride as uint8 bit words. The
   round-trip must be exact and the bit order must be the documented one, or
   the loss re-masks with rows the behaviour policy never drew under and the
   PPO ratio silently stops being 1 at epoch 0.

2. DROPPING `face_delta_tokens` (NOT landed). The claim was that it equals
   the next step's `delta_tokens`. It does not.
   ``test_a_face_chunk_carries_the_previous_faces_approximation_echo_not_its_own``
   shows why: face f's chunk is face f-1's approximation echo followed by
   face f's own contraction, so the concatenation of a vertex's chunks stops
   at the last face's contraction and the last face's approximation echo --
   which IS in the step emission -- is in no chunk. The stored buffer has its
   own length and its own padding.

   WHICH DECISIONS EMIT AN ECHO. graphax writes an ``approx`` block only when
   ``core._apply_face_transform`` records a micro-action: a literal
   ``Diag`` / ``Compress`` / ``Quant``, or a CHOOSER callable that RETURNS
   one. Until 2026-09-13 alphagrad installed neither --
   ``env.make_slot_frame_hook`` was a plain tensor-returning callable, so
   graphax took the ``out = _chosen`` branch, applied the transform and called
   ``_record_micro`` for nobody; the face sink got no record and
   ``last_face_segments`` reported ``split == end``. NO (vertex, slot, rule)
   triple on ANY graph could put an echo into a chunk through the slot wire,
   which is why jobs 65344/65346 came back empty and why probe 65347 found
   head == 0 even for rows that visibly changed the emission (finding 64).
   The hook is a CHOOSER now: it decides and hands the action back, graphax
   applies AND records it. So BOTH face decision channels emit an echo, and
   the test below decides with each in turn -- ``skip``
   (``face_skips[f] == 1`` becomes ``graphax.SKIP_FACE``, which
   ``core._eliminate_vertex`` records directly as ``approx SKIP {}``) and
   ``rule`` (a slot wire row) -- reading the echo it expects off the emission
   env's OWN builder produces.

3. THE NARROW TOKEN WIRE (landed 2026-09-13). Token ids ride as uint8 at a
   vocabulary of 256, the delta count rides in its own little-endian int32
   over the first four byte slots of the wire, and the equation-id buffer is
   GONE together with the palimpsa relational forget gate it fed. What used to
   be pinned here as a REFUSAL ("token ids do not fit in a byte at 512") is
   now pinned as the positive property.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                               # noqa: E402
import pytest                                                    # noqa: E402

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from alphagrad.approx.ppo import (                               # noqa: E402
    MASK_BIT_ORDER,
    pack_mask_bits,
    unpack_mask_bits,
)


# --------------------------------------------------------------------------
# 1. bit-packed masks
# --------------------------------------------------------------------------

def _random_mask(shape, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, 2, size=shape).astype(np.float32))


def test_a_packed_face_mask_round_trips_to_the_same_zero_one_values():
    """Exactness is the whole contract: the loss must re-mask with the SAME
    values the rollout masked with."""
    for shape in ((7, 3, 5, 5), (7, 5), (7, 3, 4), (7,), (1920, 8, 8)):
        m = _random_mask(shape, seed=sum(shape))
        packed = pack_mask_bits(m, shape)
        assert packed.dtype == jnp.uint8
        back = unpack_mask_bits(packed, shape)
        assert back.dtype == jnp.float32
        assert back.shape == m.shape
        np.testing.assert_array_equal(np.asarray(back), np.asarray(m))


def test_packing_leaves_the_leading_axes_alone_and_shrinks_the_mask_thirty_two_fold():
    """The scan's step axis and the vmap's environment axis sit LEFT of the
    declared mask shape. If packing touched them, the loss's
    ``x.reshape(-1, *x.shape[2:])`` would mean something else."""
    shape = (1920, 8, 8)
    lead = (4, 3)
    m = _random_mask(lead + shape, seed=11)
    packed = pack_mask_bits(m, shape)
    bits = 1920 * 8 * 8
    assert packed.shape == lead + ((bits + 7) // 8,)
    # float32 -> 1 bit: 32x on the stored bytes.
    assert m.nbytes == 32 * packed.nbytes
    back = unpack_mask_bits(packed, shape)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(m))


def test_the_bit_order_is_little_so_element_k_is_bit_k_mod_eight_of_word_k_div_eight():
    """The layout is stated once, above ``pack_mask_bits``. This is the
    statement, as a test: a reader of the packed buffer that assumed the
    other bit order would read every word reversed."""
    assert MASK_BIT_ORDER == "little"
    m = jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 1.0], jnp.float32)
    packed = np.asarray(pack_mask_bits(m, (10,)))
    assert packed.shape == (2,)
    # bit 0 and bit 3 of word 0; bit 1 of word 1.
    assert int(packed[0]) == 0b00001001
    assert int(packed[1]) == 0b00000010


def test_packing_a_mask_whose_trailing_shape_is_not_the_declared_one_raises():
    """A wrong shape here is otherwise silent -- the buffer would unpack into
    the wrong rows and only the PPO ratio would know."""
    m = _random_mask((7, 3, 5), seed=3)
    with pytest.raises(ValueError, match="does not end in"):
        pack_mask_bits(m, (3, 6))


def test_unpacking_an_array_that_was_never_packed_raises():
    m = _random_mask((7, 5), seed=4)
    with pytest.raises(ValueError, match="expected uint8 bit words"):
        unpack_mask_bits(m, (7, 5))


def test_unpacking_with_a_mask_shape_the_words_cannot_hold_raises():
    packed = pack_mask_bits(_random_mask((7, 5), seed=5), (7, 5))
    with pytest.raises(ValueError, match="packs into"):
        unpack_mask_bits(packed, (7, 9))


def test_the_packed_mask_survives_a_jit_boundary():
    """The rollout packs inside a ``lax.scan`` under ``jit``; the loss unpacks
    inside the gradient. Both must be traceable."""
    shape = (16, 8, 8)
    m = _random_mask(shape, seed=7)

    @jax.jit
    def _round_trip(x):
        return unpack_mask_bits(pack_mask_bits(x, shape), shape)

    np.testing.assert_array_equal(np.asarray(_round_trip(m)), np.asarray(m))


# --------------------------------------------------------------------------
# 2. the face chunks are a PREFIX of the step delta
# --------------------------------------------------------------------------

def _perceptron():
    import jax.random as jrand
    from graphax.examples import Perceptron

    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    args = (x, y, W1, b1, W2, b2, gamma, beta)
    cj = jax.make_jaxpr(Perceptron)(*args)
    return cj.jaxpr, cj.literals, args


def _reference_step(jaxpr, argnums, consts, args, vertex, rows, skips,
                    vocab=None):
    """``(tokens, [(start, split, end)])`` of ONE vertex's elimination.

    Built through ``env._face_dict_for_vertex`` -- THE builder the measurement
    and ``LiveFaceStream._decided`` both use -- so this reference carries the
    same transforms the stream replays, and the echo it reports is graphax's
    own, not a re-derivation. ``[split:end]`` of a segment IS the face's
    approximation part (``IncrementalPathTokenizer.last_face_segments``).
    """
    from types import SimpleNamespace

    from graphax import IncrementalPathTokenizer

    from alphagrad.approx.common.token_vocab import incr_token_vocab
    from alphagrad.approx.env import _face_dict_for_vertex

    ref = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                   vocab_size=incr_token_vocab(vocab))
    ref.base_tokens()
    keys = list(ref.ij.faces(int(vertex)))
    ft = _face_dict_for_vertex(SimpleNamespace(jaxpr=jaxpr), ref.ij,
                               int(vertex), rows, skips, keys=keys)
    toks = [int(t) for t in ref.eliminate(int(vertex), (), ft)]
    return toks, ref.last_face_segments()


def _quant_row(dtype="bfloat16"):
    """The wire row for ``QUANT <dtype>`` -- the slot RULE this test decides
    with. A cast is legal on any slot that stores values, so it finds an
    applied rule on every face without a search over geometry."""
    from graphax.sparse.micro_actions import QUANT_DTYPES
    from alphagrad.approx.env import QUANT_SENTINEL

    return (QUANT_SENTINEL, QUANT_DTYPES.index(dtype), 0)


@pytest.mark.parametrize("channel", ["skip", "rule"])
def test_a_face_chunk_carries_the_previous_faces_approximation_echo_not_its_own(
        channel):
    """THE REASON `face_delta_tokens` IS NOT THE NEXT STEP'S `delta_tokens`.

    ``LiveFaceStream.chunk`` returns face ``f-1``'s approximation echo
    followed by face ``f``'s own contraction, and reports the echo's length as
    ``head``. So every face's echo is carried by its SUCCESSOR's chunk, and
    the last face of a vertex has no successor: its echo is in the step's
    emission and in none of the stored chunks.

    BOTH FACE DECISION CHANNELS ARE CHECKED. ``skip`` is the wire bit graphax
    records itself, legal on every face by construction. ``rule`` is a slot
    wire row -- a QUANT on the ``lhs`` slot -- which until 2026-09-13 emitted
    NO block at all, because ``env.make_slot_frame_hook`` returned a tensor and
    graphax records a chooser's result only when the chooser returns the
    micro-action itself (module docstring, section 2). The hook is a chooser
    now, so a rule decision has an echo exactly as a skip does, and the
    prefix property below is the same statement for both. The VERTEX is still
    searched: a vertex with one face has no successor to carry anything, and
    for ``rule`` every face must actually apply the row.

    Every claim is checked against the emission ``_reference_step`` gets from
    env's own builder, token for token, so a change in what graphax emits
    fails here rather than being absorbed.
    """
    from alphagrad.approx.env import wire_slots
    from alphagrad.approx.live_faces import LiveFaceStream

    jaxpr, consts, args = _perceptron()
    argnums = (2, 3, 4, 5)
    V = len(jaxpr.eqns)
    MR, F, S = 8, 8, wire_slots()
    VOCAB = None            # the one resolver's answer (256)

    lfs = LiveFaceStream(jaxpr, argnums, consts, args, vocab=VOCAB,
                         max_faces=F, max_axes=8, window=8192)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, MR, 3), np.int32)
    vspecs = -np.ones((MR, 3), np.int32)
    exact = -np.ones((F, S, 3), np.int32)
    no_skip = np.zeros((F,), np.int32)
    row = _quant_row()

    def _wire(upto):
        """Faces ``0..upto-1`` decided, the rest exact -- exactly the prefix
        ``_decided`` replays when the stream is asked for face ``upto``."""
        rows = exact.copy()
        skips = np.zeros((F,), np.int32)
        if channel == "skip":
            skips[:upto] = 1
        else:
            for g in range(upto):
                rows[g, 0] = row
        return rows, skips

    def _all_faces_decide(cand, k):
        """Does every face of ``cand`` emit a block once decided? For ``skip``
        that is true by construction; for ``rule`` the row has to apply."""
        rows, skips = _wire(k)
        try:
            _toks, segs = _reference_step(
                jaxpr, argnums, consts, args, cand, rows, skips, VOCAB)
        except Exception:
            return False
        return len(segs) == k and all(e > sp for _s, sp, e in segs)

    vertex, n_faces = None, 0
    for cand in range(1, V + 1):
        lfs._chunks.clear()
        # `chunk` returns (tokens, count, n_faces, ends, head) -- FIVE
        # values since the equation-id buffer was removed, so `n_faces` is
        # index 2.
        k = int(lfs.chunk(order, specs, 0, cand, vspecs, exact, no_skip, 0)[2])
        if k >= 2 and _all_faces_decide(cand, k):
            vertex, n_faces = cand, k
            break
    assert vertex is not None, (
        f"no vertex of this graph has two or more faces that all emit a "
        f"{channel} block, so no face has a successor to read an echo off")

    # An EXACT vertex emits no approximation at all: no face has an echo, so
    # no chunk has a head. This is the control -- without it a head of 0 on
    # the decided run could not be told from "the stream never reports one".
    exact_toks, exact_segs = _reference_step(
        jaxpr, argnums, consts, args, vertex, exact, no_skip, VOCAB)
    assert len(exact_segs) == n_faces
    assert all(sp == e for _s, sp, e in exact_segs), (
        f"vertex {vertex} emitted an approximation with no decision made: "
        f"{exact_segs}")
    for f in range(n_faces):
        lfs._chunks.clear()
        assert int(lfs.chunk(order, specs, 0, vertex, vspecs, exact,
                             no_skip, f)[4]) == 0

    # Now DECIDE. Face f's chunk is read with faces 0..f-1 decided and face f
    # still undecided, so the emission it comes from is `_wire(f)`.
    heads = []
    for f in range(n_faces):
        rows_f, sk = _wire(f)
        toks_f, segs_f = _reference_step(
            jaxpr, argnums, consts, args, vertex, rows_f, sk, VOCAB)
        lfs._chunks.clear()
        tok, cnt, nf, _ends, head = lfs.chunk(
            order, specs, 0, vertex, vspecs, rows_f, sk, f)
        cnt, head = int(cnt), int(head)
        assert int(nf) == n_faces
        assert cnt > 0, (
            f"face {f} of vertex {vertex} handed back an empty chunk -- the "
            f"stream fell to its soft-failure path, so nothing below is a "
            f"statement about the echo. stats={dict(lfs.stats)}")
        heads.append(head)
        if f == 0:
            # No predecessor, so nothing to echo.
            assert head == 0
            continue
        start, split, end = segs_f[f - 1]
        assert end > split, (
            f"face {f - 1} of vertex {vertex} was decided ({channel}) but "
            f"emitted no approximation block ({segs_f}); graphax no longer "
            f"records this decision, so this test can say nothing about the "
            f"echo")
        assert head == end - split, (
            f"face {f}'s chunk reports a {head}-token echo against face "
            f"{f - 1}'s {end - split}-token approximation block")
        assert [int(t) for t in tok[:head]] == toks_f[split:end], (
            f"face {f}'s chunk does not OPEN on face {f - 1}'s approximation "
            f"block")
        # ... and the rest of the chunk is face f's OWN contraction, which
        # starts where face f's segment starts.
        assert [int(t) for t in tok[head:cnt]] == toks_f[
            segs_f[f][0]:segs_f[f][1]]

    # THE GAP. With every face decided, the emission ends on the LAST face's
    # approximation block -- and the chunk that would carry it is chunk
    # `n_faces`, which does not exist.
    all_rows, all_sk = _wire(n_faces)
    all_toks, all_segs = _reference_step(
        jaxpr, argnums, consts, args, vertex, all_rows, all_sk, VOCAB)
    last_start, last_split, last_end = all_segs[n_faces - 1]
    assert last_end > last_split, (
        "the last face emitted no approximation block, so this vertex cannot "
        "show the gap")
    assert last_end == len(all_toks)
    lfs._chunks.clear()
    past = lfs.chunk(order, specs, 0, vertex, vspecs, all_rows, all_sk,
                     n_faces)
    assert int(past[1]) == 0 and int(past[4]) == 0, (
        "there is a chunk past the last face, so the last face's echo would "
        "have a carrier after all")

    # Counted: the chunks carry the echoes of faces 0..n-2 and no other. The
    # last face's echo is emitted and stored by nothing, which is the gap that
    # makes `face_delta_tokens` a different buffer from the step emission.
    echoes = [e - sp for _s, sp, e in all_segs]
    assert heads == [0] + echoes[:-1]
    assert sum(heads) == sum(echoes) - echoes[-1] < sum(echoes)


# --------------------------------------------------------------------------
# 3. THE NARROW TOKEN WIRE (landed 2026-09-13)
# --------------------------------------------------------------------------
#
# Token ids ride as uint8 at a vocabulary of 256, the delta count rides in its
# own little-endian int32 across the first four byte slots of the wire, and the
# equation-id buffer is GONE -- with the palimpsa relational forget gate it fed
# and with the (MAX_EQNS,) histogram that reproduced the gate inside the
# recurrence. The tests below pin all three.

_DIGIT_BASE = 10


_ENCODER = "Encoder"


def _encoder_tokenizer(vocab_size):
    """The repo's own ``Encoder`` target, tokenized at ``vocab_size``.

    A transformer ENCODER BLOCK, which is the smallest thing in the example
    set that is shaped like the campaign's ``TransformerLM`` target (measured
    base blocks: Encoder 1323 tokens, TransformerLM 1406). Built through
    ``common.examples`` so it is the same jaxpr the trainer would tokenize.
    """
    import jax.random as jrand
    from graphax import IncrementalPathTokenizer

    from alphagrad.approx.common.examples import (
        get_args, get_fn, infer_argnums,
    )

    args = tuple(get_args(_ENCODER, jrand.PRNGKey(0)))
    cj = jax.make_jaxpr(get_fn(_ENCODER))(*args)
    return IncrementalPathTokenizer(
        cj.jaxpr, tuple(infer_argnums(_ENCODER)), list(cj.literals),
        list(args), vocab_size=vocab_size)


def test_the_configured_token_vocabulary_is_two_hundred_fifty_six():
    """THE decision, as a test.

    One resolver, one default. Every producer of the token stream -- the
    observation path, the base block, the face chunks, the AZ plan tokenizer
    -- calls ``incr_token_vocab``, because a base tokenized at one vocabulary
    and a delta tokenized at another do not concatenate.
    """
    from alphagrad.approx.common import token_vocab as _tv

    assert _tv.INCR_TOKEN_VOCAB_DEFAULT == 256
    assert _tv.incr_token_vocab() == 256


def test_at_the_configured_vocabulary_every_token_id_fits_in_one_byte():
    """THE POSITIVE PIN the byte budget earns.

    This file used to assert the opposite -- that a byte-wide id space was
    unreachable at the 512 the runs used. The vocabulary is 256 now, so the
    tokenizer's own static cap is 255 and a transformer-shaped graph's base
    block, which is where the largest ids appear before a single elimination,
    stays inside it.
    """
    from alphagrad.approx.common.token_vocab import (
        DELTA_TOKEN_MAX, incr_token_vocab,
    )

    tk = _encoder_tokenizer(incr_token_vocab())
    assert tk.max_token_id() == DELTA_TOKEN_MAX == 255
    base = [int(t) for t in tk.base_tokens()]
    assert base, "the Encoder example emitted no base tokens"
    assert max(base) <= DELTA_TOKEN_MAX
    assert min(base) >= 0


def test_a_byte_wide_id_space_leaves_a_small_but_legal_name_alphabet():
    """The number the decision was made on.

    graphax appends vocabulary tokens by design ("New markers MUST be
    appended"), so the reserved count moves: it was 230 before the byte-only
    catalog and is 223 after it. The test therefore reads the count rather
    than spelling it, and pins the CONSEQUENCE: a 256-wide id space leaves
    fewer than 32 name symbols but more than the 2 graphax requires, so names
    are spelled from several atoms and the stream is longer. That is the
    trade the owner took.
    """
    from alphagrad.approx.common.token_vocab import (
        incr_token_vocab, reserved_token_slots,
    )

    reserved = reserved_token_slots(_DIGIT_BASE)
    names = incr_token_vocab() - reserved
    assert 2 <= names < 32, (
        f"the reserved vocabulary is {reserved} slots including "
        f"{_DIGIT_BASE} digits, so a byte leaves {names} name symbols")


def test_a_vocabulary_the_tokenizer_cannot_fit_raises_from_the_resolver():
    """UN-SWALLOWED. graphax raises this too, but it raises from inside a
    per-step host callback where ``LiveFaceStream._tokenizer_at`` and
    ``env._incremental_stream_tokens`` catch ``Exception`` and turn the
    failure into an empty chunk plus a bumped ``failures`` counter. Resolving
    through ``incr_token_vocab`` moves the raise to the caller."""
    from alphagrad.approx.common.token_vocab import (
        incr_token_vocab, reserved_token_slots,
    )

    too_small = reserved_token_slots(_DIGIT_BASE) + 1
    with pytest.raises(ValueError, match="name symbols"):
        incr_token_vocab(too_small)


def test_a_vocabulary_wider_than_a_byte_raises_because_the_wire_would_wrap():
    from alphagrad.approx.common.token_vocab import incr_token_vocab

    with pytest.raises(ValueError, match="uint8"):
        incr_token_vocab(512)


def test_the_live_face_stream_refuses_a_vocabulary_wider_than_a_byte():
    """The face chunks are a slice of the SAME emission, so they carry the
    same tokens and must be built at the same id space."""
    from alphagrad.approx.live_faces import LiveFaceStream

    jaxpr, consts, args = _perceptron()
    with pytest.raises(ValueError, match="uint8"):
        LiveFaceStream(jaxpr, (2, 3, 4, 5), consts, args, vocab=512,
                       max_faces=8, max_axes=8, window=256)


def test_the_policy_embedding_is_checked_against_the_tokenizer_vocabulary():
    """JAX CLAMPS an out-of-range gather instead of raising, so an embedding
    with fewer rows than the tokenizer has ids reads the LAST row for every id
    past the table and the policy learns from a collision it never sees.
    ``ppo.main`` refuses that combination before it builds anything."""
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    from alphagrad.approx.ppo import check_embedding_covers_tokenizer

    tok = incr_token_vocab()
    # Equal is enough, and wider is allowed (the surplus rows are simply
    # never gathered).
    assert check_embedding_covers_tokenizer(tok) == tok
    assert check_embedding_covers_tokenizer(tok + 64) == tok
    with pytest.raises(ValueError, match="smaller than the tokenizer"):
        check_embedding_covers_tokenizer(tok - 1)


def test_the_trainer_default_embedding_width_is_the_tokenizer_vocabulary():
    """The two defaults are one number, so the check above passes by
    construction unless somebody overrides one of them."""
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    from alphagrad.approx.ppo import make_argparser

    args = make_argparser().parse_args([])
    assert int(args.vocab_size) == incr_token_vocab()


def test_the_delta_count_header_is_its_own_int32_outside_the_token_buffer():
    """THE HEADER SCHEME.

    The count used to ride in slot 0 of the token buffer (``t[0] = n``). A
    count up to ``MAX_DELTA_TOKENS`` does not fit in a byte, so it moved out:
    the first ``DELTA_HEADER_SLOTS`` slots of the wire are one little-endian
    int32 and the tokens start after them.
    """
    from alphagrad.approx import env as _env

    t = np.asarray(_env._delta_observation(list(range(20)), 5))
    assert t.dtype == np.uint8
    assert t.shape == (_env.DELTA_HEADER_SLOTS + _env.MAX_DELTA_TOKENS,)
    assert int(_env.decode_delta_header(t)) == 15
    H = _env.DELTA_HEADER_SLOTS
    assert list(t[H:H + 15]) == list(range(5, 20))
    assert int(t[H + 15]) == _env.DELTA_TOKEN_PAD


def test_a_delta_count_above_a_byte_and_above_int_sixteen_survives_the_header():
    """THE ROUND TRIP THE HEADER EXISTS FOR.

    256 is the first count a byte cannot hold and 32768 is the first int16
    cannot; both are real delta lengths at the default budget. The header must
    return them exactly, or the encoder reads a truncated delta and its
    recurrence desyncs from the stream for the rest of the episode.
    """
    from alphagrad.approx import env as _env

    for n in (0, 1, 255, 256, 257, 32767, 32768,
              int(_env.MAX_DELTA_TOKENS)):
        if n > _env.MAX_DELTA_TOKENS:
            continue
        t = np.asarray(_env._delta_observation([1] * n, 0))
        got = int(_env.decode_delta_header(t))
        assert got == n, f"a delta of {n} tokens came back as {got}"


def test_the_header_codec_round_trips_every_count_the_budget_allows():
    from alphagrad.approx import env as _env

    for n in (0, 1, 255, 256, 65535, 65536, 1 << 24, (1 << 31) - 1, 1 << 31,
              (1 << 32) - 1):
        enc = _env.encode_delta_header(n)
        assert enc.dtype == np.uint8
        assert enc.shape == (_env.DELTA_HEADER_SLOTS,)
        assert int(_env.decode_delta_header(jnp.asarray(enc))) == n
    # The header is an UNSIGNED 32-bit count (owner ruling 2026-09-14), so
    # 2**32 is the first value refused, on the host, before anything is sent.
    with pytest.raises(ValueError, match="uint32 header"):
        _env.encode_delta_header(1 << 32)


def test_token_id_zero_is_a_real_token_so_padding_is_read_from_the_count():
    """WHY THE COUNT IS THE ONLY LENGTH.

    graphax's token 0 is the literal '-', which occurs INTERIOR to real
    streams. A reader that recovered a length by scanning for the pad value
    would stop at the first negative number in the delta. Nothing does: the
    header carries the tokenizer's own ``len()``, and this test shows a delta
    whose interior is all zeros coming back at its full length.
    """
    from graphax.jaxpr import get_vocab

    from alphagrad.approx import env as _env

    _vocab, n_vocab, _ = get_vocab(_DIGIT_BASE)
    assert n_vocab[0] == "-", (
        "token 0 is no longer '-'; recheck whether 0 can still occur inside a "
        "real stream before relaxing anything that depends on this")

    t = np.asarray(_env._delta_observation([0] * 7, 0))
    assert int(_env.decode_delta_header(t)) == 7
    H = _env.DELTA_HEADER_SLOTS
    assert list(t[H:H + 7]) == [0] * 7


def test_a_token_id_above_a_byte_raises_on_the_host():
    """THE RAISE THAT REPLACES A SILENT WRAP. A token past 255 means the
    tokenizer was built at a vocabulary the wire cannot carry."""
    from alphagrad.approx import env as _env

    with pytest.raises(ValueError, match="outside .0, 255."):
        _env._delta_observation([1, 256, 3], 0)
    # 255 itself is fine: the bound is inclusive.
    t = np.asarray(_env._delta_observation([1, 255, 3], 0))
    assert int(t[_env.DELTA_HEADER_SLOTS + 1]) == 255


# --------------------------------------------------------------------------
# 3b. the equation ids are GONE, at both ends
# --------------------------------------------------------------------------

def test_the_equation_id_budget_knob_raises_instead_of_being_ignored():
    """A knob whose name still reads like a budget but which nothing consults
    is how a run gets mis-read, so ``ALPHAGRAD_MAX_EQNS`` raises -- the same
    policy ``ALPHAGRAD_MAX_TOKENS`` got when it was deleted. Checked in a
    FRESH interpreter, because ``ppo`` reads it once at import."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env["JAX_PLATFORMS"] = "cpu"
    env["ALPHAGRAD_MAX_EQNS"] = "4096"
    env["PATH"] = os.environ.get("PATH", "")
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    env["HOME"] = os.environ.get("HOME", "/tmp")
    r = subprocess.run(
        [sys.executable, "-c",
         "import alphagrad.approx.ppo"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0, "ALPHAGRAD_MAX_EQNS was accepted silently"
    assert "ALPHAGRAD_MAX_EQNS is GONE" in r.stderr, r.stderr[-2000:]


def test_the_encoder_carry_has_three_leaves_and_no_equation_histogram():
    """``cumhist`` was ``(MAX_EQNS,) = 4096`` float32 PER STEP PER SAMPLE in
    the trajectory, and ``nvalid`` a scalar beside it. Both existed only to
    reproduce the relational forget-gate features inside the recurrence."""
    from alphagrad.approx.ppo import EncCarry

    assert EncCarry._fields == ("M", "I", "pos")


def test_the_palimpsa_mixers_have_no_relational_gate_and_take_no_equation_ids():
    """THE CONSUMER, removed. A ``rel_gate`` left behind would be a dead
    parameter the optimizer still updates, and an ``eqn_ids`` argument left
    behind would be a silent no-op for any caller that still passed one."""
    import inspect

    from alphagrad.transformer.palimpsa_encoder import (
        BiPalimpsaMixer, PalimpsaEncoder, PalimpsaEncoderLayer, PalimpsaMixer,
    )

    for cls in (PalimpsaMixer, BiPalimpsaMixer):
        assert not hasattr(cls, "_relational_gate_mod")
        assert "rel_gate" not in cls.__annotations__
        assert "eqn_ids" not in inspect.signature(cls.__call__).parameters
    for cls in (PalimpsaEncoderLayer, PalimpsaEncoder):
        assert "eqn_ids" not in inspect.signature(cls.__call__).parameters


def test_a_built_palimpsa_encoder_refuses_equation_ids():
    """The removal reaches the INSTANCE, not only the class: a caller that
    still hands this encoder ids gets a TypeError, loudly."""
    import jax.random as jrand

    from alphagrad.transformer import make_encoder

    enc = make_encoder("palimpsa", 1, 2, 8, 16, key=jrand.PRNGKey(0))
    xs = jnp.zeros((4, 8), jnp.float32)
    out = enc(xs, key=jrand.PRNGKey(1))
    assert out.shape == (4, 8)
    with pytest.raises(TypeError):
        enc(xs, eqn_ids=jnp.zeros((4,), jnp.int32), key=jrand.PRNGKey(1))


def test_no_trajectory_or_train_batch_leaf_carries_equation_ids():
    """THE STORE. Four leaves went: ``delta_eqns``, ``face_delta_eqns``,
    ``enc_cumhist`` and ``enc_nvalid``."""
    from alphagrad.approx.ppo import Trajectory, TrainBatch

    for cls in (Trajectory, TrainBatch):
        names = set(cls._fields)
        assert "delta_eqns" not in names
        assert "face_delta_eqns" not in names
        assert "enc_cumhist" not in names
        assert "enc_nvalid" not in names
        assert "delta_offset" in names
        assert "delta_count" in names


def test_the_env_state_carries_tokens_and_a_count_and_no_equation_buffer():
    """THE ROLLOUT STORE. ``reset`` is the one place the buffer is built from
    nothing, so it is where a widened dtype or a resurrected id buffer would
    reappear."""
    from alphagrad.approx import env as _env

    assert "delta_eqns" not in _env.EnvState._fields
    e = _make_delta_env()
    st = e.reset()
    assert np.asarray(st.delta_tokens).dtype == np.uint8
    assert np.asarray(st.delta_count).dtype == np.int32
    assert np.asarray(st.delta_tokens).shape == (_env.MAX_DELTA_TOKENS,)


def test_the_callback_wire_is_two_arrays_of_the_declared_width_and_dtype():
    """THE WIRE, as ``io_callback`` enforces it. The Ray measurement pool
    preallocates at ``env.obs_width`` / ``env.wire_token_dtype`` and stacks
    ``env.wire_arity`` arrays, so these descriptions of one buffer are checked
    against each other here."""
    from alphagrad.approx import env as _env

    e = _make_delta_env()
    assert e.obs_width == _env.DELTA_HEADER_SLOTS + _env.MAX_DELTA_TOKENS
    assert e.wire_token_dtype is _env.DELTA_TOKEN_DTYPE
    shp = e._callback_shape
    assert e.wire_arity == len(shp) == 2, (
        "the delta wire is (tokens, reward); a third array means the "
        "equation-id buffer came back")
    assert shp[0].shape == (e.obs_width,)
    assert shp[0].dtype == jnp.uint8
    assert shp[1].shape == (_env.NUM_REWARDS,)
    # And the env refuses to invent an equation dtype it does not have.
    with pytest.raises(RuntimeError, match="no equation-id buffer"):
        e.wire_eqn_dtype


def test_the_loss_reads_the_delta_through_a_cast_so_the_gather_stays_int32():
    """THE LOSS END. ``encode_extend`` is the single reader of the stored
    buffer; it casts to int32 before the embedding gather, so a byte-wide
    store cannot change the arithmetic. Its signature also no longer takes an
    equation buffer."""
    import inspect

    from alphagrad.approx.ppo import Agent

    src = inspect.getsource(Agent.encode_extend)
    assert 'fill_value=0).astype(jnp.int32)' in src
    assert "eqn" not in inspect.signature(Agent.encode_extend).parameters
    assert "eqn_ids_buf" not in src


def _make_delta_env():
    """A ``delta_obs`` env over the Perceptron example, no measurement.

    ``terminal_rewards_only`` keeps every non-terminal step a pure tokenizer
    step, so nothing here compiles or times a Jacobian.
    """
    from alphagrad.approx.env import EnvConfig, VertexEliminationEnv

    jaxpr, consts, args = _perceptron()
    cfg = EnvConfig(
        jaxpr=jaxpr, argnums=(2, 3, 4, 5), has_aux=False, sparse=False,
        cmp_type="flops", mem_type="peak_memory",
        terminal_rewards_only=True, delta_obs=True,
    )
    return VertexEliminationEnv(cfg, args=tuple(args), consts=list(consts),
                                num_envs=0)
