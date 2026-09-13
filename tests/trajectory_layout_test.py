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

   WHICH DECISION ACTUALLY EMITS AN ECHO, and why the old form of that test
   could never find one. graphax writes an ``approx`` block only when
   ``core._apply_face_transform`` records a micro-action: a literal
   ``Diag`` / ``Compress`` / ``Quant``, or a CHOOSER callable that returns
   one. alphagrad installs neither. ``env.make_slot_frame_hook`` is a plain
   tensor-returning callable, so graphax takes the ``out = _chosen`` branch,
   applies the transform and calls ``_record_micro`` for nobody -- the face
   sink gets no record and ``last_face_segments`` reports ``split == end``.
   So NO (vertex, slot, rule) triple on ANY graph can put an echo into a
   chunk through the slot wire; the old search was hunting something the
   engine cannot produce, which is why jobs 65344/65346 came back empty and
   why probe 65347 finds head == 0 even for rows that visibly change the
   emission. The one face decision that DOES emit an approximation block on
   this path is the SKIP channel: ``face_skips[f] == 1`` becomes
   ``graphax.SKIP_FACE``, which ``core._eliminate_vertex`` records directly
   as ``approx SKIP {}``. The test below therefore decides with SKIP, and
   reads the echo it expects off the emission env's OWN builder produces.

3. NARROWER ID DTYPES (NOT landed). Token ids do not fit in uint8 at the
   vocabulary the runs use, and the delta count rides in slot 0 of the SAME
   buffer as the equation ids, one value above what int16 can hold at the
   default budget. Both are pinned below.
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
                    vocab=512):
    """``(tokens, [(start, split, end)])`` of ONE vertex's elimination.

    Built through ``env._face_dict_for_vertex`` -- THE builder the measurement
    and ``LiveFaceStream._decided`` both use -- so this reference carries the
    same transforms the stream replays, and the echo it reports is graphax's
    own, not a re-derivation. ``[split:end]`` of a segment IS the face's
    approximation part (``IncrementalPathTokenizer.last_face_segments``).
    """
    from types import SimpleNamespace

    from graphax import IncrementalPathTokenizer

    from alphagrad.approx.env import _face_dict_for_vertex

    ref = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                   vocab_size=vocab)
    ref.base_tokens()
    keys = list(ref.ij.faces(int(vertex)))
    ft = _face_dict_for_vertex(SimpleNamespace(jaxpr=jaxpr), ref.ij,
                               int(vertex), rows, skips, keys=keys)
    toks = [int(t) for t in ref.eliminate(int(vertex), (), ft)]
    return toks, ref.last_face_segments()


def test_a_face_chunk_carries_the_previous_faces_approximation_echo_not_its_own():
    """THE REASON `face_delta_tokens` IS NOT THE NEXT STEP'S `delta_tokens`.

    ``LiveFaceStream.chunk`` returns face ``f-1``'s approximation echo
    followed by face ``f``'s own contraction, and reports the echo's length as
    ``head``. So every face's echo is carried by its SUCCESSOR's chunk, and
    the last face of a vertex has no successor: its echo is in the step's
    emission and in none of the stored chunks.

    The decision is the SKIP channel, which is the only face decision that
    emits an approximation block at all on this path (module docstring,
    section 2) and is legal on every face by construction -- it is a wire bit,
    not a masked rule, so nothing here can pass by silently skipping a
    decision the engine refused. The VERTEX is still searched: a vertex with
    one face has no successor to carry anything.

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
    VOCAB = 512

    lfs = LiveFaceStream(jaxpr, argnums, consts, args, vocab=VOCAB,
                         max_faces=F, max_axes=8, window=8192)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, MR, 3), np.int32)
    vspecs = -np.ones((MR, 3), np.int32)
    exact = -np.ones((F, S, 3), np.int32)
    no_skip = np.zeros((F,), np.int32)

    def _skips(upto):
        """Faces ``0..upto-1`` skipped, the rest exact -- exactly the prefix
        ``_decided`` replays when the stream is asked for face ``upto``."""
        s = np.zeros((F,), np.int32)
        s[:upto] = 1
        return s

    vertex, n_faces = None, 0
    for cand in range(1, V + 1):
        lfs._chunks.clear()
        k = int(lfs.chunk(order, specs, 0, cand, vspecs, exact, no_skip, 0)[3])
        if k >= 2:
            vertex, n_faces = cand, k
            break
    assert vertex is not None, (
        "no vertex of this graph has two faces, so no face has a successor "
        "and the echo cannot be read off it")

    # An EXACT vertex emits no approximation at all: no face has an echo, so
    # no chunk has a head. This is the control -- without it a head of 0 on
    # the skipped run could not be told from "the stream never reports one".
    exact_toks, exact_segs = _reference_step(
        jaxpr, argnums, consts, args, vertex, exact, no_skip, VOCAB)
    assert len(exact_segs) == n_faces
    assert all(sp == e for _s, sp, e in exact_segs), (
        f"vertex {vertex} emitted an approximation with no decision made: "
        f"{exact_segs}")
    for f in range(n_faces):
        lfs._chunks.clear()
        assert int(lfs.chunk(order, specs, 0, vertex, vspecs, exact,
                             no_skip, f)[5]) == 0

    # Now SKIP. Face f's chunk is read with faces 0..f-1 decided and face f
    # still undecided, so the emission it comes from is `_skips(f)`.
    heads = []
    for f in range(n_faces):
        sk = _skips(f)
        toks_f, segs_f = _reference_step(
            jaxpr, argnums, consts, args, vertex, exact, sk, VOCAB)
        lfs._chunks.clear()
        tok, _ids, cnt, nf, _ends, head = lfs.chunk(
            order, specs, 0, vertex, vspecs, exact, sk, f)
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
            f"face {f - 1} of vertex {vertex} was SKIPPED but emitted no "
            f"approximation block ({segs_f}); graphax no longer records a "
            f"skip, so this test can say nothing about the echo")
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
    all_toks, all_segs = _reference_step(
        jaxpr, argnums, consts, args, vertex, exact, _skips(n_faces), VOCAB)
    last_start, last_split, last_end = all_segs[n_faces - 1]
    assert last_end > last_split, (
        "the last face emitted no approximation block, so this vertex cannot "
        "show the gap")
    assert last_end == len(all_toks)
    lfs._chunks.clear()
    past = lfs.chunk(order, specs, 0, vertex, vspecs, exact,
                     _skips(n_faces), n_faces)
    assert int(past[2]) == 0 and int(past[5]) == 0, (
        "there is a chunk past the last face, so the last face's echo would "
        "have a carrier after all")

    # Counted: the chunks carry the echoes of faces 0..n-2 and no other. The
    # last face's echo is emitted and stored by nothing, which is the gap that
    # makes `face_delta_tokens` a different buffer from the step emission.
    echoes = [e - sp for _s, sp, e in all_segs]
    assert heads == [0] + echoes[:-1]
    assert sum(heads) == sum(echoes) - echoes[-1] < sum(echoes)


# --------------------------------------------------------------------------
# 3. id dtypes
# --------------------------------------------------------------------------

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


def test_a_byte_wide_id_space_leaves_too_few_name_symbols():
    """The number the byte budget is computed from.

    graphax appends vocabulary tokens by design ("New markers MUST be
    appended"), so the reserved count moves: it was 230 before the byte-only
    catalog and is 223 after it (job 65344). The test therefore pins the
    CONSEQUENCE, not the count: a 256-wide id space leaves fewer than 32 name
    symbols, against 279 at the 512 the env and the launchers run, and a name
    alphabet that small spells names out of several atoms and lengthens every
    delta (see the ALPHAGRAD_INCR_TOKEN_VOCAB comment in ``env.py``).
    """
    from graphax.jaxpr import get_vocab

    vocab, _, _ = get_vocab(_DIGIT_BASE)
    reserved = len(vocab) + _DIGIT_BASE
    names_at_256 = 256 - reserved
    names_at_512 = 512 - reserved
    assert 0 < names_at_256 < 32, (
        f"the reserved vocabulary is {len(vocab)} slots plus {_DIGIT_BASE} "
        f"digits, so a byte leaves {names_at_256} name symbols")
    assert names_at_512 >= 8 * names_at_256


def test_the_incremental_token_ids_at_the_configured_vocabulary_do_not_fit_in_a_byte():
    """WHY TOKEN IDS ARE NOT uint8.

    The env and the campaign launchers run the tokenizer at 512
    (``ALPHAGRAD_INCR_TOKEN_VOCAB`` default, ``--vocab-size 512``). The
    tokenizer's own cap is then 511, and a transformer-shaped graph really
    does emit ids above 255 in its BASE block alone -- before a single
    elimination has added a name.
    """
    tk = _encoder_tokenizer(512)
    assert tk.max_token_id() == 511
    base = [int(t) for t in tk.base_tokens()]
    assert max(base) > 255, (
        f"the base block tops out at {max(base)}; this graph no longer "
        "reaches past a byte, so it cannot stand for the transformer.")


def test_narrowing_the_token_vocabulary_to_a_byte_lengthens_every_delta():
    """WHY THE ANSWER IS NOT "just set the vocabulary to 256".

    Names are positional sequences over the name alphabet. 512 leaves 272
    symbols, 256 leaves 16, and a 16-symbol alphabet spells later names out of
    several atoms instead of one. The stream -- which is exactly what
    MAX_DELTA_TOKENS budgets -- gets longer, which is the opposite of what
    narrowing the dtype was for.
    """
    big = _encoder_tokenizer(512)
    small = _encoder_tokenizer(256)
    assert small.max_token_id() == 255
    n_big = len(list(big.base_tokens()))
    n_small = len(list(small.base_tokens()))
    assert n_small > n_big, (
        f"a 256-wide vocabulary spelled the base in {n_small} tokens against "
        f"{n_big} at 512 -- this graph is too small to show the cost.")


def test_the_delta_count_header_rides_in_slot_zero_of_both_transport_buffers():
    """WHY EQUATION IDS ARE NOT int16 AS THE CODE STANDS.

    ``_delta_observation`` writes the exact host-side token count into slot 0
    of BOTH wire arrays. ``env.step`` reads it from the token buffer only, but
    the equation buffer carries the same value -- and at the default
    ``MAX_DELTA_TOKENS`` of 32768 that value is one above what int16 holds.
    Narrowing the equation buffer therefore means first taking the count out
    of its slot 0, which is a change to the callback's shared wire form.
    """
    from alphagrad.approx import env as _env

    stream = list(range(20))
    seg_ids = [0] * 20
    t, e = _env._delta_observation(stream, seg_ids, 5)
    assert int(t[0]) == 15
    assert int(e[0]) == 15
    assert int(e[1 + 15]) == -1          # pad sentinel, not representable in uint8
    assert _env.MAX_DELTA_TOKENS >= 1
    if _env.MAX_DELTA_TOKENS > np.iinfo(np.int16).max:
        assert _env.MAX_DELTA_TOKENS == 32768, (
            "the default budget moved; recheck the int16 header argument.")
