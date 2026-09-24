"""Per-face approximation policy on the 94-output head. NO ``_emit``.

The per-vertex ``UnifiedMicroPolicy`` needed ``_emit`` because its nine
independent axis gates could fire any number of times, so one decision had to
be expanded into up to ``max_substeps`` rule rows -- with a truncation rule and
a zero-axes fallback (``_canonical_axes``) to keep the expansion legal. The
94-head draws the reduce axis from a SOFTMAX, so each slot is exactly one rule
row and there is nothing to expand: the wire fields are written directly.

Per vertex: encode the axis set ONCE, then one head call per face. That is
1 encoder call + F head calls, against ``FacePathPolicy``'s F*(S+1) = 32
encoder calls and F*S = 24 head calls.

The sample/evaluate contract mirrors :class:`FacePathPolicy` exactly so the env
decoder, ``FaceAction`` wire format and the PPO loss are untouched. The joint
log-prob is returned as a SCALAR and stored as ``face_old_logp`` -- never
reconstructed from per-slot distributions, which is the failure that made the
per-vertex ratio 2.3e23 at epoch 0.
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.approx.face_action import FaceAction
from alphagrad.approx import face_action as _rec
from alphagrad.approx.heads import (
    AxisTokenFeatures, FactorTables, MAX_PRIMES, OP_COMPRESS, OP_DIAG, OP_END,
    OP_QUANT, _approx_allowed, _compute_axis_masks, _compute_op_legality,
    quant_hardware_masks,
)
from alphagrad.approx.common.masks import (
    FACE_QUANT_NARROW, NUM_FACE_QUANT_DTYPES)
from alphagrad.approx.unified_micro import FACE_DTYPE_SLOTS, _KIND_MAP
from alphagrad.approx.unified_face_head import (
    CONTRACTION_LAYOUT, FACE_SLOTS, MAX_PAIR_IDX, NUM_APPROX_OPS,
    NUM_REDUCE_AXES, OP_BLOCKDIAG, OP_NONE, OP_REDUCE, QUANT_SLOTS, S_OP,
    UnifiedFaceHead, FaceFields,
)


#: Record field -> the ``FaceFields`` attribute that holds it, for the per-FACE
#: fields only (the per-slot ones go through ``_rows``' op-dependent encoding).
#: ``skip`` is here for completeness; ``_wire_row`` skips it because the caller
#: already receives it. A record field with no entry RAISES in ``_wire_row``
#: rather than being dropped -- adding a per-face field to the declaration
#: without telling the head which logit drew it must not silently store a zero.
_FIELDS_OF_RECORD = {"skip": "skip", "quant": "quant", "join": "join"}

#: Head op (blockdiag, reduce, none) -> the wire's ``op_type`` (heads.OP_*),
#: and back. The wire's OP_QUANT is not a head op: it is the face's ``quant``
#: bit written onto the two operand slots, and it decodes to ``none`` because
#: that is what the head forced the slot's structural pick to.
_OP_WIRE = jnp.asarray([OP_DIAG, OP_COMPRESS, OP_END], jnp.int32)
_OP_HEAD = jnp.zeros((4,), jnp.int32).at[OP_DIAG].set(OP_BLOCKDIAG).at[
    OP_COMPRESS].set(OP_REDUCE).at[OP_QUANT].set(OP_NONE).at[OP_END].set(
    OP_NONE)
#: The wire's dtype column for a face whose bit is set: the narrow float's
#: index in the runtime catalog, on lhs and rhs alike.
_NARROW_SLOT = FACE_DTYPE_SLOTS[FACE_QUANT_NARROW]
#: The wire's ``(NUM_OPS,)`` profile override, read in the head's op order.
_OVERRIDE_HEAD_ORDER = jnp.asarray([OP_DIAG, OP_COMPRESS, OP_END], jnp.int32)


#: DIAGNOSTIC ONLY (ticket dsnn-dfw, the zero-approximation absorber).
#: ``ALPHAGRAD_FACE_DEBUG=1`` makes :meth:`UnifiedFacePolicy.sample_face` emit
#: one ``FACEDBG`` line per sampled face with the legality masks it drew
#: under, the skip probability, and the classes it actually drew. Nothing
#: else changes: the draw itself is untouched, the flag is read once at
#: import time and the print is compiled out when it is unset.
FACE_DEBUG = bool(int(os.environ.get("ALPHAGRAD_FACE_DEBUG", "0") or 0))

#: Host-side counters the debug sink accumulates. Keys are plain ints and
#: floats, so the sink costs one numpy pass per call and nothing on device.
FACE_DEBUG_COUNTS: dict = {}

_FACE_DEBUG_CLASSES = ("diag", "reduce", "none")


def _face_debug_sink(fv, aok, sk, om, ops, psk):
    """Accumulate one call's per-face draws into :data:`FACE_DEBUG_COUNTS`.

    Called through ``jax.debug.callback``, so the arrays arrive either
    unbatched (one face) or with a leading vmap axis (one face per
    environment). Both are flattened the same way. READ-ONLY: nothing here
    feeds back into the draw.
    """
    import numpy as _np
    fv = _np.asarray(fv).reshape(-1)
    b = int(fv.shape[0])
    aok = _np.asarray(aok).reshape(b)
    sk = _np.asarray(sk).reshape(b)
    psk = _np.asarray(psk).reshape(b)
    om = _np.asarray(om).reshape(b, -1, NUM_APPROX_OPS)
    ops = _np.asarray(ops).reshape(b, -1)
    live = fv > 0.5
    n_live = int(live.sum())
    if n_live == 0:
        return
    c = FACE_DEBUG_COUNTS
    c["faces"] = c.get("faces", 0) + n_live
    c["slots"] = c.get("slots", 0) + n_live * int(ops.shape[1])
    c["skip_drawn"] = c.get("skip_drawn", 0) + int((sk[live] > 0).sum())
    c["psk_sum"] = c.get("psk_sum", 0.0) + float(psk[live].sum())
    c["approx_ok_zero"] = (c.get("approx_ok_zero", 0)
                           + int((aok[live] < 0.5).sum()))
    for k, name in enumerate(_FACE_DEBUG_CLASSES):
        c["legal_" + name] = (c.get("legal_" + name, 0)
                              + int((om[live][:, :, k] > 0.5).sum()))
        c["drawn_" + name] = (c.get("drawn_" + name, 0)
                              + int((ops[live] == k).sum()))


def face_debug_report(tag: str = "") -> str:
    """Format and CLEAR the accumulated counters. Empty string when off."""
    c = FACE_DEBUG_COUNTS
    if not c:
        return ""
    n = max(int(c.get("faces", 0)), 1)
    s = max(int(c.get("slots", 0)), 1)
    parts = [f"[facedbg{(' ' + tag) if tag else ''}]",
             f"faces={c.get('faces', 0)}",
             f"slots={c.get('slots', 0)}",
             f"skip_drawn={c.get('skip_drawn', 0)}",
             f"p_skip_mean={c.get('psk_sum', 0.0) / n:.4f}",
             f"approx_ok_zero={c.get('approx_ok_zero', 0)}"]
    for name in _FACE_DEBUG_CLASSES:
        parts.append(f"legal_{name}={c.get('legal_' + name, 0)}"
                     f"({c.get('legal_' + name, 0) / s:.3f})")
    for name in _FACE_DEBUG_CLASSES:
        parts.append(f"drawn_{name}={c.get('drawn_' + name, 0)}")
    c.clear()
    return " ".join(parts)


class UnifiedFacePolicy(eqx.Module):
    """One decision per face; three operand slots, no unrolling.

    THE FACE'S INPUT IS ITS OWN PALIMPSA LATENT, AND NOTHING ELSE
    (2026-08-15). The head reads ``face_latent`` -- the parameter-free
    scatter of the palimpsa rows of that face's own token chunk, keyed by
    face, the same primitive the vertex slots are keyed by vertex.

    WHAT WAS REMOVED AND WHY. The input was ``[ctx_i || ctx_j ||
    face_latent]``: the two endpoint vertices' contexts gathered from the
    pointer, 3E wide. The endpoint contexts are a SECOND route from the same
    encoder into the same head, and this rewrite exists to leave exactly one
    of everything; the owner's instruction was "remove all ctx, and extents
    for now". So this is the MINIMAL baseline, deliberately: E in, 94 out.

    STATE THE EXPECTED RESULT HONESTLY. Every tokens-only face target
    measured ~0 within-step R2 (best 0.104 with shapes explicitly tokenised,
    against a 0.528 bar), and that was WITH full end-to-end gradient -- so
    fixing the gradient does not rescue the face side. Extents cleared the
    bar (main effect +0.51) and message passing added +0.19 on top. This
    class is what those get added back to, not a claim that they are
    unnecessary.

    What that replaces: an ``AxisSetEncoder`` re-run per face over the SAME
    vertex-level axis tokens, seeded with ``vertex_context + face_context``.
    Every face of a vertex fed it identical axis features (``face_sizes`` is
    deferred), so the encoder's only per-face input was the additive
    ``face_context`` it then mixed back down to one vector -- an encoder
    where a concatenation was the honest operation. The axis features are
    still built (``_face_feats_1``) because the LEGALITY MASKS are derived
    from them; they simply no longer go through an attention block.
    """

    head: UnifiedFaceHead

    max_faces: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    endpoint_read: bool = eqx.field(static=True)
    edge_mem: bool = eqx.field(static=True)
    allow_skip: bool = eqx.field(static=True, default=False)

    def __init__(self, embd_dim: int, num_heads: int, max_faces: int = 8,
                 num_encoder_layers: int = 1, max_groups: int = 16, *, key,
                 use_group_embedding: bool = False,
                 endpoint_read: bool = False,
                 edge_mem: bool = False,
                 allow_skip: bool = False,
                 approx_add: str = CONTRACTION_LAYOUT.mode):
        # num_heads / num_encoder_layers / max_groups / use_group_embedding
        # configured the deleted per-face AxisSetEncoder. They stay in the
        # signature because every trainer builds this policy positionally
        # from the shared arch defaults; they are INERT.
        self.embd_dim = embd_dim
        self.max_faces = max_faces
        self.endpoint_read = bool(endpoint_read)
        self.edge_mem = bool(edge_mem)
        self.allow_skip = bool(allow_skip)
        keys = jrand.split(key, 3)
        # in_dim = E: the face's own latent -- UNLESS --face-endpoint-read
        # (docs/FACE_LATENT_INFO_LOSS.md section 4), where the input is
        # [chunk_mean || slot_i || slot_j] and the head widens to 3E, and/or
        # --face-edge-mem (section 8), which appends the two EDGE-keyed rows
        # [.. || emem_lhs || emem_rhs] (+2E). Widths: E / 3E / 3E / 5E for
        # neither / one / the other / both. The key stream is untouched
        # either way, so a flag-off build is bit-identical to the pre-flag
        # policy.
        _in = embd_dim * (1 + (2 if self.endpoint_read else 0)
                          + (2 if self.edge_mem else 0))
        # `approx_add` sets the head's OUTPUT WIDTH: 94 logits under
        # lossy/lossless, 95 under choose, 125 under learned1, 156 under
        # learned2 (``unified_face_head`` module docstring). It is an
        # architecture parameter like `embd_dim`, PLUMBED from
        # ``ppo._build_agent``'s ``args.approx_add``, and that builder
        # cross-checks it against ``env.approx_add()`` so the head, the wire and
        # the engine cannot be built at three different widths.
        self.head = UnifiedFaceHead(embd_dim, in_dim=_in,
                                    key=keys[1], approx_add=approx_add)

    # ------------------------------------------------------------- the width
    @property
    def layout(self):
        """The head's :class:`FaceHeadLayout` -- THE source of the slot count.

        ``FACE_SLOTS`` is 3 and keeps meaning "the CONTRACTION slots", so the
        loops that iterate them stay correct at every width. Every SHAPE in
        this module asks ``self.n_slots`` instead, which is the head's own
        ``--approx-add`` width (3 / 3 / 3 / 4 / 5). Mixing the two is exactly
        how a learned join row gets applied to a tensor nothing was decided
        for.
        """
        return self.head.layout

    @property
    def n_slots(self) -> int:
        return self.head.layout.n_slots

    @property
    def approx_add(self) -> str:
        return self.head.layout.mode

    # ------------------------------------------------------------ masks
    @staticmethod
    def _fit(v, n):
        v = jnp.asarray(v, jnp.float32).ravel()
        return jnp.concatenate([v, jnp.zeros((n,), jnp.float32)])[:n]

    @staticmethod
    def _pair_sizes(features):
        sz = jnp.asarray(features.size, jnp.int32)
        return jnp.concatenate([sz, jnp.ones((MAX_PAIR_IDX,), jnp.int32)]
                               )[:MAX_PAIR_IDX]

    def _face_masks(self, features, pair_valid_f, comp_valid_f,
                    quant_legality_mask, op_override, tables):
        """(op, i, j, axis, pair_ok) masks, broadcast to the three slots.

        Every slot sees the SAME face, so the legality is the same for all
        three; they are stacked rather than shared so the head's per-slot
        signature stays honest if that ever stops being true.

        IT STOPPED BEING TRUE (ticket .18, D3; finding 54): the three slots
        hold three different tensors. When ``features`` is a LIST of
        ``self.n_slots`` per-slot features -- with ``pair_valid_f`` (S, N, N),
        ``comp_valid_f`` (S, N) and ``quant_legality_mask`` (S, D) to match,
        all from ``LiveFaceStream.face_slot_legality`` -- each slot's masks
        come from ITS OWN legality (:meth:`_slot_masks_1`) and are stacked.
        A single ``features`` keeps the historical broadcast, byte for byte.
        (A LIST, specifically: ``AxisTokenFeatures`` is itself a tuple.)
        """
        if not isinstance(features, list):
            op_legal = _compute_op_legality(
                features, pair_valid=pair_valid_f,
                compress_valid=comp_valid_f,
                quant_legality_mask=quant_legality_mask,
                op_override=op_override)
            i_diag, i_compress, j_diag, _ = _compute_axis_masks(
                features, pair_valid=pair_valid_f,
                compress_valid=comp_valid_f)
            _S = self.n_slots
            op_legal = jnp.asarray(op_legal, jnp.float32)
            om = jnp.broadcast_to(op_legal[_OVERRIDE_HEAD_ORDER],
                                  (_S, NUM_APPROX_OPS))
            im = jnp.broadcast_to(self._fit(i_diag, MAX_PAIR_IDX),
                                  (_S, MAX_PAIR_IDX))
            jm = jnp.broadcast_to(self._fit(j_diag, MAX_PAIR_IDX),
                                  (_S, MAX_PAIR_IDX))
            am = jnp.broadcast_to(self._fit(i_compress, NUM_REDUCE_AXES),
                                  (_S, NUM_REDUCE_AXES))
            # Non-coprime (i, j) table: gcd == 1 admits only factor 1, a no-op.
            #
            # PER-FACE UNDER --per-face-masks. `features` here is `_face_feats_1`'s
            # output, so once `face_sizes_f` is supplied these are THIS FACE's live
            # logical sizes (`masks.dim_logical_sizes`) rather than the vertex's
            # nominal ones -- the same numbers `masks.face_masks_and_sizes` screened
            # the pair with, and the same ones `_rows` derives the factor from. With
            # `face_sizes_f=None` it is the historical per-VERTEX table.
            sz = self._pair_sizes(features)
            g = jnp.gcd(sz[:, None], sz[None, :])
            pair_ok = jnp.broadcast_to((g > 1).astype(jnp.float32),
                                       (_S, MAX_PAIR_IDX, MAX_PAIR_IDX))
            # THE FACE BIT'S LEGALITY: the narrow float is a legal cast on
            # this face (the hardware mask times the face's own), and the
            # profile admits QUANT (op_legal carries the override).
            qm = (op_legal[OP_QUANT]
                  * self._fit(quant_legality_mask,
                              NUM_FACE_QUANT_DTYPES)[FACE_QUANT_NARROW])
            return om, im, jm, am, pair_ok, qm
        outs = [self._slot_masks_1(features[s], pair_valid_f[s],
                                   comp_valid_f[s], quant_legality_mask[s],
                                   op_override)
                for s in range(self.n_slots)]
        # The bit narrows lhs AND rhs and is legal iff the narrow float is a
        # real cast on at least one of them: the slot hook holds it as an
        # identity cast on an operand that is narrow already or has no value.
        qm = jnp.max(jnp.stack([outs[s][5] for s in QUANT_SLOTS]))
        return tuple(jnp.stack([o[k] for o in outs]) for k in range(5)) + (qm,)

    def _slot_masks_1(self, features, pair_valid, comp_valid,
                      quant_legality_mask, op_override):
        """ONE slot's (op, i, j, axis, pair_ok, q_narrow) from ONE slot's
        legality.

        Bottom-up hierarchical legality (ticket .59): a parent op is legal only
        if at least one of its sub-argument choices is legal.
          - Diag is legal only if at least one pair has gcd > 1 and pair_valid;
            im and jm are column/row projections of pair_ok, eliminating invalid
            axis fallbacks.
          - Reduce is legal only if at least one reduce axis is valid.
          - None is unconditionally legal.
          - ``q_narrow`` is this slot's share of the face's Quant bit: the
            narrow float is a legal, non-idempotent cast here (D4 identity
            quant masking), and the profile admits QUANT. The bit is legal
            when one operand slot's share is set.
        The resulting op_legal mask is composed with op_override.
        """
        sz = self._pair_sizes(features)
        g = jnp.gcd(sz[:, None], sz[None, :])
        pv = jnp.asarray(pair_valid, jnp.float32)
        pv = jnp.pad(pv, ((0, MAX_PAIR_IDX), (0, MAX_PAIR_IDX))
                     )[:MAX_PAIR_IDX, :MAX_PAIR_IDX]
        pair_ok = (g > 1).astype(jnp.float32) * pv
        im = (jnp.sum(pair_ok, axis=-1) > 0.0).astype(jnp.float32)
        jm = (jnp.sum(pair_ok, axis=0) > 0.0).astype(jnp.float32)
        diag_legal = (jnp.sum(pair_ok) > 0.0).astype(jnp.float32)

        _, i_compress, _, _ = _compute_axis_masks(
            features, pair_valid=pair_valid, compress_valid=comp_valid)
        am = self._fit(i_compress, NUM_REDUCE_AXES)
        reduce_legal = (jnp.sum(am) > 0.0).astype(jnp.float32)

        dm = self._fit(quant_legality_mask, NUM_FACE_QUANT_DTYPES)
        q_narrow = dm[FACE_QUANT_NARROW]

        none_legal = jnp.ones((), jnp.float32)

        if op_override is None:
            oo = jnp.ones((NUM_APPROX_OPS,), jnp.float32)
            q_over = jnp.ones((), jnp.float32)
        else:
            oo_wire = jnp.asarray(op_override, jnp.float32)
            oo = oo_wire[_OVERRIDE_HEAD_ORDER]
            q_over = oo_wire[OP_QUANT]
        om = jnp.stack([diag_legal, reduce_legal, none_legal]) * oo
        return om, im, jm, am, pair_ok, q_narrow * q_over

    def _slot_inputs(self, features, pair_valid_f, comp_valid_f,
                     quant_legality_mask, face_sizes_f, face_quant_f):
        """Per-slot inputs, or ``None`` when every input is per-FACE.

        The per-slot arrays come from ``face_slot_legality``
        (``--face-slot-frames``): sizes (S, N), quant (S, K), pair (S, N, N),
        comp (S, N). Any one of them present switches the whole face to the
        per-slot path; the others are broadcast to match, so an oracle-path
        pair mask can still travel with live per-slot sizes.
        """
        per = ((face_sizes_f is not None and jnp.ndim(face_sizes_f) == 2)
               or jnp.ndim(pair_valid_f) == 3 or jnp.ndim(comp_valid_f) == 2
               or (face_quant_f is not None and jnp.ndim(face_quant_f) >= 1))
        if not per:
            return None
        S = self.n_slots

        def _rows_of(x, rank):
            x = jnp.asarray(x)
            return x if x.ndim == rank + 1 else jnp.broadcast_to(
                x, (S,) + x.shape)

        sizes = None if face_sizes_f is None else _rows_of(face_sizes_f, 1)
        ff = [self._face_feats_1(features,
                                 None if sizes is None else sizes[s])
              for s in range(S)]
        pv = _rows_of(pair_valid_f, 2)
        cv = _rows_of(comp_valid_f, 1)
        fq = (None if face_quant_f is None
              else _rows_of(jnp.asarray(face_quant_f, jnp.float32),
                            0 if jnp.ndim(face_quant_f) <= 1 else 1))
        qm = jnp.stack([self._quant_mask_1(
            quant_legality_mask, None if fq is None else fq[s])
            for s in range(S)])
        return ff, pv, cv, qm

    def _face_feats(self, features, face_sizes, f):
        return self._face_feats_1(
            features, None if face_sizes is None else face_sizes[f])

    def _face_feats_1(self, features, sizes_f):
        """AxisTokenFeatures for ONE face's LIVE contraction.

        This is what makes a per-face decision mean anything. Without it the
        head sees the vertex's static features plus a face EMBEDDING -- a
        label, not information -- so every face looks identical and the head is
        approximating a contraction it never read. face_sizes comes from
        LiveVertexMaskOracle.face_features, which keeps the shape of the
        SparseTensor the mask probe already built.

        face_sizes is None falls back to the shared vertex features, which
        is the OLD behaviour -- kept so the blind path stays reachable for an
        A/B, not because it is correct.

        WHICH SIZES (--per-face-masks). The array wired in is
        ``LiveVertexMaskOracle.face_dim_sizes`` -- the LIVE ``logical_size`` of
        ``out_dims ++ primal_dims`` -- NOT ``face_features``' ``val.shape``.
        Only the former is indexed the way ``Diag(i, j)``, the (N, N) pair
        mask and ``rule_is_legal`` are indexed, so only the former makes the
        head's ``gcd``-derived factor mean the same thing as the mask that
        admitted the pair. ``tag_bits`` / ``group_id`` still come from the
        vertex: they are structural flags the env stamps after a per-vertex
        DIAG lands, which under --live-faces never happens (the vertex rule
        rows are always exact), and the oracle's ``pair_valid_f`` is AND-ed in
        downstream and is authoritative either way.
        """
        if sizes_f is None:
            return features
        sz = jnp.asarray(sizes_f, jnp.int32)
        n = features.size.shape[0]
        sz = jnp.concatenate([sz, jnp.zeros((n,), jnp.int32)])[:n]
        valid = (sz > 0).astype(jnp.float32)
        return AxisTokenFeatures(
            size=jnp.maximum(sz, 1),
            log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
            tag_bits=features.tag_bits,
            group_id=features.group_id,
            valid_mask=valid,
        )

    # ------------------------------------------------------------ wire
    def _rows(self, fields: FaceFields, features, tables):
        """FaceFields -> the env's per-slot wire fields. Direct, not expanded.

        Slot semantics, matching ``decode_vertex_rule_specs``:
          DIAG      -> i = axis-pair index i, j = index j, factor/exponents set
          COMPRESS  -> i carries the AXIS, compress_kind carries the fn
          END       -> all zero
          the face's ``quant`` bit -> a QUANT row (op_type OP_QUANT,
                       quant_dtype = the narrow float) on lhs AND rhs, never
                       on one of them; their structural picks are none.
        A skipped face forces every slot to END; the face's ``skip`` row is
        what becomes ``graphax.SKIP_FACE``.
        """
        keep = (fields.skip == 0)
        op = jnp.where(keep, fields.op, OP_NONE).astype(jnp.int32)
        _S = self.n_slots
        quant_slot = jnp.zeros((_S,), bool).at[jnp.asarray(QUANT_SLOTS)].set(
            True)
        is_qt = quant_slot & keep & (jnp.asarray(fields.quant) > 0)
        op = jnp.where(is_qt, OP_NONE, op).astype(jnp.int32)
        is_bd = op == OP_BLOCKDIAG
        is_rd = op == OP_REDUCE

        # factor = gcd(N_i, N_j): the LARGEST legal factor, i.e. the SMALLEST
        # blocks. Square pairs therefore give a pure diagonal. Sampling the
        # factor was removed upstream because the env clamped it to a divisor
        # of this gcd anyway -- the clamp, not the head, decided it.
        #
        # THE FACTOR IS NOT AN ACTION, AND UNDER --per-face-masks IT DOES NOT
        # NEED TO BE. `features` is the FACE's features there, so this gcd is
        # taken over the live operand's own logical sizes, and the mask
        # (`face_masks_and_sizes`, per_face=True) admits (i, j) only when
        # `base == 1` and `span % g_face == 0` on every dispatch mode. A free
        # pair has `base = 1, span = gcd`, so `factor = g_face` is exactly
        # `base * span` -- the finest end of `diag_pair_legal_factors`, LEGAL
        # BY CONSTRUCTION rather than by luck. That is the documented
        # deterministic factor rule; the 94-slot head layout is UNCHANGED (no
        # factor field, no head-shape change), and a learned factor field would
        # replace this one line without moving anything else.
        #
        # PER SLOT (ticket .18): a list of per-slot features gives each slot
        # its OWN sizes, so the factor is the gcd on the tensor the Diag hits.
        if isinstance(features, list):
            sz = jnp.stack([self._pair_sizes(f) for f in features])
            _s = jnp.arange(self.n_slots)
            N_i = sz[_s, jnp.clip(fields.i, 0, MAX_PAIR_IDX - 1)]
            N_j = sz[_s, jnp.clip(fields.j, 0, MAX_PAIR_IDX - 1)]
        else:
            sz = self._pair_sizes(features)
            N_i = sz[jnp.clip(fields.i, 0, MAX_PAIR_IDX - 1)]
            N_j = sz[jnp.clip(fields.j, 0, MAX_PAIR_IDX - 1)]
        g = jnp.gcd(N_i, N_j)
        exps = tables.max_exps[g].astype(jnp.int32)                # (S, P)
        primes = tables.primes[g]                                  # (S, P)
        factor = jnp.maximum(
            jnp.prod(jnp.where(primes > 0, primes, 1) ** exps, axis=-1)
            .astype(jnp.int32), 1)

        row = dict(
            op_type=jnp.where(is_qt, OP_QUANT, _OP_WIRE[op]).astype(jnp.int32),
            i=jnp.where(is_rd, fields.axis, jnp.where(is_bd, fields.i, 0)
                        ).astype(jnp.int32),
            j=jnp.where(is_bd, fields.j, 0).astype(jnp.int32),
            exponents=jnp.where(is_bd[:, None], exps,
                                jnp.zeros_like(exps)).astype(jnp.int32),
            factor=jnp.where(is_bd, factor, 0).astype(jnp.int32),
            compress_kind=jnp.where(is_rd, _KIND_MAP[fields.reduce_fn], 0
                                    ).astype(jnp.int32),
            quant_dtype=jnp.where(is_qt, _NARROW_SLOT, 0).astype(jnp.int32),
            quant_scale_sign=jnp.ones((_S,), jnp.int32),
            quant_scale_frac=jnp.zeros((_S,), jnp.float32),
        )
        # THE DECLARATION IS THE KEY SET. A key it does not know is a field
        # nothing stores; a declared key missing here is a field `sample`
        # leaves at a default and `evaluate` then scores. Both raise.
        _rec.check_wire_row_keys(row, self.approx_add, where="_rows")
        return row

    # ------------------------------------------- the codec, as ONE PAIR
    #
    # `_wire_row` and `_fields_of` are the encode/decode halves of the SAME
    # mapping, and they live next to each other because the third use of the
    # action record (`Agent._face_replay` -> `evaluate_face`) is the one whose
    # omission is SILENT: score a field sample never drew, or skip one it did,
    # and the PPO ratio leaves 1 with no error anywhere.
    #
    # What makes them provably a pair is not this comment but
    # `tests/face_action_record_test.py`, which round-trips
    # FaceFields -> _wire_row -> _fields_of -> _wire_row at every
    # `--approx-add` width and requires bit equality -- on the DERIVED fields
    # too, which `score` does not read and which therefore have to be
    # RE-DERIVED identically rather than carried.

    def _wire_row(self, fields: FaceFields, features, tables):
        """ONE face's complete slice of the action record.

        The per-slot wire fields (:meth:`_rows`) PLUS every per-FACE field the
        running width declares other than ``skip``, which the caller already
        has. Under ``choose`` that is the join bit, which rides the wire as
        ``face_join[f]`` -- the channel ``env._face_dict_for_vertex`` decodes
        with ``join_mode_of_bit``.
        """
        row = self._rows(fields, features, tables)
        for name in _rec.per_face_names(self.approx_add):
            if name == "skip":
                continue
            try:
                attr = _FIELDS_OF_RECORD[name]
            except KeyError:
                raise KeyError(
                    f"face_action.FACE_ACTION_FIELDS declares the per-face "
                    f"field {name!r} but unified_face_policy."
                    f"_FIELDS_OF_RECORD does not say which FaceFields "
                    f"attribute holds it, so `sample` would store a default "
                    f"and `evaluate` would score it. Add the mapping (and the "
                    f"logit in UnifiedFaceHead) before declaring the field."
                ) from None
            v = getattr(fields, attr)
            if v is None:
                raise ValueError(
                    f"the {self.approx_add!r} head declares the per-face "
                    f"record field {name!r} but FaceFields carries None for "
                    f"it. A default written here would be a decision the head "
                    f"never drew.")
            row[name] = jnp.asarray(v, _rec.field(name).np_dtype)
        return row

    def _fields_of(self, fa: FaceAction, f) -> FaceFields:
        """Face ``f`` of a STORED record, back as the head's own ``FaceFields``.

        The exact inverse of :meth:`_wire_row` for the SCORED fields, which are
        the only ones `score` reads: COMPRESS parked its axis in ``i``, DIAG
        parked the pair, the face's ``quant`` bit is its own record field.
        The DERIVED fields (``factor`` / ``exponents``, ``quant_dtype`` and
        the two quant-scale constants) are deliberately NOT read: they carry
        no decision, and reading them back would invent a variable for the
        loss to score. A wire OP_QUANT decodes to ``none``: that is what the
        head forced the operand slot's structural pick to.
        """
        op = _OP_HEAD[fa.op_type[f]].astype(jnp.int32)
        is_rd = op == OP_REDUCE
        if fa.quant is None:
            raise ValueError(
                "a stored face record carries quant=None: the rollout dropped "
                "the face's quant bit between `sample` and the replay. "
                "Scoring 0 here would make `evaluate` score a decision the "
                "behaviour policy never drew -- the ratio would not be 1 at "
                "epoch 0.")
        join = None
        if self.layout.has_choose:
            if fa.join is None:
                raise ValueError(
                    "--approx-add 'choose' decides the join PER FACE and this "
                    "stored record has join=None: the rollout dropped the bit "
                    "between `sample` and the replay. Scoring JOIN_LOSSY here "
                    "would make `evaluate` score a decision the behaviour "
                    "policy never drew -- the ratio would not be 1 at epoch 0.")
            join = jnp.asarray(fa.join[f], jnp.int32)
        elif fa.join is not None:
            raise ValueError(
                f"a stored face record carries a join bit but --approx-add "
                f"{self.approx_add!r} has no choose logit to score it against.")
        return FaceFields(
            skip=fa.skip[f],
            quant=jnp.asarray(fa.quant[f], jnp.int32),
            op=op,
            i=jnp.where(is_rd, 0, fa.i[f]).astype(jnp.int32),
            j=fa.j[f].astype(jnp.int32),
            axis=jnp.where(is_rd, fa.i[f], 0).astype(jnp.int32),
            reduce_fn=jnp.argmax(
                (_KIND_MAP[None, :] == fa.compress_kind[f][:, None]
                 ).astype(jnp.int32), axis=-1).astype(jnp.int32),
            join=join,
        )

    # -------------------------------------------------- ONE face at a time
    def _repr(self, face_latent):
        """The head's input: the face's own latent, E wide -- widened under
        ``endpoint_read`` (+2E: [.. || slot_i || slot_j]) and/or ``edge_mem``
        (+2E: [.. || emem_lhs || emem_rhs]); the caller concatenates, this
        module never gathers.

        NO face_embedding and no learned index table -- a label is not
        information. An absent latent is a ZERO vector, never a substitute:
        an empty chunk must leave the input at 0, exactly as
        ``Agent._face_encode``'s skip does.
        """
        if face_latent is None:
            n = self.embd_dim * (1 + (2 if self.endpoint_read else 0)
                                 + (2 if self.edge_mem else 0))
            return jnp.zeros((n,), jnp.float32)
        return face_latent

    @staticmethod
    def _quant_mask_1(quant_legality_mask, face_quant_f):
        """The hardware dtype mask narrowed by THIS face's QUANT legality.

        ``face_quant_f`` is either a scalar float (0/1) or a ``(K,)`` float
        array over ``masks.FACE_QUANT_DTYPES``. The hardware mask is indexed by
        the runtime catalog, so it is GATHERED at the face set's runtime
        indices (``FACE_DTYPE_SLOTS``), never sliced positionally.
        """
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        qhw = jnp.asarray(quant_legality_mask, jnp.float32)[FACE_DTYPE_SLOTS]
        if face_quant_f is None:
            return qhw
        fq = jnp.asarray(face_quant_f, jnp.float32)
        if fq.ndim == 0:
            return qhw * fq
        return qhw * fq[:NUM_FACE_QUANT_DTYPES]

    def sample_face(self, features: AxisTokenFeatures,
                    tables: FactorTables, key, f: int, pair_valid_f,
                    comp_valid_f, face_valid_f, *, face_context=None,
                    face_sizes_f=None, face_quant_f=None,
                    quant_legality_mask=None,
                    op_legality_override=None,
                    allow_skip=None):
        """Draw face ``f``'s decision. ``(skip, row, logp, ent, arity,
        skip_prob, op_dist)`` -- one slice of what :meth:`sample` stacks."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        per_slot = self._slot_inputs(
            features, pair_valid_f, comp_valid_f, quant_legality_mask,
            face_sizes_f, face_quant_f)
        if per_slot is None:
            quant_legality_mask = self._quant_mask_1(quant_legality_mask,
                                                     face_quant_f)
            ff = self._face_feats_1(features, face_sizes_f)
            om, im, jm, am, pair_ok, qm = self._face_masks(
                ff, pair_valid_f, comp_valid_f, quant_legality_mask,
                op_legality_override, tables)
        else:
            ff, _pv, _cv, _qm = per_slot
            om, im, jm, am, pair_ok, qm = self._face_masks(
                ff, _pv, _cv, _qm, op_legality_override, tables)
        ctx_f = self._repr(face_context)
        approx_ok = (jnp.asarray(allow_skip, dtype=jnp.float32)
                     if allow_skip is not None
                     else (1.0 if getattr(self, "allow_skip", False)
                           else _approx_allowed(op_legality_override)))
        z, fields, lp, e, ar = self.head.sample(
            ctx_f, key, op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
            quant_mask=qm, pair_ok=pair_ok, face_valid=face_valid_f > 0.5,
            approx_ok=approx_ok)
        if FACE_DEBUG:
            self._debug_line(om, z, fields, face_valid_f, approx_ok)
        return (fields.skip, self._wire_row(fields, ff, tables), lp, e, ar,
                jax.nn.sigmoid(z[0]), self._op_dist(z))

    def _debug_line(self, om, z, fields, face_valid_f, approx_ok):
        """Ship one sampled face to the host counters (ALPHAGRAD_FACE_DEBUG).

        ``om`` is the op legality the draw ran under, in the head's own
        order (blockdiag, reduce, quant, none); ``fields.op`` is what the
        draw returned, same order. ``psk`` is the skip Bernoulli's
        probability BEFORE the ``face_valid`` / ``approx_ok`` gate and
        ``fields.skip`` is what the gate left. Read-only.
        """
        jax.debug.callback(
            _face_debug_sink,
            jnp.asarray(face_valid_f, jnp.float32),
            jnp.asarray(approx_ok, jnp.float32),
            jnp.asarray(fields.skip, jnp.int32),
            jnp.asarray(om, jnp.float32),
            jnp.asarray(fields.op, jnp.int32),
            jax.nn.sigmoid(z[0]),
        )

    def evaluate_face(self, features: AxisTokenFeatures,
                      tables: FactorTables, fa: FaceAction, f: int,
                      pair_valid_f, comp_valid_f, face_valid_f, *,
                      face_context=None, face_sizes_f=None,
                      face_quant_f=None,
                      quant_legality_mask=None, op_legality_override=None,
                      allow_skip=None):
        """Score the stored face ``f`` under current params and STORED masks.
        Mirrors :meth:`sample_face` gate for gate -- anything less and the
        ratio is not 1 at epoch 0. That includes ``face_sizes_f`` and
        ``face_quant_f``: both enter the MASKS, so a replay that drops them
        scores a different distribution and the ratio silently leaves 1."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        per_slot = self._slot_inputs(
            features, pair_valid_f, comp_valid_f, quant_legality_mask,
            face_sizes_f, face_quant_f)
        if per_slot is None:
            quant_legality_mask = self._quant_mask_1(quant_legality_mask,
                                                     face_quant_f)
            ff = self._face_feats_1(features, face_sizes_f)
            om, im, jm, am, pair_ok, qm = self._face_masks(
                ff, pair_valid_f, comp_valid_f, quant_legality_mask,
                op_legality_override, tables)
        else:
            ff, _pv, _cv, _qm = per_slot
            om, im, jm, am, pair_ok, qm = self._face_masks(
                ff, _pv, _cv, _qm, op_legality_override, tables)
        ctx_f = self._repr(face_context)
        z = self.head.logits(ctx_f)
        # The stored record, back as the head's own decision -- `_wire_row`'s
        # inverse, and the one place the PPO ratio's "same variable" claim is
        # made good.
        fields = self._fields_of(fa, f)
        approx_ok = (jnp.asarray(allow_skip, dtype=jnp.float32)
                     if allow_skip is not None
                     else (1.0 if getattr(self, "allow_skip", False)
                           else _approx_allowed(op_legality_override)))
        lp, e, ar = self.head.score(
            z, fields, op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
            quant_mask=qm, pair_ok=pair_ok, face_valid=face_valid_f > 0.5,
            approx_ok=approx_ok)
        return lp, e, ar, jax.nn.sigmoid(z[0]), self._op_dist(z)

    def _op_dist(self, z):
        # `layout.slot_base`, not a restated `1 + 31*s`: one source of truth
        # for the offsets, and it raises rather than slicing if the band ever
        # moves. EVERY slot the width has, not the three contraction ones: a
        # learned slot whose op distribution nobody reported would be invisible
        # in the telemetry while the engine applied its row.
        _b = self.layout.slot_base
        return jax.nn.softmax(
            jnp.stack([z[_b(s) + S_OP:_b(s) + S_OP + NUM_APPROX_OPS]
                       for s in range(self.n_slots)]), axis=-1)

    # ------------------------------------------------------------ sample
    def sample(self, vertex_context, features: AxisTokenFeatures,
               tables: FactorTables, key, face_pair_valid, face_comp_valid,
               face_valid, quant_legality_mask=None,
               op_legality_override=None, face_sizes=None,
               face_quant=None, allow_skip=None):
        """``(FaceAction, joint_logp, joint_entropy, arity, skip_probs,
        op_dists, quant_logps)`` -- FacePathPolicy's contract, verbatim."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        F = self.max_faces
        keys = jrand.split(key, F)

        logp = jnp.array(0.0)
        ent = jnp.array(0.0)
        arity = jnp.array(0.0)
        skips, skip_probs, rows, op_dists, q_lps = [], [], [], [], []
        for f in range(F):
            # BLIND PATH (no live-faces stream): there are no per-face
            # tokens, so the latent is zero and every face of a vertex sees
            # the same (empty) input. Stated instead of implied -- this path
            # exists for the no-stream A/B, not because it can decide
            # anything. `vertex_context` is accepted and IGNORED: the head
            # reads one thing, and it is the face's own latent.
            sk, row, lp, e, ar, sp, od = self.sample_face(
                features, tables, keys[f], f,
                face_pair_valid[f], face_comp_valid[f], face_valid[f],
                face_sizes_f=None if face_sizes is None else face_sizes[f],
                face_quant_f=None if face_quant is None else face_quant[f],
                quant_legality_mask=quant_legality_mask,
                op_legality_override=op_legality_override,
                allow_skip=allow_skip)
            logp = logp + lp
            ent = ent + e
            arity = arity + ar
            skips.append(sk)
            skip_probs.append(sp)
            rows.append(row)
            op_dists.append(od)
            q_lps.append(jnp.zeros((self.n_slots,), jnp.float32))

        # THE RECORD, stacked over faces from the declaration's own key set:
        # `skip` plus whatever `_wire_row` carried, per-slot and per-face
        # alike. Nothing is named twice, so a field added to
        # FACE_ACTION_FIELDS arrives here with no edit.
        fa = FaceAction(
            skip=jnp.stack(skips),
            **{k: jnp.stack([r[k] for r in rows]) for k in rows[0]},
        )
        _rec.check(fa, self.approx_add, F, where="UnifiedFacePolicy.sample")
        return (fa, logp, ent, arity, jnp.stack(skip_probs),
                jnp.stack(op_dists), jnp.stack(q_lps))

    # ---------------------------------------------------------- evaluate
    def evaluate(self, vertex_context, features: AxisTokenFeatures,
                 tables: FactorTables, fa: FaceAction, face_pair_valid,
                 face_comp_valid, face_valid, quant_legality_mask=None,
                 op_legality_override=None, face_sizes=None,
                 face_quant=None, allow_skip=None):
        """Score a stored FaceAction under CURRENT parameters and the STORED
        masks. Mirrors :meth:`sample` gate for gate -- anything less and the
        ratio is not 1 at epoch 0."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        F = self.max_faces

        logp = jnp.array(0.0)
        ent = jnp.array(0.0)
        arity = jnp.array(0.0)
        skip_probs, op_dists, q_lps = [], [], []
        for f in range(F):
            lp, e, ar, sp, od = self.evaluate_face(
                features, tables, fa, f,
                face_pair_valid[f], face_comp_valid[f], face_valid[f],
                face_sizes_f=None if face_sizes is None else face_sizes[f],
                face_quant_f=None if face_quant is None else face_quant[f],
                quant_legality_mask=quant_legality_mask,
                op_legality_override=op_legality_override,
                allow_skip=allow_skip)
            logp = logp + lp
            ent = ent + e
            arity = arity + ar
            skip_probs.append(sp)
            op_dists.append(od)
            q_lps.append(jnp.zeros((self.n_slots,), jnp.float32))
        return (logp, ent, arity, jnp.stack(skip_probs),
                jnp.stack(op_dists), jnp.stack(q_lps))
