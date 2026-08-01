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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.approx.heads import (
    AxisSetEncoder, AxisTokenFeatures, FaceAction, FactorTables, MAX_PRIMES,
    _approx_allowed, _compute_axis_masks, _compute_op_legality,
    quant_hardware_masks,
)
from alphagrad.approx.unified_micro import _BF16_SLOT, _F32_SLOT, _KIND_MAP
from alphagrad.approx.unified_face_head import (
    FACE_SLOTS, MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES,
    OP_BLOCKDIAG, OP_NONE, OP_QUANT, OP_REDUCE, UnifiedFaceHead, FaceFields,
)


class UnifiedFacePolicy(eqx.Module):
    """One 94-output decision per face; three operand slots, no unrolling."""

    encoder: AxisSetEncoder
    head: UnifiedFaceHead

    max_faces: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, num_heads: int, max_faces: int = 8,
                 num_encoder_layers: int = 1, max_groups: int = 16, *, key,
                 use_group_embedding: bool = False):
        self.embd_dim = embd_dim
        self.max_faces = max_faces
        keys = jrand.split(key, 3)
        self.encoder = AxisSetEncoder(
            embd_dim, num_heads, num_layers=num_encoder_layers,
            max_groups=max_groups, key=keys[0],
            use_group_embedding=use_group_embedding)
        self.head = UnifiedFaceHead(embd_dim, key=keys[1])

    # ------------------------------------------------------------ masks
    def _face_masks(self, features, pair_valid_f, comp_valid_f,
                    quant_legality_mask, op_override, tables):
        """(op, i, j, axis, pair_ok) masks, broadcast to the three slots.

        Every slot sees the SAME face, so the legality is the same for all
        three; they are stacked rather than shared so the head's per-slot
        signature stays honest if that ever stops being true.
        """
        op_legal = _compute_op_legality(
            features, pair_valid=pair_valid_f, compress_valid=comp_valid_f,
            quant_legality_mask=quant_legality_mask, op_override=op_override)
        i_diag, i_compress, j_diag, _ = _compute_axis_masks(
            features, pair_valid=pair_valid_f, compress_valid=comp_valid_f)

        def _fit(v, n):
            v = jnp.asarray(v, jnp.float32).ravel()
            return jnp.concatenate([v, jnp.zeros((n,), jnp.float32)])[:n]

        om = jnp.broadcast_to(_fit(op_legal, NUM_APPROX_OPS),
                              (FACE_SLOTS, NUM_APPROX_OPS))
        im = jnp.broadcast_to(_fit(i_diag, MAX_PAIR_IDX),
                              (FACE_SLOTS, MAX_PAIR_IDX))
        jm = jnp.broadcast_to(_fit(j_diag, MAX_PAIR_IDX),
                              (FACE_SLOTS, MAX_PAIR_IDX))
        am = jnp.broadcast_to(_fit(i_compress, NUM_REDUCE_AXES),
                              (FACE_SLOTS, NUM_REDUCE_AXES))
        # Non-coprime (i, j) table: gcd == 1 admits only factor 1, a no-op.
        sz = jnp.asarray(features.size, jnp.int32)
        sz = jnp.concatenate([sz, jnp.ones((MAX_PAIR_IDX,), jnp.int32)]
                             )[:MAX_PAIR_IDX]
        g = jnp.gcd(sz[:, None], sz[None, :])
        pair_ok = jnp.broadcast_to((g > 1).astype(jnp.float32),
                                   (FACE_SLOTS, MAX_PAIR_IDX, MAX_PAIR_IDX))
        return om, im, jm, am, pair_ok

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
          QUANT     -> quant_dtype carries the dtype slot
          END       -> all zero
        A skipped face forces every slot to END; the face's ``skip`` row is
        what becomes ``graphax.SKIP_FACE``.
        """
        keep = (fields.skip == 0)
        op = jnp.where(keep, fields.op, OP_NONE).astype(jnp.int32)
        is_bd = op == OP_BLOCKDIAG
        is_rd = op == OP_REDUCE
        is_qt = op == OP_QUANT

        # factor = gcd(N_i, N_j): the LARGEST legal factor, i.e. the SMALLEST
        # blocks. Square pairs therefore give a pure diagonal. Sampling the
        # factor was removed upstream because the env clamped it to a divisor
        # of this gcd anyway -- the clamp, not the head, decided it.
        sz = jnp.asarray(features.size, jnp.int32)
        sz = jnp.concatenate([sz, jnp.ones((MAX_PAIR_IDX,), jnp.int32)]
                             )[:MAX_PAIR_IDX]
        N_i = sz[jnp.clip(fields.i, 0, MAX_PAIR_IDX - 1)]
        N_j = sz[jnp.clip(fields.j, 0, MAX_PAIR_IDX - 1)]
        g = jnp.gcd(N_i, N_j)
        exps = tables.max_exps[g].astype(jnp.int32)                # (S, P)
        primes = tables.primes[g]                                  # (S, P)
        factor = jnp.maximum(
            jnp.prod(jnp.where(primes > 0, primes, 1) ** exps, axis=-1)
            .astype(jnp.int32), 1)

        zeros = jnp.zeros((FACE_SLOTS,), jnp.int32)
        return dict(
            op_type=op,
            i=jnp.where(is_rd, fields.axis, jnp.where(is_bd, fields.i, 0)
                        ).astype(jnp.int32),
            j=jnp.where(is_bd, fields.j, 0).astype(jnp.int32),
            exponents=jnp.where(is_bd[:, None], exps,
                                jnp.zeros_like(exps)).astype(jnp.int32),
            factor=jnp.where(is_bd, factor, 0).astype(jnp.int32),
            compress_kind=jnp.where(is_rd, _KIND_MAP[fields.reduce_fn], 0
                                    ).astype(jnp.int32),
            quant_dtype=jnp.where(
                is_qt, jnp.where(fields.dtype_idx > 0, _BF16_SLOT, _F32_SLOT),
                0).astype(jnp.int32),
            quant_scale_sign=jnp.ones((FACE_SLOTS,), jnp.int32),
            quant_scale_frac=jnp.zeros((FACE_SLOTS,), jnp.float32),
        )

    # -------------------------------------------------- ONE face at a time
    def _ctx(self, vertex_context, face_context, ff, f):
        """The head's input for face ``f``.

        ``face_context`` (the palimpsa summary of the tokens emitted for this
        face) is added to the vertex context BEFORE the axis-set encoder, so
        it conditions the axis tokens rather than only shifting the head's
        input. ``None`` reproduces the old vertex-only context exactly.
        """
        base = (vertex_context if face_context is None
                else vertex_context + face_context)
        # NO face_embedding: the face's identity is its header tokens
        # (`path <central> & <in> & <out>` opens every chunk), so a learned
        # index table was a label where content already exists -- and a
        # cap-sized parameter in a design whose width is per-graph.
        return self.encoder(ff, base)[1]

    def sample_face(self, vertex_context, features: AxisTokenFeatures,
                    tables: FactorTables, key, f: int, pair_valid_f,
                    comp_valid_f, face_valid_f, *, face_context=None,
                    face_sizes_f=None, quant_legality_mask=None,
                    op_legality_override=None):
        """Draw face ``f``'s decision. ``(skip, row, logp, ent, arity,
        skip_prob, op_dist)`` -- one slice of what :meth:`sample` stacks."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        ff = self._face_feats_1(features, face_sizes_f)
        ctx_f = self._ctx(vertex_context, face_context, ff, f)
        om, im, jm, am, pair_ok = self._face_masks(
            ff, pair_valid_f, comp_valid_f, quant_legality_mask,
            op_legality_override, tables)
        z, fields, lp, e, ar = self.head.sample(
            ctx_f, key, op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
            pair_ok=pair_ok, face_valid=face_valid_f > 0.5,
            approx_ok=_approx_allowed(op_legality_override))
        return (fields.skip, self._rows(fields, ff, tables), lp, e, ar,
                jax.nn.sigmoid(z[0]), self._op_dist(z))

    def evaluate_face(self, vertex_context, features: AxisTokenFeatures,
                      tables: FactorTables, fa: FaceAction, f: int,
                      pair_valid_f, comp_valid_f, face_valid_f, *,
                      face_context=None, face_sizes_f=None,
                      quant_legality_mask=None, op_legality_override=None):
        """Score the stored face ``f`` under current params and STORED masks.
        Mirrors :meth:`sample_face` gate for gate -- anything less and the
        ratio is not 1 at epoch 0."""
        if quant_legality_mask is None:
            quant_legality_mask = quant_hardware_masks()[0]
        ff = self._face_feats_1(features, face_sizes_f)
        ctx_f = self._ctx(vertex_context, face_context, ff, f)
        om, im, jm, am, pair_ok = self._face_masks(
            ff, pair_valid_f, comp_valid_f, quant_legality_mask,
            op_legality_override, tables)
        z = self.head.logits(ctx_f)
        # Read the fields back off the stored wire rows -- the inverse of
        # _rows. COMPRESS parked the axis in `i`, DIAG parked the pair.
        op = fa.op_type[f]
        is_rd = op == OP_REDUCE
        fields = FaceFields(
            skip=fa.skip[f],
            op=op,
            i=jnp.where(is_rd, 0, fa.i[f]).astype(jnp.int32),
            j=fa.j[f].astype(jnp.int32),
            axis=jnp.where(is_rd, fa.i[f], 0).astype(jnp.int32),
            reduce_fn=jnp.argmax(
                (_KIND_MAP[None, :] == fa.compress_kind[f][:, None]
                 ).astype(jnp.int32), axis=-1).astype(jnp.int32),
            dtype_idx=(fa.quant_dtype[f] == _BF16_SLOT).astype(jnp.int32),
        )
        lp, e, ar = self.head.score(
            z, fields, op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
            pair_ok=pair_ok, face_valid=face_valid_f > 0.5,
            approx_ok=_approx_allowed(op_legality_override))
        return lp, e, ar, jax.nn.sigmoid(z[0]), self._op_dist(z)

    @staticmethod
    def _op_dist(z):
        return jax.nn.softmax(
            jnp.stack([z[1 + 31 * s:1 + 31 * s + NUM_APPROX_OPS]
                       for s in range(FACE_SLOTS)]), axis=-1)

    # ------------------------------------------------------------ sample
    def sample(self, vertex_context, features: AxisTokenFeatures,
               tables: FactorTables, key, face_pair_valid, face_comp_valid,
               face_valid, quant_legality_mask=None,
               op_legality_override=None, face_sizes=None):
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
            sk, row, lp, e, ar, sp, od = self.sample_face(
                vertex_context, features, tables, keys[f], f,
                face_pair_valid[f], face_comp_valid[f], face_valid[f],
                face_sizes_f=None if face_sizes is None else face_sizes[f],
                quant_legality_mask=quant_legality_mask,
                op_legality_override=op_legality_override)
            logp = logp + lp
            ent = ent + e
            arity = arity + ar
            skips.append(sk)
            skip_probs.append(sp)
            rows.append(row)
            op_dists.append(od)
            q_lps.append(jnp.zeros((FACE_SLOTS,), jnp.float32))

        fa = FaceAction(
            skip=jnp.stack(skips),
            **{k: jnp.stack([r[k] for r in rows]) for k in rows[0]},
        )
        return (fa, logp, ent, arity, jnp.stack(skip_probs),
                jnp.stack(op_dists), jnp.stack(q_lps))

    # ---------------------------------------------------------- evaluate
    def evaluate(self, vertex_context, features: AxisTokenFeatures,
                 tables: FactorTables, fa: FaceAction, face_pair_valid,
                 face_comp_valid, face_valid, quant_legality_mask=None,
                 op_legality_override=None, face_sizes=None):
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
                vertex_context, features, tables, fa, f,
                face_pair_valid[f], face_comp_valid[f], face_valid[f],
                face_sizes_f=None if face_sizes is None else face_sizes[f],
                quant_legality_mask=quant_legality_mask,
                op_legality_override=op_legality_override)
            logp = logp + lp
            ent = ent + e
            arity = arity + ar
            skip_probs.append(sp)
            op_dists.append(od)
            q_lps.append(jnp.zeros((FACE_SLOTS,), jnp.float32))
        return (logp, ent, arity, jnp.stack(skip_probs),
                jnp.stack(op_dists), jnp.stack(q_lps))
