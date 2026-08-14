"""#109: bucket-compiled face widths for the Gumbel-AZ search hot path.

The face dimension of the per-decision jitted call (``az_gumbel._face_plan``)
is the configured ``MAX_FACES`` -- a topology-derived bound (2538 on the TLM
targets, even at seq16/dm64) while actual live-face counts per vertex are
~1-8. The face while_loop already runs the ACTUAL count, so the loop body
never visits padding faces; what the bound still costs on every single search
draw is (a) the device upload of the ``(prefix, MAX_FACES, ...)`` face-history
wires, (b) MAX_FACES-wide while_loop carries and pure_callback operands, and
(c) the MAX_FACES-wide outputs shipped back to the host.

Bucketing pads each draw to the smallest width in ``FACE_BUCKETS`` (final
fallback: the configured ``MAX_FACES``) that covers the live face count, and
jits once per bucket. Semantics are untouched by construction:

* the while_loop trip count is the live count either way;
* per-face RNG keys are ``fold_in(key, f)`` -- width-independent;
* the static sampling masks are per-face slices (``f_pair[f]``), identical
  for every real face at any width >= the live count;
* padding faces are never iterated, so their outputs are deterministic
  constants -- ``pad_face_outputs`` reproduces them bitwise on the host.

The caller re-pads outputs to the full width, so replay storage, the loss
and the PlanTokenizer wires see byte-identical arrays.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "FACE_BUCKETS",
    "bucket_width",
    "with_face_width",
    "pad_face_outputs",
    "hist_face_width",
]

# Ascending. The configured MAX_FACES is always the implicit final bucket
# (fallback for counts above the largest explicit bucket).
FACE_BUCKETS = (2, 4, 8, 32)


def bucket_width(n, max_faces, buckets=FACE_BUCKETS):
    """Smallest bucket >= ``n``; the configured ``max_faces`` as fallback.

    Never returns a width above ``max_faces`` (a bucket wider than the
    provable bound would claim face slots that cannot exist) and never one
    below the smallest bucket (``n <= 0`` is a legal empty enumeration --
    the while_loop then runs zero times at the smallest width).
    """
    n = int(n)
    mf = int(max_faces)
    for b in buckets:
        b = int(b)
        if n <= b <= mf:
            return b
    return mf


def with_face_width(agent, width):
    """``agent`` with ``face_path_policy.max_faces`` rebound to ``width``.

    ``max_faces`` is a STATIC eqx field -- part of the treedef, invisible to
    ``eqx.tree_at`` -- so the policy is shallow-copied field by field with
    the one static changed. Every parameter array is SHARED (no copy), and
    ``eqx.filter_jit`` sees a distinct static signature, i.e. exactly one
    compile per width. Identity when the width already matches or there is
    no face head (the exact arm).
    """
    import equinox as eqx

    pol = getattr(agent, "face_path_policy", None)
    if pol is None or int(pol.max_faces) == int(width):
        return agent
    pol2 = object.__new__(type(pol))
    for f in dataclasses.fields(pol):
        object.__setattr__(pol2, f.name, getattr(pol, f.name))
    object.__setattr__(pol2, "max_faces", int(width))
    return eqx.tree_at(lambda a: a.face_path_policy, agent, pol2)


# The translator's unused row: `micro_actions_to_rule_specs_jax` writes
# ``[-1, -1, 0]`` for OP_END sub-steps, so every face the while_loop never
# visited carries exactly this row through `to_env_action_dynamic`.
_END_SPEC_ROW = np.array([-1, -1, 0], dtype=np.int32)


def pad_face_outputs(max_faces, fr, fs, fa, f_pair, f_comp, f_valid, f_cnt,
                     f_ends=None):
    """Host-side re-pad of a bucketed draw's outputs to ``max_faces``.

    Fill values are the EXACT bytes the unbucketed call produces for faces
    the while_loop never visits:

    * ``fr``       END translator rows ``[-1, -1, 0]`` per slot;
    * ``fs``       0 (no skip);
    * ``fa``       `_face_loop`'s wire0 init (END ops, zeros, scale sign 1);
    * ``f_pair`` / ``f_comp``  the static sampling masks, which are a
      broadcast of ONE per-vertex row over the face axis -- so the padding
      rows equal row 0;
    * ``f_valid`` / ``f_cnt``  0;
    * ``f_ends``   0 -- the "no vertex" endpoint id, which gathers a zero
      context, exactly what an unvisited face slot has.

    ``fa`` is a ``heads.FaceAction`` of numpy arrays; returns the same tuple
    shape padded, all numpy.
    """
    from alphagrad.approx.heads import FaceAction, OP_END
    from alphagrad.approx.env import FACE_SLOTS

    F = int(max_faces)
    fr = np.asarray(fr, np.int32)
    fb = int(fr.shape[0])
    if fb >= F:
        _out = (fr, np.asarray(fs, np.int32), fa, np.asarray(f_pair),
                np.asarray(f_comp), np.asarray(f_valid),
                np.asarray(f_cnt))
        return _out if f_ends is None else _out + (
            np.asarray(f_ends, np.int32),)

    def _fill(x, fill):
        x = np.asarray(x)
        out = np.full((F,) + x.shape[1:], fill, dtype=x.dtype)
        out[:fb] = x
        return out

    fr2 = np.broadcast_to(
        _END_SPEC_ROW, (F, int(FACE_SLOTS), 3)).astype(np.int32).copy()
    fr2[:fb] = fr
    fs2 = _fill(np.asarray(fs, np.int32), 0)
    fa2 = FaceAction(
        skip=_fill(fa.skip, 0),
        op_type=_fill(fa.op_type, int(OP_END)),
        i=_fill(fa.i, 0),
        j=_fill(fa.j, 0),
        exponents=_fill(fa.exponents, 0),
        factor=_fill(fa.factor, 0),
        compress_kind=_fill(fa.compress_kind, 0),
        quant_dtype=_fill(fa.quant_dtype, 0),
        quant_scale_sign=_fill(fa.quant_scale_sign, 1),
        quant_scale_frac=_fill(fa.quant_scale_frac, 0.0),
    )
    fp = np.asarray(f_pair)
    fp2 = np.broadcast_to(fp[0], (F,) + fp.shape[1:]).copy()
    fp2[:fb] = fp
    fc = np.asarray(f_comp)
    fc2 = np.broadcast_to(fc[0], (F,) + fc.shape[1:]).copy()
    fc2[:fb] = fc
    fv2 = _fill(np.asarray(f_valid), np.asarray(f_valid).dtype.type(0))
    fn2 = _fill(np.asarray(f_cnt, np.int32), 0)
    if f_ends is None:
        return fr2, fs2, fa2, fp2, fc2, fv2, fn2
    fe2 = _fill(np.asarray(f_ends, np.int32), 0)
    return fr2, fs2, fa2, fp2, fc2, fv2, fn2, fe2


def hist_face_width(face_hist, skip_hist, n):
    """Highest face slot the committed prefix actually USES, over its first
    ``n`` decisions (0 when the prefix is empty).

    A history row matters to the prefix replay iff it carries a decided wire
    (``row[..., 0] >= 0`` -- both the ``EXACT_FACE_ROWS`` fill (-1) and the
    END translator row (-1) are excluded) or a skip. A decided face whose
    slots are ALL exact-END with no skip is indistinguishable from padding
    here, and slicing it away is value-identical: it decodes to no rules and
    contributes nothing to ``LiveFaceStream._decided`` (whose ``upto`` is
    bounded by the vertex's own key list anyway).
    """
    n = int(n)
    if n <= 0:
        return 0
    fh = np.asarray(face_hist[:n])
    sh = np.asarray(skip_hist[:n])
    used = np.any(fh[..., 0] >= 0, axis=2) | (sh != 0)      # (n, F) bool
    if not used.any():
        return 0
    F = used.shape[1]
    last = np.where(used.any(axis=1),
                    F - np.argmax(used[:, ::-1], axis=1), 0)
    return int(last.max())
