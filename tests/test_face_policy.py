#!/usr/bin/env python3
"""sample() == evaluate() for UnifiedFacePolicy, and no _emit anywhere.

The parity test is the one that matters: sample returns the joint log-prob that
gets STORED as face_old_logp, and evaluate recomputes it as the new side of the
PPO ratio. If they disagree by any amount, the ratio is not 1 at epoch 0 --
which is exactly the bug that ran the per-vertex head to 2.3e23.

The round trip through the wire fields is where it can break: _rows parks the
COMPRESS axis in `i` and the DIAG pair in `i`/`j`, so evaluate has to invert
that correctly or it scores a different variable than sample drew.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np

from alphagrad.approx.heads import AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables
from alphagrad.approx.unified_face_policy import UnifiedFacePolicy
from alphagrad.approx.unified_face_head import FACE_SLOTS

FAIL = []


def ck(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def feats(sizes):
    sz = jnp.asarray(sizes, jnp.int32)
    n = len(sizes)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def main():
    E, F = 32, 8
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F, key=jrand.PRNGKey(0))
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))
    f = feats([8, 8, 4, 16, 6, 4])
    N = f.size.shape[0]

    fpv = jnp.ones((F, N, N), jnp.float32)
    fcv = jnp.ones((F, N), jnp.float32)
    fval = jnp.concatenate([jnp.ones((5,)), jnp.zeros((F - 5,))]).astype(jnp.float32)

    print("=== A. no _emit / no max_substeps in the new path ===")
    import alphagrad.approx.unified_face_policy as P
    import alphagrad.approx.unified_face_head as H
    import inspect
    # Strip docstrings: both modules NAME these in prose to explain why they
    # are gone, so a raw substring search reports its own documentation.
    import ast, re
    def code_only(mod):
        t = ast.parse(inspect.getsource(mod))
        for node in ast.walk(t):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        return ast.unparse(t)
    src = code_only(P) + code_only(H)
    ck("no _emit", "_emit" not in src)
    ck("no _canonical_axes", "_canonical_axes" not in src)
    ck("no max_substeps", "max_substeps" not in src)

    print("\n=== B. shapes ===")
    fa, lp, ent, ar, sp, od, ql = pol.sample(
        ctx, f, tables, jrand.PRNGKey(2), fpv, fcv, fval)
    ck("skip is (F,)", fa.skip.shape == (F,), str(fa.skip.shape))
    ck("op_type is (F, 3)", fa.op_type.shape == (F, FACE_SLOTS),
       str(fa.op_type.shape))
    ck("one rule row per slot (no substep axis)", fa.i.ndim == 2,
       f"i.shape={fa.i.shape}")

    print("\n=== C. sample() logp == evaluate() logp ===")
    d = []
    for s in range(200):
        fa, lp, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(100 + s),
                                fpv, fcv, fval)
        lp2, *_ = pol.evaluate(ctx, f, tables, fa, fpv, fcv, fval)
        d.append(abs(float(lp) - float(lp2)))
    d = np.array(d)
    ck("max |dlogp| < 1e-5", d.max() < 1e-5, f"max={d.max():.3e}")
    ck("ratio == 1 at epoch 0",
       float(np.exp(np.clip(d, -700, 700)).max()) < 1.00002,
       f"max ratio={np.exp(d).max():.8f}")

    print("\n=== D. padding faces are inert ===")
    fa, lp_a, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(7), fpv, fcv, fval)
    ck("padding faces drew skip=0", bool(np.all(np.asarray(fa.skip)[5:] == 0)),
       f"{np.asarray(fa.skip)}")

    print("\n=== E. one encoder + one head call PER FACE ===")
    # This used to assert ONE encoder call for the whole vertex. That was only
    # true while every face shared a context -- which is exactly the blindness
    # test F rules out: a shared context means the encoder cannot have read the
    # face. Per-face contexts (live tokens, live shapes) require a call each.
    # The property worth pinning is that it is one per FACE and not one per
    # (face, sub-step): FacePathPolicy ran the encoder 32x and the head 24x.
    calls = [0]
    orig = type(pol.encoder).__call__
    def counted(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)
    type(pol.encoder).__call__ = counted
    try:
        pol.sample(ctx, f, tables, jrand.PRNGKey(3), fpv, fcv, fval)
    finally:
        type(pol.encoder).__call__ = orig
    ck("encoder called once per face, not per sub-step "
       "(FacePathPolicy: 32)", calls[0] == F, f"{calls[0]} calls, F={F}")

    print("\n=== F. the head SEES the face (not just a label) ===")
    # Two faces with DIFFERENT live contraction shapes must produce different
    # logits. If they do not, the head is reading an embedding index and the
    # per-face decision is decoration.
    fs = np.zeros((F, N), np.int32)
    fs[0, :3] = [8, 4, 2]
    fs[1, :3] = [16, 6, 3]
    fs[2, :2] = [4, 4]
    fsz = jnp.asarray(fs)
    ff0 = pol._face_feats(f, fsz, 0)
    ff1 = pol._face_feats(f, fsz, 1)
    ck("per-face features differ",
       not bool(jnp.all(ff0.size == ff1.size)),
       f"{np.asarray(ff0.size)} vs {np.asarray(ff1.size)}")
    # face_embedding is DELETED: identity comes from the face's own tokens
    # (the chunk opens with `path <central> & <in> & <out>`), so the "label
    # only" baseline below is now genuinely zero-information -- two faces
    # with identical features are indistinguishable, which is the point.
    z0 = pol.head.logits(pol.encoder(ff0, ctx)[1])
    z1 = pol.head.logits(pol.encoder(ff1, ctx)[1])
    # and against the BLIND path: same features, only the embedding differs
    b0 = pol.head.logits(pol.encoder(f, ctx)[1])
    b1 = pol.head.logits(pol.encoder(f, ctx)[1])
    d_seeing = float(jnp.max(jnp.abs(z0 - z1)))
    d_blind = float(jnp.max(jnp.abs(b0 - b1)))
    ck("seeing the contraction changes the logits more than the label alone",
       d_seeing > d_blind, f"seeing={d_seeing:.4f} blind(label only)={d_blind:.4f}")
    fa_s, lp_s, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(11), fpv, fcv,
                                fval, face_sizes=fsz)
    lp_s2, *_ = pol.evaluate(ctx, f, tables, fa_s, fpv, fcv, fval,
                             face_sizes=fsz)
    ck("parity still exact with per-face features",
       abs(float(lp_s) - float(lp_s2)) < 1e-5,
       f"d={abs(float(lp_s)-float(lp_s2)):.3e}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
