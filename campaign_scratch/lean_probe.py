"""Does gradient reach PALIMPSA'S BASE ENCODE? -- one script, both builds.

THE CLAIM UNDER TEST
--------------------
The vertex representation is produced by palimpsa reading the BASE token
stream. If that read happens OUTSIDE the differentiated region, the encoder
weights that produced it get NO cotangent from it: whatever reads the result
(the identity pool, the pointer) trains, but the thing that WROTE it does not.

HOW IT IS MEASURED (not asserted)
---------------------------------
The base encode is run with a SECOND COPY of the agent, ``agent_b``, whose
values are identical to ``agent``'s. Gradient w.r.t. ``agent_b`` is by
construction exactly "the gradient that flows back through the base encode",
separated from the delta-encode gradient that shares the same weights. Three
topologies are differentiated:

  PROD    what the build actually runs.
  INSIDE  the base encode moved inside the differentiated region.
  DELTA   the K=1 step-delta encode (the only palimpsa gradient the old
          build had), for scale.

On the OLD build PROD contains no base encode at all, so ``d/d agent_b`` is
structurally zero -- and INSIDE reports the magnitude that is being thrown
away. On the NEW build PROD == INSIDE.

The script auto-detects which API it is running against, so the SAME file
produces the before and the after number.

Run:  JAX_PLATFORMS=cpu uv run --no-sync python lean_probe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the gate module for its env pins + build_case (the same tiny branched
# MLP the regression gate drives, so this probe exercises the real policy path
# on a graph small enough to differentiate on a CPU).
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
import policy_regression_gate as G  # noqa: E402  (pins env vars at import)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphagrad.approx.common import carry_stream as CS  # noqa: E402

LEGACY = hasattr(CS, "base_identity_stream")


def _leaf_norms(tree, prefix):
    """(count, l2) over inexact leaves whose path contains ``prefix``."""
    params = eqx.filter(tree, eqx.is_inexact_array)
    tot, n = 0.0, 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        p = jax.tree_util.keystr(path)
        if prefix in p:
            tot += float(jnp.sum(jnp.asarray(leaf) ** 2))
            n += int(np.asarray(leaf).size)
    return n, float(np.sqrt(tot))


def _nparams(tree):
    params = eqx.filter(tree, eqx.is_inexact_array)
    return sum(int(np.asarray(x).size)
               for x in jax.tree_util.tree_leaves(params))


def main():
    case = G.build_case()
    env, agent = case["env"], case["agent"]
    total_v = case["total_v"]
    EMBD = G.EMBD
    W = case["window"]

    state = env.reset()
    base_tok, base_n = env.base_observation()
    base_w = max(int(base_n), 1)
    base_tok = base_tok[:base_w]
    try:
        base_own = env.base_owners()
    except Exception:
        base_own = None

    print(f"[probe] API = {'LEGACY (identity_stream)' if LEGACY else 'LEAN (base_mem)'}")
    print(f"[probe] base stream = {int(base_n)} tokens, total_v = {total_v}, "
          f"embd_dim = {EMBD}")
    print(f"[probe] agent params = {_nparams(agent)}")
    for name, sub in (("encoder", ".encoder"), ("vertex_policy", ".vertex_policy"),
                      ("face head", ".face_path_policy"),
                      ("identity_pool", ".identity_pool"),
                      ("ctx_proj", ".ctx_proj")):
        n, _ = _leaf_norms(agent, sub)
        if n:
            print(f"[probe]   {name:14s} {n}")

    def _base(a):
        return CS.init_carry(a, base_tok, base_n, window=base_w,
                             total_v=total_v, embd_dim=EMBD,
                             base_owners=base_own)

    enc0, m0s, m0c = _base(agent)
    if LEGACY:
        vs0, vc0 = m0s, m0c                 # old: init_carry seeds the memory
        base_mem = None
        ident = CS.base_identity_stream(
            agent, base_tok, base_n, window=base_w,
            total_v=total_v, base_owners=base_own)
    else:
        vs0, vc0 = CS.zero_memory(total_v, EMBD)
        base_mem = (m0s, m0c)
        ident = None

    part = jnp.zeros((total_v + 1,), jnp.float32).at[0].set(1.0)
    # At `env.reset()` the step delta is EMPTY (the base stream has just been
    # consumed), so scoring the delta path there measures nothing. The scale
    # reference is therefore ONE delta of the base stream's own length --
    # real tokens, the real `advance` path, a realistic count.
    _dw = int(case["window"])
    dtok = jnp.concatenate([base_tok, jnp.zeros((_dw,), jnp.int32)])[:_dw]
    dcnt = jnp.asarray(min(int(base_n), _dw), jnp.int32)
    owner = jnp.asarray(0, jnp.int32)

    def _heads(a, vs, vc, bm, idt):
        if LEGACY:
            return CS.heads(a, vs, vc, identity_stream=idt, preference=None)
        return CS.heads(a, vs, vc, base_mem=bm, preference=None)

    def _scalar(triple):
        lg, ctx, val = triple
        lg = jnp.where(jnp.isfinite(lg), lg, 0.0)
        return (jnp.sum(lg ** 2) + jnp.sum(ctx ** 2) + jnp.sum(val ** 2))

    # ---------------------------------------------------------------- PROD
    # Exactly what the build runs: the base encode already happened, with
    # `agent`, outside. `agent_b` therefore appears nowhere on the OLD build.
    def f_prod(agent_b):
        if LEGACY:
            return _scalar(_heads(agent, vs0, vc0, None, ident))
        # LEAN: the production loss recomputes the base memory under gradient.
        _e, bs, bc = _base(agent_b)
        c2, vs2, vc2 = CS.advance(
            agent, enc0, vs0, vc0, dtok, dcnt, owner,
            window=_dw, participants=part, chunk=0)
        return _scalar(_heads(agent, vs2, vc2, (bs, bc), None))

    # -------------------------------------------------------------- INSIDE
    def f_inside(agent_b):
        if LEGACY:
            idt = CS.base_identity_stream(
                agent_b, base_tok, base_n, window=base_w,
                total_v=total_v, base_owners=base_own)
            _e, bs, bc = _base(agent_b)
            c2, vs2, vc2 = CS.advance(
                agent, enc0, bs, bc, dtok, dcnt, owner,
                window=_dw, participants=part, chunk=0)
            return _scalar(_heads(agent, vs2, vc2, None, idt))
        return f_prod(agent_b)

    # --------------------------------------------------------------- DELTA
    def f_delta(agent_d):
        c2, vs2, vc2 = CS.advance(
            agent_d, enc0, vs0, vc0, dtok, dcnt, owner,
            window=_dw, participants=part, chunk=0)
        return _scalar(_heads(agent, vs2, vc2, base_mem, ident))

    rows = []
    for tag, fn in (("PROD", f_prod), ("INSIDE", f_inside),
                    ("DELTA(K=1)", f_delta)):
        g = eqx.filter_grad(fn)(agent)
        n_all, l2_all = _leaf_norms(g, "")
        _, l2_enc = _leaf_norms(g, ".encoder")
        _, l2_emb = _leaf_norms(g, ".embedding")
        rows.append((tag, l2_all, l2_enc, l2_emb))

    print()
    print("  topology       ||grad|| (all)   ||grad|| encoder   ||grad|| embedding")
    for tag, a, e, m in rows:
        print(f"  {tag:13s} {a:16.6e} {e:18.6e} {m:20.6e}")

    prod_enc = rows[0][2]
    inside_enc = rows[1][2]
    delta_enc = rows[2][2]
    print()
    print(f"GRADIENT REACHING THE BASE ENCODE (production topology): "
          f"{prod_enc:.6e}")
    print(f"GRADIENT AVAILABLE THROUGH THE BASE ENCODE            : "
          f"{inside_enc:.6e}")
    print(f"GRADIENT THROUGH THE K=1 STEP DELTA (for scale)       : "
          f"{delta_enc:.6e}")
    print(f"[probe] (the DELTA row uses one synthetic delta of "
          f"{int(base_n)} real tokens: at env.reset() the step delta is "
          f"empty and would measure nothing)")
    if prod_enc == 0.0:
        print("VERDICT: the base encode receives ZERO gradient in production.")
    else:
        print(f"VERDICT: the base encode receives gradient; it is "
              f"{prod_enc / max(delta_enc, 1e-30):.2f}x the K=1 delta path.")


if __name__ == "__main__":
    main()
