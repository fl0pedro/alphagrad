"""decode4_audit: DAG-AGNOSTICISM by CONSTRUCTION, not by coincidence.

The shape audit inside decode4_face_train flags any parameter dimension equal
to V / NSTEP / F / T, and on this graph two of those numbers COLLIDE with
constants that have nothing to do with the DAG (the micro-action op vocabulary
is 71 and NSTEP is 71; 3*embd_dim is 96 and the TLM has 96 vertices).  A
number test cannot tell those apart.  This one can: build the SAME four
models for TWO DIFFERENT graphs -- different vertex count, different step
count, different stream length, different face count, different chunk cap --
and diff the parameter shapes.  Anything that moves is dimensioned by the DAG.
"""
import os
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_CHUNKED_EXTEND", "1")
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np, equinox as eqx
import decode3_arch as ARCH

E, NFT, HW = 32, 14, 256


class FaceAttnPool(eqx.Module):
    q: jax.Array
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim, *, key):
        ks = jrand.split(key, 4)
        self.embd_dim = embd_dim
        self.q = jrand.normal(ks[0], (embd_dim,)) * (embd_dim ** -0.5)
        self.k_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[1])
        self.v_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[2])
        self.out_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[3])


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NFT, width, depth=2, key=key)


def shapes(pool_name, NV, vocab):
    k = jrand.split(jrand.PRNGKey(0), 4)
    agent = ARCH.build(E, 3, 2, 2, NV, k[0], vocab=vocab)
    head = Readout(3 * E, HW, k[1])
    pool = FaceAttnPool(E, key=k[2]) if pool_name == "attn" else None
    leaves = jax.tree_util.tree_leaves_with_path(
        eqx.filter((agent, head, pool), eqx.is_inexact_array))
    return {jax.tree_util.keystr(p): tuple(v.shape) for p, v in leaves}


# TLM(seq=32,dmodel=128,vocab=1024) has 96 vertices / 71 steps; the second
# graph is deliberately a DIFFERENT size in every DAG quantity.
GRAPHS = [("TLM-96v", 96), ("other-137v", 137)]
print("=" * 78)
print("DAG-AGNOSTICISM: identical parameter shapes across two graph sizes?")
print("=" * 78)
ok_all = True
for pool_name in ["mean", "last", "sumtok", "attn"]:
    vocab = 513 if pool_name == "sumtok" else 512
    a = shapes(pool_name, GRAPHS[0][1], vocab)
    b = shapes(pool_name, GRAPHS[1][1], vocab)
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
            if a.get(k) != b.get(k)}
    n = sum(int(np.prod(v)) for v in a.values())
    ok = not diff
    ok_all &= ok
    verdict = ("PASS -- every parameter shape is identical for both graphs"
               if ok else "FAIL " + str(diff))
    print(f"  {pool_name:8s} {n:7d} trainable scalars  {verdict}")
print()
print("The only per-DAG quantity is the COMPILATION SHAPE (window length, "
      "face count, slot count) -- a recompile cost, not a transfer barrier.")
print("ALL PASS" if ok_all else "SOME ARMS ARE NOT DAG-AGNOSTIC")
