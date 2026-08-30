"""decode_bench: cost of one palimpsa encode+backward over a long stream."""
import os, sys, time
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, jax.random as jrand
from types import SimpleNamespace

from alphagrad.approx.ppo import _build_agent
from alphagrad.approx.common import carry_stream as cs
from alphagrad.approx import vertex_memory as vmem

L = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
E = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 3
B = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.environ.setdefault("ALPHAGRAD_CHUNKED_EXTEND", "1")
print("chunked_extend", os.environ.get("ALPHAGRAD_CHUNKED_EXTEND"),
      "block", os.environ.get("ALPHAGRAD_CHUNK_BLOCK", "128"))

V = 96
args = SimpleNamespace(vocab_size=512, embd_dim=E, op_embd_dim=8, num_layers=NL,
                       num_heads=2, hidden_dim=64, value_dims="64,32",
                       set_pointer=True, set_pointer_blocks=2,
                       dynamic_substeps=False, no_approx_head=True,
                       live_faces=False, face_actions=False,
                       unified_head=False, unified_face_head=False,
                       max_substeps=1)
agent = _build_agent(args, V, 1, 1, jrand.PRNGKey(0))
print("params", sum(x.size for x in jax.tree.leaves(eqx.filter(agent, eqx.is_inexact_array))))

rng = np.random.default_rng(0)
toks = jnp.asarray(rng.integers(1, 470, size=(B, L)), jnp.int32)
eqns = jnp.asarray(rng.integers(0, 200, size=(B, L)), jnp.int32)
owners = jnp.asarray(rng.integers(-1, V, size=(B, L)), jnp.int32)
vfeat = jnp.asarray(rng.normal(size=(V, 10)).astype(np.float32))
vfeat = vfeat.at[:, 0].set(jnp.abs(vfeat[:, 0]) * 3)
tgt = jnp.asarray(rng.normal(size=(B, V)).astype(np.float32))


def one(agent, tok, eqn, own):
    c0 = agent.carry_init()
    c1, rows, valid, eq = agent.encode_extend(c0, tok, eqn, L, window=L, start=0, chunk=0)
    s = jnp.zeros((V + 1, E), jnp.float32); c = jnp.zeros((V + 1,), jnp.float32)
    s, c = vmem.update_ids(s, c, rows, own, valid)
    logits, ctx, val = agent.heads_from_memory(s, c, vertex_features=vfeat)
    return ctx


def loss(agent):
    ctx = jax.vmap(one, in_axes=(None, 0, 0, 0))(agent, toks, eqns, owners)
    return jnp.mean((ctx[..., 0] - tgt) ** 2)


f = eqx.filter_jit(eqx.filter_value_and_grad(loss))
t = time.time(); v, g = f(agent); v.block_until_ready(); print(f"compile+first {time.time()-t:.1f}s loss {v:.4f}")
t = time.time()
for _ in range(3):
    v, g = f(agent); v.block_until_ready()
print(f"per fwd+bwd step ({B} traj, L={L}): {(time.time()-t)/3:.2f}s")
