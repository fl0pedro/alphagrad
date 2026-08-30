"""Is the learned per-vertex IDENTITY a real, discriminative channel?

Three questions, on the policy gate's own graph and agent:
  1. at init it must be EXACTLY zero (out_proj is zeroed, like every other
     additive context path here), so the swap cannot move the initial policy;
  2. with a non-zero out_proj the per-vertex identities must DIFFER from each
     other -- otherwise the pool is a constant and cannot tell candidates
     apart;
  3. it must move the pointer's logits, and gradient must reach the pool.
"""
import sys
sys.argv = [sys.argv[0]]
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import tests.policy_regression_gate as G
from alphagrad.approx.common import carry_stream as CS

case = G.build_case()
agent, env, total_v = case["agent"], case["env"], case["total_v"]
bt, be, bn = env.base_observation()
bw = max(int(bn), 1)
own = env.base_owners()
enc, vs, vc = CS.init_carry(agent, bt[:bw], be[:bw], bn, window=bw,
                            total_v=total_v, embd_dim=G.EMBD, base_owners=own)
stream = CS.base_identity_stream(agent, bt[:bw], be[:bw], bn, window=bw,
                                 total_v=total_v, base_owners=own)
rows, ids, valid = stream
print("base rows:", rows.shape, "owned rows:", int(jnp.sum(ids >= 0)),
      "distinct owners:", int(jnp.unique(ids).shape[0]))

ident0 = agent.identity(stream, total_v + 1)
print("Q1 identity at init: max|.| = %.3e (expect exactly 0)"
      % float(jnp.max(jnp.abs(ident0))))

key = jax.random.PRNGKey(0)
w = jax.random.normal(key, agent.identity_pool.out_proj.weight.shape) * 0.5
agent2 = eqx.tree_at(lambda a: a.identity_pool.out_proj.weight, agent, w)
ident = np.asarray(agent2.identity(stream, total_v + 1))
occ = np.abs(ident).sum(-1) > 0
print("Q2 non-empty slots:", int(occ.sum()), "of", ident.shape[0])
d = [[float(np.linalg.norm(ident[i] - ident[j])) for j in range(len(ident))]
     for i in range(len(ident))]
off = [d[i][j] for i in range(len(d)) for j in range(len(d)) if i != j
       and occ[i] and occ[j]]
print("Q2 pairwise identity distance: min %.4f mean %.4f max %.4f"
      % (min(off), sum(off) / len(off), max(off)))

l0 = CS.heads(agent, vs, vc, identity_stream=stream)[0]
l1 = CS.heads(agent2, vs, vc, identity_stream=stream)[0]
print("Q3 vertex logits without/with identity:",
      np.round(np.asarray(l0), 4), np.round(np.asarray(l1), 4))
print("Q3 max logit shift: %.4f" % float(jnp.max(jnp.abs(l1 - l0))))


def _loss(a):
    lg, ctx, val = CS.heads(a, vs, vc, identity_stream=stream)
    return jnp.sum(jax.nn.log_softmax(lg) ** 2) + jnp.sum(val ** 2)


g = eqx.filter_grad(_loss)(agent2)
gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(
    eqx.filter(g.identity_pool, eqx.is_inexact_array)))))
print("Q3 grad norm into identity_pool: %.4e (must be > 0)" % gn)
