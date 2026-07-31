#!/usr/bin/env python3
"""Batched callback must be bit-identical to the per-env one.

Run twice (ALPHAGRAD_BATCHED_CALLBACK=0 / =1) and diff the digests.
"""
import hashlib, os, sys
import jax, jax.numpy as jnp, numpy as np

from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX, StepAction, VertexEliminationEnv)

E = 4


def build():
    from graphax import examples
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    cj = jax.make_jaxpr(examples.Helmholtz)(x)
    return VertexEliminationEnv.from_jaxpr(
        cj, args=(x,), argnums=(0,), num_envs=0,
        target_fun=examples.Helmholtz, cmp_type="flops",
        mem_type="peak_memory", measure_latency=False)


def main():
    env = build()
    st = env.reset()
    vv = list(env.valid_vertices)
    # E distinct actions so no two envs share inputs (rules out CSE dedup).
    tv = jnp.asarray([vv[i % len(vv)] for i in range(E)], dtype=jnp.int32)
    specs = jnp.broadcast_to(
        jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32).at[..., 2].set(0),
        (E, MAX_RULES_PER_VERTEX, 3))
    acts = StepAction(target_vertex=tv, rule_specs=specs)
    def _b(x):
        a = jnp.asarray(x)
        return jnp.broadcast_to(a, (E,) + a.shape)
    states = jax.tree_util.tree_map(_b, st)

    out = jax.vmap(env.step, in_axes=(0, 0))(states, acts)
    r = np.asarray(out.reward)
    tok = np.asarray(out.state.tokens)
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(r).tobytes())
    h.update(np.ascontiguousarray(tok).tobytes())
    flag = os.environ.get("ALPHAGRAD_BATCHED_CALLBACK", "0")
    print(f"BATCHED={flag} reward_shape={r.shape} tokens_shape={tok.shape}")
    print(f"BATCHED={flag} digest={h.hexdigest()}")
    print(f"BATCHED={flag} reward_row0={np.round(r[0], 6).tolist()}")
    print(f"BATCHED={flag} reward_row{E-1}={np.round(r[E-1], 6).tolist()}")


if __name__ == "__main__":
    main()
