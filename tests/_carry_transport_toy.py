"""A two-graph toy of the carry transport: one step body, two carry blocks.

The policy's graph attaches two carried values through CONTRACTIONS, the
dense container's form ``s + J @ d``, and the measured graph through ROW
SUMS, the diag container's ``s + sum(J * d)``, with the second state
attached first; ``d`` is ``w - stop_gradient(w)`` on both, as in
``graphax.examples.neuromorphic.attach_rsnn_past``. The step body reads the
two attached states and is the same on both, so ``carry_plan._variant_of``
aligns the two programs the way it aligns RSNN_SHD's containers, in
milliseconds.

    vertex   policy's graph             measured graph
    1        stop_gradient(w)           stop_gradient(w)
    2        sub: d                     sub: d
    3        dot_general(J, d)          broadcast_in_dim(d)  \\
    4        add: s' (attached)         mul(K, .)             | rewritten
    5        dot_general(K, d)          reduce_sum           /
    6        add: u' (attached)         add: u' (attached)
    7        mul(s', u')  <- step body  broadcast_in_dim(d)  \\
    8        mul(w, .)                  mul(J, .)             | rewritten
    9        sin                        reduce_sum           /
    10       mul(., x)                  add: s' (attached)
    11       reduce_sum (the output)    mul(s', u')  <- step body
    12                                  mul(w, .)
    13                                  sin
    14                                  mul(., x)
    15                                  reduce_sum (the output)

Vertices 1, 2, 4 and 6 of the policy's graph have counterparts (1, 2, 10 and
6), 3 and 5 have none. A face's position is its rank in the live graph's
variable order, so the two faces of vertex 7 through the attached states,
``(s', 7, .)`` and ``(u', 7, .)``, are faces 0 and 1 on the policy's graph and
1 and 0 on the measured one.
"""

import types

import jax
import jax.numpy as jnp
from graphax.examples.neuromorphic import snn_carry_scope

N = 3
ARGNUMS = (3,)
#: policy vertex -> measured vertex: the step body with its output, and the
#: carry vertices that have a counterpart
BODY = {7: 11, 8: 12, 9: 13, 10: 14}
OUTPUT = {11: 15}
ATTACHED = {4: 10, 6: 6}
DIFFERENCE = {1: 1, 2: 2}
REWRITTEN = (3, 5)


def policy_target(x, s, u, w, J, K):
    with snn_carry_scope():
        d = w - jax.lax.stop_gradient(w)
        s_att = s + J @ d
        u_att = u + K @ d
    return jnp.sum(jnp.sin(w * (s_att * u_att)) * x)


def measured_target(x, s, u, w, J, K):
    with snn_carry_scope():
        d = w - jax.lax.stop_gradient(w)
        u_att = u + jnp.sum(K * d[:, None], axis=-1)
        s_att = s + jnp.sum(J * d[:, None], axis=-1)
    return jnp.sum(jnp.sin(w * (s_att * u_att)) * x)


def program(fn, args):
    """``{"config", "args", "consts"}``, the form ``carry_plan`` keeps a
    program in."""
    cj = jax.make_jaxpr(fn)(*args)
    return {"config": types.SimpleNamespace(jaxpr=cj.jaxpr, argnums=ARGNUMS),
            "args": tuple(args), "consts": tuple(cj.literals)}


def variant(container="diag"):
    """The measured program as ``carry_plan`` holds a variant, against the
    policy's program as its ``base``."""
    from alphagrad.approx.common import carry_plan as CP
    f32 = jnp.float32
    x = jnp.arange(1.0, N + 1.0, dtype=f32)
    s = jnp.linspace(0.1, 0.3, N, dtype=f32)
    u = jnp.linspace(0.7, 0.4, N, dtype=f32)
    w = jnp.linspace(0.5, 0.9, N, dtype=f32)
    dense = (0.5 * jnp.eye(N) + 0.1).astype(f32)
    base = program(policy_target, (x, s, u, w, dense, 0.5 * dense))
    alt = program(measured_target, (x, s, u, w, jnp.full((N, 1), 0.6, f32),
                                    jnp.full((N, 1), 0.3, f32)))
    return CP._variant_of(container, alt["config"], alt["args"],
                          alt["consts"], base)
