"""END-TO-END repro: drive the REAL MicroActionPolicy over the nn256 graph the
failing PPO job uses, decode its actions exactly as env._callback does, run the
measurement, and count "TRANSFORM DID NOT FIT".

ALPHAGRAD_LIVE_MASKS=0 -> the pre-fix behaviour (policy masked from tag bits
only).  =1 (default) -> the live-edge oracle masks are handed to the policy.
"""
import os, sys, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jax.numpy as jnp, jax.random as jrand, numpy as np
from graphax import inline_call_primitives
from graphax.core import vertex_elimination_jaxpr

from alphagrad.approx.common import get_args, get_fn
from alphagrad.approx.common.masks import LiveVertexMaskOracle
from alphagrad.approx.env import (
    MAX_AXES_PER_VERTEX, MAX_RULES_PER_VERTEX,
    compute_static_axis_state, micro_actions_to_rule_specs_jax,
    rule_specs_to_transforms,
)
from alphagrad.approx.heads import (
    MicroActionPolicy, precompute_factor_tables)
from alphagrad.approx.ppo import _axis_features_from_state

USE_MASKS = os.environ.get("ALPHAGRAD_LIVE_MASKS", "1") != "0"
N_EPISODES = int(os.environ.get("REPRO_EPISODES", "40"))
MAX_SUBSTEPS = int(os.environ.get("REPRO_SUBSTEPS", "1"))


def traced_inlined(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    from jax.extend.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def main():
    key = jrand.PRNGKey(7)
    _, args_key, agent_key = jrand.split(key, 3)
    fn = get_fn("VmappedNeuralNetwork")
    xs = get_args("VmappedNeuralNetwork", args_key, dataset="mnist")
    cj = traced_inlined(fn, xs)
    jaxpr, consts = cj.jaxpr, cj.literals
    total_v = len(jaxpr.eqns)
    argnums = (2, 3, 4, 5)

    axis_state_np, axis_valid_np = compute_static_axis_state(jaxpr, total_v)
    axis_state_j = jnp.asarray(axis_state_np)
    axis_valid_j = jnp.asarray(axis_valid_np)
    max_axis_size = int(max(
        int(s) for e in jaxpr.eqns
        for v in list(e.outvars) + list(e.invars)
        if hasattr(v, "aval") for s in (v.aval.shape or (1,))
    ))
    tables = precompute_factor_tables(max_axis_size)
    policy = MicroActionPolicy(
        embd_dim=32, num_heads=2, max_substeps=MAX_SUBSTEPS,
        num_encoder_layers=1, key=agent_key)
    v_ctx = jnp.zeros((32,), jnp.float32)

    sample = jax.jit(lambda feats, k, pv, cv: policy.sample(
        v_ctx, feats, tables, k, pair_valid=pv, compress_valid=cv))
    to_specs = jax.jit(micro_actions_to_rule_specs_jax)

    fails = collections.Counter()
    n_ep = n_step = n_rule = 0
    probe_s = 0.0
    import time
    for ep in range(N_EPISODES):
        k_ep = jrand.PRNGKey(1000 + ep)
        order = list(np.random.RandomState(ep).permutation(
            np.arange(1, total_v + 1)))
        oracle = LiveVertexMaskOracle(jaxpr, consts, xs, argnums,
                                      max_axes=MAX_AXES_PER_VERTEX)
        specs = np.full((total_v, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
        specs[:, :, 2] = 0
        for t, v in enumerate(order):
            v = int(v)
            k_ep, k = jrand.split(k_ep)
            if USE_MASKS:
                t0 = time.time()
                pv_all, cv_all = oracle.masks([v])
                probe_s += time.time() - t0
                pv, cv = jnp.asarray(pv_all[v]), jnp.asarray(cv_all[v])
            else:
                pv = cv = None
            feats = _axis_features_from_state(
                axis_state_j[v - 1], axis_valid_j[v - 1])
            actions, *_ = sample(feats, k, pv, cv)
            row = to_specs(
                actions.op_type.astype(jnp.int32),
                actions.i.astype(jnp.int32),
                actions.j.astype(jnp.int32),
                actions.factor.astype(jnp.int32),
                axis_state_j[v - 1],
                compress_kinds=actions.compress_kind.astype(jnp.int32),
                quant_dtypes=actions.quant_dtype.astype(jnp.int32),
            )
            specs[t] = np.asarray(row)
            n_step += 1

            o_list = [int(x) for x in order[:t + 1]]
            tf = rule_specs_to_transforms(
                jaxpr, o_list, specs[:t + 1].tolist())
            n_rule += sum(len(r) for _, r in tf)
            try:
                vertex_elimination_jaxpr(
                    jaxpr, o_list, consts, *xs, argnums=argnums,
                    count_ops=True, transforms=tf)
            except ValueError as exc:
                msg = str(exc)
                if "TRANSFORM DID NOT FIT" in msg:
                    head = msg.split(" on edge")[0]
                    head = head[head.index("TRANSFORM"):]
                    fails[head] += 1
                # a partial order legitimately raises the full-order guard;
                # everything else is a real failure we want to see
                elif "un-eliminated" not in msg and "left" not in msg:
                    fails[f"OTHER: {type(exc).__name__}: {msg[:90]}"] += 1
            except Exception as exc:                      # noqa: BLE001
                fails[f"OTHER: {type(exc).__name__}: {str(exc)[:90]}"] += 1
            # keep the oracle in lockstep with the env
            if USE_MASKS:
                tmap = dict(tf)
                oracle.advance(v, tmap.get(v, ()))
        n_ep += 1

    print("=" * 72)
    print(f"live masks: {'ON' if USE_MASKS else 'OFF'}   "
          f"episodes={n_ep} steps={n_step} emitted-rules={n_rule} "
          f"substeps={MAX_SUBSTEPS}")
    print(f"TRANSFORM DID NOT FIT occurrences: {sum(fails.values())}")
    for k, c in fails.most_common(15):
        print(f"   {c:5d}  {k}")
    if USE_MASKS:
        print(f"mask probe cost: {probe_s:.2f}s total "
              f"({probe_s/max(n_step,1)*1e3:.1f} ms/step)")
    print("=" * 72)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
