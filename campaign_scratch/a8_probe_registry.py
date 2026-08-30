"""Per-family probe: raw output structure, traced-target graph, loss value,
and (mode=grad) jacve-vs-jax.grad relative error.

Run twice: --mode graph (campaign dims, graph table) and --mode grad (small
dims, the Jacobian===grad acceptance table).
"""
import os, sys, json, traceback

MODE = sys.argv[1] if len(sys.argv) > 1 else "graph"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "4096")
if MODE == "grad":
    os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "8")
    os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "8")
    os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "16")
    os.environ.setdefault("ALPHAGRAD_SNN_TRUNC", "1")
else:
    os.environ.setdefault("ALPHAGRAD_SNN_TRUNC", "1")

from types import SimpleNamespace as NS
import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common import examples as ex
from alphagrad.approx import env as envmod

NAMES = [
    "NeuralNetwork", "VmappedNeuralNetwork",
    "Perceptron", "VmappedPerceptron",
    "TransformerLM", "TransformerLM3",
    "Encoder", "VmappedEncoder",
    "EncoderDecoder", "VmappedEncoderDecoder",
    "ConvNet", "VmappedConvNet",
    "MoE", "VmappedMoE",
    "ViT", "VmappedViT",
    "LIF_SNN", "ADALIF_SNN", "ADALIF_SNN_SEQ", "LIF_SNN_SHD",
    "Simple", "Lighthouse", "Helmholtz", "RobotArm_6DOF",
    "RoeFlux_1d", "RoeFlux_3d", "BlackScholes_Jacobian",
]
if len(sys.argv) > 2:
    NAMES = sys.argv[2].split(",")


def traced_inlined(fn, xs):
    from graphax import inline_call_primitives
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:
        from jax.extend.core import ClosedJaxpr
    except ImportError:
        from jax._src.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def n_valid(jaxpr, args, consts, argnums):
    _, _, _, vo = envmod._build_graph(jaxpr, args, consts, argnums)
    valid = []
    for i, eqn in enumerate(jaxpr.eqns, 1):
        if eqn.outvars[0] not in jaxpr.outvars or i in vo:
            valid.append(i)
    return len(valid)


def shapes(t):
    return [list(np.shape(l)) for l in jax.tree_util.tree_leaves(t)]


for name in NAMES:
    rec = {"example": name}
    try:
        fn = ex.get_fn(name)
        xs = ex.get_args(name, jax.random.PRNGKey(0), dataset=None)
        raw = fn(*xs)
        rec["raw_n_leaves"] = len(jax.tree_util.tree_leaves(raw))
        rec["raw_is_tuple"] = isinstance(raw, (tuple, list))
        rec["raw_shapes"] = shapes(raw)
        try:
            target, txs, argnums = ex.grad_target_setup(
                NS(measure_grad=False, seed_vertices=False), fn, xs, name)
            rec["argnums"] = list(argnums)
            cj = traced_inlined(target, txs)
            rec["out_shapes"] = [list(a.shape) for a in cj.out_avals]
            rec["n_eqns"] = len(cj.jaxpr.eqns)
            rec["n_valid"] = n_valid(cj.jaxpr, txs, cj.literals, argnums)
            try:
                rec["max_faces"] = int(envmod.derived_max_faces(
                    cj.jaxpr, argnums, cj.literals, txs))
            except Exception as e:
                rec["max_faces"] = f"ERR {type(e).__name__}: {e}"[:120]
            v = target(*txs)
            lv = jax.tree_util.tree_leaves(v)
            if len(lv) == 1 and np.ndim(lv[0]) == 0:
                rec["loss"] = float(lv[0])
            else:
                rec["loss"] = "nonscalar:" + str([list(np.shape(l)) for l in lv])
        except Exception as e:
            rec["target_error"] = f"{type(e).__name__}: {e}"[:300]

        if MODE == "grad" and "target_error" not in rec:
            try:
                from graphax import jacve
                g = jax.jit(jax.grad(target, argnums=tuple(argnums)))(*txs)
                j = jax.jit(jacve(target, order="rev", argnums=tuple(argnums)))(*txs)
                gl = jax.tree_util.tree_leaves(g)
                jl = jax.tree_util.tree_leaves(j)
                worst = 0.0
                for a, b in zip(jl, gl):
                    a = jnp.reshape(a, b.shape)
                    sc = float(jnp.max(jnp.abs(b)))
                    worst = max(worst, float(jnp.max(jnp.abs(a - b))) / (sc if sc else 1.0))
                rec["n_leaves"] = len(gl)
                rec["rel_err"] = worst
                fa = np.concatenate([np.asarray(jnp.reshape(a, b.shape)).ravel()
                                     for a, b in zip(jl, gl)])
                fb = np.concatenate([np.asarray(b).ravel() for b in gl])
                rec["rel_frob"] = float(np.linalg.norm(fa - fb)
                                        / (np.linalg.norm(fb) or 1.0))
                rec["per_leaf"] = [
                    round(float(jnp.max(jnp.abs(jnp.reshape(a, b.shape) - b))
                                / (float(jnp.max(jnp.abs(b))) or 1.0)), 9)
                    for a, b in zip(jl, gl)]
            except Exception as e:
                rec["grad_error"] = f"{type(e).__name__}: {e}"[:300]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["tb"] = traceback.format_exc()[-400:]
    print("ROW " + json.dumps(rec), flush=True)
