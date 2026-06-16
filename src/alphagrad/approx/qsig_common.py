"""Shared infra for quality-signal experiments. Each candidate signal imports
this and only implements its per-rule scoring. Run on CPU.

Provides: build_env(), capture_gfn(env,ev,seq), init_w(seed), align(g,e),
predict(x,*W), load_labels(), rules(), mnist(), build_pool(exact,...), and
spearman()/pspearman() + report() helpers.
"""
import os, glob, json
import numpy as np

import alphagrad.approx.common.datasets as ds
ds.NN_VMAP_BATCH = 16
import jax, jax.numpy as jnp, equinox as eqx, optax
from alphagrad.approx.env import VertexEliminationEnv, _callback
import alphagrad.approx.common.compile_cache as ccmod
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.datasets import load_dataset
from alphagrad.approx.verify_pareto_solution import build_order_specs

ARGN = infer_argnums("VmappedNeuralNetwork")
B = 16
LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
EXACT = jax.value_and_grad(LOSS, argnums=ARGN)


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=16)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=1, reps_per_point=1, percentile_keep=0.6,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, 1)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


def capture_gfn(env, ev, seq):
    order, specs, _ = build_order_specs(seq, env)
    cap = {}; orig = ccmod.cached_compile
    def cc(key, fn):
        out = orig(key, fn)
        if isinstance(key, (bytes, bytearray)) and key.startswith(b"approx:"): cap["fn"] = out
        return out
    ccmod.cached_compile = cc
    try:
        _callback(env.config, env.args, env.consts, jnp.asarray(order), jnp.asarray(specs), len(order), *ev)
    finally:
        ccmod.cached_compile = orig
    return cap["fn"]


def init_w(seed=0, ind=784, hid=256, out=10):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return (jax.random.normal(k1, (hid, ind)) / np.sqrt(ind), jnp.zeros(hid),
            jax.random.normal(k2, (out, hid)) / np.sqrt(hid), jnp.zeros(out))


def align(g, e):
    """Layout-align an approx grad leaf to its exact leaf (transpose-back)."""
    g = np.asarray(g)
    if g.shape == e.shape: return g
    if g.ndim == 2 and g.shape == e.shape[::-1]: return g.T
    return None


def align_grads(ga, ge):
    out = []
    for a, e in zip(ga, ge):
        al = align(a, e)
        out.append(jnp.asarray(al if al is not None else np.zeros_like(np.asarray(e))))
    return tuple(out)


@jax.jit
def predict(x, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return jnp.tanh(a1 @ W2.T + b2)


def accuracy(W, x, y):
    pr = []
    for i in range(0, x.shape[0], 2000):
        pr.append(jnp.argmax(predict(x[i:i+2000], *W), -1))
    return float(jnp.mean(jnp.concatenate(pr) == jnp.argmax(y, -1)))


_MNIST = {}
def mnist():
    if not _MNIST:
        xtr, ytr = load_dataset("mnist", None, "train")
        xte, yte = load_dataset("mnist", None, "test")
        _MNIST.update(xtr=jnp.asarray(xtr), ytr=jnp.asarray(ytr),
                      xte=jnp.asarray(xte), yte=jnp.asarray(yte))
    return _MNIST["xtr"], _MNIST["ytr"], _MNIST["xte"], _MNIST["yte"]


def load_labels():
    lab = {}
    for s in ("cmorl", "mogfn"):
        for f in glob.glob(os.path.expanduser(f"~/dsnn/train_exp/{s}_shard*.json")):
            try: d = json.load(open(f))
            except Exception: continue
            for r in d.get("results", []):
                if "acc_mean" in r and "idx" in r:
                    lab[(s, r["idx"])] = dict(acc=r["acc_mean"], accs=r.get("accs", []),
                                              lat=-r["front_obj"]["latency_ns"],
                                              front_cos=r["front_obj"]["cosine_sim"])
    return lab


def rules():
    """Yield (src, idx, seq, label_dict) for every labeled rule."""
    lab = load_labels()
    for s in ("cmorl", "mogfn"):
        front = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{s}_front.json")))["front"]
        for idx, p in enumerate(front):
            if (s, idx) in lab:
                yield s, idx, p["seq"], lab[(s, idx)]


def build_pool(M=16, n_steps=1200, lr=1e-3, seed=0):
    """Arc-length-stratified weight checkpoints from a reverse-mode trajectory.
    Returns list of (xb,yb,W_jnp,exact_grad_np)."""
    rng = np.random.default_rng(seed)
    xtr, ytr, _, _ = mnist()
    W = init_w(0); opt = optax.adam(lr); ost = opt.init(W)
    snaps, losses = [], []
    for step in range(n_steps):
        i = rng.integers(0, xtr.shape[0], size=B)
        xb, yb = xtr[i], ytr[i]
        val, g = EXACT(xb, yb, *W); upd, ost = opt.update(g, ost, W); W = optax.apply_updates(W, upd)
        losses.append(float(val))
        if step % 20 == 0: snaps.append(tuple(np.asarray(w) for w in W))
    losses = np.array(losses[::20][:len(snaps)])
    arc = np.concatenate([[0], np.cumsum(np.abs(np.diff(losses)))])
    pick = sorted(set(int(np.argmin(np.abs(arc - t))) for t in np.linspace(0, arc[-1], M)))
    cache = []
    for j in pick:
        i = rng.integers(0, xtr.shape[0], size=B)
        xb, yb = xtr[i], ytr[i]
        Wj = tuple(jnp.asarray(w) for w in snaps[j])
        _, ge = EXACT(xb, yb, *Wj)
        cache.append((xb, yb, Wj, [np.asarray(g) for g in ge]))
    return cache


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return pear(rank(x[m]), rank(y[m]))
def pspearman(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z)); m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rank(x[m]), rank(y[m]), rank(z[m])
    def res(t): A = np.vstack([rz, np.ones_like(rz)]).T; return t - A @ np.linalg.lstsq(A, t, rcond=None)[0]
    return pear(res(rx), res(ry))


def report(name, recs):
    """recs: list of dict with keys src, sig, acc, lat. Prints Spearman table + saves json."""
    cm = [r for r in recs if r["src"] == "cmorl"]; mo = [r for r in recs if r["src"] == "mogfn"]
    g = lambda rs, k: np.array([r[k] for r in rs], float)
    print(f"\n==== {name} : Spearman vs accuracy (ceiling ~0.95; aligned_cos=0.79) ====")
    print(f"  C-MORL={spearman(g(cm,'sig'),g(cm,'acc')):+.3f}  MOGFN={spearman(g(mo,'sig'),g(mo,'acc')):+.3f}  "
          f"BOTH={spearman(g(recs,'sig'),g(recs,'acc')):+.3f}  BOTH|lat={pspearman(g(recs,'sig'),g(recs,'acc'),g(recs,'lat')):+.3f}  (n={len(recs)})")
    json.dump(recs, open(os.path.expanduser(f"~/dsnn/train_exp/qsig_{name}.json"), "w"))
