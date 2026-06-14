"""Pool version of the quality-signal ablation: score each rule's layout-aligned
cosine + sign-agreement over a POOL of weight checkpoints (arc-length-stratified
along a reverse-mode trajectory), aggregated as mean and worst-region (p10), and
compare to the single-init signal. Spearman vs 20-seed test accuracy.

  JAX_PLATFORMS=cpu uv run signal_pool_ablation.py
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
NPOOL = 24


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    tfn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=16)
    closed = jax.make_jaxpr(tfn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=tfn,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=1, reps_per_point=1, percentile_keep=0.6,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, 1)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


def capture_gfn(env, ev, order, specs):
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
    k = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(k)
    return (jax.random.normal(k1, (hid, ind)) / np.sqrt(ind), jnp.zeros(hid),
            jax.random.normal(k2, (out, hid)) / np.sqrt(hid), jnp.zeros(out))


def align(g, e):
    g = np.asarray(g)
    if g.shape == e.shape: return g
    if g.ndim == 2 and g.shape == e.shape[::-1]: return g.T
    return None


def build_pool(exact, xt, yt, rng):
    """Reverse-mode trajectory; snapshot every 25 steps over 1500; pick NPOOL
    checkpoints evenly spaced in cumulative |loss change| (arc length)."""
    W = init_w(0); opt = optax.adam(1e-3); ost = opt.init(W)
    snaps, losses = [], []
    for step in range(1500):
        idx = rng.integers(0, xt.shape[0], size=B)
        xb, yb = jnp.asarray(np.asarray(xt)[idx]), jnp.asarray(np.asarray(yt)[idx])
        val, g = exact(xb, yb, *W)
        upd, ost = opt.update(g, ost, W); W = optax.apply_updates(W, upd)
        losses.append(float(val))
        if step % 25 == 0:
            snaps.append(tuple(np.asarray(w) for w in W))
    losses = np.array(losses[::25][:len(snaps)])
    arc = np.concatenate([[0], np.cumsum(np.abs(np.diff(losses)))])
    targets = np.linspace(0, arc[-1], NPOOL)
    pick = [int(np.argmin(np.abs(arc - t))) for t in targets]
    pool = [snaps[i] for i in sorted(set(pick))]
    # cache exact grad (one fixed batch per checkpoint)
    cache = []
    for Wm in pool:
        idx = rng.integers(0, xt.shape[0], size=B)
        xb, yb = jnp.asarray(np.asarray(xt)[idx]), jnp.asarray(np.asarray(yt)[idx])
        Wj = tuple(jnp.asarray(w) for w in Wm)
        _, ge = exact(xb, yb, *Wj)
        cache.append((xb, yb, Wj, [np.asarray(g) for g in ge]))
    return cache


def score_rule(gfn, cache):
    """Return (aligned_cos, signfrac) per checkpoint."""
    acs, sgs = [], []
    for xb, yb, Wj, ge in cache:
        _, ga = gfn(xb, yb, *Wj); ga = [np.asarray(g) for g in ga]
        al, eg, sm, sn = [], [], 0, 0
        for li in (0, 2):
            a = align(ga[li], ge[li])
            if a is None: a = np.zeros_like(ge[li])
            al.append(a.ravel()); eg.append(ge[li].ravel())
            sm += int(np.sum(np.sign(a) == np.sign(ge[li]))); sn += a.size
        fa = np.concatenate(al); fe = np.concatenate(eg)
        acs.append(float(fa @ fe / (np.linalg.norm(fa) * np.linalg.norm(fe) + 1e-30)))
        sgs.append(sm / sn)
    return np.array(acs), np.array(sgs)


def load_labels():
    lab = {}
    for s in ("cmorl", "mogfn"):
        for f in glob.glob(os.path.expanduser(f"~/dsnn/train_exp/{s}_shard*.json")):
            try: d = json.load(open(f))
            except Exception: continue
            for r in d.get("results", []):
                if "acc_mean" in r and "idx" in r:
                    lab[(s, r["idx"])] = (r["acc_mean"], r["lat_ns_mean"])
    return lab


def rank(x): x = np.asarray(x, float); return np.argsort(np.argsort(x)).astype(float)
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y); return pear(rank(np.asarray(x)[m]), rank(np.asarray(y)[m]))
def pspear(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rank(np.asarray(x)[m]), rank(np.asarray(y)[m]), rank(np.asarray(z)[m])
    def res(t): A = np.vstack([rz, np.ones_like(rz)]).T; return t - A @ np.linalg.lstsq(A, t, rcond=None)[0]
    return pear(res(rx), res(ry))


def main():
    env, ev = build_env()
    loss_fn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    exact = jax.value_and_grad(loss_fn, argnums=ARGN)
    xt, yt = load_dataset("mnist", None, "train")
    rng = np.random.default_rng(0)
    print("[pool] building reverse-mode arc-length checkpoint pool...", flush=True)
    cache = build_pool(exact, xt, yt, rng)
    print(f"[pool] {len(cache)} checkpoints", flush=True)
    lab = load_labels()
    rows = {"cmorl": [], "mogfn": []}
    for s in ("cmorl", "mogfn"):
        front = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{s}_front.json")))["front"]
        for idx, p in enumerate(front):
            if (s, idx) not in lab: continue
            acc, lat = lab[(s, idx)]
            try:
                order, specs, _ = build_order_specs(p["seq"], env)
                gfn = capture_gfn(env, ev, order, specs)
                acs, sgs = score_rule(gfn, cache)
            except Exception as e:
                print(f"  {s}#{idx} ERR {type(e).__name__}", flush=True); continue
            rows[s].append(dict(
                idx=idx, acc=acc, lat=lat,
                single_ac=float(acs[0]), mean_ac=float(np.mean(acs)), p10_ac=float(np.percentile(acs, 10)),
                single_sg=float(sgs[0]), mean_sg=float(np.mean(sgs)), p10_sg=float(np.percentile(sgs, 10))))
        print(f"[{s}] {len(rows[s])} rules done", flush=True)
    json.dump(rows, open(os.path.expanduser("~/dsnn/train_exp/signals_pool.json"), "w"))

    def col(rs, k): return np.array([r[k] for r in rs], float)
    allr = rows["cmorl"] + rows["mogfn"]
    print("\n========= SPEARMAN vs accuracy (pool checkpoints) =========")
    print(f"{'signal':>12} | {'C-MORL':>8} | {'MOGFN':>8} | {'BOTH':>8} | {'BOTH|lat':>8}")
    for k in ("single_ac", "mean_ac", "p10_ac", "single_sg", "mean_sg", "p10_sg"):
        c = spear(col(rows['cmorl'], k), col(rows['cmorl'], 'acc'))
        m = spear(col(rows['mogfn'], k), col(rows['mogfn'], 'acc'))
        b = spear(col(allr, k), col(allr, 'acc'))
        pb = pspear(col(allr, k), col(allr, 'acc'), col(allr, 'lat'))
        print(f"{k:>12} | {c:>+8.3f} | {m:>+8.3f} | {b:>+8.3f} | {pb:>+8.3f}")


if __name__ == "__main__":
    main()
