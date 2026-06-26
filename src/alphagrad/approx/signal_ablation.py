"""3-number Spearman ablation: does layout-aligned cosine / sign-agreement predict
downstream training accuracy better than the current flattened cosine?
Self-contained; runs both sources; prints the Spearman table.

  JAX_PLATFORMS=cpu uv run signal_ablation.py
"""
import os, glob, json
import numpy as np

import alphagrad.approx.common.datasets as ds
ds.NN_VMAP_BATCH = 16
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback
import alphagrad.approx.common.compile_cache as ccmod
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.datasets import load_dataset
from alphagrad.approx.verify_pareto_solution import build_order_specs

ARGN = infer_argnums("VmappedNeuralNetwork")
NBATCH = 4
B = 16


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


def align(g, p):  # transpose-back only (skip non-transpose mismatches as None)
    g = np.asarray(g)
    if g.shape == p.shape: return g
    if g.ndim == 2 and g.shape == p.shape[::-1]: return g.T
    return None


def load_labels():
    lab = {}
    for s in ("cmorl", "mogfn"):
        for f in glob.glob(os.path.expanduser(f"~/dsnn/train_exp/{s}_shard*.json")):
            try: d = json.load(open(f))
            except Exception: continue
            for r in d.get("results", []):
                if "acc_mean" in r and "idx" in r:
                    lab[(s, r["idx"])] = (r["acc_mean"], r["lat_ns_mean"], r["front_obj"]["cosine_sim"])
    return lab


def signals_for(env, ev, exact, seq, xb_list, yb_list, W):
    order, specs, _ = build_order_specs(seq, env)
    gfn = capture_gfn(env, ev, order, specs)
    raw_c, al_c, sgn, nrm = [], [], [], []
    for xb, yb in zip(xb_list, yb_list):
        _, ge = exact(xb, yb, *W); ge = [np.asarray(g) for g in ge]
        _, ga = gfn(xb, yb, *W); ga = [np.asarray(g) for g in ga]
        # raw flattened cosine (front-style) only if all shapes match
        if all(a.shape == e.shape for a, e in zip(ga, ge)):
            fa = np.concatenate([a.ravel() for a in ga]); fe = np.concatenate([e.ravel() for e in ge])
            raw_c.append(float(fa @ fe / (np.linalg.norm(fa) * np.linalg.norm(fe) + 1e-30)))
        # aligned: weight leaves only (W1 idx0, W2 idx2), weighted concat
        al, eg, sg, se = [], [], 0, 0
        for li in (0, 2):
            a = align(ga[li], ge[li])
            if a is None: a = np.zeros_like(ge[li])
            al.append(a.ravel()); eg.append(ge[li].ravel())
            sg += int(np.sum(np.sign(a) == np.sign(ge[li]))); se += a.size
        fa = np.concatenate(al); fe = np.concatenate(eg)
        al_c.append(float(fa @ fe / (np.linalg.norm(fa) * np.linalg.norm(fe) + 1e-30)))
        sgn.append(sg / se)
        na = np.linalg.norm(np.concatenate([np.asarray(a).ravel() for a in ga]))
        ne = np.linalg.norm(np.concatenate([e.ravel() for e in ge]))
        nrm.append(na / (ne + 1e-30))
    return (float(np.mean(raw_c)) if raw_c else float("nan"),
            float(np.mean(al_c)), float(np.mean(sgn)), float(np.mean(nrm)))


def rank(x):
    x = np.asarray(x, float); return np.argsort(np.argsort(x)).astype(float)


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return pear(rank(np.asarray(x)[m]), rank(np.asarray(y)[m])), int(m.sum())


def pspear(x, y, z):  # partial spearman controlling for z
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rank(np.asarray(x)[m]), rank(np.asarray(y)[m]), rank(np.asarray(z)[m])
    def resid(t):
        A = np.vstack([rz, np.ones_like(rz)]).T
        c = np.linalg.lstsq(A, t, rcond=None)[0]; return t - A @ c
    return pear(resid(rx), resid(ry))


def main():
    env, ev = build_env()
    loss_fn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    exact = jax.value_and_grad(loss_fn, argnums=ARGN)
    xt, yt = load_dataset("mnist", None, "train")
    rng = np.random.default_rng(0)
    idxb = [rng.integers(0, xt.shape[0], size=B) for _ in range(NBATCH)]
    xb_list = [jnp.asarray(np.asarray(xt)[i]) for i in idxb]
    yb_list = [jnp.asarray(np.asarray(yt)[i]) for i in idxb]
    W = init_w(0)
    lab = load_labels()
    rows = {"cmorl": [], "mogfn": []}
    for s in ("cmorl", "mogfn"):
        front = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{s}_front.json")))["front"]
        for idx, p in enumerate(front):
            if (s, idx) not in lab: continue
            acc, lat, fcos = lab[(s, idx)]
            try:
                rc, ac, sg, nr = signals_for(env, ev, exact, p["seq"], xb_list, yb_list, W)
            except Exception as e:
                print(f"  {s}#{idx} ERR {type(e).__name__}", flush=True); continue
            rows[s].append(dict(idx=idx, acc=acc, lat=lat, front_cos=fcos,
                                raw_cos=rc, aligned_cos=ac, signfrac=sg, normr=nr))
        print(f"[{s}] {len(rows[s])} rules done", flush=True)
    json.dump(rows, open(os.path.expanduser("~/dsnn/train_exp/signals.json"), "w"))

    def col(rs, k): return np.array([r[k] for r in rs], float)
    print("\n================ SPEARMAN vs 20-seed test accuracy ================")
    print(f"{'signal':>14} | {'C-MORL':>16} | {'MOGFN':>16} | {'BOTH':>16} | {'BOTH|lat':>9}")
    allr = rows["cmorl"] + rows["mogfn"]
    for k in ("front_cos", "raw_cos", "aligned_cos", "signfrac", "normr"):
        c = spear(col(rows["cmorl"], k), col(rows["cmorl"], "acc"))
        m = spear(col(rows["mogfn"], k), col(rows["mogfn"], "acc"))
        b = spear(col(allr, k), col(allr, "acc"))
        pb = pspear(col(allr, k), col(allr, "acc"), col(allr, "lat"))
        print(f"{k:>14} | {c[0]:>+.3f} (n={c[1]:>3}) | {m[0]:>+.3f} (n={m[1]:>3}) | {b[0]:>+.3f} (n={b[1]:>3}) | {pb:>+.3f}")
    # gated: drop collapse rules (normr<0.05) then re-score signfrac
    print("\n--- signfrac with collapse gate (normr>=0.05) ---")
    for s, rs in (("cmorl", rows["cmorl"]), ("mogfn", rows["mogfn"]), ("both", allr)):
        g = [r for r in rs if r["normr"] >= 0.05]
        if g:
            sp = spear(col(g, "signfrac"), col(g, "acc"))
            print(f"  {s}: signfrac rho={sp[0]:+.3f} (n={sp[1]} of {len(rs)})")


if __name__ == "__main__":
    main()
