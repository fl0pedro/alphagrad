"""Measure the RL-found orders of ONE variant with a sampler-style 10x8
distribution (per-pass latency/peak, per-point cosine/frob, + deterministic
muls/flops/io/bytes) via env._callback(raw_sink=...). Output mirrors the
sampler jsonl so the existing plotters work.

Usage: python measure_found.py --variant diag_factor --num-passes 10
"""
import argparse, json, glob, os, time
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--num-passes", type=int, default=10)
ap.add_argument("--hidden-dim", type=int, default=256)
ap.add_argument("--out-dir", default=os.path.expanduser("~/dsnn/found_measured"))
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

# hidden-dim override (sampler-style, before any graph build)
if a.hidden_dim:
    import alphagrad.approx.common.examples as _ex
    _ex.NN_HIDDEN_DIM = int(a.hidden_dim)

import jax.numpy as jnp
from graphax.sparse.micro_actions import COMPRESS_KINDS, QUANT_DTYPES
from alphagrad.approx.env import _callback, MAX_RULES_PER_VERTEX
from alphagrad.approx.heads import OP_DIAG, OP_COMPRESS, OP_QUANT, OP_END
from alphagrad.approx.sampler_full import make_args_dict, build_gen_context, make_encoder

_CK = list(COMPRESS_KINDS)
_QD = [str(d) for d in QUANT_DTYPES]

def variant_of(d):
    try:
        c = yaml.safe_load(open(os.path.join(d, "files", "config.yaml")))
        v = c.get("variant"); return (v.get("value") if isinstance(v, dict) else v)
    except Exception:
        return None

# ---- find this variant's newest matrix run + collect distinct found seqs ----
run_dir = None
for d in sorted(glob.glob(os.path.expanduser("~/dsnn/wandb/run-*")), key=os.path.getmtime, reverse=True):
    if variant_of(d) == a.variant:
        run_dir = d; break
assert run_dir, f"no matrix run found for variant {a.variant}"
b = json.load(open(os.path.join(run_dir, "files", "best_sequences.json")))
raw_seqs = []
bo = b.get("best_overall") or {}
if bo.get("seq"): raw_seqs.append(("best_overall", bo["seq"]))
for ch, qd in (b.get("quantile_sequences") or {}).items():
    for lab, e in (qd.items() if isinstance(qd, dict) else []):
        if e.get("seq"): raw_seqs.append((f"{ch}/{lab}", e["seq"]))

def seq_key(seq):
    return tuple((v.get("vertex"), tuple(sorted((o.get("op"), o.get("i"), o.get("j"),
                 o.get("factor"), o.get("axis"), o.get("kind"), o.get("dtype"))
                 for o in v.get("ops", []) if isinstance(o, dict)))) for v in seq if isinstance(v, dict))
uniq = {}
for label, seq in raw_seqs:
    k = seq_key(seq)
    if k not in uniq: uniq[k] = (label, seq)
seqs = list(uniq.values())
print(f"variant={a.variant} run={os.path.basename(run_dir)} distinct_seqs={len(seqs)}")

# ---- gen context (graph is variant-independent; we feed arbitrary specs) ----
class _NS: pass
ns = _NS()
for k, v in dict(example="VmappedNeuralNetwork", dataset="mnist", dataset_size=None,
                 num_data_points=8, num_eval_samples=8, max_exec_seconds=0.0,
                 seed=12345, hidden_dim=a.hidden_dim, exec_on_gpu=False,
                 actor_num_gpus=0.0).items():
    setattr(ns, k, v)
ad = make_args_dict(ns)
srv, vv, axis_state, axis_valid = build_gen_context(ad, ns.seed)
enc = make_encoder()
S = int(MAX_RULES_PER_VERTEX)

def build_order_specs(seq):
    """typed records -> (order 1-indexed, specs (N,S,3))."""
    recs = [v for v in seq if isinstance(v, dict)]
    N = len(recs)
    order = np.array([int(v["vertex"]) + 1 for v in recs], dtype=np.int32)  # 0->1 indexed
    specs = np.full((N, S, 3), -1, dtype=np.int32); specs[..., 2] = 0
    for vidx, v in enumerate(recs):
        rv = int(v["vertex"])  # 0-indexed for axis_state lookup
        op = np.full(S, OP_END, dtype=np.int32)
        i = np.zeros(S, np.int32); j = np.zeros(S, np.int32); fac = np.full(S, -1, np.int32)
        ck = np.zeros(S, np.int32); qd = np.zeros(S, np.int32)
        for s, o in enumerate(v.get("ops", [])[:S]):
            nm = o.get("op")
            if nm == "Diag":
                op[s] = OP_DIAG; i[s] = o.get("i", 0); j[s] = o.get("j", 0); fac[s] = o.get("factor", -1)
            elif nm == "Compress":
                op[s] = OP_COMPRESS; i[s] = o.get("axis", o.get("i", 0))
                k = o.get("kind"); ck[s] = _CK.index(k) if k in _CK else 0
            elif nm == "Quant":
                op[s] = OP_QUANT; dt = str(o.get("dtype"))
                qd[s] = _QD.index(dt) if dt in _QD else 0
        specs[vidx] = enc(op, i, j, fac, axis_state[rv], ck, qd)
    return order, specs

out_path = os.path.join(a.out_dir, f"{a.variant}.jsonl")
n_ok = n_bad = 0
with open(out_path, "w") as fout:
    for idx, (label, seq) in enumerate(seqs):
        try:
            order, specs = build_order_specs(seq)
        except Exception as exc:
            n_bad += 1; continue
        nops = sum(len(v.get("ops", [])) for v in seq if isinstance(v, dict))
        for p in range(a.num_passes):
            raw = {}
            try:
                _, _, reward = _callback(
                    srv._config, srv._args, srv._consts,
                    jnp.asarray(order, dtype=jnp.int32), jnp.asarray(specs, dtype=jnp.int32),
                    int(len(order)), *srv._eval_samples,
                    init=False, point_idx=-1, raw_sink=raw,
                )
                raw.update(idx=idx, **{"pass": p}, label=label, n_ops=nops,
                           reward_vec=[float(x) for x in np.asarray(reward)], invalid=False)
            except Exception as exc:
                raw = dict(idx=idx, **{"pass": p}, label=label, n_ops=nops,
                           invalid=True, err=f"{type(exc).__name__}: {str(exc)[:160]}")
            fout.write(json.dumps(raw) + "\n")
        n_ok += 1
        if idx < 3 or idx % 10 == 0:
            cps = raw.get("cosine_sim_per_point", [])
            print(f"  [{idx}] {label} nops={nops} cos~{(np.median(cps) if cps else float('nan')):.3f} "
                  f"lat~{(np.median(raw.get('latency_ns_samples',[0]))/1e6):.3f}ms flops={raw.get('flops')}")
print(f"measured {n_ok} seqs ({n_bad} unbuildable) -> {out_path}")
