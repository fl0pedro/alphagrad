# The GPU A/B of merge batch 11: graphax trees (core-v2 as the base, the batch-11 head, the head without the
# barrier at narrow private sums) on the same plans, timed in ONE process with the measurement's own compile and
# timing instrument.
# tools/b11_gpu_ab.sbatch runs the three steps; each step can run alone:
#   plans   --target T --out DIR                     the plan set of target T (under the head graphax)
#   compile --target T --variant V --out DIR         each plan's executable under graphax V, through _compile_measure
#                                                    on the measurement's own lowering, serialized with its HLO
#   time    --out DIR --variants base,v2,... [...]   load every variant of every plan and time them paired
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
TARGETS = ("memobj_rtrl", "memobj_bptt", "nn256", "tlm", "rsnn_rtrl", "rsnn_bptt")
# mem_objective_test's RSNN_SHD program, on which dsnn-dfw.299, .304 and .305 were measured.
_MEMOBJ = {"--example": "RSNN_SHD", "--dataset": "none", "--step-position": "7"}
_THESIS = {"nn256": "nn256", "tlm": "tlm", "rsnn_rtrl": "rsnn_rtrl", "rsnn_bptt": "rsnn_bptt"}
_KEEP_ENV = re.compile(r"^(ALPHAGRAD_|GRAPHAX_|DSNN_)")
_STATIC = {"ALPHAGRAD_SKIP_COST_ANALYSIS": "1", "ALPHAGRAD_SKIP_COUNT_OPS": "1",
           "ALPHAGRAD_DIRECT_MEASURE": "1", "ALPHAGRAD_QUALITY_METRIC": "none",
           "ALPHAGRAD_COST_FORM": "paired-log", "ALPHAGRAD_PAIRED_COST_FLOOR": "byte",
           "ALPHAGRAD_MEM_CHANNEL": "temp", "ALPHAGRAD_PLAN_LOG": "1"}


class _Captured(BaseException):
    pass


def _generator():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _HERE / "gen_fq_launchers.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def target_setup(target):
    # The environment and the target flags of the thesis row, as the generator renders it.
    if target.startswith("memobj_"):
        return dict(_STATIC), {**_MEMOBJ, "--temporal-rule": target.split("_")[1]}
    gen = _generator()
    node = "pgi15-gpu19" if target in ("tlm", "rsnn_rtrl") else "pgi15-gpu18"
    # The stack a launcher needs (owner ruling 2026-09-27) is the directory of this clone; only the exports are read.
    txt = gen.render(gen.thesis_arm(arm="C_popart", target=_THESIS[target], seed="250197", node=node),
                     str(_HERE.parent.parent))
    env = {k: v.strip('"') for k, v in re.findall(r"^export ([A-Z_][A-Z0-9_]*)=(\S+)", txt, re.M)
           if _KEEP_ENV.match(k)}
    env.update(_STATIC)
    cli = {}
    for flag in ("--example", "--dataset", "--temporal-rule"):
        m = re.search(r"^\s+" + re.escape(flag) + r" (\S+)$", txt, re.M)
        if m:
            cli[flag] = m.group(1)
    return env, cli


def _apply_env(env, overrides):
    for k, v in env.items():
        os.environ[k] = v
    for kv in overrides:
        k, v = kv.split("=", 1)
        os.environ[k] = v


def _build(target, cli_over):
    import alphagrad.approx.tools.landscape_map as lm
    _env, cli = target_setup(target)
    cli.update(cli_over)
    argv = [x for kv in cli.items() for x in kv] + [
        "--seed", "250197", "--num-eval-samples", "1", "--num-data-points", "1",
        "--reps-per-point", "1", "--latency-inner-reps", "1", "--out-dir", "/tmp/b11_gpu_ab"]
    a = lm.make_argparser().parse_args(argv)
    env, ev, _cj = lm.build_env(a)
    return lm, env, ev


def _order(lm, env, kind):
    o = lm.rev_order(env) if kind == "reverse" else lm.markowitz_order(env)
    return [int(v) for v in list(o)]


class _Spy:
    # Keeps the candidate's arguments and program of one measurement and stops it at its first compile,
    # after that compile when compile_it is set.
    def __init__(self, compile_it=False):
        import alphagrad.approx.common.rsnn_shd as rs
        import alphagrad.approx.env as envmod
        self.rs, self.envmod = rs, envmod
        self.real_args, self.real_prog = rs.measure_args, envmod.measured_program
        self.args, self.fn, self.lowered = None, None, None
        self.compile_it, self.compiled, self.compile_s = compile_it, None, None

    def __enter__(self):
        def _args(config, args):
            out = self.real_args(config, args)
            if self.args is None:
                self.args = out
            return out

        def _prog(*a, **kw):
            fn = self.real_prog(*a, **kw)
            self.fn = fn
            return fn

        def _stop(lowered, compile_tuple=None):
            self.lowered = lowered
            if self.compile_it:
                t0 = time.perf_counter()
                self.compiled = self._compile(lowered, compile_tuple)
                self.compile_s = time.perf_counter() - t0
            raise _Captured()

        self.rs.measure_args, self.envmod.measured_program = _args, _prog
        self._compile = self.envmod._compile_measure
        self.envmod._compile_measure = _stop
        return self

    def __exit__(self, *exc):
        self.rs.measure_args, self.envmod.measured_program = self.real_args, self.real_prog
        self.envmod._compile_measure = self._compile
        return False


def capture(lm, env, ev, order, wires, compile_it=False):
    plan = {"specs": None, "face_specs": None, "face_skips": None, "wires": list(wires)}
    import jax
    with _Spy(compile_it) as spy:
        try:
            lm.measure(env, ev, order, plan)
        except _Captured:
            got = [tuple(x.shape) for x in jax.tree_util.tree_leaves(spy.args)]
            want = [tuple(x.shape) for x in jax.tree_util.tree_leaves(spy.lowered.args_info)]
            if got != want:
                raise RuntimeError(f"the kept arguments {got} are not the compiled program's {want}")
            return spy
    raise RuntimeError("the measurement ended without compiling the plan's program")


def _join(lm, env, ev, order, singles, limit):
    # Singletons on distinct faces, in id order, joined while the joined plan still builds.
    wires, faces = [], set()
    for pid in sorted(singles):
        w = singles[pid]["wires"]
        key = {(int(x["k"]), int(x["f"])) for x in w}
        if key & faces:
            continue
        try:
            capture(lm, env, ev, order, wires + w)
        except Exception as e:
            print(f"[plans]   {pid} does not join: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        wires, faces = wires + w, faces | key
        if len(faces) >= limit:
            break
    return wires


def cmd_plans(a):
    env_vars, _cli = target_setup(a.target)
    _apply_env(env_vars, a.env)
    import numpy as np
    lm, env, ev = _build(a.target, dict(kv.split("=", 1) for kv in a.cli))
    rsnn = a.target.startswith(("memobj_", "rsnn_"))
    plans = []
    rev = _order(lm, env, "reverse")
    plans.append({"pid": "exact_rev", "order": "reverse", "wires": []})
    if not rsnn:
        plans.append({"pid": "exact_mk", "order": "markowitz", "wires": []})
    okind = "reverse" if rsnn else "markowitz"
    order = rev if rsnn else _order(lm, env, "markowitz")
    inv = lm.face_inventory(env, np.asarray(order, dtype=np.int32), capture_tensors=True)
    if a.target.endswith("rtrl"):
        sys.path.insert(0, str(_HERE.parent / "tests"))
        import mem_objective_test as M
        plans.append({"pid": "diag_container", "order": "reverse",
                      "wires": M._diag_on_the_carried_face(lm, env, rev)})
    want = {"memobj_rtrl": ("quant", "reduce"), "memobj_bptt": ("quant", "reduce", "diag"),
            "nn256": ("quant", "reduce", "diag"), "tlm": ("quant", "diag"),
            "rsnn_rtrl": (), "rsnn_bptt": ("quant",)}[a.target]
    for op in want:
        sweep, _orders = lm.build_singleton_sweep_plans(env, order, inv, ops=(op,))
        if op == "quant":
            singles = {p: v for p, v in sweep.items() if ":face:" in p}
            wires = [w for p in sorted(singles) for w in singles[p]["wires"]]
            name = f"quant_all{len(singles)}"
        else:
            wires = _join(lm, env, ev, order, sweep, a.join)
            name = f"{op}_join{len({(w['k'], w['f']) for w in wires})}"
        if wires:
            plans.append({"pid": name, "order": okind, "wires": wires})
    if a.target == "nn256":
        # Quant and Reduce together, the kind of compress_e993: its batch sums read narrow edges.
        q = next(p for p in plans if p["pid"].startswith("quant_all"))
        r = next(p for p in plans if p["pid"].startswith("reduce_join"))
        taken = {(w["k"], w["f"], w["slot"]) for w in q["wires"]}
        plans.append({"pid": "quant_reduce", "order": okind,
                      "wires": q["wires"] + [w for w in r["wires"] if (w["k"], w["f"], w["slot"]) not in taken]})
    for p in plans:
        print(f"[plans] {a.target} {p['pid']}: {p['order']} order, {len(p['wires'])} wires", flush=True)
    out = Path(a.out) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.target}.json").write_text(json.dumps(plans))


def cmd_compile(a):
    env_vars, _cli = target_setup(a.target)
    _apply_env(env_vars, a.env)
    import pickle
    import jax
    import numpy as np
    import graphax
    from jax.experimental.serialize_executable import serialize
    lm, env, ev = _build(a.target, dict(kv.split("=", 1) for kv in a.cli))
    plans = json.loads((Path(a.out) / "plans" / f"{a.target}.json").read_text())
    out = Path(a.out) / "exe" / a.variant / a.target
    out.mkdir(parents=True, exist_ok=True)
    hlo_dir = Path(a.out) / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    for p in plans:
        t0 = time.perf_counter()
        order = _order(lm, env, p["order"])
        row = {"target": a.target, "pid": p["pid"], "variant": a.variant, "graphax": graphax.__file__,
               "device": str(jax.devices()[0].device_kind)}
        try:
            spy = capture(lm, env, ev, order, p["wires"], compile_it=True)
            ex = spy.compiled
            payload, in_tree, out_tree = serialize(ex)
            host = [np.asarray(x) for x in jax.tree_util.tree_leaves(spy.args)]
            if in_tree.num_leaves != len(host):
                raise RuntimeError(f"{in_tree.num_leaves} inputs against {len(host)} kept arguments")
            (out / f"{p['pid']}.exe").write_bytes(pickle.dumps((payload, in_tree.num_leaves, out_tree.num_leaves)))
            digest = hashlib.sha256(b"".join(x.tobytes() for x in host)).hexdigest()[:16]
            args_file = Path(a.out) / "args" / a.target / f"{p['pid']}.npz"
            if not args_file.exists():
                # The three trees compile side by side and export the same inputs (one args_sha).
                # Each writes a file of its own and renames it into place, so the timing never
                # reads a file two writers interleaved (job 68485: Bad CRC-32 in nn256 quant_reduce).
                args_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = args_file.with_name(f".{args_file.stem}.{os.getpid()}.npz")
                np.savez(tmp, *host)
                os.replace(tmp, args_file)
            hlo = ex.as_text()
            with gzip.open(hlo_dir / f"{a.target}.{p['pid']}.{a.variant}.hlo.txt.gz", "wt") as fh:
                fh.write(hlo)
            ma = ex.memory_analysis()
            text = spy.lowered.as_text()
            row.update(ok=True, args_sha=digest, n_args=len(host), compile_s=round(spy.compile_s, 2),
                       temp=int(ma.temp_size_in_bytes), arg_bytes=int(ma.argument_size_in_bytes),
                       out_bytes=int(ma.output_size_in_bytes),
                       barriers=text.count("stablehlo.optimization_barrier"),
                       dots=text.count("stablehlo.dot_general"), **_concats(hlo))
        except Exception as e:
            row.update(ok=False, err=f"{type(e).__name__}: {str(e)[:600]}")
        row["s"] = round(time.perf_counter() - t0, 1)
        print("[compile] " + json.dumps(row), flush=True)
        with open(out / "manifest.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


def _ci(ratios, rng, n=2000):
    import numpy as np
    r = np.asarray(ratios, dtype=np.float64)
    boots = np.median(r[rng.integers(0, len(r), size=(n, len(r)))], axis=1)
    return float(np.median(r)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "f64": 8, "s32": 4, "s64": 8, "pred": 1, "u8": 1, "s8": 1}
_INSTR = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = (\w+)\[([0-9,]*)\]\S* ([\w-]+)\(")


def _concats(hlo):
    # The largest concatenation of the optimized program, and whether two carry stacks meet in one
    # (the width 89600 + 16384 or 700 + 128 of dsnn-dfw.304).
    largest, joined = 0, False
    for line in hlo.splitlines():
        m = _INSTR.match(line)
        if not m or not (m.group(4) == "concatenate" or "concatenate" in m.group(1)):
            continue
        dims = [int(d) for d in m.group(3).split(",") if d]
        largest = max(largest, math.prod(dims) * _BYTES.get(m.group(2), 4))
        joined = joined or any(d in (105984, 828) for d in dims)
    return {"largest_concat_mb": round(largest / 2 ** 20, 1), "carry_stacks_joined": joined}


def _rel(a, b):
    import numpy as np
    if a.shape != b.shape:
        return None
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    scale = float(np.max(np.abs(a64))) if a64.size else 0.0
    return float(np.max(np.abs(a64 - b64)) / scale) if scale > 0 else float(np.max(np.abs(b64)) if b64.size else 0.0)


def _manifest(root, variant, target):
    f = root / "exe" / variant / target / "manifest.jsonl"
    if not f.exists():
        return {}
    return {r["pid"]: r for r in map(json.loads, f.read_text().splitlines())}


def cmd_time(a):
    os.environ.update({k: v for k, v in _STATIC.items() if k == "ALPHAGRAD_DIRECT_MEASURE"})
    import pickle
    import jax
    import numpy as np
    import alphagrad.approx.env as envmod
    from jax.experimental.serialize_executable import deserialize_and_load
    rng = np.random.default_rng(250197)
    dev = jax.devices()[0]
    root = Path(a.out)
    variants = a.variants.split(",")
    base = variants[0]
    rows_path = root / "rows.jsonl"
    summary = []
    print(f"[time] device {dev.device_kind}, jax {jax.__version__}, variants {variants}, base {base}", flush=True)
    for target in a.targets.split(","):
        plans_file = root / "plans" / f"{target}.json"
        if not plans_file.exists():
            continue
        for p in json.loads(plans_file.read_text()):
            pid = p["pid"]
            args_file = root / "args" / target / f"{pid}.npz"
            if not args_file.exists():
                summary.append(f"{target:12} {pid:22} no arguments exported")
                continue
            with np.load(args_file) as z:
                host = [z[f"arr_{i}"] for i in range(len(z.files))]
            args = [jax.device_put(x, dev) for x in host]
            exes, recs = {}, {}
            for v in variants + [base + "#2"]:
                src = v.split("#")[0]
                f = root / "exe" / src / target / f"{pid}.exe"
                made = _manifest(root, src, target).get(pid, {})
                rec = {"target": target, "pid": pid, "variant": v,
                       **{k: made[k] for k in ("compile_s", "temp", "arg_bytes", "out_bytes", "barriers",
                                               "dots", "largest_concat_mb", "carry_stacks_joined", "args_sha")
                          if k in made}}
                recs[v] = rec
                if not f.exists():
                    rec["err"] = made.get("err", "not compiled")
                    continue
                try:
                    payload, n_in, n_out = pickle.loads(f.read_bytes())
                    if n_in != len(args):
                        raise RuntimeError(f"{n_in} inputs against {len(args)} arguments")
                    ex = deserialize_and_load(
                        payload, jax.tree_util.tree_structure((tuple(range(n_in)), {})),
                        jax.tree_util.tree_structure(tuple(range(n_out))))
                    out = ex(*args)
                    jax.block_until_ready(out)
                    rec["_out"] = [np.asarray(x) for x in out]
                    exes[v] = ex
                except Exception as e:
                    rec["err"] = f"{type(e).__name__}: {str(e)[:400]}"
                    print(f"[time] {target} {pid} {v}: {rec['err']}", flush=True)
            if base not in exes:
                summary.append(f"{target:12} {pid:22} base failed: {recs[base].get('err')}")
                for rec in recs.values():
                    rec.pop("_out", None)
                    with open(rows_path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                continue
            for v, rec in recs.items():
                if "_out" in rec and v != base:
                    rels = [_rel(x, y) for x, y in zip(recs[base]["_out"], rec["_out"])]
                    rec["max_rel_vs_base"] = (None if any(r is None for r in rels) or len(rels) != len(recs[base]["_out"])
                                              else max(rels, default=0.0))
            t_one = envmod._time_one_rep(exes[base], args, [dev], 1)[0] * 1e-9
            inner = int(min(50, max(1, math.ceil(a.window_s / max(t_one, 1e-6)))))
            per_round = len(exes) * inner * t_one
            rounds = int(min(a.max_rounds, max(a.min_rounds, a.plan_budget_s // max(per_round, 1e-6))))
            lat = {v: [] for v in exes}
            peak = {v: [] for v in exes}
            names = list(exes)
            for r in range(rounds):
                for v in names[r % len(names):] + names[:r % len(names)]:
                    ns, pk, _src, _o = envmod._time_one_rep(exes[v], args, [dev], inner)
                    lat[v].append(ns)
                    peak[v].append(pk)
            for v, rec in recs.items():
                rec.pop("_out", None)
                if v in lat:
                    rec.update(inner=inner, rounds=rounds, lat_ns_median=float(np.median(lat[v])),
                               watermark_median=float(np.median(peak[v])))
                    ratios = [x / y for x, y in zip(lat[v], lat[base])]
                    rec["lat_ratio"], rec["ci_lo"], rec["ci_hi"] = _ci(ratios, rng)
                    if "temp" in recs[base] and "temp" in rec and recs[base]["temp"]:
                        rec["temp_ratio"] = rec["temp"] / recs[base]["temp"]
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            drift = recs.get(base + "#2", {})
            floor = (abs(drift["lat_ratio"] - 1.0) if "lat_ratio" in drift else float("nan"))
            rb = recs[base]
            summary.append(f"{target:12} {pid:22} inner {inner:2d} rounds {rounds:2d} base "
                           f"{rb['lat_ns_median'] / 1e3:11.1f} us  temp {rb['temp'] / 2 ** 20:9.2f} MB  "
                           f"wm {rb['watermark_median'] / 2 ** 20:9.2f} MB  concat>= {rb.get('largest_concat_mb', 0):.1f} MB "
                           f"joined={rb.get('carry_stacks_joined')}  drift floor {floor:.4f}")
            for v in variants[1:]:
                rec = recs[v]
                if "lat_ratio" not in rec:
                    summary.append(f"{'':36}{v:10} FAILED {rec.get('err', '')[:120]}")
                    continue
                flag = "  SLOWER" if rec["ci_lo"] > 1.0 + floor else ""
                mr = rec.get("max_rel_vs_base")
                summary.append(
                    f"{'':36}{v:10} lat {rec['lat_ratio']:.4f} [{rec['ci_lo']:.4f}, {rec['ci_hi']:.4f}]  "
                    f"temp {rec['temp'] / 2 ** 20:9.2f} MB ({rec.get('temp_ratio', float('nan')):.3f})  "
                    f"wm {rec['watermark_median'] / 2 ** 20:9.2f} MB  "
                    f"concat>= {rec.get('largest_concat_mb', 0):8.1f} MB joined={rec.get('carry_stacks_joined')}  "
                    f"values {('%.2e' % mr) if mr is not None else 'layout differs'}{flag}")
            print("\n".join(summary[-(len(variants)):]), flush=True)
            del exes
    (root / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n[time] SUMMARY\n" + "\n".join(summary), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plans", "compile"):
        s = sub.add_parser(name)
        s.add_argument("--target", choices=TARGETS, required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--env", action="append", default=[], metavar="K=V")
        s.add_argument("--cli", action="append", default=[], metavar="FLAG=V")
        if name == "plans":
            s.add_argument("--join", type=int, default=4)
        else:
            s.add_argument("--variant", required=True)
    s = sub.add_parser("time")
    s.add_argument("--out", required=True)
    s.add_argument("--variants", required=True)
    s.add_argument("--targets", default=",".join(TARGETS))
    s.add_argument("--window-s", type=float, default=0.05)
    s.add_argument("--plan-budget-s", type=float, default=90.0)
    s.add_argument("--min-rounds", type=int, default=8)
    s.add_argument("--max-rounds", type=int, default=40)
    a = ap.parse_args(argv)
    {"plans": cmd_plans, "compile": cmd_compile, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()
