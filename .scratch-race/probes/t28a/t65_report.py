#!/usr/bin/env python
"""Summarize one t65_probe.jsonl as markdown tables (ratios and bytes).

    python t65_report.py RUN_DIR/t65_probe.jsonl
"""
import json
import sys

recs = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in recs:
    by.setdefault(r["phase"], []).append(r)

imp = by["import"][0]
tgt = by["target"][0]
print(f"### {imp['example']} on {tgt['device']} ({imp['host']}), "
      f"{tgt['n_vertices']} vertices, sparse={tgt['sparse']}, "
      f"inner={tgt['inner']} points={tgt['n_points']} reps={tgt['n_reps']}, "
      f"rounds={imp['rounds']}\n")
print("orders:", {k: v[:6] + ["..."] for k, v in tgt["orders"].items()})
for f in by.get("faces", []):
    print(f"live faces on {f['order']}: {f['live_faces']} (steps with faces {f['steps_with_faces']})")
for p in by.get("plan", []):
    print(f"  plan {p['order']}:{p['plan']}: faces_with_hooks={p['faces_with_hooks']} "
          f"n_faces_approx={p['n_faces_approx']} n_slot_rows={p['n_slot_rows']}")

fl = by["floor"][0]


def g(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


print(f"\n**Drift floor** (rev-exact tiled compiled twice): latency "
      f"{g(fl['lat_median'])} ({g(fl['lat_min'])}..{g(fl['lat_max'])}).  "
      f"**Planner reference / tiled reference** (rev-exact, both engines): latency "
      f"{g(fl['planner_ref_lat_median'])} ({g(fl['planner_ref_lat_min'])}.."
      f"{g(fl['planner_ref_lat_max'])}), temp {g(fl['planner_ref_temp_ratio'])}, "
      f"fusions {fl['planner_ref_fusions']}, same HLO {fl['planner_ref_hlo_same']}.\n")

comp = {}
for r in by["compile"]:
    if "error" in r:
        print(f"COMPILE ERROR {r['tag']} {r['engine']} sparse={r['sparse']}: {r['error'][:300]}")
        continue
    comp[(r["tag"], r["engine"], r["sparse"])] = r

ref = comp[("ref_rev_exact", "tiled", True)]
print(f"reference rev-exact tiled: temp {ref['temp_bytes']} B, "
      f"triton-temp {ref.get('temp_triton_bytes')}, output {ref['output_bytes']} B, "
      f"fusions {ref['hlo_fusions']}, paths {ref['paths']}")
refpl = comp[("refpl_rev_exact", "planner", True)]
print(f"reference rev-exact planner: temp {refpl['temp_bytes']} B, "
      f"triton-temp {refpl.get('temp_triton_bytes')}, fusions {refpl['hlo_fusions']}, "
      f"paths {refpl['paths']}, lower {refpl['lower_stats']}\n")

for r in by.get("value", []):
    v = r["vs_jax_grad"]
    lay = r["layout"]
    if v.get("rel_l2") is None:
        print(f"{r['tag']}: vs jax.grad {v}; param layout "
              f"{sum(1 for x in lay if x.get('param_layout'))}/{len(lay)}; "
              f"leaves {[(x.get('sizes') or x.get('shape'), x.get('axes'), x.get('val_shape')) for x in lay]}")
        continue
    print(f"{r['tag']}: vs jax.grad max_abs={v.get('max_abs'):.3e} rel_l2={v.get('rel_l2'):.3e} "
          f"cos={v.get('cosine')}; param layout {sum(1 for x in lay if x.get('param_layout'))}/{len(lay)}")

print("\n#### Latency (paired ratios against rev-exact tiled; pl/ti measured back to back)\n")
print("| order:plan | faces | tiled lat | planner lat | planner/tiled lat | tiled fus | planner fus | same HLO |")
print("|---|---|---|---|---|---|---|---|")


def f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


for s in by["summary"]:
    print(f"| {s['tag']} | {s.get('n_faces_approx')} | {f(s.get('tiled:lat_ratio_median'))} "
          f"({f(s.get('tiled:lat_ratio_min'))}..{f(s.get('tiled:lat_ratio_max'))}) | "
          f"{f(s.get('planner:lat_ratio_median'))} ({f(s.get('planner:lat_ratio_min'))}.."
          f"{f(s.get('planner:lat_ratio_max'))}) | {f(s.get('planner_over_tiled:lat_median'))} "
          f"({f(s.get('planner_over_tiled:lat_min'))}..{f(s.get('planner_over_tiled:lat_max'))}) | "
          f"{s.get('tiled:fusions')} | {s.get('planner:fusions')} | {s.get('planner_over_tiled:hlo_same')} |")

print("\n#### Memory (XLA static temp bytes of the sparse executable; triton = compiled with Triton GEMM on)\n")
print("| order:plan | tiled temp | planner temp | planner/tiled | tiled triton | planner triton | tiled out | planner out | tiled paths | planner paths |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in by["summary"]:
    t = comp.get((s["tag"], "tiled", True), {})
    p = comp.get((s["tag"], "planner", True), {})
    print(f"| {s['tag']} | {t.get('temp_bytes')} | {p.get('temp_bytes')} | "
          f"{f(s.get('planner_over_tiled:temp_ratio'))} | {t.get('temp_triton_bytes')} | "
          f"{p.get('temp_triton_bytes')} | {t.get('output_bytes')} | {p.get('output_bytes')} | "
          f"{t.get('paths')} | {p.get('paths')} |")

print("\n#### Values (dense oracle = the same plan with sparse_representation False)\n")
print("| order:plan | tiled sparse vs dense | planner sparse vs dense | tiled vs planner (sparse) | tiled vs planner (dense) | tiled sparse vs jax.grad | planner sparse vs jax.grad | tiled param layout | planner param layout |")
print("|---|---|---|---|---|---|---|---|---|")


def ag(d):
    if d is None:
        return "n/a"
    if "shape_mismatch" in d:
        return f"SHAPE {d['shape_mismatch']}"
    if d.get("bit_identical"):
        return "bit-identical"
    return f"rel {d['rel_l2']:.2e} cos {d['cosine']:.6f}" if d.get("rel_l2") is not None else str(d)


for v in by["values"]:
    print(f"| {v['tag']} | {ag(v.get('tiled:sparse_vs_dense'))} | {ag(v.get('planner:sparse_vs_dense'))} | "
          f"{ag(v.get('tiled_vs_planner:sparse'))} | {ag(v.get('tiled_vs_planner:dense'))} | "
          f"{ag(v.get('tiled:sparse_vs_jax_grad'))} | {ag(v.get('planner:sparse_vs_jax_grad'))} | "
          f"{v.get('tiled:layout_param')} | {v.get('planner:layout_param')} |")

print("\n#### Layout census of the sparse outputs (axes per returned gradient; param layout = axis == position)\n")
for v in by["values"]:
    for e in ("tiled", "planner"):
        lay = v.get(f"{e}:layout")
        if not lay:
            continue
        odd = [(i, x["axes"], x.get("val_shape")) for i, x in enumerate(lay)
               if x.get("kind") == "sparse" and not x.get("param_layout")]
        print(f"{v['tag']} {e}: {len(lay)} leaves, not in parameter layout: {odd if odd else 'none'}")
