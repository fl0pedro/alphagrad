#!/usr/bin/env python
"""Summarize one t28b_engine_probe JSONL as markdown (ticket dsnn-3qm.67).

    python t28b_report.py RUN_DIR/t65_probe.jsonl

Three arms: candidate (the lazy tiled executor), incumbent (the tiled executor
of graphax 1f3d404 under GRAPHAX_TILED_LEGACY=1) and planner (untouched). Every
latency number is a ratio against the INCUMBENT rev-exact arm measured back to
back with it; the drift floor stands beside them.
"""
import json
import sys

ARMS = ("candidate", "incumbent", "planner")
recs = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in recs:
    by.setdefault(r["phase"], []).append(r)

imp = by["import"][0]
tgt = by["target"][0]
print(f"### {imp['example']} on {tgt['device']} ({imp['host']}), "
      f"{tgt['n_vertices']} vertices, inner={tgt['inner']} "
      f"points={tgt['n_points']} reps={tgt['n_reps']}, rounds={imp['rounds']}\n")


def g(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


fl = by["floor"][0]
print(f"**Drift floor** (the incumbent rev-exact arm compiled twice): "
      f"{g(fl['lat_median'])} ({g(fl['lat_min'])}..{g(fl['lat_max'])}).\n")

comp = {}
for r in by["compile"]:
    if "error" in r:
        print(f"COMPILE ERROR {r['tag']} {r['engine']} sparse={r['sparse']}: "
              f"{r['error'][:300]}")
        continue
    comp[(r["tag"], r["engine"], r["sparse"])] = r

ref = comp[("ref_rev_exact", "incumbent", True)]
print(f"reference rev-exact incumbent: temp {ref['temp_bytes']} B, triton-temp "
      f"{ref.get('temp_triton_bytes')}, fusions {ref['hlo_fusions']}, "
      f"paths {ref['paths']}\n")

print("#### Latency, ratios against the rev-exact incumbent\n")
print("| order:plan | candidate | incumbent | planner | cand/incu |")
print("|---|---|---|---|---|")
for s in by["summary"]:
    row = [s["tag"]]
    for a in ARMS:
        row.append(f"{g(s.get(f'{a}:lat_ratio_median'))} "
                   f"({g(s.get(f'{a}:lat_ratio_min'))}.."
                   f"{g(s.get(f'{a}:lat_ratio_max'))})")
    row.append(f"{g(s.get('candidate_over_incumbent:lat_median'))} "
               f"({g(s.get('candidate_over_incumbent:lat_min'))}.."
               f"{g(s.get('candidate_over_incumbent:lat_max'))})")
    print("| " + " | ".join(row) + " |")

print("\n#### XLA static temp bytes of the sparse executable\n")
print("| order:plan | candidate | incumbent | planner | cand/incu | "
      "cand triton | incu triton | plan triton | fusions ca/in/pl |")
print("|---|---|---|---|---|---|---|---|---|")
for s in by["summary"]:
    m = {a: comp.get((s["tag"], a, True), {}) for a in ARMS}
    print(f"| {s['tag']} | " + " | ".join(
        str(m[a].get("temp_bytes")) for a in ARMS) + " | "
        + g(s.get("candidate_over_incumbent:temp_ratio")) + " | "
        + " | ".join(str(m[a].get("temp_triton_bytes")) for a in ARMS) + " | "
        + "/".join(str(m[a].get("hlo_fusions")) for a in ARMS) + " |")


def ag(d):
    if d is None:
        return "n/a"
    if "shape_mismatch" in d:
        return f"SHAPE {d['shape_mismatch']}"
    if d.get("empty"):
        return "empty"
    if d.get("bit_identical"):
        return "bit-identical"
    if d.get("rel_l2") is None:
        return str(d)[:60]
    return f"rel {d['rel_l2']:.2e}"


print("\n#### Values\n")
print("| order:plan | cand sparse vs own dense | cand vs incumbent | "
      "cand vs planner | cand vs jax.grad | incu vs jax.grad | "
      "plan vs jax.grad | param layout ca/in/pl |")
print("|---|---|---|---|---|---|---|---|")
for v in by["values"]:
    print(f"| {v['tag']} | {ag(v.get('candidate:sparse_vs_dense'))} | "
          f"{ag(v.get('candidate_vs_incumbent:sparse'))} | "
          f"{ag(v.get('candidate_vs_planner:sparse'))} | "
          f"{ag(v.get('candidate:sparse_vs_jax_grad'))} | "
          f"{ag(v.get('incumbent:sparse_vs_jax_grad'))} | "
          f"{ag(v.get('planner:sparse_vs_jax_grad'))} | "
          + "/".join(str(v.get(f"{a}:layout_param")) for a in ARMS) + " |")

print("\n#### Gradients not in parameter layout, and the Reduce extents\n")
for v in by["values"]:
    for a in ARMS:
        lay = v.get(f"{a}:layout")
        if not lay:
            continue
        odd = [(i, x["axes"]) for i, x in enumerate(lay)
               if x.get("kind") == "sparse" and not x.get("param_layout")]
        sizes = [x.get("sizes") for x in lay if x.get("kind") == "sparse"]
        print(f"{v['tag']} {a}: {len(lay)} leaves, "
              f"not in parameter layout {len(odd)}"
              + (f" {odd}" if odd else "")
              + (f"; sizes {sizes}" if imp["example"] != "TransformerLM" else ""))
