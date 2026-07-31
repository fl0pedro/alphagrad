#!/usr/bin/env python3
"""--no-approx-head must REMOVE the heads, not mask them."""
import equinox as eqx, jax, jax.random as jrand
from jax.tree_util import tree_leaves_with_path, keystr
from alphagrad.approx.ppo import _build_agent, make_argparser

BASE = ["--example", "VmappedNeuralNetwork", "--dynamic-substeps"]

def build(extra):
    a = make_argparser().parse_args(BASE + extra)
    return _build_agent(a, total_v=13, num_factors=6, max_rules=16,
                        key=jrand.PRNGKey(0))

def report(tag, ag):
    leaves = tree_leaves_with_path(eqx.filter(ag, eqx.is_inexact_array))
    n = sum(x.size for _, x in leaves)
    paths = [keystr(p) for p, _ in leaves]
    approx = [p for p in paths
              if "micro_action_policy" in p or "face_path_policy" in p]
    napx = sum(x.size for p, x in leaves
               if "micro_action_policy" in keystr(p)
               or "face_path_policy" in keystr(p))
    print(f"  {tag:22s} micro={ag.micro_action_policy is not None} "
          f"face={ag.face_path_policy is not None} "
          f"params={n:,} approx_params={napx:,} approx_leaves={len(approx)}")
    return n, napx, len(approx)

print("=== agent construction ===")
n_full, a_full, l_full = report("unified-head+faces",
                                build(["--unified-head", "--face-actions"]))
n_mask, a_mask, l_mask = report("ve_only (MASKED)",
                                build(["--unified-head", "--face-actions",
                                       "--variant", "ve_only"]))
n_gone, a_gone, l_gone = report("--no-approx-head", build(["--no-approx-head"]))

print("\n=== assertions ===")
ok = True
def ck(name, cond, d=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {d}")
    ok &= bool(cond)

ck("masked variant still HOLDS approx params", a_mask > 0,
   f"{a_mask:,} params, {l_mask} leaves — masking does not remove")
ck("--no-approx-head holds ZERO approx params", a_gone == 0, f"{a_gone}")
ck("--no-approx-head has no approx leaves", l_gone == 0, f"{l_gone}")
ck("micro_action_policy is None", build(["--no-approx-head"]).micro_action_policy is None)
ck("face_path_policy is None",
   build(["--no-approx-head", "--face-actions"]).face_path_policy is None,
   "even with --face-actions passed")
ck("total params strictly smaller", n_gone < n_mask,
   f"{n_gone:,} < {n_mask:,}  (saved {n_mask - n_gone:,})")
print("\n" + ("ALL PASS" if ok else "FAILURES"))
raise SystemExit(0 if ok else 1)
