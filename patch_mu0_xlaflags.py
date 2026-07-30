#!/usr/bin/env python3
"""Forward XLA_FLAGS to the mu0 SPMD actor's runtime_env. Without it the actor
autotunes and SIGSEGVs in the XLA autotuner on Blackwell (--xla_gpu_autotune_level=0
never reaches it). The actor env fires before `import jax`, so this is the right knob."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/mu0_ray.py")
s = p.read_text()
if 'spmd_env_vars["XLA_FLAGS"]' in s:
    print("ALREADY PATCHED"); sys.exit(0)
anchor = "    for k, v in os.environ.items():\n        if k.startswith(\"ALPHAGRAD_\") or k.startswith(\"JAX_COMPILATION_\"):"
ins = ("    # Forward XLA_FLAGS (e.g. --xla_gpu_autotune_level=0) so the SPMD actor\n"
       "    # does NOT segfault in the XLA autotuner on Blackwell during mctx compile.\n"
       "    if os.environ.get(\"XLA_FLAGS\"):\n"
       "        spmd_env_vars[\"XLA_FLAGS\"] = os.environ[\"XLA_FLAGS\"]\n")
assert anchor in s, "anchor not found"
s = s.replace(anchor, ins + anchor, 1)
p.write_text(s)
print("patched: XLA_FLAGS forwarded to SPMD actor")
