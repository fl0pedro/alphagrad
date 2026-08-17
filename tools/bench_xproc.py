"""Cross-process executable shipping via CUDA_VISIBLE_DEVICES. One question:

    Does a blob compiled in a process pinned to physical GPU 0 load and run
    in a process pinned to physical GPU 1 -- no rebind, because BOTH sides
    see their card as device id 0?

This is the owner's route, and it is the load-bearing cell for any
process-based compile pool: in-process cross-id rebinding is refuted
(61455: CUDA client ignores executable_devices, 18/18), but production's
per-process pinning sidesteps ids entirely. compile_cache.py exercises the
SAME-card round trip daily; the CROSS-card cell has never been tested --
"same arch so it should hold" is exactly what the ship arm said before it
failed.

Mechanics: the parent runs twice as a subprocess of itself.
  role=compile  CVD=0: lower one TLM plan, compile (production compiler
                opts + parallel LLVM), serialize -> /tmp blob + trees +
                the OUTPUT computed locally (the correctness reference).
  role=run      CVD=1: deserialize (its device 0 IS physical GPU 1),
                execute the SAME inputs, compare against the reference.
PASS = load succeeds AND outputs allclose. Numerical identity across two
identical Blackwell cards is expected; allclose(1e-5) guards against
nondeterministic reductions, and bitwise equality is reported when true.
"""

import os
import pickle
import subprocess
import sys
import time

BLOB = "/tmp/xproc_blob.pkl"
REF = "/tmp/xproc_ref.pkl"


def child(role, visible):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible
    r = subprocess.run(
        [sys.executable, "-u", __file__, role],
        env=env, capture_output=True, text=True, timeout=1200)
    print(f"--- {role} (CVD={visible}) rc={r.returncode} ---")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-3000:])
    return r.returncode


def main_compile():
    import jax
    import numpy as np
    from jax.experimental import serialize_executable as sx
    import bench_compile as bc

    fn = bc.get_fn("TransformerLM")
    xs = bc.get_args("TransformerLM", jax.random.PRNGKey(250197))
    jaxpr = jax.make_jaxpr(fn)(*xs)
    order = [int(v) for v in
             np.random.default_rng(250197).permutation(
                 np.arange(1, len(jaxpr.eqns) + 1))]
    t0 = time.perf_counter()
    lowered = jax.jit(bc.jacve(fn, order, argnums=tuple(range(len(xs))))
                      ).lower(*xs)
    exe = lowered.compile(compiler_options=bc.PAR_OPTS)
    print(f"compiled on {jax.devices()[0].device_kind} id=0 "
          f"in {time.perf_counter() - t0:.1f}s")
    out = exe(*xs)
    blob, in_tree, out_tree = sx.serialize(exe)
    with open(BLOB, "wb") as f:
        pickle.dump((blob, in_tree, out_tree), f)
    with open(REF, "wb") as f:
        pickle.dump([np.asarray(o) for o in jax.tree_util.tree_leaves(out)],
                    f)
    print(f"blob={len(blob)} bytes; reference output saved")


def main_run():
    import jax
    import numpy as np
    from jax.experimental import serialize_executable as sx
    import bench_compile as bc

    with open(BLOB, "rb") as f:
        blob, in_tree, out_tree = pickle.load(f)
    with open(REF, "rb") as f:
        ref = pickle.load(f)
    t0 = time.perf_counter()
    loaded = sx.deserialize_and_load(blob, in_tree, out_tree)
    print(f"deserialize_and_load on physical GPU1 (local id 0) "
          f"in {time.perf_counter() - t0:.2f}s -- NO recompile")
    xs = bc.get_args("TransformerLM", jax.random.PRNGKey(250197))
    out = loaded(*xs)
    got = [np.asarray(o) for o in jax.tree_util.tree_leaves(out)]
    assert len(got) == len(ref)
    bitwise = all(np.array_equal(g, r) for g, r in zip(got, ref))
    close = all(np.allclose(g, r, rtol=1e-5, atol=1e-5)
                for g, r in zip(got, ref))
    print(f"outputs: bitwise={bitwise} allclose={close}")
    if not close:
        raise SystemExit("FAIL: cross-card execution differs")
    print("PASS: cross-process ship via CUDA_VISIBLE_DEVICES works")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        main_compile()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        main_run()
    else:
        rc = child("compile", "0")
        if rc == 0:
            rc = child("run", "1")
        sys.exit(rc)
