#!/usr/bin/env python3
"""Pins the properties of LiveFaceStream whose violation is SILENT.

1. Chunks are non-empty and grow the stream -- an all-zero chunk would leave
   the palimpsa carry frozen and the head blind, which reads exactly like the
   bug this replaces.
2. Concatenating the chunks of faces 0..n-1 with the trailing approximation
   reproduces the tokens a single `eliminate` with the same decisions emits.
   This is the whole claim: the head reads the REAL stream, in order.
3. The snapshot restores. Ten repeated calls with the same arguments return
   byte-identical tokens, and the tokenizer that produced them still emits the
   same NEXT step as one that never saw a speculative run -- the name-generator
   rewind is what makes that true, and without it the two diverge.
4. A decision on face f-1 CHANGES face f's chunk. If it does not, the loop is
   decorative: the head would be reading the same contraction whatever it did.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from alphagrad.approx.live_faces import LiveFaceStream           # noqa: E402
from alphagrad.approx.env import COMPRESS_SENTINEL               # noqa: E402


def build():
    from graphax.examples import Perceptron
    import jax.random as jrand

    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    args = (x, y, W1, b1, W2, b2, gamma, beta)
    cj = jax.make_jaxpr(Perceptron)(*args)
    return cj.jaxpr, cj.literals, args


def main():
    jaxpr, consts, args = build()
    argnums = (2, 3, 4, 5)
    V = len(jaxpr.eqns)
    MR, F, S = 8, 8, 3

    lfs = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                         max_faces=F, max_axes=8, window=8192)

    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, MR, 3), np.int32)
    vspecs = -np.ones((MR, 3), np.int32)
    rows = -np.ones((F, S, 3), np.int32)
    skips = np.zeros((F,), np.int32)

    # a vertex with more than one face, please
    v, nf = None, 0
    for cand in range(1, V + 1):
        _t, _i, _c, n = lfs.chunk(order, specs, 0, cand, vspecs, rows,
                                  skips, 0)
        if int(n) > nf:
            v, nf = cand, int(n)
        if nf >= 3:
            break
    print(f"probe: vertex {v} has {nf} faces")
    assert v is not None and nf >= 2, "need a multi-face vertex to test"

    ok = True

    # --- 1. chunks carry tokens ------------------------------------------
    counts = []
    for f in range(nf):
        t, i, c, n = lfs.chunk(order, specs, 0, v, vspecs, rows, skips, f)
        counts.append(int(c))
        print(f"  face {f}: {int(c):4d} tokens  n_faces={int(n)}")
    if all(c > 0 for c in counts):
        print("PASS 1  every face chunk is non-empty")
    else:
        print(f"FAIL 1  empty chunk(s): {counts}")
        ok = False

    # --- 2. chunks reconstruct the real stream ---------------------------
    # With NO decisions the concatenation of the per-face contractions must be
    # the whole step, because there are no approximation blocks to interleave.
    from graphax import IncrementalPathTokenizer
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    full = [int(t) for t in tk.eliminate(v, ())]
    cat = []
    for f in range(nf):
        t, _i, c, _n = lfs.chunk(order, specs, 0, v, vspecs, rows,
                                  skips, f)
        cat += [int(x) for x in t[:int(c)]]
    if cat == full:
        print(f"PASS 2  chunks concatenate to the real step ({len(full)} tok)")
    else:
        print(f"FAIL 2  {len(cat)} vs {len(full)} tokens; "
              f"first diff at {next((k for k in range(min(len(cat), len(full))) if cat[k] != full[k]), min(len(cat), len(full)))}")
        ok = False

    # --- 3. the snapshot restores ----------------------------------------
    base = lfs.chunk(order, specs, 0, v, vspecs, rows, skips, 0)[0].copy()
    lfs._chunks.clear()
    same = True
    for _ in range(10):
        lfs._chunks.clear()
        same &= bool(np.array_equal(
            lfs.chunk(order, specs, 0, v, vspecs, rows, skips, 0)[0], base))
    # ... and the tokenizer is still good for the NEXT step
    tk_used = lfs._prefix[(order[:0].tobytes(), specs[:0].tobytes())]
    nxt_used = [int(t) for t in tk_used.eliminate(v, ())]
    tk_clean = IncrementalPathTokenizer(
        jaxpr, argnums, list(consts), list(args), vocab_size=512)
    tk_clean.base_tokens()
    nxt_clean = [int(t) for t in tk_clean.eliminate(v, ())]
    if same and nxt_used == nxt_clean:
        print("PASS 3  repeat calls identical AND the tokenizer is unpolluted")
    else:
        print(f"FAIL 3  repeat_identical={same} "
              f"next_step_identical={nxt_used == nxt_clean} "
              f"({len(nxt_used)} vs {len(nxt_clean)} tokens)")
        ok = False

    # --- 4. face f-1's decision changes face f's chunk --------------------
    # Searched, not assumed: on a small graph most faces hand their `pre` slot
    # a SCALAR, where nothing is legal and make_live_masked_hook correctly
    # skips whatever it is given. Asserting on a fixed (vertex, rule) would
    # then pin the fail-soft path and call it a pass.
    found = None
    for cand in range(1, V + 1):
        lfs._chunks.clear()
        _t, _i, _c, nfc = lfs.chunk(order, specs, 0, cand, vspecs, rows,
                                    skips, 0)
        if int(nfc) < 2:
            continue
        lfs._chunks.clear()
        plain = lfs.chunk(order, specs, 0, cand, vspecs, rows, skips, 1)
        trials = [np.array([COMPRESS_SENTINEL, ax, 0], np.int32)
                  for ax in range(6)]
        trials += [np.array([i, j, -1], np.int32)
                   for i in range(4) for j in range(4) if i != j]
        for slot in range(S):
            for tr in trials:
                dec = rows.copy()
                dec[0, slot] = tr
                lfs._chunks.clear()
                got = lfs.chunk(order, specs, 0, cand, vspecs, dec, skips, 1)
                if not np.array_equal(plain[0], got[0]):
                    found = (cand, slot, tuple(int(x) for x in tr),
                             int(plain[2]), int(got[2]))
                    break
            if found:
                break
        if found:
            break
    if found:
        cand, slot, tr, n0, n1 = found
        print(f"PASS 4  vertex {cand} slot {slot} rule {tr}: face 1 reads "
              f"{n0} -> {n1} tokens once face 0 is approximated")
    else:
        print("FAIL 4  no decision on face 0 changed what face 1 reads")
        ok = False

    print()
    print(f"stats: {lfs.consume_stats()}")
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
