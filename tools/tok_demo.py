#!/usr/bin/env python3
"""Tokenizer walkthrough: BASE, then one FACE at a time, with approximations.

Reuses tools/tok_pretty.py's Decoder so the naming is the canonical hex-ordinal
one. Adds three things the plain dump does not show:

  * the BASE section spelled out -- what the model reads before any decision;
  * a random approximation per vertex, so the face bodies show what a rule
    actually does to the emitted arithmetic;
  * an explicit HANDOFF marker per face: the tokens in that block are the delta
    Palimpsa encodes before the next head is asked for its decision.
"""
import os, sys, random
import jax
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tok_pretty import Decoder, GRAPHS

from graphax.sparse.micro_actions import Compress


def main():
    which = os.environ.get("GRAPH", "dag")
    vocab = int(os.environ.get("VOCAB", "512"))
    seed = int(os.environ.get("SEED", "0"))
    rng = random.Random(seed)

    from graphax.jaxpr import IncrementalPathTokenizer
    fn, mk, argnums = GRAPHS[which]
    args = mk()
    cj = jax.make_jaxpr(fn)(*args)
    jaxpr = cj.jaxpr
    nv = len(jaxpr.eqns)

    order = list(range(nv, 0, -1))
    # One random approximation per vertex. COMPRESS over a physical axis is the
    # cheapest legal thing to demonstrate; ~half the vertices get nothing so
    # the exact and approximated face bodies sit side by side.
    KINDS = ("mean", "max", "abs_max")
    transforms, chosen = [], {}
    for v in order:
        if rng.random() < 0.5:
            kind = rng.choice(KINDS)
            transforms.append((v, [Compress((0,), kind)]))
            chosen[v] = f"compress('{kind}', axis 0)"
        else:
            chosen[v] = None

    # A COMPRESS needs a physical axis to reduce; on a scalar edge (val.ndim
    # == 0) graphax now RAISES rather than skipping silently -- the silent skip
    # desynced the two edges of a shared variable and made a no-op
    # approximation report cos = 1.0. So drop the ones that do not fit and SAY
    # which, instead of pretending they were applied.
    dropped = []
    while True:
        tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), cj.literals,
                                      list(args), vocab_size=vocab)
        try:
            toks = [int(t) for t in
                    tk.capture_stream(order, transforms or None)]
            break
        except ValueError as exc:
            import re as _re
            m = _re.search(r'at vertex (\d+)', str(exc))
            if not m or not transforms:
                raise
            bad = int(m.group(1))
            transforms = [t for t in transforms if t[0] != bad]
            dropped.append(bad)
            chosen[bad] = None
    dec = Decoder(tk, canon=True)

    p_id = tk.vocab["path"]
    starts = [i for i, t in enumerate(toks) if t == p_id]
    head_end = starts[0] if starts else len(toks)
    spans = [(starts[k], starts[k + 1] if k + 1 < len(starts) else len(toks))
             for k in range(len(starts))]
    faces = list(tk.ij.all_faces())
    dec.render(toks)                      # one pass fixes the display numbering

    W = 78
    print("=" * W)
    print(f"  {which}    {fn.__doc__.splitlines()[0].strip()}")
    print(f"  vocab {vocab}   order {order}   seed {seed}")
    if dropped:
        print(f"  NOTE: compress illegal on scalar edges at v{dropped} "
              f"-- those vertices run EXACT (graphax raises rather than")
        print("        silently skipping, which used to fake cos = 1.0).")
    print("=" * W)

    print("\n" + "-" * W)
    print("  BASE  --  emitted ONCE, before any elimination decision.")
    print("  Names are hex ordinals in issue order. 0x0/0x1 are the two")
    print("  primal inputs; the digits 0..9 own atoms 0..9, so pool names")
    print("  start at 0xa. The `outputs` line seeds the Jacobian: one seed")
    print("  per (output, input) pair.")
    print("-" * W)
    for ln in dec.render(toks[:head_end]):
        print("   " + ln)
    print(f"\n   BASE = {head_end} tokens.  Palimpsa encodes this, then the")
    print( "   vertex head picks the first elimination.")

    # PIPELINE VIEW.
    #   =>  a token HANDOFF: new tokens are appended to the stream and
    #       Palimpsa re-encodes it.
    #   ->  an internal DELEGATION: the summary is routed to a head.
    #
    # The VE head is queried once per VERTEX; the approximation head once per
    # FACE. Crucially the approximation for face k is emitted TOGETHER with the
    # contraction of face k+1 -- the chunks are interleaved, not one per face.
    ap_id = tk.vocab.get("approx")

    def split(a, b):
        """(contraction, approximated contraction) halves of a face span."""
        if ap_id is not None and ap_id in toks[a:b]:
            m = a + toks[a:b].index(ap_id)
            return (a, m), (m, b)
        return (a, b), (b, b)

    by_central = {}
    for fr, sp in zip(faces, spans):
        by_central.setdefault(fr.central, []).append((fr, sp))

    def emit(label, parts):
        n = sum(b - a for a, b in parts if b > a)
        print(f"\n   [{label}]   {n} tokens")
        for a, b in parts:
            if b > a:
                for ln in dec.render(toks[a:b], base_indent="      "):
                    print(ln)

    def hand(to):
        print(f"\n   ==> palimpsa  ->  {to}")

    print("\n" + "=" * W)
    print("  PIPELINE   ( => token handoff / re-encode,  -> head delegation )")
    print("=" * W)
    emit("base", [(0, head_end)])
    hand("VE head")

    step = 0
    for k, eqn in enumerate(jaxpr.eqns, start=1):
        grp = by_central.get(eqn.outvars[0])
        if not grp:
            continue                        # output vertex: no faces
        step += 1
        n = len(grp)
        halves = [split(a, b) for _, (a, b) in grp]
        print(f"\n   --- vertex v{k} ({eqn.primitive.name}), {n} face(s); "
              f"approx = {chosen.get(k) or 'NONE (exact)'} ---")
        # face 1's contraction arrives on its own...
        emit(f"face {step},1 contraction", [halves[0][0]])
        for f in range(n):
            hand("approximation head")
            nxt = [halves[f][1]]
            lbl = f"approximated contraction {step},{f + 1}"
            if f + 1 < n:                  # ...then approx_k rides with face k+1
                nxt.append(halves[f + 1][0])
                lbl += f"  &  face {step},{f + 2} contraction"
            emit(lbl, nxt)
        hand("VE head" if step < len([1 for e in jaxpr.eqns
                                      if by_central.get(e.outvars[0])])
             else "done (order complete)")

    print("\n" + "=" * W)
    print(f"  STREAM {len(toks)} tokens = {head_end} base "
          f"+ {len(toks) - head_end} appended by {step} eliminations")
    print("=" * W)


if __name__ == "__main__":
    main()
