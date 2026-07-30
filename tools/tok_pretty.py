#!/usr/bin/env python3
"""Readable rendering of graphax's elimination token stream, for any function.

    GRAPH=dag|chain|diamond|mlp|all   VOCAB=512   CANON=1   python tok_pretty3.py

Renders `capture_stream`, which is base -> outputs -> one block per face. Face
blocks carry their own token counts; output vertices produce no faces and so do
not appear at all (nothing routes THROUGH an output -- it is kept as a pseudo
output edge of the Jacobian).

TWO NAME UNIVERSES (CANON=1 fixes this in the RENDER only)
----------------------------------------------------------
The tokenizer pre-registers every ORIGINAL jaxpr variable in `__init__`, so the
inputs and the eliminated vertices get names 0x0, 0x1, 0x2, ... But the base
EQUATIONS come from `IncrementalJaxpr.base_eqns()`, whose variables are the
re-traced frame's own `Var` objects -- different Python objects, so they draw
FRESH names from the same pool. The raw stream therefore reads

    inputs 0x0 0x1 { 0x6 mul 0x7 _ 0x8   0x9 sin 0x6  ... }

where 0x7 and 0x8 ARE x1 and x2, but under different names than the header just
introduced, and 0x0/0x1 never appear in any equation. The face headers use the
ORIGINAL names (`face 0x3 & 0x2 & 0x4`) while the face bodies use trace names,
so the two universes are interleaved with the correspondence never stated.

`IncrementalJaxpr.env` holds exactly that correspondence (original Var -> frame
Var). With CANON=1 this script pushes every trace name back through it and then
renumbers by first appearance, so the base reads

    inputs 0x0 0x1 { 0x2 mul 0x0 _ 0x1   0x3 sin 0x2  ... } 0x4 | 0x5 # 0x6 ...

That is a RENDERING fix. The real fix belongs in `_var_name`, which should
canonicalise through `ij.env` before drawing a new name -- otherwise the model
has to infer the aliasing from context, and the pool burns two names per node.

NAMES, AND WHY NOTHING STARTS AT 0
----------------------------------
    token id  <  L                 -> vocabulary word   (mul, {, fns, ...)
    token id  == L + d,  d < base  -> DIGIT d           (base 10 -> 0..9)
    token id  == L + a,  a >= base -> NAME ATOM

Digits own the bottom of the atom range, so the first NAME atom is `base`
itself -- decimal 10. `#0`..`#9` were the literal digits; `#2`..`#9` were absent
only because these graphs' elementals use nothing but 0, 1 and -1.

Names are positional sequences over a bounded alphabet, which is NOT prefix-
free: with alphabet V, name #0 is (a,) and name #V is (a,a), so `a a` is legally
one name or two. At the production vocab_size=512 the alphabet is wide enough
that every name here is a single atom and the question never arises. VOCAB=248
reproduces the ambiguity.
"""
from __future__ import annotations

import os
import re

import jax
import jax.numpy as jnp


# --------------------------------------------------------------- test graphs
def simple_dag(x1, x2):
    """f(x1,x2) = ( log(sin(x1*x2)), sin(x1*x2) - x1*x2 )"""
    v1 = x1 * x2
    v2 = jnp.sin(v1)
    return jnp.log(v2), v2 - v1


def chain(x):
    """f(x) = log(exp(sin(x)))  -- pure chain, 1 in / 1 out"""
    return jnp.log(jnp.exp(jnp.sin(x)))


def diamond(x, y):
    """f(x,y) = tanh(x*y + x)  -- x reaches the output by two routes"""
    return jnp.tanh(x * y + x)


def scalar_mlp(x, W, b):
    """f(x,W,b) = sum(tanh(W@x + b))  -- parameterized op, so `fns` fires"""
    return jnp.sum(jnp.tanh(W @ x + b))


GRAPHS = {
    "dag": (simple_dag, lambda: (jnp.array(1.3), jnp.array(0.7)), (0, 1)),
    "chain": (chain, lambda: (jnp.array(0.6),), (0,)),
    "diamond": (diamond, lambda: (jnp.array(0.4), jnp.array(1.1)), (0, 1)),
    "mlp": (scalar_mlp,
            lambda: (jnp.ones((3,)), jnp.ones((2, 3)) * 0.5, jnp.zeros((2,))),
            (1, 2)),
}


class NameBook:
    """Atom tuple <-> pool ordinal (inverse of `name_gen_python_style`)."""

    def __init__(self, start: int, alphabet: int):
        self.start, self.V = start, alphabet

    def ordinal(self, atoms) -> int:
        off = sum(self.V ** m for m in range(1, len(atoms)))
        n = 0
        for a in atoms:
            n = n * self.V + (int(a, 16) - self.start)
        return off + n


class Decoder:
    """Token list -> readable lines, names re-grouped and (optionally) canonical."""

    def __init__(self, tk, canon: bool = True):
        self.tk = tk
        self.L = tk._L
        self.base = tk.digit_base
        self.book = NameBook(tk.digit_base, tk._name_alphabet)
        self.ambiguous = 0
        self.multi_atom = 0
        self._alias = {}        # trace-var atoms -> original-var atoms
        self._disp = {}         # atoms -> display ordinal (first appearance)
        if canon:
            self._build_alias()

    def _build_alias(self):
        """original jaxpr Var -> frame Var, via `ij.env` and `trace.to_jaxpr`.

        `ij.env` maps an original Var to a TRACER, not to the frame `Var` the
        tokenizer names things by; `to_jaxpr` is what resolves a tracer list to
        frame vars (the same route `_value_output_vars` takes). Anything that
        does not resolve is simply not aliased.
        """
        from jax._src import core as jcore
        ij = self.tk.ij
        env = getattr(ij, "env", None) or {}
        names = self.tk._names
        originals = [v for v in self.tk.jaxpr.invars if v in env]
        originals += [ov for eqn in self.tk.jaxpr.eqns for ov in eqn.outvars
                      if ov in env]
        if not originals:
            return
        try:
            jx = ij.trace.to_jaxpr([env[v] for v in originals], ij.dbg, ij.si)[0]
        except Exception:
            return
        for v, fv in zip(originals, jx.outvars):
            if not isinstance(fv, jcore.Var):
                continue
            a_tr, a_or = names.get(fv), names.get(v)
            if a_tr is not None and a_or is not None and a_tr != a_or:
                self._alias[a_tr] = a_or

    def _label(self, atoms):
        atoms = self._alias.get(atoms, atoms)
        if atoms not in self._disp:
            self._disp[atoms] = len(self._disp)
        return f"0x{self._disp[atoms]:x}"

    def _atom(self, tok):
        if tok < self.L:
            return "w", self.tk.n_vocab.get(tok, f"<{tok}>")
        a = tok - self.L
        return ("d", a) if a < self.base else ("a", a)

    def _issued(self):
        by_len = {}
        for nm in set(self.tk._names.values()) | set(self.tk._fns.values()):
            by_len.setdefault(len(nm), set()).add(nm)
        return by_len

    def items(self, toks):
        by_len = self._issued()
        maxlen = max(by_len, default=1)
        out, i, n = [], 0, len(toks)
        while i < n:
            k, v = self._atom(int(toks[i]))
            if k == "w":
                out.append(("w", v))
                i += 1
            elif k == "d":
                j, val = i, 0
                while j < n:
                    kk, vv = self._atom(int(toks[j]))
                    if kk != "d":
                        break
                    val = val * self.base + vv
                    j += 1
                out.append(("n", str(val)))
                i = j
            else:
                hit = None
                for ln in range(min(maxlen, n - i), 0, -1):
                    cand, ok = [], True
                    for t in toks[i:i + ln]:
                        kk, vv = self._atom(int(t))
                        if kk != "a":
                            ok = False
                            break
                        cand.append(hex(vv))
                    if ok and tuple(cand) in by_len.get(ln, ()):
                        hit = tuple(cand)
                        break
                if hit is None:
                    out.append(("?", f"?{v:x}"))
                    i += 1
                    continue
                if len(hit) > 1:
                    self.multi_atom += 1
                    for ln in range(len(hit) - 1, 0, -1):
                        if tuple(hit[:ln]) in by_len.get(ln, ()):
                            self.ambiguous += 1
                            break
                out.append(("v", self._label(hit)))
                i += len(hit)
        return out

    _BREAK = {"fns", "path", "face", "outputs", "inputs", "approx"}

    def render(self, toks, indent="    ", base_indent=""):
        lines, cur, depth = [], "", 0

        def flush():
            nonlocal cur
            if cur.strip():
                lines.append(base_indent + indent * depth + cur.strip())
            cur = ""

        for kind, text in self.items(toks):
            if kind == "w" and text in self._BREAK:
                flush()
                cur = "face" if text == "path" else text
            elif kind == "w" and text == "{":
                cur += " {"
                flush()
                depth += 1
            elif kind == "w" and text == "}":
                flush()
                depth = max(0, depth - 1)
                lines.append(base_indent + indent * depth + "}")
            elif kind == "w" and text == "\n":
                flush()
            else:
                cur += (" " if cur else "") + text
        flush()
        return lines


# ------------------------------------------------------- constant-op analysis
_NUM = re.compile(r"^-?\d+$")


def const_stats(lines):
    """How much of the stream is arithmetic on literal constants.

    An elemental partial of `sub` is +1 / -1, of a `mul` it is the other
    operand, and the accumulation scaffolding pads with 0. graphax materialises
    all of those as REAL equations -- `mul 0 _ 1`, `mul 1 _ 1`, `mul - 1 _ 1` --
    instead of folding them, so they cost tokens here and (unless XLA folds
    them later) work at run time.
    """
    tot = dead = ident = 0
    for ln in lines:
        s = ln.strip()
        m = re.match(r"^\S+\s+(mul|add|sub)\s+(.+)$", s)
        if not m:
            continue
        tot += 1
        ops = [o.strip() for o in m.group(2).split("_")]
        ops = [o for o in ops if o]
        if ops and all(_NUM.match(o.replace(" ", "")) for o in ops):
            dead += 1                       # every operand is a literal
        elif m.group(1) == "mul" and any(o == "1" for o in ops):
            ident += 1                      # x * 1
    return tot, dead, ident


# ----------------------------------------------------------------------- run
def show(name, fn, args, argnums, vocab_size, canon):
    from graphax.jaxpr import IncrementalPathTokenizer

    cj = jax.make_jaxpr(fn)(*args)
    jaxpr = cj.jaxpr
    nv = len(jaxpr.eqns)
    outset = set(jaxpr.outvars)

    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)

    tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), cj.literals,
                                  list(args), vocab_size=vocab_size)
    order = list(range(nv, 0, -1))
    toks = [int(t) for t in tk.capture_stream(order)]
    dec = Decoder(tk, canon=canon)

    # Face token spans: `path` is a unique word id, so its positions cut the
    # stream into base+outputs followed by exactly one block per face.
    p_id = tk.vocab["path"]
    starts = [i for i, t in enumerate(toks) if t == p_id]
    head_end = starts[0] if starts else len(toks)
    spans = [(starts[k], starts[k + 1] if k + 1 < len(starts) else len(toks))
             for k in range(len(starts))]
    faces = list(tk.ij.all_faces())
    assert len(faces) == len(spans), (len(faces), len(spans))

    all_lines = dec.render(toks)          # one pass: fixes the display numbering
    print()
    for ln in dec.render(toks[:head_end]):
        print("   " + ln)

    print()
    by_central = {}
    for fr, sp in zip(faces, spans):
        by_central.setdefault(fr.central, []).append((fr, sp))
    for k, eqn in enumerate(jaxpr.eqns, start=1):
        ov = eqn.outvars[0]
        grp = by_central.get(ov)
        if not grp:                        # OUTPUT vertex: no faces, not printed
            continue
        n_tok = sum(b - a for _, (a, b) in grp)
        print(f"   eliminate v{k}   {eqn.primitive.name}   "
              f"({len(grp)} faces, {n_tok} tokens)")
        for fr, (a, b) in grp:
            for j, ln in enumerate(dec.render(toks[a:b], base_indent="      ")):
                print(ln + (f"      ({b - a} tokens)" if j == 0 else ""))
        print()

    tot, dead, ident = const_stats(all_lines)
    print(f"   stream {len(toks)} tokens;  {tot} arithmetic equations, of which "
          f"{dead} have ALL-LITERAL operands and {ident} more are `x * 1`")
    print(f"   -> {100.0 * (dead + ident) / max(tot, 1):.0f}% of the arithmetic "
          f"is constant-foldable")

    issued = list(tk._names.values()) + list(tk._fns.values())
    ords = [dec.book.ordinal(nm) for nm in issued]
    ok = sorted(ords) == list(range(len(ords)))
    print(f"   {'PASS' if ok else 'FAIL'}  {len(ords)} pool names -> dense "
          f"0..{len(ords) - 1};  {len(dec._alias)} trace names aliased to "
          f"originals;  {dec.multi_atom} multi-atom, {dec.ambiguous} ambiguous")
    print()
    return ok


def main():
    which = os.environ.get("GRAPH", "dag")
    vocab = int(os.environ.get("VOCAB", "512"))
    canon = os.environ.get("CANON", "1") == "1"
    keys = list(GRAPHS) if which == "all" else which.split(",")
    allok = True
    for k in keys:
        fn, mk, argnums = GRAPHS[k]
        allok &= show(f"{k}    {fn.__doc__.splitlines()[0].strip()}",
                      fn, mk(), argnums, vocab, canon)
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
