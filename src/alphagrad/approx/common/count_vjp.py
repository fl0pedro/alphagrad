# -*- coding: utf-8 -*-
"""A chunk loop whose BACKWARD pass costs the real chunk count.

THE PROBLEM
-----------
The rollout walks a step's delta with ``lax.while_loop`` and stops after
``ceil(count / C)`` chunks. The LOSS cannot: reverse-mode AD has no transpose
rule for ``lax.while_loop``, so the differentiated form is a ``lax.scan`` of
``nb = ceil(window / C)`` iterations with a ``lax.cond`` inside that skips the
chunks past the batch-wide ``budget``. The skipped chunks are numerically free
-- ``_step`` freezes the whole carry and emits a zero row, so their value and
their cotangent are exactly zero -- but the SCAN STILL RUNS. At the 32768 cap
and a fold chunk of 1024 that is 32 outer iterations to do the work of 4, in
the forward pass and again in the backward pass, per ``advance``, per sample,
per K step.

THE FIX
-------
``lax.while_loop`` has no transpose rule, but a ``jax.custom_vjp`` never asks
for one: JAX differentiates the custom rule, not the primal. So the loop is
legal in BOTH directions as long as we write the reverse sweep ourselves.

``count_loop`` runs ``body(i, carry)`` for ``i`` in ``[0, nb_live)`` with a
``while_loop``, saving each iteration's INPUT carry into a stacked buffer of
the static length ``nb``. The backward runs a second ``while_loop`` from
``nb_live - 1`` down to ``0``; at each index it reloads that iteration's input
carry, rebuilds the chunk's VJP with ``jax.vjp`` (so the chunk's forward is
recomputed, exactly as ``jax.checkpoint`` recomputes it today), applies the
incoming cotangent, and accumulates the loop-invariant cotangents.

WHAT IS THE SAME NUMBER, AND WHAT IS NOT
----------------------------------------
The FORWARD is bit-identical, measured, at every count -- folded and
unfolded, scalar and vmapped. A chunk at ``i >= nb_live`` is
``lax.cond(False, run, identity)`` today, which is the identity on the carry;
not running it at all is the same map, and the live chunks run the same
arithmetic in the same order.

The GRADIENT is bit-identical, at every count, on ``_extend_sequential``'s
budget form and on the fold's SEQUENTIAL chunk interior
(``ALPHAGRAD_FOLD_PARALLEL=0``) outside vmap. On the PARALLEL chunk interior,
which is the shipped default, and on either interior under ``vmap``, it
differs by a few float32 ulp once more than one chunk is live. The measured
worst case is under 2 ulp of the leaf's own magnitude; the test pins 8.

THE CAUSE IS THE ``lax.cond``, NOT THE ``while_loop``. A toy probe with no
alphagrad in it (``probe_cvjp.py``, job 65523 section B) measures five loop
forms against the shipped one. ``count_loop``'s gradient is bitwise equal to a
plain ``lax.scan`` of the same body at every live count. What differs is the
``lax.cond`` the shipped body wraps around the chunk: ``scan`` and
``scan`` + ``cond`` disagree in the gradient's last bits on their own, with no
``while_loop`` anywhere. ``cond``'s transpose joins both branches and forms
the same sum in a different shape, and float32 addition is not associative.

Deleting that cond IS the change, so the two forms cannot agree bit for bit.
Matching the transpose (``jax.checkpoint`` on the chunk in the backward,
``ALPHAGRAD_COUNT_VJP_REMAT``) was tried and does not close it, which is what
you would expect once the cause is the cond.

So ``ALPHAGRAD_COUNT_VJP`` DEFAULTS TO OFF and the shipped path is unchanged.
Turn it on for a run that is allowed to move its last bits, and expect the
loss time to stop tracking the window bin.

MEMORY
------
The residual is ``nb`` stacked input carries, which is exactly what
``lax.scan`` of a ``jax.checkpoint``-ed body already stacks (a checkpointed
body's residual IS its input). So the backward's residual claim does not move.
The stacked buffer is allocated at the static ``nb`` and only its first
``nb_live`` slots are ever written, so the ALLOCATION still follows the window
bin while the WORK follows the count.

THE CARRY MUST BE ALL-INEXACT
-----------------------------
``count_loop``'s carry may only hold floating-point leaves. An integer leaf
would need a ``float0`` cotangent threaded through a hand-written
``while_loop``, and the two callers do not need it: the palimpsa encode
carry's only integer leaf is ``pos``, the chunk body never reads it (each
chunk reads its own buffer from an explicit start), and its final value is one
clipped addition the caller does outside the loop.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import lax

try:                                        # jax.core is the long-lived name
    _eval_jaxpr = jax.core.eval_jaxpr
except AttributeError:                      # pragma: no cover - version drift
    from jax._src import core as _jcore
    _eval_jaxpr = _jcore.eval_jaxpr


def enabled() -> bool:
    """``ALPHAGRAD_COUNT_VJP`` -- the count-proportional backward pass.

    OFF by default. On the shipped parallel chunk interior the gradient it
    produces differs from the ``lax.scan`` + ``lax.cond`` form by a few
    float32 ulp (see the module docstring), and a path that moves the numbers
    does not become the default on its own.
    """
    return os.environ.get("ALPHAGRAD_COUNT_VJP", "0") != "0"


def _is_inexact(x) -> bool:
    return jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)


def split_inexact(tree):
    """``(float_leaves, other_leaves, spec)`` -- the inverse is `merge`."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    mask = tuple(_is_inexact(l) for l in leaves)
    f = [l for l, m in zip(leaves, mask) if m]
    o = [l for l, m in zip(leaves, mask) if not m]
    return f, o, (treedef, mask)


def merge_inexact(f, o, spec):
    """Rebuild a tree `split_inexact` took apart."""
    treedef, mask = spec
    fi, oi = iter(f), iter(o)
    return jax.tree_util.tree_unflatten(
        treedef, [next(fi) if m else next(oi) for m in mask])


def hoist_closure(f, *example_args):
    """``(converted, consts)`` -- like ``jax.closure_convert``, but total.

    ``jax.closure_convert`` hoists only the INEXACT closed-over values and
    leaves the integer ones captured in the returned function's Python
    closure. Inside a ``custom_vjp``'s BACKWARD rule that is a tracer leak
    under ``vmap``: the backward runs after the forward's batching context has
    closed, and a captured ``BatchTracer`` escapes it (measured -- the token
    stream, ``int32[E, C]``, came out as an ``UnexpectedTracerError``).
    Everything the backward touches has to arrive through ``res``, so
    everything is hoisted here and the caller decides which half is
    differentiated.

    ``converted(args, consts)`` takes the example args as one tuple.
    """
    leaves, in_tree = jax.tree_util.tree_flatten(example_args)
    seen = []

    def _flat(*flat_in):
        args = jax.tree_util.tree_unflatten(in_tree, list(flat_in))
        out = f(*args)
        o_leaves, o_tree = jax.tree_util.tree_flatten(out)
        seen.append(o_tree)
        return o_leaves

    closed = jax.make_jaxpr(_flat)(*leaves)
    o_tree = seen[0]

    def converted(args, consts):
        flat_in = jax.tree_util.tree_leaves(args)
        out = _eval_jaxpr(closed.jaxpr, list(consts), *flat_in)
        return jax.tree_util.tree_unflatten(o_tree, out)

    return converted, list(closed.consts)


def count_loop(body, carry, *, nb, nb_live, y_struct=None):
    """Run ``body(i, carry) -> (carry, y)`` for ``i`` in ``[0, nb_live)``.

    ``nb`` is the STATIC upper bound on the trip count -- it sizes the saved
    carry stack and the stacked ``y`` output, so it is still the window's
    ``ceil(W / C)``. ``nb_live`` is the TRACED real trip count and is what
    both sweeps actually run.

    ``carry`` must be a pytree of floating-point arrays (see the module
    docstring). ``y_struct`` describes ONE iteration's ``y`` as a pytree of
    ``jax.ShapeDtypeStruct``; the returned ``ys`` is that stacked to ``nb``,
    with the untaken slots left at zero -- which is what the ``lax.cond``
    form's skip branch writes there today. Pass ``None`` when the body has no
    per-iteration output; ``body`` must then return ``(carry, None)``.

    ``nb_live`` MUST be unbatched under any surrounding ``vmap``. It is the
    while_loop's predicate, and a batched predicate makes ``lax.while_loop``
    run to the batch-wide maximum with a select on every lane, which would
    compute the skipped chunks anyway. That is the same contract the
    ``budget`` argument already carries.
    """
    nb = int(nb)
    if nb <= 0:
        raise ValueError("count_loop needs a positive static bound, got %r"
                         % (nb,))
    _f, _o, _ = split_inexact(carry)
    if _o:
        raise TypeError(
            "count_loop's carry must be all-inexact; got %d non-inexact "
            "leaf/leaves (%s). Thread integer state outside the loop."
            % (len(_o), ", ".join(str(jnp.asarray(x).dtype) for x in _o)))

    has_y = y_struct is not None

    def _flat(c, i):
        c2, y = body(i, c)
        return c2, (y if has_y else ())

    conv, consts = hoist_closure(_flat, carry, jnp.zeros((), jnp.int32))
    # The inexact consts are what the gradient is FOR (the agent's
    # parameters). The rest -- the token stream, the scatter ids, the counts
    # -- have no cotangent and only need to reach the backward, which they do
    # through `res`.
    cs_f, cs_o, cs_spec = split_inexact(consts)
    live = jnp.asarray(nb_live, jnp.int32)

    def _ys0():
        if not has_y:
            return ()
        return jax.tree_util.tree_map(
            lambda s: jnp.zeros((nb,) + tuple(s.shape), s.dtype), y_struct)

    def _sweep(c0, cf, co, lv, save):
        cs = merge_inexact(cf, co, cs_spec)
        stack0 = jax.tree_util.tree_map(
            lambda x: jnp.zeros((nb,) + tuple(jnp.shape(x)),
                                jnp.asarray(x).dtype), c0) if save else None

        def _cond(st):
            return st[0] < lv

        def _body(st):
            if save:
                i, c, ys, sk = st
                sk = jax.tree_util.tree_map(lambda b, x: b.at[i].set(x), sk, c)
            else:
                i, c, ys = st
            c2, y = conv((c, i), cs)
            if has_y:
                ys = jax.tree_util.tree_map(
                    lambda b, v: b.at[i].set(v), ys, y)
            return (i + 1, c2, ys, sk) if save else (i + 1, c2, ys)

        init = (jnp.zeros((), jnp.int32), c0, _ys0())
        if save:
            out = lax.while_loop(_cond, _body, init + (stack0,))
            return out[1], out[2], out[3]
        out = lax.while_loop(_cond, _body, init)
        return out[1], out[2], None

    @jax.custom_vjp
    def _loop(c0, cf):
        c_f, ys_f, _ = _sweep(c0, cf, cs_o, live, save=False)
        return c_f, ys_f

    def _loop_fwd(c0, cf):
        c_f, ys_f, stack = _sweep(c0, cf, cs_o, live, save=True)
        # EVERYTHING the backward reads travels in `res`. Nothing traced is
        # closed over -- see `hoist_closure`.
        return (c_f, ys_f), (stack, cf, cs_o, live)

    def _loop_bwd(res, ct):
        stack, cf, co, lv = res
        ct_c, ct_ys = ct
        g_cf0 = [jnp.zeros_like(x) for x in cf]

        def _cond(st):
            return st[0] >= 0

        def _body(st):
            i, g_c, g_cf = st
            c_i = jax.tree_util.tree_map(lambda b: b[i], stack)

            def _one(c, f):
                return conv((c, i), merge_inexact(f, co, cs_spec))

            # MATCH THE SHIPPED PATH'S TRANSPOSE, not just its arithmetic.
            # The scan form differentiates a `jax.checkpoint`-ed body, so its
            # per-chunk backward is `remat_transpose`: recompute the forward,
            # then transpose. A plain `jax.vjp` here is `linearize` +
            # `transpose`, which is the same function through a different
            # jaxpr -- and on the parallel chunk interior
            # (`_extend_parallel`'s associative scan) the two forms' add
            # trees came out about one float32 ulp apart. Checkpointing
            # `_one` puts this backward on the same transpose as the old one.
            _one_t = (jax.checkpoint(_one)
                      if os.environ.get("ALPHAGRAD_COUNT_VJP_REMAT", "1") != "0"
                      else _one)
            _out, vjp = jax.vjp(_one_t, c_i, cf)
            g_y = (jax.tree_util.tree_map(lambda b: b[i], ct_ys)
                   if has_y else ())
            g_c2, g_cf2 = vjp((g_c, g_y))
            return (i - 1, g_c2, [a + b for a, b in zip(g_cf, g_cf2)])

        _i, g_c, g_cf = lax.while_loop(
            _cond, _body, (lv - 1, ct_c, g_cf0))
        return g_c, g_cf

    _loop.defvjp(_loop_fwd, _loop_bwd)
    c_f, ys_f = _loop(carry, cs_f)
    return c_f, (ys_f if has_y else None)
