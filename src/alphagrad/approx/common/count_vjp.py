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

WHY THIS IS THE SAME NUMBER
---------------------------
* FORWARD. A chunk at ``i >= nb_live`` is ``lax.cond(False, run, identity)``
  today, which is the identity on the carry. Not running it at all is the
  same map. The per-chunk arithmetic, the chunk order and the accumulation
  order inside ``fold_fn`` are untouched.
* BACKWARD. The skipped chunks' cotangent contribution is exactly ``+0.0``
  today, because ``cond``'s transpose feeds the untaken branch a zero
  cotangent. The live chunks are transposed in the same reverse order, from
  the same recomputed forward, so every nonzero partial sum is formed from
  the same terms in the same order. The ONE difference is that the leading
  ``+0.0`` additions are gone; adding ``+0.0`` to a float32 accumulator is
  exact for every value except ``-0.0``, which becomes ``+0.0``. That is a
  sign-of-zero difference and is zero ulp in magnitude.

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


def enabled() -> bool:
    """``ALPHAGRAD_COUNT_VJP`` -- the count-proportional backward pass.

    On by default. ``0`` restores the ``lax.scan`` + ``lax.cond`` form, which
    is the only other way to differentiate through the chunk loop.
    """
    return os.environ.get("ALPHAGRAD_COUNT_VJP", "1") != "0"


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
    _live = jnp.asarray(nb_live, jnp.int32)

    def _flat(c, i):
        c2, y = body(i, c)
        return c2, (y if has_y else ())

    # HOIST THE DIFFERENTIABLE CLOSURE. `body` closes over the agent's
    # parameters, and a value closed over by a `custom_vjp` primal is a
    # CONSTANT of that primal -- it would silently receive a zero cotangent.
    # `jax.closure_convert` turns exactly the inexact closed-over values into
    # explicit arguments, which is the documented way to write a custom_vjp
    # over a closure. Integer closures (the token stream, the scatter ids)
    # stay closed over, and correctly so: they have no cotangent.
    conv, consts = jax.closure_convert(_flat, carry, jnp.zeros((), jnp.int32))
    consts = list(consts)

    def _zeros_like_stack(tree):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((nb,) + tuple(jnp.shape(x)),
                                jnp.asarray(x).dtype), tree)

    def _ys0():
        if not has_y:
            return ()
        return jax.tree_util.tree_map(
            lambda s: jnp.zeros((nb,) + tuple(s.shape), s.dtype), y_struct)

    def _sweep(c0, cs, save):
        stack0 = _zeros_like_stack(c0) if save else None

        def _cond(st):
            return st[0] < _live

        def _body(st):
            if save:
                i, c, ys, sk = st
                sk = jax.tree_util.tree_map(lambda b, x: b.at[i].set(x), sk, c)
            else:
                i, c, ys = st
            c2, y = conv(c, i, *cs)
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
    def _loop(c0, cs):
        c_f, ys_f, _ = _sweep(c0, cs, save=False)
        return c_f, ys_f

    def _loop_fwd(c0, cs):
        c_f, ys_f, stack = _sweep(c0, cs, save=True)
        return (c_f, ys_f), (stack, cs)

    def _loop_bwd(res, ct):
        stack, cs = res
        ct_c, ct_ys = ct
        ct_cs0 = [jnp.zeros_like(x) for x in cs]

        def _cond(st):
            return st[0] >= 0

        def _body(st):
            i, g_c, g_cs = st
            c_i = jax.tree_util.tree_map(lambda b: b[i], stack)

            def _one(c, cc):
                return conv(c, i, *cc)

            _out, vjp = jax.vjp(_one, c_i, cs)
            g_y = (jax.tree_util.tree_map(lambda b: b[i], ct_ys)
                   if has_y else ())
            g_c2, g_cs2 = vjp((g_c, g_y))
            return (i - 1, g_c2, [a + b for a, b in zip(g_cs, g_cs2)])

        _i, g_c, g_cs = lax.while_loop(
            _cond, _body, (_live - 1, ct_c, ct_cs0))
        return g_c, g_cs

    _loop.defvjp(_loop_fwd, _loop_bwd)
    c_f, ys_f = _loop(carry, consts)
    return c_f, (ys_f if has_y else None)
