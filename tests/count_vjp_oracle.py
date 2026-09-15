"""THE OLD DIFFERENTIATED CHUNK LOOP, KEPT AS A GRADIENT ORACLE.

Owner ruling 2026-09-15 made ``common/count_vjp.count_loop`` the only loop on
the budget path. Until then the budget path was a ``lax.scan`` of
``nb = ceil(window / chunk)`` iterations with a ``jax.checkpoint``-ed body and
a ``lax.cond`` inside that skipped every chunk past the batch-wide budget.

THAT BODY IS REPRODUCED HERE, ONCE, AND NOWHERE ELSE. It is deleted from
``ppo.py`` and from ``common/delta_fold.py``. It survives here for one job: to
say what the shipped loop's gradient is measured against. A change that moves
the gradient away from this body is then a fact somebody has to look at,
instead of a number nobody can reproduce.

HOW IT IS INSTALLED
-------------------
``delta_fold`` and ``ppo`` both reach the loop as an attribute of the
``count_vjp`` MODULE, so replacing that one attribute puts the old body under
both call sites at once::

    monkeypatch.setattr(count_vjp, "count_loop", scan_cond_loop)

``scan_cond_loop`` takes the same arguments and returns the same pair, so
neither caller can tell the difference except in the numbers.

WHAT THE TWO FORMS AGREE ON
---------------------------
The FORWARD is bit-identical. A chunk at ``i >= nb_live`` is
``lax.cond(False, run, identity)``, which is the identity on the carry, and
not running it at all is the same map.

The GRADIENT differs in its last bits on the shipped parallel chunk interior
and under ``vmap``. ``cond``'s transpose joins both branches and forms the
same sum in a different shape, and float32 addition is not associative. That
is the whole difference; see ``common/count_vjp.py`` for the probe that
isolates it with no alphagrad in it at all.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def scan_cond_loop(body, carry, *, nb, nb_live, y_struct=None):
    """``count_loop``'s signature, the OLD body behind it.

    ``lax.scan`` over the static ``nb``, a ``lax.cond`` on ``i < nb_live``
    inside it, and ``jax.checkpoint`` around the whole body -- which is what
    ``ALPHAGRAD_FOLD_REMAT`` and ``ALPHAGRAD_LOSS_EXTEND_REMAT`` used to
    switch on, and both defaulted to on.

    Unlike ``count_loop`` this form has no all-inexact requirement on the
    carry, because ``scan`` and ``cond`` carry integers themselves. The
    callers still hand it the float-only carry, because that is the shape
    ``count_loop`` fixed for them, and the frozen integer leaf is added back
    outside the loop either way.
    """
    nb = int(nb)
    if nb <= 0:
        raise ValueError("scan_cond_loop needs a positive static bound, got "
                         f"{nb!r}")
    has_y = y_struct is not None
    live = jnp.asarray(nb_live, jnp.int32)

    def _skip_y():
        return jax.tree_util.tree_map(
            lambda s: jnp.zeros(tuple(s.shape), s.dtype), y_struct)

    def _step(state, i):
        def _run(s):
            c2, y = body(i, s)
            return c2, (y if has_y else ())

        def _skip(s):
            return s, (_skip_y() if has_y else ())

        return lax.cond(i < live, _run, _skip, state)

    c_f, ys = lax.scan(jax.checkpoint(_step), carry,
                       jnp.arange(nb, dtype=jnp.int32))
    return c_f, (ys if has_y else ())


def max_rel_diff(a, b):
    """The largest relative difference between two pytrees of arrays.

    Per leaf: ``max|a - b| / max(max|b|, tiny)``, with ``b`` the reference.
    Leaves whose reference is identically zero are compared absolutely, so a
    non-zero difference against a zero reference is still reported and is not
    divided away. Returns ``(worst, leaf_index)``.
    """
    la = [x for x in jax.tree_util.tree_leaves(a)]
    lb = [x for x in jax.tree_util.tree_leaves(b)]
    if len(la) != len(lb) or not la:
        raise ValueError(f"trees have {len(la)} and {len(lb)} leaves")
    worst, where = 0.0, -1
    for i, (x, y) in enumerate(zip(la, lb)):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise ValueError(f"leaf {i}: {x.shape} != {y.shape}")
        scale = float(np.max(np.abs(y)))
        d = float(np.max(np.abs(x - y)))
        rel = d if scale == 0.0 else d / scale
        if rel > worst:
            worst, where = rel, i
    return worst, where
