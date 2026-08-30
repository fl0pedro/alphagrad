"""Plumbing check: do the probe's HOST targets exist and VARY within a step?

If stat_ln_i/stat_ln_j are constant across the faces of a step group, the
within-step R2 is 0 BY CONSTRUCTION and no amount of probe training moves it.
This answers that without training anything.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
import math
import jax, jax.numpy as jnp, numpy as np
from alphagrad.approx.common.masks import LiveVertexMaskOracle as LVMO
from alphagrad.approx.common import feature_probe as FP
from alphagrad.approx.env import MAX_AXES_PER_VERTEX

def f(x, y):
    z = jnp.sin(x) * jnp.cos(y)
    w = jnp.tanh(z @ z.T)
    return w @ z

A = (jnp.ones((4, 4)) * 0.3, jnp.ones((4, 4)) * 0.7)
AN = (0, 1)
closed = jax.make_jaxpr(f)(*A)
jaxpr = closed.jaxpr
V = len(jaxpr.eqns)
N = int(MAX_AXES_PER_VERTEX)
lnv = np.zeros((V + 2,), np.float32)
for i, eq in enumerate(jaxpr.eqns, start=1):
    n = 1
    if eq.outvars and hasattr(eq.outvars[0], 'aval'):
        for s in eq.outvars[0].aval.shape:
            n *= int(s)
    lnv[i] = float(np.log2(max(n, 1)))
print('total_v', V, 'ln_of_vertex', np.round(lnv, 2))

o = LVMO(jaxpr, list(closed.literals), list(A), AN, max_axes=N)
for v in range(1, V + 1):
    try:
        faces = o.probe_faces(v, approx=True)
    except Exception as e:
        print('v', v, 'probe_faces FAILED', type(e).__name__, str(e)[:80]); continue
    nf = len(faces)
    # endpoints: ppo stores them from the face loop; here approximate with a
    # scan over the oracle's own enumeration if available.
    ends = np.zeros((max(nf, 1), 2), np.int32)
    ep = getattr(o, 'face_endpoints', None)
    if callable(ep):
        try:
            ends = np.asarray(ep(v), np.int32)
        except Exception:
            pass
    t, e, n_out = FP.face_targets_host(o, jaxpr, v, 32, N,
                                       ln_of_vidx=lnv, endpoints=ends)
    print('v=%d n_faces=%d' % (v, n_out))
    if n_out:
        for c, nm in enumerate(FP.FACE_NAMES):
            col = t[:n_out, c]
            print('    %-10s min=%.3f max=%.3f std=%.3f' %
                  (nm, col.min(), col.max(), col.std()))
