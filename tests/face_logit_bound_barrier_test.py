import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import unified_face_head as U             # noqa: E402

EMBD = 8
CLAMP = 15.0


@pytest.fixture(autouse=True)
def _keep_the_clamp():
    keep = U.LOGIT_CLAMP[0]
    try:
        yield
    finally:
        U.set_logit_clamp(keep)


def _logits_jaxpr(clamp):
    U.set_logit_clamp(clamp)
    head = U.UnifiedFaceHead(EMBD, key=jrand.PRNGKey(0))
    return jax.make_jaxpr(head.logits)(jnp.ones((EMBD,), jnp.float32)).jaxpr


def _all_eqns(jaxpr):
    for e in jaxpr.eqns:
        yield e
        for p in e.params.values():
            for s in (p if isinstance(p, (tuple, list)) else (p,)):
                sub = getattr(s, "jaxpr", s)
                if hasattr(sub, "eqns"):
                    yield from _all_eqns(sub)


def _is_var(v):
    return v is not None and type(v).__name__ not in ("Literal", "DropVar")


# dsnn-dfw.95: the GPU loss program lost the division when the bound fused with the projection.
def test_the_bound_reads_the_projection_through_the_barrier():
    jaxpr = _logits_jaxpr(CLAMP)
    made_by = {v: e for e in jaxpr.eqns for v in e.outvars if _is_var(v)}
    tanh = [e for e in jaxpr.eqns if e.primitive.name == "tanh"]
    assert len(tanh) == 1, [e.primitive.name for e in jaxpr.eqns]
    path, v = [], tanh[0].invars[0]
    while _is_var(v) and v in made_by:
        e = made_by[v]
        path.append(e.primitive.name)
        if e.primitive.name in ("optimization_barrier", "dot_general"):
            break
        v = next((x for x in e.invars if _is_var(x)), None)
    assert path and path[-1] == "optimization_barrier", (
        f"the bound's tanh reads the projection without a barrier: {path}")


def test_no_bound_traces_neither_barrier_nor_tanh():
    names = {e.primitive.name for e in _all_eqns(_logits_jaxpr(0.0))}
    assert "optimization_barrier" not in names, names
    assert "tanh" not in names, names
