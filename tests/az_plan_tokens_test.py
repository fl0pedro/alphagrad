"""The tokenizer AZ now shares with PPO: legal set, branching, golden stream.

Three properties, one per silent-failure mode this swap was meant to close:

1. ``PlanTokenizer.legal`` == the parallel ``_build_graph``/``_eliminate_vertex``
   model AZ used to carry, at EVERY decision of an episode. That equality is
   the licence to delete the second model; keeping two is the two-MDPs hazard.
2. ``PlanTokenizer.branch()`` restores exactly: a speculative chain of any
   length leaves the tokenizer emitting byte-identical tokens afterwards.
3. base ++ concat(per-decision deltas) == ``env._incremental_stream_tokens``
   on the committed plan, BITWISE -- tokens AND eqn ids.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common.examples import (  # noqa: E402
    get_args, get_fn, infer_argnums)
from alphagrad.approx.common.plan_tokens import PlanTokenizer  # noqa: E402

TASK = "Lighthouse"     # small, resolvable without a dataset


@pytest.fixture(scope="module")
def setup():
    fn = get_fn(TASK)
    argnums = infer_argnums(TASK)
    xs = get_args(TASK, jax.random.PRNGKey(0))
    closed = jax.make_jaxpr(fn)(*xs)
    jaxpr = closed.jaxpr
    pt = PlanTokenizer(jaxpr, argnums, list(closed.literals), list(xs))
    pt.base()
    return jaxpr, argnums, list(closed.literals), list(xs), pt


def _parallel_model(jaxpr, xs, consts, argnums):
    """The graph model az_gumbel used to carry alongside the tokenizer."""
    from graphax.core import _build_graph, _prune_graph
    _eg, g, tg, vo = _build_graph(jaxpr, xs, consts, argnums)
    _prune_graph(g, tg, jaxpr, argnums)
    return g, tg, vo


def test_legal_set_matches_the_deleted_graph_model(setup):
    from graphax.core import _eliminate_vertex

    jaxpr, argnums, consts, xs, pt = setup
    g, tg, vo = _parallel_model(jaxpr, xs, consts, argnums)
    valid = [i for i in range(1, len(jaxpr.eqns) + 1)
             if jaxpr.eqns[i - 1].outvars[0] in g]

    def legal_old(graph):
        return [i for i in valid if jaxpr.eqns[i - 1].outvars[0] in graph]

    rng = np.random.default_rng(0)
    steps = 0
    with pt.branch():
        while True:
            old, new = legal_old(g), pt.legal(valid)
            assert old == new, (
                f"step {steps}: tokenizer legal set {new} != parallel model "
                f"{old} -- the search and the tokens would live in two MDPs")
            if not new:
                break
            v = int(new[rng.integers(len(new))])
            _eliminate_vertex(v, jaxpr, g, tg, vo, count_ops=False,
                              transforms=())
            pt.eliminate(v)
            steps += 1
    assert steps >= 2   # Lighthouse: 2 eliminable interior vertices


def test_branch_restores_the_tokenizer_exactly(setup):
    jaxpr, argnums, consts, xs, pt = setup
    valid = pt.legal(range(1, len(jaxpr.eqns) + 1))
    v0 = valid[0]

    ref, ref_ids = None, None
    with pt.branch():
        ref, ref_ids = pt.eliminate(v0)

    # A LONG speculative chain, then the same first elimination again.
    with pt.branch():
        chain = pt.legal(valid)
        for v in chain[:4]:
            if v in pt.legal(valid):
                pt.eliminate(v)

    with pt.branch():
        again, again_ids = pt.eliminate(v0)

    assert again == ref, "branch() did not restore the tokenizer's tokens"
    assert again_ids == ref_ids, "branch() did not restore the eqn ids"


def test_golden_stream_equivalence(setup):
    from alphagrad.approx.env import (
        MAX_RULES_PER_VERTEX, _incremental_stream_tokens)

    jaxpr, argnums, consts, xs, pt = setup
    valid = pt.legal(range(1, len(jaxpr.eqns) + 1))
    rng = np.random.default_rng(1)

    base_toks, base_ids = pt.base()
    stream, seg = list(base_toks), list(base_ids)
    order = []
    with pt.branch():
        while True:
            legal = pt.legal(valid)
            if not legal:
                break
            v = int(legal[rng.integers(len(legal))])
            t, i = pt.eliminate(v, is_last=(len(legal) == 1))
            stream += t
            seg += i
            order.append(v)

    n = len(order)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.jaxpr = jaxpr
    cfg.argnums = tuple(argnums)
    ref, ref_ids, _ft, _ls = _incremental_stream_tokens(
        cfg, consts, xs, order, specs.tolist(), {}, honor_last_compress=True)

    assert list(ref) == stream, (
        f"streamed {len(stream)} tokens vs env {len(ref)}")
    assert list(ref_ids) == seg, "eqn ids diverge from the env's stream"


def test_eqn_ids_are_not_all_zero(setup):
    """The relational gate was INERT on AZ: VEJaxpr's eqn array was all zeros
    while PPO fed real segment ids. `last_eqn_ids()` must carry real ids."""
    jaxpr, argnums, consts, xs, pt = setup
    _t, ids = pt.base()
    assert any(int(i) >= 0 for i in ids)
    assert len(set(int(i) for i in ids)) > 1, (
        "base eqn ids collapsed to a single value -- the relational gate "
        "would be inert, which is exactly the VEJaxpr defect")
