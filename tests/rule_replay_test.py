"""A4 — the oracle replay must use the ACTUALLY-APPLIED rules.

``decode_vertex_rule_specs`` is the single wire→transform translation shared by
the measurement (``_callback``) and the mask-oracle bridge (``ppo.py``). These
pin (a) faithful decoding of each row type -- including a COMPRESS that is
honored at EVERY position of the plan, no longer only on the terminal vertex --
and the -1 (joint gcd) factor sentinel, and (b) that advancing the oracle WITH the
applied Diag yields different downstream masks than the old structural
(rules=()) replay — the desync the fix removes.
"""
import jax
import jax.numpy as jnp
import numpy as np

from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.env import (
    COMPRESS_SENTINEL,
    MAX_RULES_PER_VERTEX,
    QUANT_SENTINEL,
    decode_vertex_rule_specs,
)
from alphagrad.approx.common.masks import LiveVertexMaskOracle


def _rows(*rows):
    out = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    for i, r in enumerate(rows):
        out[i] = r
    return out


def _mk(fn, args, argnums=(0, 1)):
    cj = jax.make_jaxpr(fn)(*args)
    return cj


def test_decode_diag_quant_and_gcd_sentinel():
    fn = lambda x, y: jnp.tanh(x @ y)          # eqn1: dot (4,6)@(6,4) -> (4,4)
    cj = _mk(fn, (jnp.ones((4, 6)), jnp.ones((6, 4))))
    # DIAG bi1=0 (out axis 0, size 4), bi2=1 (primal axis 1: 6 / 4), factor -1
    # -> joint gcd(4, 6, 4) = 2
    rules = decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, -1]))
    assert len(rules) == 1 and isinstance(rules[0], Diag)
    assert rules[0].factor == 2, f"joint gcd expected 2, got {rules[0].factor}"
    # QUANT row decodes regardless of position
    rules = decode_vertex_rule_specs(
        cj.jaxpr, 1, _rows([QUANT_SENTINEL, 0, 0]))
    assert len(rules) == 1 and isinstance(rules[0], Quant)
    # factor 1 no-op and non-dividing factors are dropped
    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 1])) == ()
    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 5])) == ()


def test_compress_honored_at_every_position():
    """COMPRESS decodes for any vertex, not only the terminal one.

    The removed `is_last` gate returned () for every non-final vertex, which
    silently discarded the action: the terminal measurement came back
    bit-identical to exact AD while the policy had been credited with an
    approximation.
    """
    fn = lambda x, y: jnp.tanh(x @ y)
    cj = _mk(fn, (jnp.ones((4, 6)), jnp.ones((6, 4))))
    row = _rows([COMPRESS_SENTINEL, 0, 0])
    got = decode_vertex_rule_specs(cj.jaxpr, 1, row)
    assert len(got) == 1 and isinstance(got[0], Compress)


def test_mid_plan_compress_reaches_the_measured_jacobian():
    """A COMPRESS on a NON-final vertex must survive into the measurement.

    Numeric, not structural: eliminate a COMPLETE order with one COMPRESS
    planted at each position in turn and compare against exact AD. Every
    position must (a) not raise -- graphax's nominal-shape asserts are
    exact-AD-only and apply_compress drops the axis POINTER, not the logical
    size -- and (b) at least one NON-FINAL position must actually move the
    Jacobian, which is precisely what the old gate suppressed.
    """
    from graphax import jacve
    from graphax.core import _build_graph
    from alphagrad.approx.common.masks import make_live_masked_hook
    from alphagrad.approx.common.examples import get_args, get_fn

    # A REAL target (the small 4-8-4 MLP), not the 2-op toy above: on the toy
    # every axis-0 COMPRESS happens to land on an edge that is already uniform
    # along that axis, so the reduction is the identity and the test could not
    # tell a working COMPRESS from a dropped one.
    fn = get_fn("NeuralNetwork")
    args = get_args("NeuralNetwork", jax.random.PRNGKey(0))
    argnums = tuple(range(len(args)))
    cj = jax.make_jaxpr(fn)(*args)
    _, _, _, vo = _build_graph(cj.jaxpr, args, list(cj.literals), argnums)
    valid = [i for i, eqn in enumerate(cj.jaxpr.eqns, 1)
             if eqn.outvars[0] not in cj.jaxpr.outvars or i in vo]
    order = list(reversed(valid))

    def _flat(t):
        return np.concatenate(
            [np.asarray(x, np.float64).ravel()
             for x in jax.tree_util.tree_leaves(t)])

    ref = _flat(jax.jit(jacve(fn, order, argnums=argnums))(*args))
    tried = moved = 0
    for v in order[:-1]:                          # NON-final positions only
        rules = ()
        for axis in range(4):
            rules = decode_vertex_rule_specs(
                cj.jaxpr, int(v), _rows([COMPRESS_SENTINEL, axis, 0]))
            if rules:
                break
        if not rules:
            continue
        tried += 1
        out = _flat(jax.jit(jacve(                # must not raise
            fn, order, argnums=argnums,
            transforms=[(int(v),
                         (make_live_masked_hook(tuple(rules)),))]))(*args))
        cos = float(out @ ref / (np.linalg.norm(out) * np.linalg.norm(ref)))
        if cos < 0.999:
            moved += 1
    assert tried > 5, f"only {tried} non-final vertices took a COMPRESS at all"
    assert moved > 0, (
        "no NON-FINAL COMPRESS changed the measured Jacobian -- the position "
        "gate is back, or the rules are being dropped somewhere downstream")


def test_rules_replay_diverges_from_structural_replay():
    """After a real, ORACLE-LEGAL Diag lands, downstream masks differ from the
    structural (rules=()) replay — the desync the old bridge suffered.

    The Diag is chosen from the oracle's own pair_valid mask (mirroring
    production, where sampled actions are mask-legal by construction); a
    hand-crafted pair may conflict with the live edge's built-in coupling and
    raise — which is itself why the replay must use real, legal rules.
    """
    from alphagrad.approx.env import diag_row_to_pair

    # MLP-shaped graph (the spec's own example structure): the dot Jacobian
    # wrt W leaves (batch=2) x (in=8) dense-dense with gcd 2 — a live-legal
    # Diag. Small elementwise graphs have NO legal pairs (fully coupled or
    # co-prime), so they can't exercise this.
    def mlp(x, W1, W2):
        return jnp.tanh(x @ W1) @ W2

    cases = [
        (mlp, (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1),
         (1, 2)),
    ]
    for fn, args, argnums in cases:
        cj = _mk(fn, args)
        probe = LiveVertexMaskOracle(cj.jaxpr, cj.literals, args, argnums)
        # Find an oracle-legal (vertex, wire-row) whose decode yields a Diag.
        found = None
        for v in range(1, len(cj.jaxpr.eqns) + 1):
            pair, _comp = probe.vertex_mask(v)
            eqn = cj.jaxpr.eqns[v - 1]
            if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
                continue
            out_len = len(eqn.outvars[0].aval.shape)
            n_primal = min(
                (len(iv.aval.shape) for iv in eqn.invars if hasattr(iv, "aval")),
                default=0,
            )
            for bi1 in range(out_len):
                for bi2 in range(n_primal):
                    i, j = diag_row_to_pair(cj.jaxpr, v, bi1, bi2)
                    if i != j and pair[i, j]:
                        rules = decode_vertex_rule_specs(
                            cj.jaxpr, v, _rows([bi1, bi2, -1]))
                        if rules and isinstance(rules[0], Diag):
                            found = (v, rules)
                            break
                if found:
                    break
            if found:
                break
        if not found:
            continue
        v, rules = found
        o_rules = LiveVertexMaskOracle(cj.jaxpr, cj.literals, args, argnums)
        o_struct = LiveVertexMaskOracle(cj.jaxpr, cj.literals, args, argnums)
        o_rules.advance(v, rules=rules)
        o_struct.advance(v, rules=())
        p_r, c_r = o_rules.masks()
        p_s, c_s = o_struct.masks()
        if (not np.array_equal(p_r, p_s)) or (not np.array_equal(c_r, c_s)):
            return  # divergence demonstrated — the fix is load-bearing
    raise AssertionError(
        "no case produced a mask divergence between rules-replay and "
        "structural replay — the fix would be vacuous on these graphs"
    )
