"""The INTERLEAVED per-face stream: emit -> decide -> emit.

The batch path emits `[base] -> [everything vertex 1 did] -> ...`, so every
decision for a vertex is made before any of its tokens exist. These pin the
spec's ordering instead: the path tokens for a face are emitted BEFORE the
skip/approx decision for that face, and the chosen action's tokens are emitted
before the next decision sees them.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from graphax import IncrementalJaxpr
from graphax.sparse.micro_actions import Diag

from alphagrad.approx.face_stream import (
    TOK_DIAG,
    TOK_PATH,
    TOK_SKIP,
    InterleavedFaceDriver,
    face_transforms_for,
)


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)
_ARGNUMS = (1, 2)


def _incr():
    cj = jax.make_jaxpr(_mlp)(*_ARGS)
    return IncrementalJaxpr(cj.jaxpr, argnums=_ARGNUMS, consts=cj.literals,
                            args=_ARGS), cj


def _drive(policy, max_substeps=1, slots=("lhs", "rhs", "res")):
    incr, cj = _incr()
    d = InterleavedFaceDriver(policy, max_substeps=max_substeps)
    for v in range(1, len(cj.jaxpr.eqns) + 1):
        incr.eliminate(v, face_transforms=face_transforms_for(incr, v, d, slots))
    return d, incr


def test_path_tokens_precede_the_decision():
    """THE contract: when the policy is asked, the path tokens for that face
    are already in the stream."""
    seen = []

    def policy(ctx):
        seen.append(list(ctx["tokens"]))
        return None                      # skip everything

    d, _ = _drive(policy)
    assert seen, "policy was never consulted"
    for snapshot in seen:
        assert TOK_PATH in snapshot, (
            "the decision was made before any path token was emitted — that is "
            "exactly the batch ordering this driver exists to replace"
        )


def test_chosen_action_is_emitted_before_the_next_decision():
    """Substep 2 must see substep 1's approx head in the stream."""
    snapshots = []

    def policy(ctx):
        snapshots.append((ctx["substep"], list(ctx["tokens"])))
        return ctx["legal"][0] if ctx["legal"] else None

    d, _ = _drive(policy, max_substeps=3)
    later = [(s, t) for s, t in snapshots if s > 0]
    if not later:
        pytest.skip("no face admitted a second substep on this graph")
    for _s, toks in later:
        assert any(t in (TOK_DIAG,) or t == TOK_SKIP for t in toks), (
            "a later substep saw no record of the earlier one"
        )


def test_skip_leaves_the_operand_exact_and_is_recorded():
    d, _ = _drive(lambda ctx: None)
    assert d.stats["faces"] > 0
    assert d.stats["applied"] == 0
    assert d.stats["skipped"] > 0
    assert TOK_SKIP in d.tokens


def test_decisions_can_differ_per_face():
    """The whole point of per-face: approximate one path, skip another."""
    def policy(ctx):
        if ctx["face_index"] % 2 == 0 and ctx["legal"]:
            return ctx["legal"][0]
        return None

    d, _ = _drive(policy)
    if d.stats["faces"] < 2:
        pytest.skip("needs >= 2 faces")
    # both outcomes present => the policy really addressed faces independently
    assert d.stats["applied"] >= 1
    assert d.stats["skipped"] >= 1


def test_only_legal_actions_are_offered():
    """Legality is read off the LIVE operand, so an illegal action is
    unrepresentable rather than merely unlikely."""
    from alphagrad.approx.common.masks import rule_is_legal

    offered = []

    def policy(ctx):
        for a in ctx["legal"]:
            offered.append((ctx["operand"], a))
        return ctx["legal"][0] if ctx["legal"] else None

    _drive(policy)
    assert offered, "no actions were ever offered"
    for st, a in offered:
        assert rule_is_legal(st, a), f"offered an illegal action: {a}"


def test_stream_is_append_only():
    """Tokens are only ever appended — the encoder carries a recurrence, so a
    rewrite of earlier tokens would silently desync it."""
    prefixes = []

    def policy(ctx):
        prefixes.append(list(ctx["tokens"]))
        return ctx["legal"][0] if ctx["legal"] else None

    d, _ = _drive(policy)
    for earlier, later in zip(prefixes, prefixes[1:]):
        assert later[: len(earlier)] == earlier, "stream was rewritten, not appended"
    assert d.tokens[: len(prefixes[-1])] == prefixes[-1]


def test_driver_runs_without_an_encoder():
    """encoder=None makes it a pure token/decision recorder (no agent needed)."""
    d, _ = _drive(lambda ctx: None)
    assert d.enc_state is None
    assert isinstance(d.tokens, list) and d.tokens
