"""STAGE 2: the env emits the DELTA, and the delta is the APPROXIMATED graph.

Stage 1 proved the per-step delta buffer equals ``stream[pos : pos + W]``
(tests/delta_buffer_equivalence_test.py). Stage 2 deleted the stream: the
callback now returns ONLY the tokens the last elimination emitted, with the
tokenizer's own length in a header slot, and the base is a host-side constant
(``VertexEliminationEnv.base_observation``).

Two things must hold, and neither is visible in any metric we log:

  1. CONCATENATION. base + delta_1 + ... + delta_k is the length-k stream,
     token for token and eqn-id for eqn-id. If it were not, the encoder's
     recurrence would be reading a different graph than the one the
     measurement builds -- silently.

  2. THE DELTA CARRIES THE APPROXIMATIONS. ``_incremental_stream_tokens``
     drives ``tk.eliminate(v, rules, per_face)``; a producer that eliminated
     BARE would emit the EXACT graph's tokens while the measurement built the
     approximated one (the divergence c83a6a1 fixed on the LiveFaceStream
     side). So the test runs the SAME order twice -- once all-exact, once with
     a DIAG on every vertex -- and requires the deltas to DIFFER. If the
     observation ever stops depending on the approximation decisions, this
     goes red.

  3. ONE VOCABULARY. ``base_observation`` builds its own tokenizer, so it
     must resolve ALPHAGRAD_INCR_TOKEN_VOCAB exactly the way
     ``_incremental_stream_tokens`` does -- otherwise the same symbol maps to
     a different id, i.e. a different embedding row, for the base only. Test 1
     pins this: any mismatch breaks the concatenation at the first base token
     that differs. It says nothing about WHICH id space is right; the
     default IS the launchers' ``--vocab-size 512`` now, so the tokenizer
     and the policy embedding name one id space (230 reserved + 10 digits
     + 272 name symbols, max id 511).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

from alphagrad.approx.env import (                                # noqa: E402
    MAX_DELTA_TOKENS, MAX_FACES, MAX_RULES_PER_VERTEX, FACE_SLOTS,
    EnvConfig, VertexEliminationEnv, _callback,
)


@pytest.fixture(autouse=True)
def _append_only(monkeypatch):
    """delta_obs only exists under the append-only tokenizer. Set it PER
    TEST and restore -- the suite shares one interpreter and `_callback`
    reads this variable on every call, so a module-level assignment would
    silently switch every later test's env to the append-only path."""
    monkeypatch.setenv("ALPHAGRAD_INCREMENTAL_TOKENS", "1")


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
_CJ = jax.make_jaxpr(_fn)(*ARGS)
_CONSTS = list(_CJ.literals)
_V = len(_CJ.jaxpr.eqns)


def _cfg(delta_obs):
    return EnvConfig(
        jaxpr=_CJ.jaxpr, argnums=(0, 1), has_aux=False, sparse=False,
        cmp_type="flops", mem_type="peak_memory",
        # Non-terminal steps return straight after tokenization, so no
        # measurement runs and the test is a tokenizer test.
        terminal_rewards_only=True,
        delta_obs=delta_obs,
    )


def _specs(diag):
    """(V, MAX_RULES, 3) rows: all-exact, or a DIAG(0,0,factor=2) per vertex."""
    rows = np.full((_V, MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    rows[..., 2] = 0
    if diag:
        rows[:, 0, :] = np.asarray([0, 0, 2], dtype=np.int32)
    return jnp.asarray(rows)


_FACES = jnp.full((_V, MAX_FACES, FACE_SLOTS, 3), -1, dtype=jnp.int32)
_SKIPS = jnp.zeros((_V, MAX_FACES), dtype=jnp.int32)
_ORDER = jnp.asarray(np.arange(1, _V + 1), dtype=jnp.int32)


def _delta_at(cfg, specs, step):
    """One step's (tokens, eqn_ids) as the delta wire, unpacked."""
    tok, eqn, _r = _callback(
        cfg, ARGS, _CONSTS, _ORDER, specs, _FACES, _SKIPS, step)
    t, e = np.asarray(tok), np.asarray(eqn)
    n = int(t[0])
    assert n == int(e[0]), "header slots disagree"
    assert 0 <= n <= MAX_DELTA_TOKENS
    return list(t[1:1 + n]), list(e[1:1 + n])


_VOCAB = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))


def _cold_stream(prefix, specs_np, vocab=_VOCAB):
    """Reference replay: tokenize exactly `prefix` with the same rule decode
    `_callback` uses, so `is_last` lands on the same vertex."""
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.env import decode_vertex_rule_specs

    tk = IncrementalPathTokenizer(
        _CJ.jaxpr, (0, 1), list(_CONSTS), list(ARGS), vocab_size=vocab)
    stream = [int(t) for t in tk.base_tokens()]
    seg = [int(g) for g in tk.last_eqn_ids()]
    last = len(prefix) - 1
    for k, v in enumerate(prefix):
        rules = decode_vertex_rule_specs(
            _CJ.jaxpr, int(v), specs_np[int(v) - 1], is_last=(k == last))
        stream += [int(t) for t in tk.eliminate(int(v), tuple(rules))]
        seg += [int(g) for g in tk.last_eqn_ids()]
    return stream, seg


@pytest.mark.parametrize("diag", [False, True])
def test_base_plus_deltas_reconstructs_the_stream(diag):
    """base_observation() + every step's delta IS the append-only stream.

    Up to the documented CLIP: a block longer than MAX_DELTA_TOKENS loses its
    tail, and -- unlike the old absolute cursor, which deferred that tail to
    the next step and attributed it to the wrong vertex -- the tail is simply
    dropped. ``diag=True`` on this graph produces a 1346-token block against
    the default 1024 budget, so the clipping branch is genuinely exercised.
    """
    if os.environ.get("ALPHAGRAD_TOKENS_MID_COMPRESS", "1") == "0":
        pytest.skip("mid-compress decode changes is_last semantics")
    specs = _specs(diag)
    specs_np = np.asarray(specs)
    cfg = _cfg(True)
    env = VertexEliminationEnv(cfg, args=ARGS, consts=_CONSTS)

    btok, beqn, bn = env.base_observation()
    rebuilt_t = list(np.asarray(btok)[:bn])
    rebuilt_e = list(np.asarray(beqn)[:bn])

    prev_t, prev_e = _cold_stream([], specs_np)
    assert rebuilt_t == prev_t, "base tokens differ from the stream's base"
    assert rebuilt_e == prev_e, "base eqn ids differ from the stream's base"

    exp_t, exp_e = list(prev_t), list(prev_e)
    clipped = 0
    # NON-TERMINAL steps only: the terminal step would run the measurement,
    # and its is_last decode is the one the prefix property does not cover.
    for k in range(1, _V):
        cur_t, cur_e = _cold_stream(list(range(1, k + 1)), specs_np)
        assert cur_t[:len(prev_t)] == prev_t, (
            f"step {k}: the reference replay lost the prefix property")
        blk_t = cur_t[len(prev_t):]
        blk_e = cur_e[len(prev_e):]
        if len(blk_t) > MAX_DELTA_TOKENS:
            clipped += 1
        blk_t, blk_e = blk_t[:MAX_DELTA_TOKENS], blk_e[:MAX_DELTA_TOKENS]

        dt, de = _delta_at(cfg, specs, k)
        assert dt == blk_t, (
            f"step {k}: the emitted delta is not this elimination's block "
            f"({len(dt)} vs {len(blk_t)} tokens)")
        assert de == blk_e, f"step {k}: delta eqn ids are not the block's"

        rebuilt_t += dt
        rebuilt_e += de
        exp_t += blk_t
        exp_e += blk_e
        prev_t, prev_e = cur_t, cur_e

    assert rebuilt_t == exp_t, "base + deltas != the (clip-aware) stream"
    assert rebuilt_e == exp_e, "eqn ids diverged"
    if diag:
        assert clipped, (
            "this graph no longer exercises the MAX_DELTA_TOKENS clip, so the "
            "clip-aware comparison above is vacuous")


def test_the_delta_describes_the_APPROXIMATED_graph():
    """Same order, different approximation -> different observation.

    A delta producer that eliminated BARE (no rules, no per-face transforms)
    would emit byte-identical tokens for both, i.e. the policy would be
    reading the exact graph while the measurement built the approximated one.
    """
    cfg = _cfg(True)
    exact = [_delta_at(cfg, _specs(False), k)[0] for k in range(1, _V)]
    diagd = [_delta_at(cfg, _specs(True), k)[0] for k in range(1, _V)]
    assert any(a != b for a, b in zip(exact, diagd)), (
        "the delta is IDENTICAL with and without a DIAG on every vertex -- "
        "the observation no longer carries the approximation decisions")


def test_full_stream_env_is_untouched():
    """delta_obs=False still returns the legacy MAX_TOKENS full stream --
    that is the observation alpha0 / gdpo / gfn / mu0 and the ray workers
    consume, and stage 2 must not have moved it."""
    from alphagrad.approx.env import MAX_TOKENS

    tok, eqn, _r = _callback(
        _cfg(False), ARGS, _CONSTS, _ORDER, _specs(True), _FACES, _SKIPS, 2)
    assert np.asarray(tok).shape == (MAX_TOKENS,)
    assert np.asarray(eqn).shape == (MAX_TOKENS,)
    ref_t, _ref_e = _cold_stream([1, 2], np.asarray(_specs(True)))
    assert list(np.asarray(tok)[:len(ref_t)]) == ref_t


def test_obs_width_matches_the_wire():
    """The Ray measurement pool preallocates at `env.obs_width`; it must be
    the width the callback actually declares."""
    from alphagrad.approx.env import MAX_TOKENS

    e_d = VertexEliminationEnv(_cfg(True), args=ARGS, consts=_CONSTS)
    e_f = VertexEliminationEnv(_cfg(False), args=ARGS, consts=_CONSTS)
    assert e_d.obs_width == 1 + MAX_DELTA_TOKENS
    assert e_f.obs_width == MAX_TOKENS
    assert e_d._callback_shape[0].shape == (e_d.obs_width,)
    assert e_f._callback_shape[0].shape == (e_f.obs_width,)


def test_reset_carries_an_empty_delta_and_no_callback():
    """Under delta_obs, reset() makes NO host callback: the base is a
    constant and step 0 has eliminated nothing."""
    env = VertexEliminationEnv(_cfg(True), args=ARGS, consts=_CONSTS,
                               num_envs=0)
    st = env.reset()
    assert int(st.delta_count) == 0
    assert st.delta_tokens.shape == (MAX_DELTA_TOKENS,)
    assert st.delta_eqns.shape == (MAX_DELTA_TOKENS,)
    assert np.all(np.asarray(st.delta_tokens) == 0)
    assert np.all(np.asarray(st.delta_eqns) == -1)
