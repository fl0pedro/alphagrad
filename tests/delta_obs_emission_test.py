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
# THIS MODULE DOES NOT PIN THE DELTA BUDGET AT IMPORT, AND MUST NOT.
#
# ``env.py`` freezes MAX_DELTA_TOKENS and _DELTA_OVERFLOW into module constants
# at its FIRST import, and ``pytest tests/`` imports EVERY test module during
# collection before it runs the first test. So the two assignments that used to
# stand here --
#
#     os.environ["ALPHAGRAD_MAX_DELTA_TOKENS"] = "1024"
#     os.environ["ALPHAGRAD_DELTA_OVERFLOW"] = "clip"
#
# -- under the comment "PINNED, not inherited (the suite runs one process per
# module, so these are ours to set)" did two wrong things and one right thing
# only when this module happened to be run alone:
#
#  1. THEY DID NOT TAKE. Any module collected earlier (a < d) imported
#     alphagrad first and froze the defaults, 32768 and "raise". The diag block
#     on this graph is 2190 tokens, so it never clipped, and
#     ``test_base_plus_deltas_reconstructs_the_stream[True]`` died on its own
#     "this graph no longer exercises the MAX_DELTA_TOKENS clip" guard -- the
#     guard working exactly as designed, on a pin that had silently lost a
#     race.
#  2. THEY LEAKED. The variables stayed set in the shared process for the rest
#     of the run, so ``tests/per_face_apply_test.py``, whose subprocess probe
#     asserts "the default MAX_DELTA_TOKENS is 32768", inherited 1024 and
#     failed too.
#
# The clip path is still exercised on every run -- see
# ``test_the_clip_is_exercised_at_the_old_budget_in_its_own_interpreter``,
# which pins 1024+clip in a FRESH interpreter, where a pin is the only thing
# that can work. The RAISE default has its own fresh-interpreter cases at the
# bottom of this file.

import pathlib                                                    # noqa: E402
import subprocess                                                 # noqa: E402
import sys                                                        # noqa: E402

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
    `_callback` uses (position-independent, so a prefix vertex and the newest
    vertex decode identically)."""
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.env import decode_vertex_rule_specs

    tk = IncrementalPathTokenizer(
        _CJ.jaxpr, (0, 1), list(_CONSTS), list(ARGS), vocab_size=vocab)
    stream = [int(t) for t in tk.base_tokens()]
    seg = [int(g) for g in tk.last_eqn_ids()]
    for v in prefix:
        rules = decode_vertex_rule_specs(
            _CJ.jaxpr, int(v), specs_np[int(v) - 1])
        stream += [int(t) for t in tk.eliminate(int(v), tuple(rules))]
        seg += [int(g) for g in tk.last_eqn_ids()]
    return stream, seg


def _check_reconstruction(diag, require_clip=False):
    """base_observation() + every step's delta IS the append-only stream.

    Up to the documented CLIP: a block longer than MAX_DELTA_TOKENS loses its
    tail, and -- unlike the old absolute cursor, which deferred that tail to
    the next step and attributed it to the wrong vertex -- the tail is simply
    dropped. Clipping is the ALPHAGRAD_DELTA_OVERFLOW=clip opt-out; the default
    since aa0774c8 is to RAISE.

    A PLAIN FUNCTION, not a test, because it is called from two places: the
    in-process test below at whatever budget ``env.py`` froze, and the
    fresh-interpreter test that pins 1024+clip -- where ``diag=True`` on this
    graph produces a 2190-token block and ``require_clip`` demands that the
    clipping branch actually ran. The in-process caller cannot demand that: it
    does not own the frozen budget (see the module header).
    """
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
    # NON-TERMINAL steps only: the terminal step would run the measurement.
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
    if require_clip:
        assert clipped, (
            f"this graph no longer exercises the MAX_DELTA_TOKENS clip at the "
            f"pinned budget {MAX_DELTA_TOKENS}, so the clip-aware comparison "
            f"above is vacuous")
    return clipped


@pytest.mark.parametrize("diag", [False, True])
def test_base_plus_deltas_reconstructs_the_stream(diag):
    """The reconstruction, at whatever delta budget this process froze.

    The CLIP half of the claim is not asserted here and cannot be: the budget
    belongs to whichever module imported ``env`` first (module header), so on a
    32768-wide budget this graph's 2190-token diag block simply fits. What this
    case does pin, at every budget, is that base + deltas is the stream token
    for token and eqn-id for eqn-id.
    ``test_the_clip_is_exercised_at_the_old_budget_in_its_own_interpreter``
    covers the clip itself, on every run.
    """
    _check_reconstruction(diag)


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


# --------------------------------------------------------------------------
# The RAISE default (aa0774c8). env.py freezes MAX_DELTA_TOKENS and
# _DELTA_OVERFLOW at first import and this module pins 1024+clip above, so
# the default behaviour can only be observed in a fresh interpreter with the
# knobs unset. Each case below runs one, with every ALPHAGRAD_* variable
# scrubbed -- pure library defaults, exactly what a bare `import` gets.


def _run_in_a_fresh_interpreter(script: str, **pins: str) -> None:
    """Run ``script`` in a child interpreter whose ALPHAGRAD_* configuration is
    EXACTLY ``pins`` -- every inherited ALPHAGRAD_* variable is scrubbed first.

    The scrub is the contract: a constant ``env.py`` freezes at its first import
    can only be set by a process that has not imported it yet, and "what the
    defaults are" can only be observed by a process that inherited none. With
    no pins this is the pure-defaults probe; with pins it is the only way this
    module can still pin a non-default budget now that it no longer mutates the
    shared process at import time (see the module header).
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.update(pins)
    r = subprocess.run([sys.executable, "-c", script],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (
        f"fresh-interpreter probe failed (rc={r.returncode}, pins={pins})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "PROBE-OK" in r.stdout, r.stdout


def _run_pure_defaults(script: str) -> None:
    _run_in_a_fresh_interpreter(script)


def test_the_clip_is_exercised_at_the_old_budget_in_its_own_interpreter():
    """THE CLIP BRANCH, on every run, at the 1024-token budget + clip opt-out.

    This is the case the module-level ``os.environ`` pins were for, moved to
    the one place a pin can still work. A fresh interpreter with
    ALPHAGRAD_MAX_DELTA_TOKENS=1024 and ALPHAGRAD_DELTA_OVERFLOW=clip freezes
    those constants before anything imports ``env``, so this graph's
    2190-token diag block genuinely overflows and ``require_clip`` fails if it
    ever stops doing so. The shared-process case above runs the same
    reconstruction at whatever budget the process happens to hold.
    """
    here = str(pathlib.Path(__file__).resolve().parent)
    _run_in_a_fresh_interpreter(
        "import sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "import delta_obs_emission_test as M\n"
        "assert M.MAX_DELTA_TOKENS == 1024, M.MAX_DELTA_TOKENS\n"
        "from alphagrad.approx.env import _DELTA_OVERFLOW\n"
        "assert _DELTA_OVERFLOW == 'clip', _DELTA_OVERFLOW\n"
        "assert M._check_reconstruction(True, require_clip=True)\n"
        "print('PROBE-OK')\n",
        ALPHAGRAD_MAX_DELTA_TOKENS="1024",
        ALPHAGRAD_DELTA_OVERFLOW="clip",
    )


def test_delta_overflow_raises_under_pure_defaults():
    """A delta that does not fit MAX_DELTA_TOKENS RAISES by default -- a
    clipped delta desyncs the recurrence for the rest of the episode, and
    #81 was exactly that loss happening behind process-blind counters."""
    _run_pure_defaults("""
import alphagrad.approx.env as E
assert E.MAX_DELTA_TOKENS == 32768, E.MAX_DELTA_TOKENS
assert E._DELTA_OVERFLOW == "raise", E._DELTA_OVERFLOW
try:
    E._record_delta_truncation(E.MAX_DELTA_TOKENS + 1)
except ValueError as e:
    assert "MAX_DELTA_TOKENS" in str(e), e
    print("PROBE-OK")
else:
    raise SystemExit("an overflowing delta did NOT raise under pure defaults")
""")


def test_delta_overflow_boundary_is_exact():
    """Fits at N, raises at N+1: raw_len == MAX_DELTA_TOKENS is NOT an
    overflow (no raise, no truncation counters), raw_len == N+1 is."""
    _run_pure_defaults("""
import alphagrad.approx.env as E
n = E.MAX_DELTA_TOKENS
before = (int(E._TOKENIZATION_TRUNCATION_COUNT[0]),
          int(E._TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0]))
E._record_delta_truncation(n)          # exactly full: fits
after = (int(E._TOKENIZATION_TRUNCATION_COUNT[0]),
         int(E._TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0]))
assert before == after, (before, after)
try:
    E._record_delta_truncation(n + 1)  # one past: raises
except ValueError:
    print("PROBE-OK")
else:
    raise SystemExit(f"raw_len == MAX_DELTA_TOKENS + 1 == {n + 1} did not raise")
""")
