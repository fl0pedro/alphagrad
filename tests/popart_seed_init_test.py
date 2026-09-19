"""PopArt seeds from the first real episode, not from warm-up rollouts
(owner ruling 2026-09-14).

Before this change, ``--popart-init-episodes`` defaulted to 3: episode 0 was
preceded by three full random-plan rollouts, MEASURED, purely to fill PopArt's
(mu, sigma) accumulator before the first gradient step. On the canary that
cost about nine minutes and stamped three ``episode 0`` records into the plan
log. Worse, the loop that ran those rollouts checked only
``--popart-init-episodes > 0`` -- not ``--advantage-norm`` -- so it ran even
under ``--advantage-norm none`` (the campaign's own setting), where nothing
downstream ever reads ``popart_m1``/``popart_m2``/``popart_w`` at all
(``_popart_update`` is called only under ``if use_popart:`` in
``_episode_update``). The canary's own log proved this: it ran
``--advantage-norm none`` and still logged three warm-up rollouts.

The fix:

* ``--popart-init-episodes`` now defaults to 0.
* The random-plan warm-up loop runs only when a POSITIVE value is passed
  AND ``--advantage-norm popart`` is set -- never under ``none``, no matter
  what value is passed.
* With ``--popart-init-episodes 0`` and ``--advantage-norm popart``, the
  first real episode seeds PopArt's DECODE (the (mu, sigma) that turns the
  value head's normalized output back into raw units) from its own returns,
  computed straight off the reward tensor via ``_popart_seed_returns`` -- no
  value bootstrap, so there is no circularity with the very statistics that
  rescale the value head. ``popart_m1``/``popart_m2``/``popart_w`` themselves
  are left untouched by the seed, so the ordinary ``_popart_update`` call
  later in the same episode still takes its "first update adopts the batch
  exactly" path (see its docstring), and the normal per-episode EMA update
  continues from there for every later episode.

What is pinned here:

1. The default is 0 (``test_popart_init_episodes_defaults_to_zero``).
2. ``_popart_seed_returns`` computes the plain discounted Monte-Carlo return
   from rewards/done/discount alone, matching a hand-rolled recursion, and
   takes no value-network input (``test_popart_seed_returns_*``).
3. ``_popart_update`` started from a cold ``(0, 0, 0)`` accumulator produces
   ``(mu, sigma)`` exactly equal to the mean/std of whatever batch it is fed,
   for any beta -- the property that makes "seed from the first episode's own
   returns" exact rather than approximate
   (``test_popart_update_from_cold_adopts_the_batch_exactly``).
4. The warm-up loop's own gate requires ``advantage_norm == "popart"``, so it
   can never fire under ``--advantage-norm none``
   (``test_warmup_loop_requires_advantage_norm_popart``).
5. End to end, on a real (small, CPU) two-episode run:
   * ``--advantage-norm popart --popart-init-episodes 0`` prints no
     ``[popart-init]`` warm-up line and reports a finite ``mu_quality`` on
     its very first ``[health ep0]`` row
     (``test_two_episode_run_seeds_popart_with_no_warmup``);
   * ``--advantage-norm none --popart-init-episodes 3`` -- explicitly
     positive, the exact shape of the canary's config -- STILL prints no
     ``[popart-init]`` line and no ``[health warmup]`` row: the warm-up does
     not run "regardless of the norm" any more
     (``test_two_episode_run_none_never_warms_up_even_if_requested``).
"""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax.numpy as jnp                                          # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

import alphagrad.approx.ppo as ppo                                # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_PPO = _ROOT / "src" / "alphagrad" / "approx" / "ppo.py"


# --------------------------------------------------------- 1. the default

def test_popart_init_episodes_defaults_to_zero():
    a = ppo.make_argparser().parse_args(["--example", "X"])
    assert a.popart_init_episodes == 0


# ------------------------------------------------ 2. _popart_seed_returns

def test_popart_seed_returns_matches_manual_mc_recursion():
    rng = np.random.default_rng(0)
    E, T, K = 2, 5, 3
    rewards = rng.normal(size=(E, T, K)).astype(np.float32)
    done = np.zeros((E, T), np.float32)
    done[:, -1] = 1.0
    discount = np.full((E, T), 0.9, np.float32)

    got = np.asarray(ppo._popart_seed_returns(
        jnp.asarray(rewards), jnp.asarray(done), jnp.asarray(discount)))

    want = np.zeros_like(rewards)
    run = np.zeros((E, K), np.float32)
    for t in range(T - 1, -1, -1):
        run = rewards[:, t, :] + discount[:, t, None] * (1.0 - done[:, t, None]) * run
        want[:, t, :] = run

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_popart_seed_returns_zeroes_after_done_carries_nothing_forward():
    """A step after `done` must not leak the next episode's reward backward
    -- the same guarantee GAE's own bootstrap relies on."""
    E, T, K = 1, 4, 1
    rewards = np.array([[[1.0], [2.0], [3.0], [4.0]]], np.float32)
    done = np.array([[0.0, 1.0, 0.0, 1.0]], np.float32)
    discount = np.full((E, T), 1.0, np.float32)
    got = np.asarray(ppo._popart_seed_returns(
        jnp.asarray(rewards), jnp.asarray(done), jnp.asarray(discount)))
    # segment [0,1]: G_1 = 2, G_0 = 1 + 2 = 3. segment [2,3]: G_3 = 4, G_2 = 3+4=7.
    np.testing.assert_allclose(got[0, :, 0], [3.0, 2.0, 7.0, 4.0])


def test_popart_seed_returns_takes_no_value_network_input():
    """Must stay purely reward-driven: taking a value/agent argument would
    make it circular with the very (mu, sigma) it is used to seed."""
    sig = inspect.signature(ppo._popart_seed_returns)
    assert list(sig.parameters) == ["rewards", "done", "discount"]


# ------------------------------------------- 3. _popart_update cold-start

def test_popart_update_from_cold_adopts_the_batch_exactly():
    """A `_popart_update` call starting from (0, 0, 0) leaves (mu, sigma)
    exactly the mean/std of the batch it was fed, for ANY beta. This is the
    existing debiasing property that makes "seed PopArt from episode 0's own
    returns, then let the normal per-episode update continue" exact rather
    than approximate, and it is what the finite/equal-to-its-returns
    assertion in the end-to-end test below rests on."""
    rng = np.random.default_rng(1)
    K = 4
    returns = (rng.normal(size=(6, 7, K)).astype(np.float32) * 5.0 + 2.0)
    zeros = jnp.zeros((K,), jnp.float32)
    flat = returns.reshape(-1, K)
    want_mu = flat.mean(axis=0)
    want_sigma = flat.std(axis=0)
    for beta in (0.001, 0.1, 0.5, 1.0):
        nm1, nm2, nw = ppo._popart_update(
            zeros, zeros, zeros, jnp.asarray(returns), beta, 1e-6, 1e12, 5.0)
        mu, sigma = ppo._popart_derive(nm1, nm2, nw, 1e-6, 1e12)
        mu, sigma = np.asarray(mu), np.asarray(sigma)
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
        np.testing.assert_allclose(mu, want_mu, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sigma, want_sigma, rtol=1e-4, atol=1e-4)


# --------------------------------------------- 4. the warm-up loop's gate

def _warmup_gate_if():
    """The ``if ep == 0 and ... popart_init_episodes ... > 0:`` node that
    guards the random-plan warm-up loop."""
    src = inspect.getsource(ppo)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            dumped = ast.dump(node.test)
            if "popart_init_episodes" in dumped and "'ep'" in dumped:
                return node
    raise AssertionError("the warm-up loop's `if ep == 0 and ...` gate "
                          "was not found")


def test_warmup_loop_requires_advantage_norm_popart():
    """The random-plan warm-up must never run under --advantage-norm none,
    no matter what --popart-init-episodes is set to. Before this ruling the
    gate checked only `popart_init_episodes > 0`, so the canary's
    `--advantage-norm none` run still paid for three warm-up rollouts that
    fed statistics nothing downstream reads."""
    node = _warmup_gate_if()
    dumped = ast.dump(node.test)
    assert "advantage_norm" in dumped, (
        "the warm-up gate no longer mentions args.advantage_norm at all")
    assert "'popart'" in dumped, (
        "the warm-up gate does not compare advantage_norm against 'popart'")


def test_warmup_loop_still_requires_a_positive_count():
    node = _warmup_gate_if()
    dumped = ast.dump(node.test)
    assert "popart_init_episodes" in dumped and "Gt" in dumped, (
        "the warm-up gate no longer requires popart_init_episodes > 0")


# --------------------------------------------------- 5. end to end (CPU)

# Mirrors tools/smoke.sh's canonical 2-episode CPU config exactly, so this
# exercises the real training loop the campaign runs, not a synthetic stand-in.
_COMMON = [
    "--variant", "full", "--face-actions", "--unified-face-head", "--live-faces",
    "--set-pointer", "--dynamic-substeps", "--max-substeps", "1",
    "--incremental-encode", "--grad-window", "0", "--dataset", "none",
    "--cmp-type", "flops", "--mem-type", "peak_memory", "--terminal-rewards-only",
    "--rewards", "cmp", "mem", "--lambda-cmp", "1", "--lambda-mem", "1",
    "--lambda-frob", "1", "--episodes", "2", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--example", "Helmholtz",
    # THE GRADIENT ORACLE IS OFF because the plan log is. The oracle reads the
    # distinct elimination orders of an oracle-due episode off the plan-log
    # records (agent/oracle-cap, dsnn-dfw.22) and ppo.py refuses the pair
    # rather than check nothing. This file is about the PopArt seeding.
    "--grad-oracle", "off",
]


def _run_ppo(name, *extra, timeout=1800):
    # Strip every inherited ALPHAGRAD_* var before setting our own -- a full
    # `pytest tests/` run leaves some behind (e.g. tests/edge_mem_test.py
    # writes ALPHAGRAD_MAX_DELTA_TOKENS=128 into os.environ at collection
    # time; it is a dead write for that module's own purposes, since env.py
    # had already frozen the real value by then, but it is a live leak into
    # any LATER subprocess that inherits os.environ, and 128 is below this
    # config's fold chunk). See tests/policy_regression_gate_test.py's
    # `_run_gate` for the same guard, for the same reason.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update({
        "JAX_PLATFORMS": "cpu",
        "ALPHAGRAD_EXTEND_CHUNK": "256",
        "ALPHAGRAD_EXTEND_UNROLL": "32",
        "ALPHAGRAD_DELTA_OVERFLOW": "clip",
        "ALPHAGRAD_POLICY": "palimpsa",
        "ALPHAGRAD_INCREMENTAL_TOKENS": "1",
        "ALPHAGRAD_UNIFIED_FACE_ENUM": "1",
        "ALPHAGRAD_SKIP_COUNT_OPS": "1",
        "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
    })
    return subprocess.run(
        [sys.executable, str(_PPO), *_COMMON, "--name", name, *extra],
        env=env, capture_output=True, text=True, timeout=timeout)


def _report(what, r):
    return (f"{what} (rc={r.returncode})\n--- stdout (tail) ---\n"
            f"{r.stdout[-6000:]}\n--- stderr (tail) ---\n{r.stderr[-3000:]}")


def _health_field(stdout, label, field):
    """Value of `field=` on the `[health <label>]` line, or None."""
    for line in stdout.splitlines():
        if f"[health {label}]" in line and "ppo=" in line:
            for tok in line.split():
                if tok.startswith(field + "="):
                    return tok.split("=", 1)[1]
    return None


@pytest.mark.slow
def test_two_episode_run_seeds_popart_with_no_warmup():
    r = _run_ppo("popart_seed_test_zero_init",
                 "--advantage-norm", "popart", "--popart-init-episodes", "0")
    assert r.returncode == 0, _report("popart-seed run failed", r)
    assert "[popart-init]" not in r.stdout, _report(
        "a warm-up rollout ran despite --popart-init-episodes 0", r)
    mu0 = _health_field(r.stdout, "ep0", "mu_quality")
    assert mu0 is not None and mu0 not in ("n/a",), _report(
        "episode 0 printed no mu_quality reading", r)
    val = float(mu0)
    assert np.isfinite(val), _report(
        f"PopArt mu_quality is non-finite after episode 0 ({mu0})", r)


@pytest.mark.slow
def test_two_episode_run_none_never_warms_up_even_if_requested():
    """The exact shape of the canary's bug: --advantage-norm none with a
    POSITIVE --popart-init-episodes must still run no warm-up at all."""
    r = _run_ppo("popart_seed_test_none_positive",
                 "--advantage-norm", "none", "--popart-init-episodes", "3")
    assert r.returncode == 0, _report("advantage-norm none run failed", r)
    assert "[popart-init]" not in r.stdout, _report(
        "a warm-up rollout ran under --advantage-norm none", r)
    assert "[health warmup]" not in r.stdout, _report(
        "a warm-up episode was logged under --advantage-norm none", r)
