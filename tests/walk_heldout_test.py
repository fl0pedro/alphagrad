"""A3 — per-episode walk data + held-out scoring.

Pins the two properties the workstream promised and that nothing else in the
suite can catch:

1. **Flag-off bit-identity.** With ``ALPHAGRAD_WALK_HELDOUT`` and
   ``ALPHAGRAD_WALK_ROTATE`` unset, the probe batch is byte-for-byte the
   pre-A3 batch (``PRNGKey(ALPHAGRAD_WALK_PROBE_SEED)``), the published
   episode index is IGNORED, and repeated calls return the identical float.
2. **Rotation actually rotates, held-out is actually held out.** Different
   episodes give different batches and different scores; the held-out batch is
   never the training batch of the same OR of any other episode.

Plus the A4 invariant this workstream must not regress: ``_walk_argnums``
excludes 0-d argnums.
"""
import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                   # noqa: E402
import jax.random as jrand                                # noqa: E402

from alphagrad.approx import env as E                     # noqa: E402


_WALK_ENV = ("ALPHAGRAD_WALK_HELDOUT", "ALPHAGRAD_WALK_ROTATE",
             "ALPHAGRAD_WALK_EPISODE", "ALPHAGRAD_WALK_STEPS",
             "ALPHAGRAD_WALK_PROBE_SEED", "ALPHAGRAD_WALK_LR",
             "ALPHAGRAD_WALK_NOISE_STD")


@pytest.fixture(autouse=True)
def _clean_walk_env():
    """Every test starts from "no A3 switch set" and leaves nothing behind.

    The walk configuration is process-wide environment state; a leaked
    ``ALPHAGRAD_WALK_ROTATE`` would silently change every later test in the
    session (and this is exactly the "two paths that must agree" bug class the
    module docstring warns about).
    """
    saved = {k: os.environ.get(k) for k in _WALK_ENV}
    for k in _WALK_ENV:
        os.environ.pop(k, None)
    E._PROBE_BATCH.clear()
    del E._WALK_FINGERPRINT[:]
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    E._PROBE_BATCH.clear()
    del E._WALK_FINGERPRINT[:]


# ---------------------------------------------------------------- fixtures
_B, _D, _H, _C = 16, 6, 8, 3


# A FIXED TEACHER. Labels must be a LEARNABLE function of the inputs, not
# random: with random labels the only way to reduce the loss is to memorise the
# batch, so a held-out score has no signal in it and the "exact" gradient
# generalises WORSE than a crippled one (measured: -0.052 vs +0.023 on a random
# -label fixture). That is the very failure mode A3 exists to expose, so the
# fixture must not reproduce it by accident.
_TEACHER = jrand.normal(jrand.PRNGKey(11), (_D, _C))


def _labels(x):
    return jax.nn.one_hot(jnp.argmax(x @ _TEACHER, axis=-1), _C)


def _data_gen(keys):
    """A cheap jitted ``keys -> (x, y)`` sampler, like ``common.examples``."""
    x = jrand.normal(keys[0], (_B, _D))
    return x, _labels(x)


def _loss(x, y, w1, w2):
    h = jnp.tanh(x @ w1)
    logits = h @ w2
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(jnp.sum(y * logp, axis=-1))


# ONE config object for the whole module. ``_probe_batch`` keys the cache on
# ``id(config.data_gen)``, exactly as the real env does with its single
# long-lived ``EnvConfig``; handing out a fresh jit wrapper per call would
# manufacture cache entries that the production path never creates.
def _config(_cache={}):
    from types import SimpleNamespace
    if not _cache:
        _cache["cfg"] = SimpleNamespace(
            data_gen=jax.jit(_data_gen),
            target_fun=_loss,
            argnums=(2, 3),
            has_aux=False,
        )
    return _cache["cfg"]


def _base_args(seed=0, _cache={}):
    if seed in _cache:
        return list(_cache[seed])
    k = jrand.PRNGKey(seed)
    k1, k3, k4 = jrand.split(k, 3)
    x = jrand.normal(k1, (_B, _D))
    y = _labels(x)
    w1 = 0.5 * jrand.normal(k3, (_D, _H))
    w2 = 0.5 * jrand.normal(k4, (_H, _C))
    _cache[seed] = [x, y, w1, w2]
    return list(_cache[seed])


def _exact_plan():
    """Stand-in for a plan's compiled executable: the EXACT gradient."""
    return jax.jit(jax.grad(_loss, argnums=(2, 3)))


def _frozen_plan():
    """Stand-in for the gradient-freezing hack: the READOUT's gradient is
    zeroed, which is structurally what skipping a face does to the parameters
    downstream of it -- they stay at their initial values for the whole walk."""
    g = jax.grad(_loss, argnums=(2, 3))

    def f(*a):
        gw1, gw2 = g(*a)
        return gw1, jnp.zeros_like(gw2)
    return jax.jit(f)


def _quality(plan, episode=None, steps=5):
    os.environ["ALPHAGRAD_WALK_STEPS"] = str(steps)
    return E._loss_drop_quality(_config(), plan, _base_args(), None,
                                episode=episode)


# ------------------------------------------------------------------- tests
def test_flag_off_probe_batch_is_the_pre_a3_batch():
    """The exact byte-level pin: default configuration reproduces the batch
    the pre-A3 code built, ``data_gen(split(PRNGKey(20260807), 5))``."""
    cfg = _config()
    got = E._probe_batch(cfg, _base_args())
    want = cfg.data_gen(jrand.split(jrand.PRNGKey(20260807), 5))
    assert E._walk_seed("train") == E._walk_probe_seed() == 20260807
    for a, b in zip(got, want):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_flag_off_ignores_the_published_episode():
    """Rotation OFF: publishing an episode must change nothing at all — not
    the seed, not the batch, not the score."""
    q0 = _quality(_exact_plan())
    E.set_walk_episode(7)
    assert E.walk_episode() == 7
    assert E._walk_seed("train") == 20260807
    q1 = _quality(_exact_plan())
    E.set_walk_episode(123456)
    q2 = _quality(_exact_plan())
    assert q0 == q1 == q2, (q0, q1, q2)
    # ...and the cache never grew past the single pre-A3 key.
    assert len(E._PROBE_BATCH) == 1


def test_flag_off_heldout_args_are_the_walk_args():
    """Held-out OFF must not even build a second batch."""
    assert not E.walk_heldout_enabled()
    _quality(_exact_plan())
    assert len(E._PROBE_BATCH) == 1


def test_rotation_changes_the_batch_and_the_score():
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    cfg = _config()
    b0 = E._probe_batch(cfg, _base_args(), "train", 0)
    b1 = E._probe_batch(cfg, _base_args(), "train", 1)
    assert not np.array_equal(np.asarray(b0[0]), np.asarray(b1[0]))
    qs = [_quality(_exact_plan(), episode=e) for e in range(4)]
    assert len(set(qs)) == len(qs), qs          # every episode differs
    assert np.std(qs) > 0.0


def test_rotation_reads_the_published_episode_when_not_passed():
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    E.set_walk_episode(3)
    assert E._walk_seed("train") == 20260807 + 3 * E._WALK_EPISODE_STRIDE
    assert E._walk_seed("train", 5) == 20260807 + 5 * E._WALK_EPISODE_STRIDE


def test_heldout_batch_is_disjoint_from_every_training_batch():
    os.environ["ALPHAGRAD_WALK_HELDOUT"] = "1"
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    cfg = _config()
    train = {E._walk_seed("train", e) for e in range(50)}
    for e in range(50):
        assert E._walk_seed("eval", e) not in train
    a = E._probe_batch(cfg, _base_args(), "train", 0)
    b = E._probe_batch(cfg, _base_args(), "eval", 0)
    assert not np.array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_heldout_scores_on_the_held_out_batch():
    """Both endpoints move to B, so the score changes; and it stays a finite
    number in [-1, 1] (the channel's contract)."""
    q_in = _quality(_exact_plan())
    os.environ["ALPHAGRAD_WALK_HELDOUT"] = "1"
    q_out = _quality(_exact_plan())
    assert q_in != q_out
    assert np.isfinite(q_out) and -1.0 <= q_out <= 1.0


def test_walk_argnums_still_excludes_zero_d():
    """A4 invariant — the 0-d tangent seed is not a weight the walk may step."""
    from types import SimpleNamespace
    args = _base_args() + [jnp.asarray(1.0)]        # 0-d seed in slot 4
    cfg = SimpleNamespace(argnums=(2, 3, 4))
    assert E._walk_argnums(cfg, args) == (2, 3)


def test_probe_cache_is_bounded_under_rotation():
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    os.environ["ALPHAGRAD_WALK_HELDOUT"] = "1"
    cfg, ba = _config(), _base_args()
    for e in range(20):
        E._probe_batch(cfg, ba, "train", e)
        E._probe_batch(cfg, ba, "eval", e)
    assert len(E._PROBE_BATCH) <= E._PROBE_BATCH_MAX


def test_fingerprint_prints_once_per_episode_with_provenance(capfd):
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    for e in (0, 0, 1):
        _quality(_exact_plan(), episode=e)
    out = capfd.readouterr().out
    lines = [ln for ln in out.splitlines() if "loss-drop walk armed" in ln]
    assert len(lines) == 2, out                      # ep 0 once, ep 1 once
    assert "episode=0" in lines[0] and "episode=1" in lines[1]
    for ln in lines:
        assert "fingerprint(probe+W0)=" in ln        # legacy field kept
        assert "train_batch=" in ln and "W0=" in ln
        assert "rotate=1" in ln


def test_flag_off_prints_exactly_one_fingerprint_line(capfd):
    """A flag-off run must not become chattier: the latch is keyed on the
    effective SEED, so publishing 5 episodes still yields ONE line."""
    for e in range(5):
        E.set_walk_episode(e)
        _quality(_exact_plan())
    out = capfd.readouterr().out
    assert len([ln for ln in out.splitlines()
                if "loss-drop walk armed" in ln]) == 1, out


def test_frozen_gradient_plan_is_scored_below_exact(capfd):
    """Sanity for the decisive experiment: a plan that zeroes one parameter's
    gradient must not tie with the exact plan on the held-out channel."""
    os.environ["ALPHAGRAD_WALK_HELDOUT"] = "1"
    os.environ["ALPHAGRAD_WALK_ROTATE"] = "1"
    q_exact = _quality(_exact_plan(), episode=0, steps=200)
    q_frozen = _quality(_frozen_plan(), episode=0, steps=200)
    assert q_exact > q_frozen, (q_exact, q_frozen)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
