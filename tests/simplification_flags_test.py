"""The radical-simplification flag set: VERIFICATION that each flag does what
its name says (element 2 of the 2026-08-26 design).

Nothing here is new machinery -- every flag below already exists at HEAD. The
tests exist because the design's whole premise is that these five knobs, set
together, remove the priors CLEAN_DESIGN_AUDIT lists, and nothing in the repo
had ever pinned that they actually do:

* ``--discount 1.0 --gae-lambda 1.0``  -> pure terminal Monte-Carlo credit:
  ``A_t == R_terminal - V(s_t)`` at EVERY step, no ``(gamma*lambda)^(T-t)``
  attenuation (audit d2, the "first elimination gets 0.31% of the signal"
  finding).
* ``--advantage-norm none``            -> PopArt is bypassed ENTIRELY, and no
  other per-batch / per-minibatch advantage standardisation exists anywhere
  (audit e1).
* ``--entropy-weight 0 --face-entropy-weight 0`` -> both entropy gradients are
  EXACTLY zero (audit f1/f2).
* ``--face-entropy-floor 0``           -> the hinge is bitwise inert (audit f3).
* ``--face-logit-clamp 0``             -> the tanh reparameterisation is off,
  i.e. 0 is already the off-value (audit a4).
"""
from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402

import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.ppo import (                              # noqa: E402
    _face_entropy_floor_penalty,
    _split_entropy_bonus,
    get_advantages,
)
from alphagrad.approx.common.gae import (                        # noqa: E402
    inverse_reward_normalization_fn,
)
from alphagrad.approx import unified_face_head as ufh            # noqa: E402


# ---------------------------------------------------------------------------
# --discount 1.0 --gae-lambda 1.0: pure terminal Monte-Carlo credit
# ---------------------------------------------------------------------------

def _terminal_only(E=3, T=12, H=3, seed=0):
    """(rewards, done, value, next_value) for a terminal-only episode."""
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(E, H)).astype(np.float32) * 3.0
    rewards = np.zeros((E, T, H), np.float32)
    rewards[:, -1, :] = R
    done = np.zeros((E, T), np.float32)
    done[:, -1] = 1.0
    value = rng.normal(size=(E, T, H)).astype(np.float32) * 0.5
    nvalue = np.concatenate([value[:, 1:, :], value[:, -1:, :]], axis=1)
    return (jnp.asarray(rewards), jnp.asarray(done), jnp.asarray(value),
            jnp.asarray(nvalue), R)


def test_discount1_lambda1_is_monte_carlo_credit_at_every_step():
    E, T, H = 3, 12, 3
    rewards, done, value, nvalue, R = _terminal_only(E, T, H)
    disc = jnp.ones((E, T), jnp.float32)
    _, estim, adv = get_advantages(rewards, done, value, nvalue, disc, 1.0)
    # the GAE variant ppo uses under --advantage-norm none decodes the value
    # head with symexp, so the baseline it subtracts is symexp(value).
    v_raw = np.asarray(inverse_reward_normalization_fn(value))
    want = np.asarray(R)[:, None, :] - v_raw            # A_t = R - V(s_t)
    np.testing.assert_allclose(np.asarray(adv), want, rtol=1e-4, atol=1e-4)
    # ... and the return target is the terminal reward at every step.
    np.testing.assert_allclose(
        np.asarray(estim), np.broadcast_to(np.asarray(R)[:, None, :],
                                           (E, T, H)),
        rtol=1e-4, atol=1e-4)
    # NO decay: |A_0| and |A_{T-1}| are the same order (they differ only by
    # the critic's own prediction, not by any (gamma*lambda)^k factor).
    a = np.asarray(adv)
    assert np.all(np.abs(a[:, 0, :] - want[:, 0, :]) < 1e-3)


def test_campaign_defaults_do_attenuate_the_terminal_signal():
    """Control for the test above: gamma=0.99, lambda=0.95 over T steps
    reproduces the audit's (gamma*lambda)^(T-t) attenuation, so the
    gamma=lambda=1 result is the flags doing the work."""
    E, T, H = 1, 40, 1
    rewards, done, value, nvalue, R = _terminal_only(E, T, H, seed=3)
    value = jnp.zeros_like(value)              # zero critic isolates the decay
    nvalue = jnp.zeros_like(nvalue)
    disc = jnp.full((E, T), 0.99, jnp.float32)
    _, _, adv = get_advantages(rewards, done, value, nvalue, disc, 0.95)
    a = np.asarray(adv)[0, :, 0]
    gl = 0.99 * 0.95
    for t in range(T):
        np.testing.assert_allclose(a[t], float(R[0, 0]) * gl ** (T - 1 - t),
                                   rtol=1e-4, atol=1e-5)
    assert abs(a[0]) < 0.15 * abs(a[-1])       # step 0 gets a sliver


# ---------------------------------------------------------------------------
# --advantage-norm none: PopArt bypassed, and NO other standardisation
# ---------------------------------------------------------------------------

def _train_episode_ast():
    """The AST of ppo.main's inner ``train_episode`` (where the advantage
    dispatch lives)."""
    src = inspect.getsource(ppo)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_episode":
            return node
    raise AssertionError("train_episode not found in ppo.py")


def _adv_dispatch(fn_ast):
    """The ``if use_popart: ... elif advantage_norm == 'none': ... else: ...``
    chain, located by its PopArt test."""
    for node in ast.walk(fn_ast):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "use_popart"):
            return node
    raise AssertionError("the advantage-normalisation dispatch was not found")


def _calls(nodes):
    """Every called name inside `nodes` (a node or a list of statements)."""
    out = []
    for root in (nodes if isinstance(nodes, list) else [nodes]):
        for n in ast.walk(root):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    out.append(f.id)
                elif isinstance(f, ast.Attribute):
                    out.append(f.attr)
    return out


def test_advantage_norm_none_branch_calls_nothing_adaptive():
    disp = _adv_dispatch(_train_episode_ast())
    # branch 2 of the chain is the `none` branch.
    assert len(disp.orelse) == 1 and isinstance(disp.orelse[0], ast.If)
    none_if = disp.orelse[0]
    src_test = ast.dump(none_if.test)
    assert "none" in src_test, "second branch is not the advantage_norm==none one"
    names = _calls(none_if.body)
    for bad in ("_popart_update", "_popart_derive", "_popart_rescale_heads",
                "_lag_basin_freeze", "std", "mean", "normalize",
                "gdpo_normalise_advantages"):
        assert bad not in names, f"{bad} is called under --advantage-norm none"


def test_popart_mutators_only_run_under_use_popart():
    fn = _train_episode_ast()
    disp = _adv_dispatch(fn)
    inside = set()
    for n in ast.walk(disp):
        if n is disp:
            continue
        inside.add(id(n))
    popart_branch = set()
    for st in disp.body:
        for n in ast.walk(st):
            popart_branch.add(id(n))
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                n.func.id in ("_popart_update", "_popart_rescale_heads"):
            assert id(n) in popart_branch, (
                f"{n.func.id} is called outside `if use_popart`")


def test_only_one_advantage_standardisation_exists():
    """The z-score branch is the ONLY place any advantage is divided by a
    batch std; nothing else in ppo.py standardises the advantage."""
    src = inspect.getsource(ppo)
    assert src.count("jnp.std(") == 1, (
        "a second jnp.std appeared in ppo.py -- check it is not a new "
        "advantage standardisation")
    disp = _adv_dispatch(_train_episode_ast())
    zscore = disp.orelse[0].orelse         # the final `else:` = zscore branch
    dumped = "\n".join(ast.dump(s) for s in zscore)
    assert "std" in dumped
    # and the GDPO aggregator (which DOES batch-normalise) is unreachable.
    assert "gdpo_normalise_advantages" not in src


# ---------------------------------------------------------------------------
# --entropy-weight 0 --face-entropy-weight 0: both gradients exactly 0
# ---------------------------------------------------------------------------

def test_entropy_bonus_and_its_gradient_are_exactly_zero():
    h_tot = jnp.asarray(1.234, jnp.float32)
    h_face = jnp.asarray(0.567, jnp.float32)

    def bonus(a, b):
        return _split_entropy_bonus(a, b, 0.0, 0.0)

    val = bonus(h_tot, h_face)
    assert float(val) == 0.0                       # bitwise
    g_tot, g_face = jax.grad(bonus, argnums=(0, 1))(h_tot, h_face)
    assert float(g_tot) == 0.0 and float(g_face) == 0.0
    # the loss composes it as `total_loss = ... - _ent_bonus`, so a zero
    # bonus contributes an exactly zero cotangent to the entropy path.
    def loss(a, b):
        return 3.0 * a + 5.0 * b - bonus(a, b)
    ga, gb = jax.grad(loss, argnums=(0, 1))(h_tot, h_face)
    assert float(ga) == 3.0 and float(gb) == 5.0


def test_entropy_weights_nonzero_do_move_the_gradient():
    """Control: the split really is `w*(H_tot - H_face) + w_face*H_face`."""
    h_tot = jnp.asarray(1.0, jnp.float32)
    h_face = jnp.asarray(0.25, jnp.float32)
    g_tot, g_face = jax.grad(
        lambda a, b: _split_entropy_bonus(a, b, 0.05, 0.005),
        argnums=(0, 1))(h_tot, h_face)
    np.testing.assert_allclose(float(g_tot), 0.05, rtol=1e-6)
    np.testing.assert_allclose(float(g_face), 0.005 - 0.05, rtol=1e-5)


# ---------------------------------------------------------------------------
# --face-entropy-floor 0: hinge bitwise inert
# ---------------------------------------------------------------------------

def test_face_entropy_floor_zero_is_bitwise_inert():
    for h in (0.0, 1e-6, 0.03, 0.3, 5.0):
        hh = jnp.asarray(h, jnp.float32)
        p = _face_entropy_floor_penalty(hh, 0.0, 10.0)
        assert float(p) == 0.0
        assert float(jax.grad(
            lambda x: _face_entropy_floor_penalty(x, 0.0, 10.0))(hh)) == 0.0
    # and the loss-site gate is STATIC (`float(...) > 0.0`), so at 0 the term
    # is not merely zero but absent from the graph.
    src = inspect.getsource(ppo)
    assert 'float(getattr(args, "face_entropy_floor", 0.0)) > 0.0' in src


def test_face_entropy_floor_positive_is_live():
    hh = jnp.asarray(0.01, jnp.float32)
    assert float(_face_entropy_floor_penalty(hh, 0.05, 10.0)) > 0.0
    assert float(jax.grad(
        lambda x: _face_entropy_floor_penalty(x, 0.05, 10.0))(hh)) < 0.0


# ---------------------------------------------------------------------------
# --face-logit-clamp 0: the tanh reparameterisation is OFF at 0
# ---------------------------------------------------------------------------

def test_face_logit_clamp_zero_is_the_off_value():
    key = jrand.PRNGKey(0)
    head = ufh.UnifiedFaceHead(embd_dim=8, key=key)
    ctx = jrand.normal(jrand.PRNGKey(1), (8,)) * 40.0    # deliberately huge
    old = ufh.LOGIT_CLAMP[0]
    try:
        ufh.set_logit_clamp(0.0)
        z_off = np.asarray(head.logits(ctx))
        raw = np.asarray(head.proj(ctx))
        np.testing.assert_array_equal(z_off, raw)       # BITWISE identity
        ufh.set_logit_clamp(15.0)
        z_on = np.asarray(head.logits(ctx))
        assert np.abs(z_on).max() < 15.0
        assert not np.array_equal(z_on, raw)
    finally:
        ufh.set_logit_clamp(old)


# ---------------------------------------------------------------------------
# --lr 0: the frozen-policy control arm (R4) really freezes, and the LR
# schedule does not divide by zero
# ---------------------------------------------------------------------------

def _ppo_optimizer(lr, episodes=250, ppo_epochs=1, minibatches=4,
                   min_mult=0.1, max_grad_norm=0.5):
    """ppo.main's optimiser construction, verbatim in the shape it matters."""
    import optax
    decay_steps = max(1, int(episodes) * int(ppo_epochs) * int(minibatches))
    schedule = optax.cosine_decay_schedule(lr, decay_steps, min_mult)
    return schedule, optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(schedule, b1=0.9, eps=1e-7),
    )


def test_lr_zero_schedule_is_finite_and_zero():
    sched, _ = _ppo_optimizer(0.0)
    for k in (0, 1, 7, 999, 1000, 10_000):
        v = float(sched(k))
        assert np.isfinite(v), f"schedule({k}) = {v}"
        assert v == 0.0
    # control: a live LR is non-zero and decays.
    sched2, _ = _ppo_optimizer(3e-4)
    assert float(sched2(0)) > float(sched2(999)) > 0.0


def test_lr_zero_leaves_every_parameter_bitwise_unchanged():
    rng = np.random.default_rng(11)
    params = {"w": jnp.asarray(rng.normal(size=(8, 5)).astype(np.float32)),
              "b": jnp.asarray(rng.normal(size=(5,)).astype(np.float32))}
    grads = {"w": jnp.asarray(rng.normal(size=(8, 5)).astype(np.float32)),
             "b": jnp.asarray(rng.normal(size=(5,)).astype(np.float32))}
    import optax
    _, opt = _ppo_optimizer(0.0)
    state = opt.init(params)
    cur = params
    for _ in range(3):                       # several steps: adam is stateful
        upd, state = opt.update(grads, state, cur)
        cur = optax.apply_updates(cur, upd)
        for k in params:
            np.testing.assert_array_equal(
                np.asarray(cur[k]), np.asarray(params[k]),
                err_msg=f"--lr 0 moved {k}")
            assert np.all(np.isfinite(np.asarray(upd[k])))
    # control: a live LR does move them.
    _, opt2 = _ppo_optimizer(3e-4)
    st2 = opt2.init(params)
    upd2, _ = opt2.update(grads, st2, params)
    moved = optax.apply_updates(params, upd2)
    assert not np.array_equal(np.asarray(moved["w"]), np.asarray(params["w"]))


def test_face_logit_clamp_module_default_is_zero():
    """az_gumbel and every test see the unclamped head unless ppo sets it.
    Read from the SOURCE (reloading the module would break the identity of
    the class other tests in the session already hold)."""
    src = inspect.getsource(ufh)
    assert "LOGIT_CLAMP = [0.0]" in src
    # and ppo hands the CLI value straight through, 0 included.
    assert ('set_logit_clamp(float(getattr(args, "face_logit_clamp", 0.0)))'
            in inspect.getsource(ppo))
