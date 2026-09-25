"""THE GRADIENT-QUALITY CHANNEL ON THE RECURRENT TARGET (dsnn-dfw.52, .51).

Two defects, both measured on the order-only RSNN_SHD arms of 2026-09-19:

dsnn-dfw.52  The cosine's EXACT REFERENCE was cached on a probe seed that did
             not carry the environment row, while the probe BATCH was drawn on
             one that did. Every environment row after the first was therefore
             scored against row 0's exact gradient -- a gradient at another
             step of the same recording, which is an unrelated vector. Job
             66642 read q med +0.0197 over [-0.2417, +1]; probe 66655
             reproduced it exactly (row 0 = 1.0, rows 1 to 3 = -0.240, -0.174,
             -0.221) and showed the elimination itself is exact under every
             free order (22 orders, float64, rel <= 9.3e-16).

dsnn-dfw.51  An UNDEFINED cosine -- the exact gradient identically zero on
             every probe batch, or a reference that would not build -- was
             scored 0.0 and the run continued. Under a quality floor that is a
             full constraint violation handed to a plan that did nothing
             wrong. It is a REFUSED measurement: missing data, counted,
             excluded from the update.

Free-order exactness is pinned here too, in float64 against ``jax.grad``, for
all three temporal rules: it is the fact that says .52 lives in the reference
and not in the elimination, and it is the test that would catch a real
elimination defect on this target.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common import examples as ex              # noqa: E402
from alphagrad.approx.common.carry_plan import (                # noqa: E402
    _traced_inlined, valid_vertices)
from alphagrad.approx.common.rsnn_shd import (                 # noqa: E402
    loss_of, rsnn_data_gen)
from graphax import jacve                                       # noqa: E402

EXAMPLE = "RSNN_SHD"
STEP = 40
N_FREE_ORDERS = 50


# ---------------------------------------------------------------------------
# the target, built the way the trainer builds it
# ---------------------------------------------------------------------------
def _build(rule, step_position=STEP):
    fn = ex.get_fn(EXAMPLE)
    xs = ex.get_args(EXAMPLE, jax.random.PRNGKey(7), dataset=None,
                     temporal_rule=rule, step_position=step_position)
    fn, xs, argnums = ex.grad_target_setup({}, fn, xs, EXAMPLE)
    cj = _traced_inlined(fn, tuple(xs))
    consts = list(cj.literals)
    valid = valid_vertices(cj.jaxpr, tuple(xs), consts, tuple(argnums))
    return fn, tuple(xs), tuple(argnums), cj, consts, tuple(valid)


def _eliminate(fn, order, argnums, cj, consts, xs):
    """THE MEASUREMENT PATH: the ``jacve`` call ``env._jacve_fn(approx=True)``
    builds for a plan with no rule and no face action."""
    gv = jacve(fn, [int(v) for v in order], argnums=tuple(argnums),
               has_aux=False, sparse_representation=False,
               transforms=[], face_transforms=None,
               jaxpr=cj.jaxpr, consts=list(consts))
    return jax.jit(gv, keep_unused=True)(*xs)


def _flat(leaves):
    return np.concatenate([np.asarray(x, np.float64).reshape(-1)
                           for x in leaves])


def _cos_rel(a_leaves, e_leaves):
    a, e = _flat(a_leaves), _flat(e_leaves)
    ne = float(np.linalg.norm(e))
    assert ne > 0.0, "the reference gradient is zero; this step is degenerate"
    return (float(np.dot(a, e) / (np.linalg.norm(a) * ne)),
            float(np.linalg.norm(a - e) / ne))


def _free_orders(valid, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = np.array(valid, dtype=np.int32).copy()
        rng.shuffle(p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# 1. every free order reproduces jax.grad, in float64, on all three rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["rtrl", "tbptt", "bptt"])
def test_every_free_order_reproduces_jax_grad_in_float64(rule):
    """50 random free orders through the measurement path against jax.grad.

    The bar is the oracle's: in float64 on the CPU an exact elimination of this
    target sits at 1e-16, so 1e-9 is four orders of magnitude of room and there
    is no noise band to hide in.
    """
    with envmod._x64_scope():
        fn, xs, argnums, cj, consts, valid = _build(rule)
        xs = tuple(jnp.asarray(x, jnp.float64)
                   if hasattr(x, "dtype") and jnp.issubdtype(x.dtype,
                                                             jnp.floating)
                   else x for x in xs)
        cj = _traced_inlined(fn, xs)
        consts = list(cj.literals)
        # The rtrl target returns (loss, *state); the reference is the loss.
        ref = list(jax.grad(lambda *a: loss_of(fn(*a)),
                            argnums=argnums)(*xs))
        worst_cos, worst_rel = 1.0, 0.0
        for o in _free_orders(valid, N_FREE_ORDERS, 20260919):
            out = _eliminate(fn, o, argnums, cj, consts, xs)
            rows = out[0] if len(cj.jaxpr.outvars) > 1 else out
            cos, rel = _cos_rel(list(rows), ref)
            worst_cos = min(worst_cos, cos)
            worst_rel = max(worst_rel, rel)
    assert worst_cos > 1.0 - 1e-9, (
        f"{rule}: a free order's gradient is not the reference gradient "
        f"(worst cosine {worst_cos!r})")
    assert worst_rel < 1e-9, (
        f"{rule}: a free order's gradient differs from the reference by "
        f"{worst_rel!r} relative")


# ---------------------------------------------------------------------------
# 2. dsnn-dfw.52: the cosine reference belongs to the environment row
# ---------------------------------------------------------------------------
@pytest.fixture
def _clean_probe_caches():
    envmod._PROBE_BATCH.clear()
    envmod._COSINE_REF.clear()
    envmod._ENV_SLOT[0] = -1
    yield
    envmod._PROBE_BATCH.clear()
    envmod._COSINE_REF.clear()
    envmod._ENV_SLOT[0] = -1


def _quality_per_env_row(rows=4):
    fn, xs, argnums, cj, consts, valid = _build("rtrl")
    gen = rsnn_data_gen(jax.random.PRNGKey(7), dataset=None,
                        temporal_rule="rtrl")
    cfg = envmod.EnvConfig(
        jaxpr=cj.jaxpr, argnums=tuple(argnums), has_aux=False, sparse=False,
        cmp_type="latency", mem_type="peak_memory", target_fun=fn,
        data_gen=gen, scalar_target=True,
        carried_outputs=len(cj.jaxpr.outvars) - 1)
    dev = jax.devices("cpu")[0]
    rev = sorted((int(v) for v in valid), reverse=True)
    ref_ex = jax.jit(
        jacve(fn, rev, argnums=tuple(argnums), has_aux=False,
              sparse_representation=False, transforms=[], face_transforms=None,
              jaxpr=cj.jaxpr, consts=list(consts)), keep_unused=True)
    cand = _free_orders(valid, 1, 4242)[0]
    cand_ex = jax.jit(
        jacve(fn, [int(v) for v in cand], argnums=tuple(argnums),
              has_aux=False, sparse_representation=False, transforms=[],
              face_transforms=None, jaxpr=cj.jaxpr, consts=list(consts)),
        keep_unused=True)
    out = {}
    for row in range(rows):
        envmod._ENV_SLOT[0] = row
        try:
            q = envmod._grad_cosine_quality(cfg, cand_ex, ref_ex, b"probe-ref",
                                            list(xs), dev, 1)
            t = int(envmod.probe_meta()["t"])
        finally:
            envmod._ENV_SLOT[0] = -1
        out[row] = (None if q is None else float(q[0]), t)
    return out


def test_every_environment_row_is_scored_against_its_own_exact_gradient(
        _clean_probe_caches):
    """dsnn-dfw.52. The rows draw different step positions, so a reference
    that is not keyed on the row is a gradient from somebody else's step."""
    seen = _quality_per_env_row(4)
    assert len({t for _, t in seen.values()}) > 1, (
        f"the four rows drew one step position, so this test could not "
        f"detect the defect it exists for: {seen}")
    for row, (q, t) in seen.items():
        assert q is not None, f"row {row} (step {t}) was refused"
        assert q > 1.0 - 1e-5, (
            f"row {row} measured step {t} and scored {q}: an exact plan was "
            f"scored against another row's exact gradient")


def test_the_probe_seed_is_one_function(_clean_probe_caches):
    """The batch and its reference must be keyed on the SAME number."""
    gen = rsnn_data_gen(jax.random.PRNGKey(7), dataset=None,
                        temporal_rule="rtrl")

    class _Cfg:
        data_gen = gen

    seeds = set()
    for row in range(4):
        envmod._ENV_SLOT[0] = row
        try:
            seeds.add(envmod._probe_seed(_Cfg, "train", None, 0))
        finally:
            envmod._ENV_SLOT[0] = -1
    assert len(seeds) == 4, f"the seed does not carry the row: {seeds}"


def test_a_generator_that_does_not_redraw_keeps_its_old_seed():
    """Flag-off bit-identity: every image and token generator is unmoved."""
    class _Cfg:
        data_gen = (lambda keys: ())

    for row in range(3):
        envmod._ENV_SLOT[0] = row
        try:
            assert (envmod._probe_seed(_Cfg, "train", None, 2)
                    == envmod._walk_seed("train", None) + 104729 * 2)
        finally:
            envmod._ENV_SLOT[0] = -1


# ---------------------------------------------------------------------------
# 3. dsnn-dfw.51: an undefined cosine is a refusal, not a 0.0
# ---------------------------------------------------------------------------
_N = 32
_rng = np.random.default_rng(0)
_W1 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_X = jnp.asarray(np.linspace(-1.0, 1.0, _N, dtype=np.float32))


def _toy(x):
    return jnp.sum(jnp.tanh(_W1 @ x) ** 2)


def _toy_gen(keys):
    return (jnp.asarray(_X),)


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv
    closed = jax.make_jaxpr(_toy)(_X)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=_toy,
        data_gen=_toy_gen, measure_latency=False,
        num_data_points=1, reps_per_point=1)


def _env_without_a_data_gen():
    """The same target with NO generator, so no probe batch can be drawn."""
    from alphagrad.approx.env import VertexEliminationEnv
    closed = jax.make_jaxpr(_toy)(_X)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=_toy,
        measure_latency=False, num_data_points=1, reps_per_point=1)


def _run_terminal(env):
    from alphagrad.approx.env import (FACE_SLOTS, MAX_FACES,
                                      MAX_RULES_PER_VERTEX, StepAction)
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for v in sorted((int(x) for x in np.asarray(env.valid_vertices)),
                    reverse=True):
        state = env.step(state, StepAction(
            jnp.asarray(v, jnp.int32), no_rules,
            jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32),
            jnp.zeros((MAX_FACES,), jnp.int32))).state
    return np.asarray(state.reward)


def test_an_undefined_cosine_is_a_refused_measurement(monkeypatch):
    """dsnn-dfw.51. It used to append 0.0 to the cosines and carry on."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setattr(envmod, "_grad_cosine_quality",
                        lambda *a, **k: None)
    envmod._WALK_UNDEFINED_WARNED.clear()
    envmod.consume_refused_counts()
    envmod.consume_plan_records()
    try:
        reward = _run_terminal(_toy_env())
        counts = envmod.consume_refused_counts()
        records = envmod.consume_plan_records()["records"]
    finally:
        envmod.consume_refused_counts()
        envmod.consume_plan_records()
        envmod._WALK_UNDEFINED_WARNED.clear()
    assert counts.get("quality-undefined") == 1, (
        f"an undefined cosine was not counted as a refusal: {counts}")
    assert counts.get("total") == 1
    assert records and records[-1].get("refused", "").startswith(
        "quality-undefined"), (
        f"the plan record does not say the measurement was refused: "
        f"{records[-1] if records else None}")
    cost = np.asarray(reward)[list(envmod.COMPUTE_REWARD_INDICES)]
    assert bool(np.all(cost <= envmod.SENTINEL_COST * 0.99)), (
        f"a refused measurement must carry the sentinel so the trainer drops "
        f"the environment, got {cost}")


def test_a_defined_cosine_is_still_a_score(monkeypatch):
    """The refusal must not swallow a measurement that HAS a cosine."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    monkeypatch.setattr(envmod, "_grad_cosine_quality",
                        lambda *a, **k: (0.75, [0.1], [0.75]))
    envmod.consume_refused_counts()
    try:
        reward = _run_terminal(_toy_env())
        counts = envmod.consume_refused_counts()
    finally:
        envmod.consume_refused_counts()
    assert not counts, f"a scored measurement was counted as refused: {counts}"
    assert float(reward[envmod.REWARD_INDEX["quality"]]) == pytest.approx(
        0.75, abs=1e-6)


def test_a_configuration_with_no_quality_channel_is_not_a_refusal(monkeypatch):
    """NO CHANNEL IS NOT MISSING DATA. A target with no data generator has no
    probe batch and therefore no gradient cosine to measure, for every plan of
    every episode -- refusing there refuses the whole run and it measures
    nothing. The refusal of dsnn-dfw.51 is for a probe batch that WAS drawn
    and whose exact gradient came back identically zero.

    Measured 2026-09-19, job 66664: the refusal reached the structural case
    and seven tests in three modules failed on it -- every terminal of the
    plan-log and reserved-slot environments carried the sentinel reward and
    the record said `refused: quality-undefined:grad_cosine`."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    envmod._WALK_UNDEFINED_WARNED.clear()
    envmod.consume_refused_counts()
    try:
        reward = _run_terminal(_env_without_a_data_gen())
        counts = envmod.consume_refused_counts()
    finally:
        envmod.consume_refused_counts()
        envmod._WALK_UNDEFINED_WARNED.clear()
    assert not counts, (
        f"a configuration with no quality channel at all was counted as a "
        f"refused measurement: {counts}")
    assert float(reward[envmod.REWARD_INDEX["quality"]]) == 0.0, (
        f"the quality slot of a target with no channel must read 0.0, got "
        f"{float(reward[envmod.REWARD_INDEX['quality']])}")


# ---------------------------------------------------------------------------
# 4. the probe width the recurrent target asks for
# ---------------------------------------------------------------------------
def test_the_recurrent_generator_asks_for_five_probe_batches(monkeypatch):
    """dsnn-dfw.51. 43 of the 99 legal step positions of the recording the
    campaign drew are silent, so one batch refuses 43 percent of every
    measurement; five takes that to 1.5 percent."""
    monkeypatch.delenv("ALPHAGRAD_GRAD_COSINE_K", raising=False)
    gen = rsnn_data_gen(jax.random.PRNGKey(7), dataset=None,
                        temporal_rule="rtrl")

    class _Cfg:
        data_gen = gen

    assert int(getattr(gen, "probe_batches")) == 5
    assert envmod._grad_cosine_k(_Cfg) == 5
    assert envmod._grad_cosine_k() == 1
    monkeypatch.setenv("ALPHAGRAD_GRAD_COSINE_K", "2")
    assert envmod._grad_cosine_k(_Cfg) == 2, "the env var must still override"


def test_the_refusal_count_is_on_the_health_line():
    """A refused terminal is dropped from the update, so the rate has to be
    readable from stdout beside the losses it silently changes."""
    import inspect

    import alphagrad.approx.ppo as ppomod
    src = inspect.getsource(ppomod)
    assert "refused=%s" in src and '_hk("refused/total"' in src


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "dsnn-dfw.55 is still true at f825993: an undefined loss-drop walk "
    "appends 0.0 to the quality slot and is not refused"))
def test_an_undefined_loss_drop_walk_is_a_refused_measurement(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "loss_drop")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setattr(envmod, "_loss_drop_quality", lambda *a, **k: None)
    envmod._WALK_UNDEFINED_WARNED.clear()
    envmod.consume_refused_counts()
    envmod.consume_plan_records()
    try:
        reward = _run_terminal(_toy_env())
        counts = envmod.consume_refused_counts()
        records = envmod.consume_plan_records()["records"]
    finally:
        envmod.consume_refused_counts()
        envmod.consume_plan_records()
        envmod._WALK_UNDEFINED_WARNED.clear()
    quality = float(reward[envmod.REWARD_INDEX["quality"]])
    refused = records[-1].get("refused") if records else None
    assert counts.get("quality-undefined") == 1, (
        f"an undefined loss-drop walk was not refused: refusal counts "
        f"{counts}, plan record refused={refused!r}, quality slot {quality}")
