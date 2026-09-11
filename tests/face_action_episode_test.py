#!/usr/bin/env python3
"""ONE EPISODE, END TO END, PER ``--approx-add`` VALUE.

WHY THIS MODULE EXISTS, stated plainly because it is a lesson and not a feature.

``tests/face_action_record_test.py`` proves the four uses of the face action
record AGREE WITH THE DECLARATION, and ``tests/test_face_head94.py`` proves
``sample`` and ``evaluate`` score the same variable at every width, exactly 0
apart. Both passed -- 39 and 188 tests -- on a branch that could not complete a
single episode, because neither of them asks the only question that matters
next:

    DOES THE REST OF THE ENV ACCEPT WHAT THE DECLARATION PRODUCES?

A record can be internally consistent, round-trip through its own codec, and
still be a shape the measurement path refuses. Nothing between the declaration
and ``env._callback`` was covered: ``StepAction`` -> ``EnvState`` -> the wire
signature -> the compile cache key -> ``_face_dict_for_vertex`` ->
``face_entry_from_slots`` -> the reward vector -> the plan log. This module
covers exactly that, per width, and it is deliberately the CHEAPEST test that
does: a 4-vertex toy, one full episode, the wire drawn from the real head.

WHAT IT ASSERTS, per value:
  1. every step of the episode is accepted -- so the rows the declaration sizes
     are the rows ``wire_slots_of_rows`` wants and the channels ``EnvState``
     keeps;
  2. the terminal REWARD VECTOR IS FINITE in every slot -- a NaN here is a
     measurement that ran on a plan nobody can score;
  3. a PLAN LOG RECORD is written, it names the value that measured it, and
     under ``choose`` it carries the per-face join bits while under every fixed
     value it carries none.

WHAT IT DELIBERATELY DOES NOT EXERCISE, and why (labelled, not hidden): the
QUALITY channel is off (``ALPHAGRAD_QUALITY_METRIC=none``). In this environment
``env._gradient_similarity`` imports ``graphax.sparse.ops.output_layout``, which
exists in NO graphax commit of ``core-v2``, so every terminal measurement that
reaches the cosine channel raises ``ModuleNotFoundError`` inside the host
callback -- on the BASE as well as here (measured: ``tests/plan_log_test.py`` is
5 failed and ``tests/reward_slot7_reserved_test.py`` 3 failed at ``9f5f643a``
too). Leaving the channel on would make this module fail for a reason that has
nothing to do with the face wire, which is the opposite of what it is for.
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx import face_action as REC                 # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    MAX_FACES, MAX_RULES_PER_VERTEX, StepAction,
    VertexEliminationEnv, wire_slots,
)

MODES = list(REC.ALL_MODES)

_M = jnp.array([[1.0, 2.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 4.0],
                [5.0, 0.0, 0.0, 1.0]])
_P = jnp.eye(4) * 2.0
_Q = jnp.eye(4) * 3.0
_X4 = jnp.linspace(0.1, 0.9, 4)


def _square(x):
    """A 4-vertex toy with a real face: two consumers of one intermediate."""
    e = _M @ x
    return _P @ e, _Q @ e


def _env():
    closed = jax.make_jaxpr(_square)(_X4)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square)


def _wire_at(mode, *, join_bit=None):
    """One vertex's face wire AT THIS WIDTH, sized from the declaration.

    The rows are all-exact END rows on purpose: what is under test is that the
    env accepts the SHAPE and the CHANNELS the declaration produces, at every
    width, not that a particular approximation applies. A value-specific rule
    would make the test about the engine.
    """
    S = REC.n_slots(mode)
    assert S == wire_slots(), (S, wire_slots())
    rows = jnp.full((MAX_FACES, S, 3), -1, jnp.int32).at[..., 2].set(0)
    skip = jnp.zeros((MAX_FACES,), jnp.int32)
    join = (None if join_bit is None
            else jnp.full((MAX_FACES,), int(join_bit), jnp.int32))
    return rows, skip, join


def _run_episode(mode, *, join_bit=None):
    """reset -> step every vertex -> (reward vector, plan records)."""
    env = _env()
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32).at[..., 2].set(0)
    rows, skip, join = _wire_at(mode, join_bit=join_bit)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules, rows, skip, join),
        ).state
    return np.asarray(state.reward), envmod.consume_plan_records()["records"]


@pytest.fixture(autouse=True)
def _quiet_quality(monkeypatch):
    # See the module docstring: the cosine channel is broken in this
    # environment for a reason that predates this wire.
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    envmod.consume_plan_records()
    yield
    envmod.consume_plan_records()


@pytest.mark.parametrize("mode", MODES)
def test_one_episode_completes_and_scores_at_every_width(monkeypatch, mode):
    """THE TEST THAT WAS MISSING. A record the env refuses fails HERE."""
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    # `choose` decides the join PER FACE, so the wire MUST carry the bit --
    # `resolve_join_mode` raises rather than defaulting, which is the point.
    bit = 1 if mode == "choose" else None
    reward, recs = _run_episode(mode, join_bit=bit)
    assert np.all(np.isfinite(reward)), (mode, reward.tolist())
    assert len(recs) == 1, (mode, [r.get("order") for r in recs])
    assert recs[0]["approx_add"] == mode, recs[0].get("approx_add")


@pytest.mark.parametrize("mode", MODES)
def test_a_SAMPLED_record_survives_the_whole_chain(monkeypatch, mode):
    """The full chain in ONE test: head -> record -> StepAction -> env -> reward.

    The test above builds the wire itself, which checks the SHAPE the
    declaration sizes. This one draws the record from the REAL head through
    ``UnifiedFacePolicy.sample`` and pushes it through the REAL
    ``Agent.to_env_action_dynamic`` before the env sees it, so the translator
    call, the per-face channel forwarding and the env's acceptance are one
    statement rather than two hops joined by a test fixture.

    ``to_env_action_dynamic`` uses no ``self``, so it is called unbound and this
    costs no Agent. The drawn rows may well be illegal on a 4-vertex toy --
    ``make_live_masked_hook`` skips a rule its own operand refuses, per slot,
    which is the designed behaviour and is not what is under test here.
    """
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, MicroAction, OP_END,
        precompute_factor_tables)
    from alphagrad.approx.ppo import Agent
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

    env = _env()
    ax = env.axis_state_static
    n_ax = int(ax.shape[1])
    sz = jnp.full((n_ax,), 4, jnp.int32)
    feats = AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((n_ax, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n_ax,), jnp.int32),
        valid_mask=jnp.ones((n_ax,), jnp.float32))
    pol = UnifiedFacePolicy(32, num_heads=2, max_faces=MAX_FACES,
                            key=jrand.PRNGKey(0), approx_add=mode)
    fa, *_ = pol.sample(
        None, feats, precompute_factor_tables(64), jrand.PRNGKey(5),
        jnp.ones((MAX_FACES, n_ax, n_ax), jnp.float32),
        jnp.ones((MAX_FACES, n_ax), jnp.float32),
        jnp.ones((MAX_FACES,), jnp.float32))
    REC.check(fa, mode, MAX_FACES, where="sampled for the episode")

    zero = jnp.zeros((1,), jnp.int32)
    micro = MicroAction(
        op_type=jnp.full((1,), OP_END, jnp.int32), i=zero, j=zero,
        exponents=jnp.zeros((1, 9), jnp.int32), factor=zero,
        compress_kind=zero, quant_dtype=zero,
        quant_scale_sign=jnp.ones((1,), jnp.int32),
        quant_scale_frac=jnp.zeros((1,), jnp.float32))

    state = env.reset()
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        sa = Agent.to_env_action_dynamic(None, v - 1, micro, ax, face_action=fa)
        assert tuple(sa.face_rows.shape) == (MAX_FACES, REC.n_slots(mode), 3)
        assert (sa.face_join is None) != ("join" in REC.names(mode))
        state = env.step(state, sa._replace(
            target_vertex=jnp.asarray(v, jnp.int32))).state
    reward = np.asarray(state.reward)
    recs = envmod.consume_plan_records()["records"]
    assert np.all(np.isfinite(reward)), (mode, reward.tolist())
    assert len(recs) == 1 and recs[0]["approx_add"] == mode


@pytest.mark.parametrize("mode", MODES)
def test_the_join_channel_exists_exactly_where_the_width_says(monkeypatch, mode):
    """The per-face channel rides the wire iff the head has the bit.

    Both directions, because both are silent: a missing bit under ``choose``
    would measure every merge under the configuration's default, and a bit under
    a fixed value would let a wire override the flag.
    """
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    has_bit = "join" in REC.names(mode)
    assert has_bit == (mode == "choose")

    reward, recs = _run_episode(mode, join_bit=1 if has_bit else None)
    assert np.all(np.isfinite(reward)), (mode, reward.tolist())
    # The plan log's column is present iff the decision is per-face.
    assert (recs[0].get("face_joins") is None) != has_bit, recs[0].get(
        "face_joins")

    # ... and the WRONG channel state raises rather than being defaulted.
    with pytest.raises(Exception):
        _run_episode(mode, join_bit=None if has_bit else 1)


def test_a_FIXED_value_carries_NO_EXTRA_LEAF_and_no_extra_positional(monkeypatch):
    """Two ways the join channel could have cost a flag-off run something.

    (1) THE PYTREE. `EnvState.face_joins` is `None` under every value that FIXES
    the join semantics, and `None` is a leaf-FREE pytree node -- but only if
    nothing re-materialises it as a zeros array on the way through. Checked by
    LEAF COUNT through the whole carry: reset, every step, and the shifted
    prefix `_replace` the terminal path builds. `choose` is the control: it MUST
    add exactly one leaf, or the channel is not being carried at all.

    (2) THE SIGNATURE. `env._callback` is called directly, POSITIONALLY, from
    about a dozen places -- six test modules plus the measure actors,
    landscape_map and az_gumbel. Inserting `face_joins` as a positional
    parameter between `face_skips` and `stop` shifted `stop` into it for every
    one of them, and turned `tests/delta_obs_emission_test.py` from 8 passed to
    4 failed. So `face_joins` is KEYWORD-ONLY, and this pins that: the eighth
    positional parameter must still be `stop`.
    """
    import inspect
    sig = inspect.signature(envmod._callback)
    pos = [n for n, prm in sig.parameters.items()
           if prm.kind is prm.POSITIONAL_OR_KEYWORD]
    assert pos[:8] == ["config", "args", "consts", "order", "sparsity_specs",
                       "face_specs", "face_skips", "stop"], pos
    assert sig.parameters["face_joins"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["face_joins"].default is None

    counts = {}
    for mode in ("lossless", "choose"):
        monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
        monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
        env = _env()
        state = env.reset()
        n0 = len(jax.tree_util.tree_leaves(state))
        no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1,
                            jnp.int32).at[..., 2].set(0)
        rows, skip, join = _wire_at(mode, join_bit=1 if mode == "choose" else None)
        assert (state.face_joins is None) == (mode != "choose")
        for v in [int(x) for x in np.asarray(env.valid_vertices)]:
            state = env.step(state, StepAction(
                jnp.asarray(v, jnp.int32), no_rules, rows, skip, join)).state
            assert len(jax.tree_util.tree_leaves(state)) == n0, (
                mode, "the step changed the carry's leaf count")
            assert (state.face_joins is None) == (mode != "choose")
        counts[mode] = n0
        envmod.consume_plan_records()
    assert counts["choose"] == counts["lossless"] + 1, (
        "the join channel must cost EXACTLY one leaf under `choose` and none "
        f"under a fixed value; got {counts}")


@pytest.mark.parametrize("mode", MODES)
def test_a_row_array_at_the_WRONG_width_is_refused_by_the_env(monkeypatch, mode):
    """The declaration's width and the env's must be ONE number.

    This is the half `face_action_record_test` cannot see: it checks the record
    against the declaration, and this checks the declaration against the env.
    """
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    S = REC.n_slots(mode)
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32).at[..., 2].set(0)
    for bad in (S - 1, S + 1):
        if bad < 1:
            continue
        env = _env()
        state = env.reset()
        rows = jnp.full((MAX_FACES, bad, 3), -1, jnp.int32).at[..., 2].set(0)
        # A DIAG on slot 0 so the face carries a live row: the width check runs
        # where the wire becomes a graphax entry, which needs something to
        # install.
        rows = rows.at[0, 0].set(jnp.asarray([1, 0, 2], jnp.int32))
        skip = jnp.zeros((MAX_FACES,), jnp.int32)
        join = (jnp.zeros((MAX_FACES,), jnp.int32)
                if "join" in REC.names(mode) else None)
        with pytest.raises(Exception):
            for v in [int(x) for x in np.asarray(env.valid_vertices)]:
                state = env.step(
                    state,
                    StepAction(jnp.asarray(v, jnp.int32), no_rules, rows,
                               skip, join)).state


def test_loss_drop_on_a_NON_SCALAR_target_is_refused_AT_SETUP_by_ppo():
    """``--quality-metric loss_drop`` needs a scalar loss, and ppo says so EARLY.

    The walk's first step is ``float(target_fun(...))``, so an analytic AD
    benchmark -- whose target is a full Jacobian -- used to reach it with a
    rank-1 output and die as ``TypeError: Only scalar arrays can be converted to
    Python scalars`` five frames inside a host callback at the first terminal
    measurement, naming neither the flag nor the example. That is how a launcher
    carrying the pair looked like a face-wire regression on every
    ``--approx-add`` value at once (it failed identically on the base).

    THE CHECK IS IN ppo, AND TWO EARLIER PLACEMENTS IN env WERE WRONG:
    ``quality_metric()`` is a pure name -> metric map that
    ``tests/quality_metric_names_test.py`` deliberately pins for BOTH target
    kinds; and ``_loss_drop_quality`` answers "the walk is undefined for this
    env" with ``None`` plus a loud warning, so raising there broke seven
    ``walk_heldout_test`` cases that hand it a stub config. ``env`` also cannot
    answer the question reliably -- ``EnvConfig.scalar_target`` is False for
    plenty of genuinely scalar targets, because it is only set when
    ``measure_grad`` / ``scalar_target`` is passed. The fact lives in
    ``ppo._has_scalar_loss(args.example)``.
    """
    import inspect
    from alphagrad.approx import ppo as _ppo
    src = inspect.getsource(_ppo.main)
    assert "needs a SCALAR-LOSS target" in src, (
        "ppo.main no longer refuses --quality-metric loss_drop on a "
        "non-scalar-loss example, so the run dies at the first terminal "
        "measurement with a TypeError instead")
    # env is UNTOUCHED on both counts: the name resolver still answers for both
    # target kinds, and the walk still answers "undefined" with None.
    for kind in (True, False):
        cfg = type("C", (), {"scalar_target": kind})()
        old = os.environ.get("ALPHAGRAD_QUALITY_METRIC")
        os.environ["ALPHAGRAD_QUALITY_METRIC"] = "loss_drop"
        try:
            assert envmod.quality_metric(cfg) == "loss_drop"
        finally:
            if old is None:
                os.environ.pop("ALPHAGRAD_QUALITY_METRIC", None)
            else:
                os.environ["ALPHAGRAD_QUALITY_METRIC"] = old
    assert "SCALAR-LOSS" not in inspect.getsource(envmod._loss_drop_quality)


def test_no_scalar_conversion_is_applied_to_a_WHOLE_record_field():
    """``float()`` / ``int()`` / ``.item()`` on an UNINDEXED record field would be
    a RANK ASSUMPTION, and the slot axis is width-dependent now.

    The coordinator's review of 2026-09-11 asked for this class of error to be
    swept, not just its one instance fixed. A per-slot field is ``(F, S)`` and
    ``S`` moves with ``--approx-add``, so a scalar conversion applied to the
    whole field is a bug that only surfaces at a width nobody ran. Indexing one
    first (``int(face_skip[f])``, ``int(fa.op_type[f, s])``) is legitimate and is
    not flagged.

    SCOPED TO THE MODULES THAT OWN THE RECORD, on purpose: ``MicroAction`` shares
    seven of these field names (``op_type``, ``factor``, ``exponents``,
    ``compress_kind``, ``quant_dtype``, the two quant-scale fields) and its
    fields are per-SUBSTEP scalars, so scalar-converting one is correct there. A
    name-based scan over the whole package would report those as offenders and
    would have to be silenced, which is how a guard stops being read.
    """
    import re
    root = pathlib.Path(envmod.__file__).resolve().parent
    owners = ["env.py", "ppo.py", "face_action.py", "unified_face_policy.py",
              "unified_face_head.py", "az_gumbel.py",
              "common/face_buckets.py"]
    names = "|".join(f.name for f in REC.FACE_ACTION_FIELDS)
    # `float(<expr>.<name>)` / `int(...)` / `<expr>.<name>.item()`, where the
    # name is NOT followed by a subscript. `face_<name>` and `face_<name>s`
    # (the carrier and env-history dialects) count too.
    pat = re.compile(
        r"(?:float|int)\(\s*[A-Za-z_][A-Za-z_0-9.]*\.(?:face_)?(?:" + names +
        r")s?\s*\)"
        r"|(?:face_)?(?:" + names + r")s?\.item\(\)")
    offenders = []
    for rel in owners:
        f = root / rel
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "a scalar conversion is applied to a WHOLE action-record field; the "
        "slot axis is width-dependent, so this is a rank assumption that only "
        "breaks at a width nobody ran:\n" + "\n".join(offenders))


def test_the_record_field_ranks_are_what_the_declaration_says(monkeypatch):
    """Per-slot fields are rank 2 (+tail), per-face rank 1, at EVERY width.

    The rank the rest of the env is entitled to assume, stated once, from the
    declaration. ``float()`` on any of the rank-1-or-more fields is the defect
    the test above forbids; this one pins the ranks it is forbidding against.
    """
    for mode in MODES:
        monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
        z = REC.zeros(mode, MAX_FACES)
        for fld in REC.fields(mode):
            got = np.ndim(getattr(z, fld.name))
            want = (2 if fld.per_slot else 1) + len(fld.tail)
            assert got == want, (mode, fld.name, got, want)
