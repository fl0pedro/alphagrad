"""Reward slot 7 is RESERVED (owner ruling 2026-09-03, ticket dsnn-3qm.15).

Gradient coverage -- the per-leaf census, the frozen-gradient HARD GUARD
(``--reject-frozen-grads``), the reward channel (``--grad-coverage-weight``)
and the optional fourth value head -- was removed from the env, the trainer,
the launcher generator and the plan log. What is pinned here:

1. THE 12-SLOT LAYOUT DID NOT MOVE. Both channel tables are identical; slot 7
   keeps its NAME (``grad_coverage``) and its ``frob_residual`` alias so
   archived plan logs and the index-keyed reward_scaling mirror keep their
   indices -- the convention slot 9 (``bkstep_acc``) already uses.
2. SLOT 7 IS NEVER POPULATED: a real measured plan emits exactly 0.0 there at
   every step, terminal included; no head, weight or symlog exemption ever
   addresses it.
3. THE MECHANISM IS GONE: no env function, no ppo ``configure_*``, no CLI
   flag, no launcher flag, and the plan-log record carries no census.
4. SPARSITY NEEDS NO GUARD: a non-zero ``--sparsity-weight`` is accepted.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.pop("ALPHAGRAD_PLAN_LOG", None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.common import reward_scaling as rs        # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)

SLOT7 = 7
LAYOUT = (
    "muls_adds_fmas", "flops", "latency_ns", "max_io_sum", "bytes_accessed",
    "peak_memory", "quality", "grad_coverage", "fidelity", "bkstep_acc",
    "sparsity", "mem_objective",
)
_HERE = pathlib.Path(__file__).resolve().parent
_ALPHAGRAD = _HERE.parent


# ------------------------------------------------------------- 1. the layout
def test_twelve_slot_layout_is_pinned_in_both_tables():
    assert NUM_REWARDS == 12
    assert REWARD_NAMES == LAYOUT
    assert rs.REWARD_NAMES == LAYOUT
    for name in LAYOUT:
        assert REWARD_INDEX[name] == rs.REWARD_INDEX[name] == LAYOUT.index(name)


def test_slot7_keeps_its_name_and_aliases():
    assert REWARD_NAMES[SLOT7] == "grad_coverage"
    assert REWARD_INDEX["frob_residual"] == SLOT7
    assert rs.REWARD_INDEX["frob_residual"] == SLOT7
    assert rs.GRAD_COVERAGE_IDX == rs.FROB_RESIDUAL_IDX == SLOT7
    # ...and slot 9 is the precedent for a reserved, never-populated slot.
    assert REWARD_NAMES[9] == "bkstep_acc"


# ------------------------------------------------ 2. never populated by env
_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _square(x):
    e = _M @ x
    return _P @ e, _Q @ e


def _make_env():
    closed = jax.make_jaxpr(_square)(_X4)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square,
    )


def _run_episode(env, skip_face_of_vertex=None):
    """Every step's reward vector, terminal last."""
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    out = []
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
        if skip_face_of_vertex is not None and v == skip_face_of_vertex:
            face_skip = face_skip.at[0].set(1)
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
        out.append(np.asarray(state.reward))
    return out


@pytest.mark.parametrize("skip", [None, 1])
def test_slot7_is_exactly_zero_on_every_step_of_a_real_measurement(skip):
    """An exact plan and a skipped-face plan (the class the guard used to
    refuse) both emit a literal 0.0 in slot 7 at every step."""
    envmod.consume_mem_parity()
    rewards = _run_episode(_make_env(), skip_face_of_vertex=skip)
    assert len(rewards) >= 1
    for r in rewards:
        assert r.shape == (NUM_REWARDS,)
        assert float(r[SLOT7]) == 0.0
    # ...and the terminal step measured SOMETHING, so this is not a zero vector.
    # Read off the measurement drain, not off the slots: since ticket .49 slot
    # 5 is the static temp, which is exactly 0 for this toy's skipped plan
    # (no temporaries survive), and the other cost slots are 0 by config.
    _mp = envmod.consume_mem_parity()
    assert _mp["measured"] == len(rewards)
    assert any(r["terminal"] for r in _mp["records"])


def test_sentinel_vector_keeps_its_shape():
    s = np.asarray(envmod._SENTINEL_BAD_REWARD)
    assert s.shape == (NUM_REWARDS,)
    assert s[SLOT7] == -1.0     # kept byte-identical to every archived sentinel


def test_plan_log_record_carries_no_census_and_names_slot7():
    os.environ["ALPHAGRAD_PLAN_LOG"] = "1"
    try:
        envmod.consume_plan_records()
        _run_episode(_make_env(), skip_face_of_vertex=1)
        recs = envmod.consume_plan_records()["records"]
        assert len(recs) == 1
        rec = recs[0]
        assert "coverage" not in rec
        assert "sentinelled" not in rec
        assert rec["reward_names"][SLOT7] == "grad_coverage"
        assert rec["rewards"][SLOT7] == 0.0
    finally:
        os.environ["ALPHAGRAD_PLAN_LOG"] = "0"
        envmod.consume_plan_records()


# ------------------------------------------------------ 3. the mechanism is gone
@pytest.mark.parametrize("name", [
    "grad_coverage_enabled", "reject_frozen_grads", "_grad_coverage",
    "_leaf_norms", "_exact_leaf_norms", "_record_grad_coverage",
    "_record_frozen_grad_plan", "consume_grad_coverage_stats",
    "consume_frozen_grad_plan_count", "_GRAD_COV_STATS", "_FROZEN_GRAD_PLANS",
    "_EXACT_LEAF_NORMS", "_plan_coverage_census", "_plan_log_max_leaves",
])
def test_env_has_no_coverage_symbol(name):
    assert not hasattr(envmod, name), name


@pytest.mark.parametrize("name", [
    "configure_grad_coverage", "GRAD_COVERAGE_HEAD",
])
def test_ppo_has_no_coverage_symbol(name):
    assert not hasattr(ppo, name), name


def test_agent_has_no_coverage_head_field():
    agents = [c for c in vars(ppo).values()
              if isinstance(c, type) and dataclasses.is_dataclass(c)
              and "value_head_fid" in {f.name for f in dataclasses.fields(c)}]
    assert agents, "no agent class with value heads found"
    for c in agents:
        assert "value_head_gcov" not in {f.name for f in dataclasses.fields(c)}


REMOVED_FLAGS = ("--reject-frozen-grads", "--no-reject-frozen-grads",
                 "--grad-coverage-weight", "--plan-log-max-leaves")


def test_ppo_source_defines_none_of_the_removed_flags():
    src = (_ALPHAGRAD / "src/alphagrad/approx/ppo.py").read_text()
    for flag in REMOVED_FLAGS:
        assert f'"{flag}"' not in src, flag


def _load_generator():
    p = _ALPHAGRAD / "tools" / "gen_fq_launchers.py"
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_generator_emits_none_of_the_removed_flags():
    gen = _load_generator()
    for flag in REMOVED_FLAGS:
        assert flag not in gen.REQUIRED_FLAGS, flag
        assert flag not in [k for k, _v in gen.SHARED_CLI], flag
    assert gen.ARMS
    for a in gen.ARMS:
        assert a["name"] != "w0_x2_screen"
        text = gen.render(a)
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for flag in REMOVED_FLAGS:
                assert flag not in line, (a["name"], line)


def test_no_in_repo_launcher_or_tool_passes_a_removed_flag():
    offenders = []
    for p in sorted(list(_ALPHAGRAD.glob("*.sh")) + list(_ALPHAGRAD.glob("*.sbatch"))
                    + list((_ALPHAGRAD / "tools").glob("*.sh"))):
        for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(f in line for f in REMOVED_FLAGS):
                offenders.append(f"{p.name}:{i}")
    assert not offenders, offenders


# ----------------------------------------- 4. no weight, head or exemption on 7
def _args(**kw):
    d = dict(fidelity_weight=0.0, cos_log_every=0, sparsity_weight=0.0,
             sparsity_log=False, symlog_channels="all", no_symlog=False,
             reward_mode="additive", rewards=["cmp", "mem", "acc"],
             lambda_cmp=1.0, lambda_mem=1.0, lambda_acc=16.0,
             cmp_type="latency", mem_type="peak_memory")
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(autouse=True)
def _restore_head_config():
    saved = (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
             ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR)
    env_saved = {k: os.environ.get(k) for k in
                 ("ALPHAGRAD_SPARSITY", "ALPHAGRAD_SPARSITY_WEIGHT",
                  "ALPHAGRAD_FIDELITY_WEIGHT", "ALPHAGRAD_COS_LOG_EVERY")}
    yield
    (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
     ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR) = saved
    for k, v in env_saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_default_heads_never_include_slot7():
    ppo.configure_fidelity(_args())
    ppo.configure_sparsity(_args())
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality")
    assert SLOT7 not in ppo.HEAD_REWARD_INDICES


def test_every_optional_head_on_still_skips_slot7():
    ppo.configure_fidelity(_args(fidelity_weight=1.0))
    ppo.configure_sparsity(_args(sparsity_weight=1.0))
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality", "fidelity",
                              "sparsity")
    assert SLOT7 not in ppo.HEAD_REWARD_INDICES


def test_symlog_mask_and_display_weights_never_touch_slot7():
    for mode in ("all", "cost", "none"):
        a = _args(symlog_channels=mode)
        ppo.configure_symlog(a)
        assert SLOT7 not in ppo._NO_SYMLOG_REWARD_INDICES, mode
        assert not bool(ppo._NO_SYMLOG_MASK_NP[SLOT7]), mode
        assert float(ppo._build_reward_weights(a)[SLOT7]) == 0.0


def test_sparsity_weight_is_accepted_without_any_guard():
    w = ppo.configure_sparsity(_args(sparsity_weight=2.0))
    assert w == 2.0
    assert ppo.SPARSITY_HEAD in ppo.VALUE_HEAD_ATTRS
