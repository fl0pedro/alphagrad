# THE MEMORY OBJECTIVE, reward slot 11 (dsnn-xvi, owner ruling 2026-09-24):
#   r_mem = -(log(temp/temp*) + log(args/args*) + log(out/out*))
# from memory_analysis() of the timed executable and of the reference (*),
# jax.grad of the target (dsnn-xta; the graphax rev-exact until 2026-09-24),
# under --cost-form paired-log, one PopArt channel, no symlog.
from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

from types import SimpleNamespace                                # noqa: E402

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                            # noqa: E402
import alphagrad.approx.ppo as ppo                               # noqa: E402
from alphagrad.approx.common import compile_cache as _cc         # noqa: E402
from alphagrad.approx.common import reward_scaling as rs         # noqa: E402
from alphagrad.approx.env import (                               # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)

MSLOT = int(REWARD_INDEX["mem_objective"])


# Owner ruling 2026-09-24 Q45: 2^-10 x the smallest nonzero reference value.
def _eps(ref):
    return 2.0 ** -10 * min(float(x) for x in ref if float(x) > 0.0)


_N = 64
_rng = np.random.default_rng(0)
_W1 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_W2 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_X = jnp.asarray(np.linspace(-1.0, 1.0, _N, dtype=np.float32))


def _toy(x):
    h = jnp.tanh(_W1 @ x)
    y = jnp.tanh(_W2 @ h)
    return jnp.sum(y * y)


def _make_env(**kw):
    closed = jax.make_jaxpr(_toy)(_X)
    kw.setdefault("measure_latency", True)
    kw.setdefault("num_data_points", 2)
    kw.setdefault("reps_per_point", 2)
    kw.setdefault("latency_inner_reps", 2)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=_toy, **kw,
    )


def _run_plan(env, order, skip_everything=False, stop_after=None):
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for k, v in enumerate(order):
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = (jnp.ones((MAX_FACES,), jnp.int32) if skip_everything
                     else jnp.zeros((MAX_FACES,), jnp.int32))
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
        if stop_after is not None and k + 1 == stop_after:
            return np.asarray(state.reward)
    return np.asarray(state.reward)


def _rev_order(env):
    return sorted(int(x) for x in np.asarray(env.valid_vertices))[::-1]


def _last_record():
    recs = envmod.consume_plan_records()["records"]
    assert recs, "no terminal plan was recorded"
    return recs[-1]


def _triple(ex):
    ma = ex.memory_analysis()
    return (float(ma.temp_size_in_bytes), float(ma.output_size_in_bytes),
            float(ma.argument_size_in_bytes))


def _formula(cand, ref):
    (t, o, a), (t_r, o_r, a_r) = cand, ref
    e = _eps(ref)
    return -(math.log((t + e) / (t_r + e)) + math.log((a + e) / (a_r + e))
             + math.log((o + e) / (o_r + e)))


def _spy_executables(monkeypatch):
    seen = {"approx": [], "paired-ref": []}
    real = _cc.cached_compile

    def _spy(key, fn):
        out = real(key, fn)
        for prefix in seen:
            if bytes(key).startswith(prefix.encode() + b":"):
                seen[prefix].append(out)
        return out
    monkeypatch.setattr(_cc, "cached_compile", _spy)
    return seen


@pytest.fixture(autouse=True)
def _paired_log_with_the_plan_log(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    envmod.consume_plan_records()
    yield
    envmod.consume_plan_records()


# ------------------------------------------------------------ 1. the layout
def test_slot_11_was_appended_in_both_tables():
    assert NUM_REWARDS == rs.NUM_REWARDS == 12
    assert MSLOT == rs.MEM_OBJECTIVE_IDX == 11
    assert REWARD_NAMES[11] == rs.REWARD_NAMES[11] == "mem_objective"
    assert REWARD_NAMES[:11] == rs.REWARD_NAMES[:11]
    # a cost stored negated, never symlogged, not one of the six refusal
    # channels, and not a sparse-terminal QUALITY channel
    assert MSLOT not in envmod.QUALITY_REWARD_INDICES
    assert MSLOT not in envmod.COMPUTE_REWARD_INDICES
    assert MSLOT in rs.COST_REWARD_INDICES
    assert MSLOT in rs.NO_SYMLOG_REWARD_INDICES
    assert MSLOT not in rs.SPARSE_TERMINAL_INDICES


# ------------------------------------------------------ 2. the pure formula
def test_mem_objective_is_the_negated_log_sum_of_three_ratios():
    same = (512.0, 256.0, 1024.0)
    e = 2.0 ** -10 * 256.0
    assert envmod.mem_objective(same, same) == (
        0.0, {"ratios": {"temp": 1.0, "args": 1.0, "out": 1.0}, "eps": e})
    v, rec = envmod.mem_objective((256.0, 256.0, 512.0), same)
    half_t, half_a = (256.0 + e) / (512.0 + e), (512.0 + e) / (1024.0 + e)
    assert v == -(math.log(half_t) + math.log(half_a) + math.log(1.0))
    assert rec == {"ratios": {"temp": half_t, "args": half_a, "out": 1.0},
                   "eps": e}
    assert v == pytest.approx(2.0 * math.log(2.0), abs=2e-3)
    # an exact 0 is finite through eps, on both sides of that term only
    v, rec = envmod.mem_objective((0.0, 256.0, 1024.0), same)
    assert v == -(math.log(e / (512.0 + e)))
    assert rec == {"ratios": {"temp": e / (512.0 + e), "args": 1.0,
                              "out": 1.0}, "eps": e}


# ------------------------------------------ 3. the reference scores exactly 0
def _the_reference_as_the_candidate(monkeypatch, env):
    real = _cc.cached_compile
    _cc._LOCAL_CACHE.clear()

    def _substitute(key, fn):
        if bytes(key).startswith(b"approx:"):
            fn = lambda: envmod._compile_measure(                # noqa: E731
                jax.jit(envmod.reference_program(env.config),
                        keep_unused=True).lower(*env.args))
        return real(key, fn)
    monkeypatch.setattr(_cc, "cached_compile", _substitute)


def test_the_reference_scores_exactly_zero(monkeypatch):
    env = _make_env()
    _the_reference_as_the_candidate(monkeypatch, env)
    r = _run_plan(env, _rev_order(env))
    _cc._LOCAL_CACHE.clear()
    assert float(r[MSLOT]) == 0.0
    rec = _last_record()
    assert rec["rewards"][MSLOT] == 0.0
    assert rec["mem_ratios"] == {"temp": 1.0, "args": 1.0, "out": 1.0}
    assert rec["mem_objective_eps"] == _eps(
        (rec["ref_temp_bytes"], rec["ref_output_bytes"], rec["ref_args_bytes"]))
    assert rec["mem_temp_bytes"] == rec["ref_temp_bytes"] > 0.0
    assert rec["mem_output_bytes"] == rec["ref_output_bytes"] > 0.0
    assert rec["mem_args_bytes"] == rec["ref_args_bytes"] > 0.0


# ---------------------------- 4. bit-for-bit the three memory_analysis ratios
def test_the_objective_is_the_three_memory_analysis_ratios_bit_for_bit(
        monkeypatch):
    seen = _spy_executables(monkeypatch)
    env = _make_env()
    # FORWARD, so the candidate is a different program from the reference.
    r = _run_plan(env, _rev_order(env)[::-1])
    assert seen["approx"] and seen["paired-ref"]
    cand = _triple(seen["approx"][-1])
    ref = _triple(seen["paired-ref"][-1])
    assert cand != ref
    expect = _formula(cand, ref)
    assert expect != 0.0
    rec = _last_record()
    e = _eps(ref)
    assert rec["mem_objective_eps"] == e
    assert rec["rewards"][MSLOT] == expect
    assert float(r[MSLOT]) == float(np.float32(expect))
    assert rec["mem_ratios"] == {"temp": (cand[0] + e) / (ref[0] + e),
                                 "args": (cand[2] + e) / (ref[2] + e),
                                 "out": (cand[1] + e) / (ref[1] + e)}
    assert (rec["mem_temp_bytes"], rec["mem_output_bytes"],
            rec["mem_args_bytes"]) == cand
    assert (rec["ref_temp_bytes"], rec["ref_output_bytes"],
            rec["ref_args_bytes"]) == ref
    print(f"[mem-objective] forward vs rev-exact: cand={cand} ref={ref} "
          f"ratios={rec['mem_ratios']} r_mem={expect:+.6f}")


def test_a_zero_temp_plan_is_finite_through_eps_on_that_term():
    env = _make_env()
    r = _run_plan(env, _rev_order(env), skip_everything=True)
    rec = _last_record()
    cand = (rec["mem_temp_bytes"], rec["mem_output_bytes"],
            rec["mem_args_bytes"])
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"],
           rec["ref_args_bytes"])
    print(f"[mem-objective] skip-everything: cand={cand} ref={ref} "
          f"ratios={rec['mem_ratios']} eps={rec['mem_objective_eps']}")
    # Dead-code elimination leaves the all-skip program with EXACTLY 0 temp
    # bytes (job 63632); eps keeps its log finite (owner ruling Q45).
    assert cand[0] == 0.0 and ref[0] > 0.0
    e = _eps(ref)
    assert rec["mem_objective_eps"] == e
    assert rec["mem_ratios"]["temp"] == e / (ref[0] + e)
    expect = -(math.log(rec["mem_ratios"]["temp"])
               + math.log(rec["mem_ratios"]["args"])
               + math.log(rec["mem_ratios"]["out"]))
    assert rec["rewards"][MSLOT] == expect
    assert float(r[MSLOT]) > 0.0                      # cheaper -> above 0


# ------------------------------------------- 5. when it is not measured: 0.0
def test_non_terminal_steps_carry_zero():
    env = _make_env()
    r = _run_plan(env, _rev_order(env), stop_after=1)
    assert float(r[MSLOT]) == 0.0


def test_the_absolute_form_reads_zero_and_records_no_ratios(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "absolute")
    env = _make_env()
    r = _run_plan(env, _rev_order(env)[::-1])
    assert float(r[MSLOT]) == 0.0
    rec = _last_record()
    assert rec["mem_ratios"] is None
    assert rec["mem_objective_eps"] is None
    assert rec["ref_args_bytes"] is None and rec["ref_output_bytes"] is None
    # the candidate's own static numbers are recorded under every form
    assert rec["mem_args_bytes"] > 0.0 and rec["mem_output_bytes"] > 0.0


# --------------------------------------------------- 6. the ppo head wiring
def _args(**kw):
    base = dict(mem_objective_weight=0.0, cost_form="paired-log",
                sparsity_weight=0.0, sparsity_log=False,
                fidelity_weight=0.0, cos_log_every=0,
                reward_mode="additive", symlog_channels="all",
                rewards=["cmp", "mem", "acc"], cmp_type="flops",
                mem_type="peak_memory", lambda_cmp=1.0, lambda_mem=1.0,
                lambda_acc=1.0, lambda_frob=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def _restore_head_globals():
    saved = (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
             ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR)
    symlog = (tuple(ppo._NO_SYMLOG_REWARD_INDICES), bool(ppo._NO_SYMLOG_ALL[0]))
    # configure_fidelity and configure_sparsity write these into the process environment.
    env_saved = {k: os.environ.get(k) for k in
                 ("ALPHAGRAD_FIDELITY_WEIGHT", "ALPHAGRAD_COS_LOG_EVERY",
                  "ALPHAGRAD_SPARSITY", "ALPHAGRAD_SPARSITY_WEIGHT")}
    yield
    (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
     ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR) = saved
    ppo._set_no_symlog_indices(symlog[0])
    ppo._NO_SYMLOG_ALL[0] = symlog[1]
    for k, v in env_saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_flag_off_is_inert(_restore_head_globals):
    assert ppo.configure_mem_objective(_args()) == 0.0
    assert ppo.MEM_OBJECTIVE_HEAD not in ppo.VALUE_HEAD_ATTRS
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality")
    assert ppo.NUM_VALUE_HEADS == 3
    ppo.configure_symlog(_args())
    assert MSLOT not in ppo._NO_SYMLOG_REWARD_INDICES
    assert rs.build_reward_weights(_args())[MSLOT] == 0.0


def test_a_weight_appends_one_head_at_the_end_raw_under_symlog(
        _restore_head_globals):
    a = _args(mem_objective_weight=0.5)
    assert ppo.configure_mem_objective(a) == 0.5
    assert ppo.HEAD_NAMES[-1] == "mem_objective"
    assert ppo.HEAD_REWARD_INDICES[-1] == MSLOT
    assert ppo.VALUE_HEAD_ATTRS[-1] == ppo.MEM_OBJECTIVE_HEAD
    assert ppo.NUM_VALUE_HEADS == len(ppo.HEAD_REWARD_INDICES) == 4
    ppo.configure_mem_objective(a)                          # idempotent
    assert ppo.NUM_VALUE_HEADS == 4
    w = ppo._build_head_weights(a)
    assert w.shape == (4,) and w[-1] == np.float32(0.5)
    ppo.configure_symlog(a)
    assert MSLOT in ppo._NO_SYMLOG_REWARD_INDICES
    assert bool(ppo._NO_SYMLOG_MASK_NP[MSLOT])
    assert rs.build_reward_weights(a)[MSLOT] == np.float32(0.5)


def test_the_head_order_is_deterministic_after_fidelity_and_sparsity(
        _restore_head_globals):
    ppo.configure_fidelity(_args(fidelity_weight=1.0))
    ppo.configure_sparsity(_args(sparsity_weight=1.0))
    ppo.configure_mem_objective(_args(mem_objective_weight=1.0))
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality", "fidelity",
                              "sparsity", "mem_objective")
    assert ppo.HEAD_REWARD_INDICES[-1] == MSLOT


def test_a_weight_under_the_absolute_form_is_refused(_restore_head_globals):
    with pytest.raises(ValueError, match="paired-log"):
        ppo.configure_mem_objective(
            _args(mem_objective_weight=1.0, cost_form="absolute"))
    assert ppo.MEM_OBJECTIVE_HEAD not in ppo.VALUE_HEAD_ATTRS


# ------------ 7. RSNN_SHD: a Diag on the carried face moves args and out
T_PIN = 7
_RSNN: dict = {}


def _rtrl_env():
    import alphagrad.approx.tools.landscape_map as lm
    from alphagrad.approx.common import carry_plan as CP
    hit = _RSNN.get("rtrl")
    if hit is not None:
        CP.register(hit["args_ns"], hit["key"], "RSNN_SHD", "rtrl",
                    hit["env"].config, hit["env"].args, hit["env"].consts,
                    dataset=None, dataset_size=-1, step_position=T_PIN)
        return lm, hit["env"], hit["eval"]
    argv = ["--example", "RSNN_SHD", "--dataset", "none",
            "--temporal-rule", "rtrl", "--step-position", str(T_PIN),
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--latency-inner-reps", "1",
            "--out-dir", "/tmp/mem_objective_test"]
    args = lm.make_argparser().parse_args(argv)
    env, eval_samples, _cj = lm.build_env(args)
    _key, args_key = jrand.split(jrand.PRNGKey(args.seed))
    _RSNN["rtrl"] = {"env": env, "args_ns": args, "key": args_key,
                     "eval": eval_samples}
    return lm, env, eval_samples


def _diag_on_the_carried_face(lm, env, order):
    from alphagrad.approx.common import carry_plan as CP
    jx = env.config.jaxpr
    mask = CP.carry_scope_mask(jx)
    inv = lm.face_inventory(env, np.asarray(order, dtype=np.int32))
    picked = [e for e in inv
              if mask[int(e["vertex"]) - 1]
              and jx.eqns[int(e["vertex"]) - 1].primitive.name == "dot_general"]
    assert picked, "the dense carry block contracts with a dot_general"
    return [{"k": int(e["k"]), "f": int(e["f"]), "slot": 0,
             "row": [0, 0, -1], "kind": "X"} for e in picked]


def _measure_rsnn(lm, env, eval_samples, order, wires, container):
    from alphagrad.approx.common import carry_plan as CP
    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "wires": wires}
    specs, faces, skips = lm.get_plan_arrays(plan, len(order))
    assert CP.container_for_plan(env.config, order, faces, skips,
                                 specs) == container
    lm.measure(env, eval_samples, order, plan)
    rec = _last_record()
    print(f"[mem-objective] RSNN_SHD rtrl rev {container}: "
          f"ratios={rec['mem_ratios']} "
          f"cand=({rec['mem_temp_bytes']:.0f}, {rec['mem_output_bytes']:.0f}, "
          f"{rec['mem_args_bytes']:.0f}) B "
          f"ref=({rec['ref_temp_bytes']:.0f}, {rec['ref_output_bytes']:.0f}, "
          f"{rec['ref_args_bytes']:.0f}) B r_mem={rec['rewards'][MSLOT]:+.4f}")
    return rec


def test_on_rsnn_shd_a_diag_on_the_carried_face_moves_the_args_term():
    lm, env, eval_samples = _rtrl_env()
    order = sorted(int(v) for v in env.valid_vertices)[::-1]
    exact = _measure_rsnn(lm, env, eval_samples, order, [], "exact")
    diag = _measure_rsnn(lm, env, eval_samples, order,
                         _diag_on_the_carried_face(lm, env, order), "diag")
    # The reference is jax.grad of the target (dsnn-xta), so the rev-exact
    # plan is a candidate like any other: its three ratios are what the two
    # programs' memory_analysis() say, and the diag plan's move is read
    # AGAINST THE EXACT PLAN, both being divided by one reference.
    assert exact["mem_objective_eps"] > 0.0
    assert exact["rewards"][MSLOT] == -(math.log(exact["mem_ratios"]["temp"])
                                        + math.log(exact["mem_ratios"]["args"])
                                        + math.log(exact["mem_ratios"]["out"]))
    assert exact["ref_args_bytes"] == diag["ref_args_bytes"]
    assert exact["ref_temp_bytes"] == diag["ref_temp_bytes"]
    assert exact["ref_output_bytes"] == diag["ref_output_bytes"]
    # Measured on RSNN_SHD rtrl at step 7 (job 67781), on the loss-only
    # program: the compact carry container reads 2,569,128 B of arguments
    # against 226,177,960 B dense, the output byte-identical. Since the
    # one-call step (owner rulings 2026-09-24 Q1c, Q11a) the program also
    # writes its next carry in its container, so the Diag carry moves the
    # output term by the same factor as the args term.
    # Since the full rollout (owner ruling 2026-09-25 Q24 a) the program's
    # arguments are the recordings and its output the gradient, whatever the
    # container, and the carry lives in the scan's state: the Diag moves the
    # TEMP term. Measured on the CPU (job 67987): 636,466,160 B of
    # temporaries exact against 13,189,760 B diag, args and output equal.
    assert diag["mem_objective_eps"] == exact["mem_objective_eps"]
    assert diag["mem_ratios"]["out"] == exact["mem_ratios"]["out"] == 1.0
    assert diag["mem_ratios"]["args"] == exact["mem_ratios"]["args"] == 1.0
    assert diag["mem_ratios"]["temp"] < exact["mem_ratios"]["temp"] / 20.0
    assert diag["rewards"][MSLOT] == -(math.log(diag["mem_ratios"]["temp"])
                                       + math.log(diag["mem_ratios"]["args"])
                                       + math.log(diag["mem_ratios"]["out"]))
    assert diag["rewards"][MSLOT] - exact["rewards"][MSLOT] > 3.0
