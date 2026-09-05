"""Gate G1-G6 telemetry (ticket dsnn-3qm.45): every documented field from a
fake episode, and each gate's arithmetic on a hand-built example.

Pure numpy over hand-built plan records (encoded with the plan log's own
``encode_wires`` so the wire shape is the real one). No env, no jax
tracing: this is the quick-script class of test.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from alphagrad.approx.common import gate_telemetry as gt
from alphagrad.approx.common.plan_log import encode_wires

HEADS = ("latency", "mem", "quality")
REWARD_NAMES = ("muls_adds_fmas", "flops", "latency_ns", "max_io_sum",
                "bytes_accessed", "peak_memory", "quality", "grad_coverage",
                "fidelity", "bkstep_acc", "sparsity")
COMPRESS_SENTINEL, QUANT_SENTINEL = -2, -3
N_RULES, MAX_FACES, SLOTS = 2, 3, 3


def _wire(order, *, skip=(), face_kind=None, rule_kind=None):
    """Dense wires for ``order``; ``skip`` = steps whose face 0 is skipped,
    ``face_kind`` = {step: sentinel} on face 0 slot 0, ``rule_kind`` =
    {step: sentinel} on rule row 0."""
    n = len(order)
    rules = np.full((n, N_RULES, 3), -1, np.int64)
    rules[:, :, 2] = 0
    faces = np.full((n, MAX_FACES, SLOTS, 3), -1, np.int64)
    skips = np.zeros((n, MAX_FACES), np.int64)
    for k in skip:
        skips[k, 0] = 1
    for k, b0 in (face_kind or {}).items():
        faces[k, 0, 0] = (b0, 0, 1)
    for k, b0 in (rule_kind or {}).items():
        rules[k, 0] = (b0, 0, 1)
    return order, rules, faces, skips


def _record(order, lat_ns, q, *, temp=1000.0, ref_lat=None, ref_temp=None,
            sentinel=False, **wire_kw):
    rec = encode_wires(*_wire(order, **wire_kw),
                       compress_sentinel=COMPRESS_SENTINEL,
                       quant_sentinel=QUANT_SENTINEL)
    rew = [0.0] * len(REWARD_NAMES)
    rew[REWARD_NAMES.index("latency_ns")] = -1e10 if sentinel else -float(lat_ns)
    rew[REWARD_NAMES.index("peak_memory")] = -1e10 if sentinel else -float(temp)
    rew[REWARD_NAMES.index("quality")] = 0.0 if sentinel else float(q)
    rec["rewards"] = rew
    rec["reward_names"] = list(REWARD_NAMES)
    rec["mem_temp_bytes"] = None if sentinel else float(temp)
    rec["mem_channel"] = "temp"
    if ref_lat is not None:
        rec[gt.REF_LATENCY_KEY] = float(ref_lat)
    if ref_temp is not None:
        rec[gt.REF_TEMP_KEY] = float(ref_temp)
    return rec


# One fake episode: 6 envs, 3 heads, reference latency 1000 ns, temp 1000 B.
REV = [4, 3, 2, 1]


def _fake_episode():
    recs = [
        # env 0: rev-exact, latency corner
        _record(REV, 1000.0, 1.0, ref_lat=1000.0, ref_temp=1000.0),
        # env 1: rev-exact again (drift 2 %), quality corner
        _record(REV, 1020.0, 1.0, ref_lat=1000.0, ref_temp=1000.0),
        # env 2: skip on vertex 3, fast, slightly worse quality, latency corner
        _record(REV, 900.0, 0.99, temp=800.0, ref_lat=1000.0, ref_temp=1000.0,
                skip=(1,)),
        # env 3: reduce on vertex 2 (face slot), mem corner
        _record(REV, 950.0, 0.98, temp=700.0, ref_lat=1000.0, ref_temp=1000.0,
                face_kind={2: COMPRESS_SENTINEL}),
        # env 4: quant micro-rule on vertex 4, destroyed Jacobian (q = 0)
        _record(REV, 980.0, 0.0, ref_lat=1000.0, ref_temp=1000.0,
                rule_kind={0: QUANT_SENTINEL}),
        # env 5: sentinelled measurement, must be ignored everywhere
        _record(REV, 0.0, 0.0, sentinel=True),
    ]
    all_rets = np.array([r["rewards"] for r in recs], dtype=np.float64)
    prefs = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.95, 0.05, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.4, 0.3, 0.3],
                      [1.0, 0.0, 0.0]], dtype=np.float32)
    T = 4
    critic_pref = np.repeat(prefs[:, None, :], T, axis=1)          # (E, T, H)
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(6, T, 3)).astype(np.float32)
    predictions = targets + 0.1 * rng.normal(size=(6, T, 3)).astype(np.float32)
    predictions[..., 1] = 0.0                                      # mem head dead
    critic = {"targets": targets, "predictions": predictions,
              "preference": critic_pref}
    winners = [{"vertex": 3, "primitive": "add", "kind": "skip"},
               {"vertex": 2, "primitive": "mul", "kind": "compress"},
               {"vertex": 1, "primitive": "dot_general", "kind": "diag"},
               {"vertex": 4, "primitive": "WRONG", "kind": "quant"}]
    vprim = {1: "dot_general", 2: "mul", 3: "add", 4: "exp"}
    # legality: 2 live faces, 6 axes; face 0 has 2 legal ordered pairs, 2
    # reduce axes, quant legal; face 1 has no pairs, 1 axis, no quant.
    N = 6
    fpair = np.zeros((MAX_FACES, N, N))
    fpair[0, 0, 1] = fpair[0, 1, 0] = 1.0
    fpair[0, 2, 2] = 1.0                       # diagonal: never a legal pair
    fcomp = np.zeros((MAX_FACES, N))
    fcomp[0, :2] = 1.0
    fcomp[1, 0] = 1.0
    fvalid = np.array([1.0, 1.0, 0.0])
    fquant = np.array([1.0, 0.0, 0.0])
    legal = gt.legal_counts_from_masks(fpair, fcomp, fvalid, fquant, None)
    return dict(records=recs, head_names=HEADS, all_rets=all_rets,
                reward_names=REWARD_NAMES, critic=critic,
                face_entropy_nats=1.2, legal=legal, winners=winners,
                vertex_primitive=vprim, quality_floor=0.985,
                offline_contrast=0.013)


# ---------------------------------------------------------------------------
# 1. The contract: every documented field, only documented fields, finite
#    where the fake episode supplies the input.
# ---------------------------------------------------------------------------
def test_fake_episode_emits_exactly_the_documented_fields():
    out = gt.episode_fields(**_fake_episode())
    documented = gt.documented_fields(HEADS)
    assert set(out) == documented, (set(out) ^ documented)
    for k, v in out.items():
        assert isinstance(v, (int, float)), (k, type(v))
    # With every input present nothing should be NaN except the fields
    # whose definition makes them so on this episode.
    nans = {k for k, v in out.items() if isinstance(v, float) and math.isnan(v)}
    assert nans == set(), nans


def test_missing_inputs_give_absent_flags_not_exceptions():
    out = gt.episode_fields([], head_names=HEADS)
    assert set(out) == gt.documented_fields(HEADS)
    assert out["paired/n"] == 0
    assert out["gate/g1/present"] == 0 and out["gate/g6/present"] == 0
    assert math.isnan(out["paired/lat_ratio_mean"])
    assert math.isnan(out["gate/g2/ev_latency"])
    assert math.isnan(out["gate/g3/uniform_floor_nats"])
    assert out["gate/g3/n_faces"] == 0


def test_documented_table_covers_every_gate_letter_and_renders():
    letters = {row[3] for row in gt.FIELD_TABLE}
    assert {"G1", "G2", "G3", "G4", "G5", "G6"} <= letters
    md = gt.render_field_table()
    for name, _u, _s, _g in gt.FIELD_TABLE + gt.FIELDS_LOGGED_ELSEWHERE + gt.CONFIG_TABLE:
        assert f"`{name}`" in md


# ---------------------------------------------------------------------------
# 2. Paired ratios: positive units, sentinels out, reference absent -> NaN.
# ---------------------------------------------------------------------------
def test_paired_ratios_are_candidate_over_rev_exact_in_positive_units():
    ep = _fake_episode()
    out = gt.episode_fields(**ep)
    # live records: 5 (the sentinelled one is out); all five carry the ref
    assert out["paired/n"] == 5 and out["paired/n_with_ref"] == 5
    lat = np.array([1000, 1020, 900, 950, 980]) / 1000.0
    assert out["paired/lat_ratio_best"] == pytest.approx(0.9)
    assert out["paired/lat_ratio_mean"] == pytest.approx(lat.mean())
    assert out["paired/lat_ratio_median"] == pytest.approx(np.median(lat))
    temp = np.array([1000, 1000, 800, 700, 1000]) / 1000.0
    assert out["paired/temp_ratio_best"] == pytest.approx(0.7)
    assert out["paired/temp_ratio_mean"] == pytest.approx(temp.mean())
    assert out["paired/grad_cosine_best"] == pytest.approx(1.0)
    assert out["paired/grad_cosine_mean"] == pytest.approx((1 + 1 + .99 + .98 + 0) / 5)


def test_reference_absent_reads_nan_and_counts_zero_with_ref():
    recs = [_record(REV, 900.0, 0.99), _record(REV, 950.0, 0.98)]
    out = gt.episode_fields(recs, head_names=HEADS)
    assert out["paired/n"] == 2 and out["paired/n_with_ref"] == 0
    assert math.isnan(out["paired/lat_ratio_mean"])
    assert math.isnan(out["paired/temp_ratio_best"])
    # quality does not need a reference
    assert out["paired/grad_cosine_best"] == pytest.approx(0.99)


def test_rev_exact_is_decided_on_the_wire():
    assert gt.record_is_rev_exact(_record(REV, 1.0, 1.0))
    assert not gt.record_is_rev_exact(_record([1, 2, 3, 4], 1.0, 1.0))
    assert not gt.record_is_rev_exact(_record(REV, 1.0, 1.0, skip=(0,)))
    assert not gt.record_is_rev_exact(
        _record(REV, 1.0, 1.0, rule_kind={0: QUANT_SENTINEL}))


# ---------------------------------------------------------------------------
# 3. G1 recovery by (vertex, primitive).
# ---------------------------------------------------------------------------
def test_g1_recovery_counts_vertex_class_pairs_and_flags_primitive_drift():
    ep = _fake_episode()
    out = gt.g1_recovery(ep["records"], ep["winners"], ep["vertex_primitive"])
    # winners: (3, skip) applied by env 2; (2, compress) by env 3; (1, diag)
    # by nobody; (4, quant) applied but its primitive label disagrees.
    assert out["gate/g1/present"] == 1
    assert out["gate/g1/n_winners"] == 4
    assert out["gate/g1/n_recovered"] == 2
    assert out["gate/g1/recovery"] == pytest.approx(0.5)
    assert out["gate/g1/recovery_skip"] == pytest.approx(1.0)
    assert out["gate/g1/recovery_compress"] == pytest.approx(1.0)
    assert out["gate/g1/recovery_diag"] == pytest.approx(0.0)
    # a winner whose primitive label drifted is NOT recoverable: it counts
    # in the denominator and in primitive_mismatch, never as a hit
    assert out["gate/g1/recovery_quant"] == pytest.approx(0.0)
    assert out["gate/g1/primitive_mismatch"] == 1


def test_g1_without_a_table_is_absent_not_zero():
    ep = _fake_episode()
    out = gt.g1_recovery(ep["records"], None)
    assert out["gate/g1/present"] == 0
    assert math.isnan(out["gate/g1/recovery"])


def test_g1_sentinelled_plans_do_not_count_as_applied():
    rec = _record(REV, 0.0, 0.0, sentinel=True, skip=(1,))
    out = gt.g1_recovery([rec], [{"vertex": 3, "primitive": None, "kind": "skip"}])
    assert out["gate/g1/n_recovered"] == 0


def test_winners_table_reads_csv_and_json(tmp_path):
    csv_p = tmp_path / "w.csv"
    csv_p.write_text("vertex,primitive,kind\nv76/add,add,skip\n12,mul,reduce\n"
                     "x,mul,quant\n")
    rows, why = gt.load_winners_table(str(csv_p))
    assert why == "" and rows == [
        {"vertex": 76, "kind": "skip", "primitive": "add"},
        {"vertex": 12, "kind": "compress", "primitive": "mul"}]
    js_p = tmp_path / "w.json"
    js_p.write_text(json.dumps([{"vertex": "v5/exp", "class": "Diag"}]))
    rows, why = gt.load_winners_table(str(js_p))
    assert rows == [{"vertex": 5, "kind": "diag", "primitive": "exp"}]
    rows, why = gt.load_winners_table(None)
    assert rows is None and "not run" in why
    rows, why = gt.load_winners_table(str(tmp_path / "missing.csv"))
    assert rows is None and "not found" in why


# ---------------------------------------------------------------------------
# 4. G2 explained variance per head.
# ---------------------------------------------------------------------------
def test_g2_explained_variance_per_head_arithmetic():
    t = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 6.0], [4.0, 5.0, 8.0]])
    p = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    out = gt.explained_variance_per_head(t, p, HEADS)
    assert out["gate/g2/ev_latency"] == pytest.approx(1.0)      # perfect
    assert math.isnan(out["gate/g2/ev_mem"])                    # constant target
    assert out["gate/g2/ev_quality"] == pytest.approx(0.0)      # constant pred
    assert out["gate/g2/n"] == 4
    # a worse-than-mean predictor is negative, as the gate expects
    p2 = p.copy()
    p2[:, 0] = [4.0, 3.0, 2.0, 1.0]
    assert gt.explained_variance_per_head(t, p2, HEADS)["gate/g2/ev_latency"] < 0


def test_g2_appended_head_and_non_finite_rows():
    heads = HEADS + ("fidelity",)
    t = np.random.default_rng(1).normal(size=(5, 3, 4))
    p = t.copy()
    p[0, 0, 3] = np.nan
    out = gt.explained_variance_per_head(t, p, heads)
    assert out["gate/g2/ev_fidelity"] == pytest.approx(1.0)
    assert out["gate/g2/n"] == 15


# ---------------------------------------------------------------------------
# 5. G3 uniform floor from the live legality masks.
# ---------------------------------------------------------------------------
def test_g3_legal_counts_mirror_the_head_leaves():
    ep = _fake_episode()
    n, skip_legal = ep["legal"]
    # face 0: 1 + 2 pairs + 2 axes x 5 fns + 1 dtype = 14 per slot
    # face 1: 1 + 0 + 1 x 5 + 0 = 6 per slot; face 2 is not live
    assert n.tolist() == [[14, 14, 14], [6, 6, 6]]
    assert skip_legal.tolist() == [True, True]


def test_g3_profile_override_removes_classes():
    N = 6
    fpair = np.zeros((1, N, N)); fpair[0, 0, 1] = 1.0
    fcomp = np.zeros((1, N)); fcomp[0, 0] = 1.0
    fvalid = np.ones(1)
    only_reduce = np.array([0.0, 1.0, 0.0, 1.0])        # DIAG, COMPRESS, QUANT, END
    n, _ = gt.legal_counts_from_masks(fpair, fcomp, fvalid, np.ones(1), only_reduce)
    assert n.tolist() == [[6, 6, 6]]                    # 1 + 1 axis x 5 fns
    skip_only = np.array([0.0, 0.0, 0.0, 1.0])
    n, _ = gt.legal_counts_from_masks(fpair, fcomp, fvalid, np.ones(1), skip_only)
    assert n.tolist() == [[1, 1, 1]]


def test_g3_uniform_floor_arithmetic():
    # one face, per-slot counts (2, 2, 2), skip legal: N = 1 + 8 = 9
    fl = gt.uniform_floor(np.array([[2, 2, 2]]), np.array([True]))
    assert fl["per_face"] == pytest.approx(math.log(9))
    # E[arity] = 1 + (8/9) * 3 * (1 - 1/2) = 1 + 4/3
    assert fl["arity_norm"] == pytest.approx(math.log(9) / (1 + 8 / 9 * 1.5))
    assert fl["n_outcomes_max"] == 9
    # skip illegal: N = 8, E[arity] = 1 + 1.5
    fl = gt.uniform_floor(np.array([[2, 2, 2]]), np.array([False]))
    assert fl["per_face"] == pytest.approx(math.log(8))
    assert fl["arity_norm"] == pytest.approx(math.log(8) / 2.5)
    # a face with nothing but None and skip: N = 2, floor log 2, arity 1
    fl = gt.uniform_floor(np.array([[1, 1, 1]]), np.array([True]))
    assert fl["per_face"] == pytest.approx(math.log(2))
    assert fl["arity_norm"] == pytest.approx(math.log(2))


def test_g3_full_94_logit_head_floor_is_stated():
    # Everything legal on a 6-axis face: 1 + 30 ordered pairs + 9 x 5 + 1 = 77
    # per slot, N = 1 + 77^3 = 456534, log N = 13.03 nats per face.
    N = 6
    fpair = np.ones((1, N, N))
    fcomp = np.ones((1, 9))
    n, sk = gt.legal_counts_from_masks(fpair, fcomp, np.ones(1), np.ones(1), None)
    assert n.tolist() == [[77, 77, 77]]
    fl = gt.uniform_floor(n, sk)
    assert fl["per_face"] == pytest.approx(math.log(1 + 77 ** 3))
    assert fl["per_face"] == pytest.approx(13.0314, abs=1e-3)


def test_g3_fields_compare_entropy_to_the_floor():
    out = gt.g3_face_entropy(0.5, np.array([[2, 2, 2]]), np.array([True]))
    floor = math.log(9) / (1 + 8 / 9 * 1.5)
    assert out["gate/g3/uniform_floor_nats"] == pytest.approx(floor)
    assert out["gate/g3/entropy_over_floor"] == pytest.approx(0.5 / floor)
    assert out["gate/g3/n_faces"] == 1
    out = gt.g3_face_entropy(None, None, None)
    assert math.isnan(out["gate/g3/face_entropy_nats"])
    assert math.isnan(out["gate/g3/entropy_over_floor"])


# ---------------------------------------------------------------------------
# 6. G4 fractions.
# ---------------------------------------------------------------------------
def test_g4_q_zero_and_feasible_fractions():
    q = np.array([1.0, 0.99, 0.0, 0.0, 0.5, np.nan])
    out = gt.g4_quality_fractions(q, tau=0.9)
    assert out["gate/g4/n"] == 5
    assert out["gate/g4/q_zero_frac"] == pytest.approx(2 / 5)
    assert out["gate/g4/q_ge_tau_frac"] == pytest.approx(2 / 5)
    assert out["gate/g4/tau"] == pytest.approx(0.9)
    out = gt.g4_quality_fractions(q, tau=None)
    assert math.isnan(out["gate/g4/q_ge_tau_frac"]) and math.isnan(out["gate/g4/tau"])
    out = gt.g4_quality_fractions([], tau=0.9)
    assert out["gate/g4/n"] == 0 and math.isnan(out["gate/g4/q_zero_frac"])


# ---------------------------------------------------------------------------
# 7. G5 corners, join, spread and drift floor.
# ---------------------------------------------------------------------------
def test_g5_records_join_envs_by_reward_vector_and_land_on_corners():
    ep = _fake_episode()
    out = gt.episode_fields(**ep)
    # latency corner: envs 0 (1.0) and 2 (0.9, pref 0.95 >= tol)
    assert out["gate/g5/latency/n"] == 2
    assert out["gate/g5/latency/best_lat_ratio"] == pytest.approx(0.9)
    assert out["gate/g5/latency/best_temp_ratio"] == pytest.approx(0.8)
    # quality corner: env 1 only
    assert out["gate/g5/quality/n"] == 1
    assert out["gate/g5/quality/best_lat_ratio"] == pytest.approx(1.02)
    # mem corner: env 3
    assert out["gate/g5/mem/n"] == 1
    assert out["gate/g5/mem/best_temp_ratio"] == pytest.approx(0.7)
    # spread = max - min over corners of the best ratios
    assert out["gate/g5/spread_lat"] == pytest.approx(1.02 - 0.9)
    assert out["gate/g5/spread_temp"] == pytest.approx(1.0 - 0.7)
    # env 4 sat in the interior: live but at no corner
    assert out["gate/g5/n_unmatched"] == 1
    # drift floor from the two rev-exact plans: 1.02 - 1.00
    assert out["gate/g5/n_rev_exact"] == 2
    assert out["gate/g5/drift_floor_lat"] == pytest.approx(0.02)
    assert out["gate/g5/drift_floor_temp"] == pytest.approx(0.0)


def test_g5_preference_corner_rule():
    assert gt.preference_corner([1, 0, 0], HEADS) == "latency"
    assert gt.preference_corner([0.05, 0.05, 0.9], HEADS) == "quality"
    assert gt.preference_corner([0.5, 0.5, 0.0], HEADS) is None
    assert gt.preference_corner([0, 0, 0], HEADS) is None
    assert gt.preference_corner([0, 170.0, 0], HEADS) == "mem"   # scale-free


def test_g5_join_uses_each_env_row_once_and_tolerates_no_rows():
    recs = [_record(REV, 900.0, 0.99), _record(REV, 900.0, 0.99)]
    rets = np.array([recs[0]["rewards"]], dtype=np.float64)
    env_of = gt.match_records_to_envs(recs, rets, REWARD_NAMES)
    assert env_of.tolist() == [0, -1]
    assert gt.match_records_to_envs(recs, None, REWARD_NAMES).tolist() == [-1, -1]


# ---------------------------------------------------------------------------
# 8. G6 placeholder and the fingerprint.
# ---------------------------------------------------------------------------
def test_g6_placeholder():
    out = gt.g6_offline_contrast(None)
    assert out["gate/g6/present"] == 0 and math.isnan(out["gate/g6/offline_contrast"])
    out = gt.g6_offline_contrast(0.012)
    assert out["gate/g6/present"] == 1 and out["gate/g6/offline_contrast"] == pytest.approx(0.012)


def test_toolchain_fingerprint_has_every_config_key(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    fp = gt.toolchain_fingerprint(sparse=True)
    ours = {name for name, *_ in gt.CONFIG_TABLE if name.startswith("toolchain/")}
    assert set(fp) == ours
    assert fp["toolchain/xla_flags"] == "--xla_gpu_autotune_level=0"
    assert fp["toolchain/sparse"] == 1
    assert gt.toolchain_fingerprint()["toolchain/sparse"] == -1


def test_critic_stash_is_pop_once():
    gt.stash_critic(np.zeros((2, 3, 3)), np.ones((2, 3, 3)), np.ones((2, 3, 3)))
    c = gt.pop_critic()
    assert c is not None and c["targets"].shape == (2, 3, 3)
    assert gt.env_preferences(c).shape == (2, 3)
    assert gt.pop_critic() is None
