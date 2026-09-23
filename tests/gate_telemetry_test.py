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
            sentinel=False, cost_form="paired-log", watermark=None,
            ref_watermark=None, **wire_kw):
    """One plan-log record, shaped like the ones env.py writes.

    ``cost_form="paired-log"`` is the settled form (ticket .9, every
    2026-09-13 arm): reward slots 2 and 5 hold NEGATED LOG-DIFFERENCES, not
    ns and bytes, and the absolute units live in ``candidate_latency_ns`` /
    ``mem_temp_bytes`` beside the reference.  Reading ns off the reward slot
    under this form is what made every paired ratio NaN in job 65321.
    """
    rec = encode_wires(*_wire(order, **wire_kw),
                       compress_sentinel=COMPRESS_SENTINEL,
                       quant_sentinel=QUANT_SENTINEL)
    rew = [0.0] * len(REWARD_NAMES)
    if cost_form == "paired-log" and not sentinel:
        _rl = float(ref_lat) if ref_lat else float(lat_ns)
        _rt = float(ref_temp) if ref_temp else float(temp)
        rew[REWARD_NAMES.index("latency_ns")] = -math.log(
            max(float(lat_ns), 1.0) / max(_rl, 1.0))
        rew[REWARD_NAMES.index("peak_memory")] = -math.log(
            max(float(temp), 1.0) / max(_rt, 1.0))
    else:
        rew[REWARD_NAMES.index("latency_ns")] = (
            -1e10 if sentinel else -float(lat_ns))
        rew[REWARD_NAMES.index("peak_memory")] = (
            -1e10 if sentinel else -float(temp))
    rew[REWARD_NAMES.index("quality")] = 0.0 if sentinel else float(q)
    rec["rewards"] = rew
    rec["reward_names"] = list(REWARD_NAMES)
    rec["cost_form"] = cost_form
    rec["mem_temp_bytes"] = None if sentinel else float(temp)
    rec["mem_channel"] = "temp"
    if not sentinel:
        rec[gt.CANDIDATE_LATENCY_KEY] = float(lat_ns)
        rec[gt.WATERMARK_KEY] = float(
            watermark if watermark is not None else temp * 2.0)
    if ref_lat is not None:
        rec[gt.REF_LATENCY_KEY] = float(ref_lat)
    if ref_temp is not None:
        rec[gt.REF_TEMP_KEY] = float(ref_temp)
        rec[gt.REF_WATERMARK_KEY] = float(
            ref_watermark if ref_watermark is not None else ref_temp * 2.0)
    return rec


def _f32(x) -> float:
    """A value that survives the float32 transport unchanged.

    Every reward slot that is NOT a paired log-difference reaches the record
    through ``float(_aggregate_samples(...))``, i.e. through a float32 jnp
    median, so it is already a float32 value. Test inputs that model those
    slots must be too, or they blame the wrong slot for the 65340 join.
    """
    return float(np.float32(x))


def _env_rows(records) -> np.ndarray:
    """The terminal reward rows AS THE TRAINER RECEIVES THEM.

    ``env._callback_measured`` hands the plan record its ``_reward_slots``
    list of PYTHON floats (float64) and returns ``jnp.array(_reward_slots,
    dtype=jnp.float32)`` to the trainer, because ``env._callback_shape``
    declares a float32 result. So an env row is the FLOAT32 IMAGE of the
    record's float64 slots, and building the rows here in float64 is exactly
    what hid the job 65340 join failure from this file.
    """
    return np.array([r["rewards"] for r in records], dtype=np.float32)


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
    all_rets = _env_rows(recs)
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
    # THE DRAIN THE FIELDS WERE COMPUTED FROM (ticket .7).  Two records the
    # trainer's own env owned, four the measure actors owned, and the
    # actors' own terminal count agrees with what arrived here.
    drain_local = {"records": recs[:2]}
    drain_pool = {"records": recs[2:], "terminals": 4, "dropped": 0,
                  "actors_seen": 2, "actors_failed": 0}
    return dict(records=recs, head_names=HEADS, all_rets=all_rets,
                reward_names=REWARD_NAMES, critic=critic,
                face_entropy_nats=1.2, legal=legal, winners=winners,
                vertex_primitive=vprim, quality_floor=0.985,
                offline_contrast=0.013, mask_source=gt.MASK_SOURCE_ORACLE,
                drain_local=drain_local, drain_pool=drain_pool)


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


def test_the_markdown_doc_carries_the_rendered_table_verbatim():
    """docs/GATE_TELEMETRY.md IS FIELD_TABLE rendered.  A field added to the
    module and not to the doc (or the other way round) is a gate reading a
    name nobody agreed to, so the doc is checked against the renderer rather
    than maintained by hand."""
    import os
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "GATE_TELEMETRY.md")
    with open(doc) as fh:
        text = fh.read()
    assert gt.render_field_table().strip() in text, (
        "docs/GATE_TELEMETRY.md is out of date; paste in:\n"
        + gt.render_field_table())


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
    n, skip_legal, quant_legal = ep["legal"]
    G = gt.face_head_geometry()
    # face 0: 1 (None) + 2 ordered pairs + 2 axes x REDUCE_FNS; face 1: 1 +
    # 0 + 1 axis x REDUCE_FNS; face 2 not live. The QUANT is ONE bit per face
    # (owner ruling 2026-09-23), legal on face 0 and not on face 1.
    # EVERY TERM IS DERIVED.  A sixth reduce fn moves both the head and this
    # expectation; a literal here would let the floor go on describing the
    # head that stopped running.
    f0 = 1 + 2 + 2 * G["n_reduce_fns"]
    f1 = 1 + 0 + 1 * G["n_reduce_fns"]
    assert n.tolist() == [[f0] * G["n_slots"], [f1] * G["n_slots"]]
    assert skip_legal.tolist() == [True, True]
    assert quant_legal.tolist() == [True, False]


def test_g3_geometry_is_derived_from_the_head_layout_not_typed():
    from alphagrad.approx.unified_face_head import (
        QUANT_SLOTS, head_layout, _LAYOUT_SPEC)
    from alphagrad.approx.common.masks import FACE_QUANT_DTYPES
    for mode in sorted(_LAYOUT_SPEC):
        G = gt.face_head_geometry(mode)
        lay = head_layout(mode)
        assert G["width"] == lay.width
        assert G["n_slots"] == lay.n_slots
        assert G["n_quant_dtypes"] == len(FACE_QUANT_DTYPES) == 2
        assert G["quant_dtypes"] == tuple(FACE_QUANT_DTYPES)
        assert G["quant_slots"] == tuple(QUANT_SLOTS)
    with pytest.raises(ValueError):
        gt.face_head_geometry("no-such-approx-add")


def test_g3_floor_follows_the_slot_count_of_the_running_width():
    """The floor is summed over the slots THIS width has.

    ``learned1`` / ``learned2`` add join slots; a floor computed over a fixed
    three would understate every one of them.
    """
    N = 6
    fpair = np.ones((1, N, N))
    fcomp = np.ones((1, 9))
    for mode in ("lossless", "learned1", "learned2"):
        n, sk, ql = gt.legal_counts_from_masks(
            fpair, fcomp, np.ones(1), np.zeros(1), None, approx_add=mode)
        G = gt.face_head_geometry(mode)
        assert n.shape == (1, G["n_slots"])
        assert ql.tolist() == [False]
        fl = gt.uniform_floor(n, sk, ql)
        assert fl["n_outcomes_max"] == 1 + int(n[0, 0]) ** G["n_slots"]
        # with the bit legal the outcomes with it set come on top: the slots
        # it does not force to none, freely chosen
        n, sk, ql = gt.legal_counts_from_masks(
            fpair, fcomp, np.ones(1), np.ones(1), None, approx_add=mode)
        assert ql.tolist() == [True]
        fl = gt.uniform_floor(n, sk, ql)
        assert fl["n_outcomes_max"] == (
            1 + int(n[0, 0]) ** G["n_slots"]
            + int(n[0, 0]) ** (G["n_slots"] - len(G["quant_slots"])))


def test_g3_a_quant_legality_bit_is_one_leaf_per_face():
    """The face's Quant is ONE bit: the oracle's (F,) array says whether it
    is legal, an (F, K) row says so through the narrow float's column, and
    neither adds a per-slot leaf."""
    from alphagrad.approx.common.masks import FACE_QUANT_NARROW
    N = 6
    fpair = np.zeros((1, N, N))
    fcomp = np.zeros((1, N))
    G = gt.face_head_geometry()
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, np.ones(1), np.ones(1))
    assert n[0, 0] == 1 and ql.tolist() == [True]
    per_dtype = np.zeros((1, G["n_quant_dtypes"]))
    per_dtype[0, FACE_QUANT_NARROW] = 1.0
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, np.ones(1), per_dtype)
    assert n[0, 0] == 1 and ql.tolist() == [True]
    per_dtype[0, FACE_QUANT_NARROW] = 0.0
    per_dtype[0, 1 - FACE_QUANT_NARROW] = 1.0      # the exact entry only
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, np.ones(1), per_dtype)
    assert n[0, 0] == 1 and ql.tolist() == [False]
    # QUANT illegal on this face: nothing added.
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, np.ones(1), np.zeros(1))
    assert n[0, 0] == 1 and ql.tolist() == [False]


def test_g3_live_slot_masks_count_each_slot_and_the_bit_on_both_operands():
    """The live path (--live-faces, every campaign arm) hands per-SLOT,
    per-DTYPE masks; the floor reads the structural leaves per slot and the
    face bit off the narrow float's column on lhs AND rhs."""
    from alphagrad.approx.common.masks import FACE_QUANT_NARROW
    G = gt.face_head_geometry()
    F, S, N, K = 3, G["n_slots"], 6, G["n_quant_dtypes"]
    pair = np.zeros((F, S, N, N))
    comp = np.zeros((F, S, N))
    quant = np.zeros((F, S, K))
    pair[0, 0, 0, 1] = 1.0                 # one ordered pair, slot 0 only
    comp[0, 1, :2] = 1.0                   # two reduce axes, slot 1 only
    quant[0, :, FACE_QUANT_NARROW] = 1.0   # bf16 legal on every slot: bit on
    quant[1, 0, FACE_QUANT_NARROW] = 1.0   # lhs only: bit off
    quant[2, 2, FACE_QUANT_NARROW] = 1.0   # the new slot only: bit off
    n, sk, ql = gt.legal_counts_from_slot_masks(pair, comp, quant, F)
    assert n[0].tolist()[:3] == [2, 1 + 2 * G["n_reduce_fns"], 1]
    assert n[1].tolist() == [1] * S        # face 1: nothing legal but None
    assert sk.tolist() == [True, True, True]
    assert ql.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        gt.legal_counts_from_slot_masks(pair[0], comp, quant, F)


def test_g3_profile_override_removes_classes():
    N = 6
    fpair = np.zeros((1, N, N)); fpair[0, 0, 1] = 1.0
    fcomp = np.zeros((1, N)); fcomp[0, 0] = 1.0
    fvalid = np.ones(1)
    only_reduce = np.array([0.0, 1.0, 0.0, 1.0])        # DIAG, COMPRESS, QUANT, END
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, fvalid, np.ones(1), only_reduce)
    assert n.tolist() == [[6, 6, 6]]                    # 1 + 1 axis x 5 fns
    assert ql.tolist() == [False]
    skip_only = np.array([0.0, 0.0, 0.0, 1.0])
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, fvalid, np.ones(1), skip_only)
    assert n.tolist() == [[1, 1, 1]]
    assert ql.tolist() == [False]
    only_quant = np.array([0.0, 0.0, 1.0, 1.0])
    n, _, ql = gt.legal_counts_from_masks(fpair, fcomp, fvalid, np.ones(1), only_quant)
    assert n.tolist() == [[1, 1, 1]]
    assert ql.tolist() == [True]


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
    # THE QUANT BIT: (2, 2, 2), skip legal, bit legal. The outcomes with the
    # bit set leave only the new slot free: N = 1 + 8 + 2 = 11. Arity: the
    # skip 1, the 8 plain outcomes 1 + 3/2 each, the 2 quant outcomes
    # 2 + 1/2 each (the face, the bit, the new slot half the time).
    fl = gt.uniform_floor(np.array([[2, 2, 2]]), np.array([True]),
                          np.array([True]))
    assert fl["n_outcomes_max"] == 11
    assert fl["per_face"] == pytest.approx(math.log(11))
    e_ar = (1 + 8 * 2.5 + 2 * 2.5) / 11
    assert fl["arity_norm"] == pytest.approx(math.log(11) / e_ar)


def test_g3_everything_legal_floor_is_the_width_s_own_arithmetic():
    """The floor of a face on which every choice is legal, stated from the
    LAYOUT.  Under the 2026-09-23 head (89 logits = 2 + 29*3, one quant bit
    per face) that is 1 + 30 ordered pairs + 9 x 5 reduce = 76 per slot,
    N = 1 + 76^3 + 76^1 (the bit set leaves the new slot free).  The
    assertion is written against ``face_head_geometry`` so it MOVES WITH THE
    LAYOUT: if the table changes and the floor code does not,
    ``max_outcomes_per_face`` and the counted floor disagree and this test
    fails."""
    N = 6
    fpair = np.ones((1, N, N))
    fcomp = np.ones((1, 9))
    G = gt.face_head_geometry()
    n, sk, ql = gt.legal_counts_from_masks(
        fpair, fcomp, np.ones(1), np.ones(1), None)
    per_slot = (1 + G["n_pair_idx"] * (G["n_pair_idx"] - 1)
                + G["n_reduce_axes"] * G["n_reduce_fns"])
    assert n.tolist() == [[per_slot] * G["n_slots"]]
    assert ql.tolist() == [True]
    fl = gt.uniform_floor(n, sk, ql)
    assert fl["n_outcomes_max"] == G["max_outcomes_per_face"]
    assert fl["per_face"] == pytest.approx(math.log(
        1 + per_slot ** G["n_slots"]
        + per_slot ** (G["n_slots"] - len(G["quant_slots"]))))


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
    # the rev-exact-CANDIDATE floor, kept for the reverse-order control:
    # the two rev-exact plans differ by 1.02 - 1.00
    assert out["gate/g5/n_rev_exact"] == 2
    assert out["gate/g5/drift_floor_lat_revexact"] == pytest.approx(0.02)
    # the floor G5 actually uses is the spread of the REPEATED REFERENCE,
    # which this episode pins at 1000 ns for every plan: zero drift.
    assert out["gate/g5/drift_floor_lat"] == pytest.approx(0.0)
    assert out["gate/g5/present"] == 1
    assert out["gate/g5/drift_floor_temp"] == pytest.approx(0.0)


def test_g5_preference_corner_rule():
    assert gt.preference_corner([1, 0, 0], HEADS) == "latency"
    assert gt.preference_corner([0.05, 0.05, 0.9], HEADS) == "quality"
    assert gt.preference_corner([0.5, 0.5, 0.0], HEADS) is None
    assert gt.preference_corner([0, 0, 0], HEADS) is None
    assert gt.preference_corner([0, 170.0, 0], HEADS) == "mem"   # scale-free


def test_g5_join_uses_each_env_row_once_and_tolerates_no_rows():
    recs = [_record(REV, 900.0, 0.99), _record(REV, 900.0, 0.99)]
    rets = _env_rows(recs[:1])
    env_of = gt.match_records_to_envs(recs, rets, REWARD_NAMES)
    assert env_of.tolist() == [0, -1]
    assert gt.match_records_to_envs(recs, None, REWARD_NAMES).tolist() == [-1, -1]


# ---------------------------------------------------------------------------
# 7b. THE JOIN ITSELF (job 65340): a complete drain that joined nothing.
# ---------------------------------------------------------------------------
def test_a_paired_log_latency_slot_is_float64_and_its_env_row_is_float32():
    """The two values that differ, in one assertion.

    Under ``--cost-form paired-log`` slot 2 is ``math.log(candidate) -
    math.log(reference)``, computed in float64 and kept in the record. The
    env row is its float32 image. They are equal in float32 and unequal in
    float64, which is the whole of the 65340 join failure.
    """
    il = REWARD_NAMES.index("latency_ns")
    rec = _record(REV, 900.0, _f32(0.99), ref_lat=1000.0, ref_temp=1000.0)
    slot = rec["rewards"][il]
    row = float(_env_rows([rec])[0][il])
    assert slot == pytest.approx(-math.log(0.9))
    assert slot != row
    assert float(np.float32(slot)) == row


def test_the_65340_shape_a_complete_drain_whose_records_joined_no_env_row():
    """Four terminals, four records, four env rows, and the pre-fix join
    matched none of them. The float32 join matches all four."""
    il = REWARD_NAMES.index("latency_ns")
    iq = REWARD_NAMES.index("quality")
    recs = [_record(REV, lat, _f32(q), ref_lat=1000.0, ref_temp=1000.0)
            for lat, q in ((900.0, 0.99), (950.0, 0.98),
                           (980.0, 0.97), (1010.0, 0.96))]
    rows = _env_rows(recs)
    # The join this replaced: float64 equality on latency_ns and quality.
    # The quality slot agrees (it is float32-exact on both sides); the
    # paired-log latency slot alone is enough to match nothing.
    wide = rows.astype(np.float64)
    for r in recs:
        assert np.any(wide[:, iq] == r["rewards"][iq])
        assert not np.any((wide[:, il] == r["rewards"][il])
                          & (wide[:, iq] == r["rewards"][iq]))
    details: dict = {}
    env_of = gt.match_records_to_envs(recs, rows, REWARD_NAMES,
                                      details=details)
    assert env_of.tolist() == [0, 1, 2, 3]
    assert details["n_joined"] == 4
    assert details["mode"] == gt.JOIN_MODE_REWARDS
    assert details["columns"] == list(REWARD_NAMES)


def test_the_fake_episode_joins_every_record_to_the_env_row_that_measured_it():
    out = gt.episode_fields(**_fake_episode())
    assert out["gate/g5/n_joined"] == 6
    assert out["gate/g5/join_mode"] == gt.JOIN_MODE_REWARDS


def test_the_join_prefers_the_env_index_a_record_carries_over_its_rewards():
    """Two records the reward vector cannot tell apart, and an identity that
    can. The identity wins and the order it gives is the one taken."""
    recs = [_record(REV, 900.0, 0.99), _record(REV, 900.0, 0.99)]
    recs[0][gt.ENV_INDEX_KEY] = 1
    recs[1][gt.ENV_INDEX_KEY] = 0
    details: dict = {}
    env_of = gt.match_records_to_envs(recs, _env_rows(recs), REWARD_NAMES,
                                      details=details)
    assert env_of.tolist() == [1, 0]
    assert details["mode"] == gt.JOIN_MODE_IDENTITY
    assert details["n_joined"] == 2


def test_an_env_index_that_no_env_row_owns_is_reported_and_never_invented():
    recs = [_record(REV, 900.0, 0.99)]
    recs[0][gt.ENV_INDEX_KEY] = 7
    details: dict = {}
    env_of = gt.match_records_to_envs(recs, _env_rows(recs), REWARD_NAMES,
                                      details=details)
    assert env_of.tolist() == [-1]
    assert details["first_unmatched"]["key"] == {gt.ENV_INDEX_KEY: 7}
    assert "out of range" in details["why"]


def test_one_record_without_the_identity_sends_the_whole_join_to_the_rewards():
    recs = [_record(REV, 900.0, 0.99), _record(REV, 950.0, 0.98)]
    recs[0][gt.ENV_INDEX_KEY] = 0
    details: dict = {}
    gt.match_records_to_envs(recs, _env_rows(recs), REWARD_NAMES,
                             details=details)
    assert details["mode"] == gt.JOIN_MODE_REWARDS


def test_the_join_compares_only_the_reward_names_both_sides_carry():
    rec = _record(REV, 900.0, _f32(0.99), ref_lat=1000.0, ref_temp=1000.0)
    rows = np.concatenate(
        [_env_rows([rec]), np.array([[1.5]], np.float32)], axis=1)
    names = list(REWARD_NAMES) + ["a_channel_the_record_predates"]
    details: dict = {}
    env_of = gt.match_records_to_envs([rec], rows, names, details=details)
    assert env_of.tolist() == [0]
    assert details["columns"] == list(REWARD_NAMES)


def test_a_slot_that_is_not_a_number_on_both_sides_still_joins():
    rec = _record(REV, 900.0, _f32(0.99), ref_lat=1000.0, ref_temp=1000.0)
    rec["rewards"][REWARD_NAMES.index("sparsity")] = float("nan")
    assert gt.match_records_to_envs(
        [rec], _env_rows([rec]), REWARD_NAMES).tolist() == [0]


def test_the_join_reports_the_first_unmatched_record_and_its_nearest_env_row():
    recs = [_record(REV, 900.0, _f32(0.99), ref_lat=1000.0, ref_temp=1000.0)]
    rows = _env_rows(recs).copy()
    rows[0, REWARD_NAMES.index("quality")] += np.float32(0.25)
    details: dict = {}
    env_of = gt.match_records_to_envs(recs, rows, REWARD_NAMES,
                                      details=details)
    assert env_of.tolist() == [-1]
    assert details["first_unmatched"]["record"] == 0
    assert details["nearest"]["env"] == 0
    assert details["nearest"]["worst_name"] == "quality"
    assert details["nearest"]["worst_diff"] == pytest.approx(0.25, abs=1e-6)


def test_the_corners_empty_line_reports_what_it_observed_and_blames_nothing(
        capsys):
    """The line this replaced named a lagging drain that job 65340 had
    already disproved. The replacement prints only measurements."""
    ep = _fake_episode()
    # Env rows from plans that are not in this episode's record set: a join
    # that finds nothing, with the drain and the preferences both intact.
    ep["all_rets"] = _env_rows(
        [_record(REV, 1234.0 + i, _f32(0.5), ref_lat=1000.0, ref_temp=1000.0)
         for i in range(6)])
    out = gt.episode_fields(**ep)
    assert out["gate/g5/n_joined"] == 0
    err = capsys.readouterr().err
    assert "G5 corners empty" in err
    assert "joined 0 of 6 plan record(s)" in err
    assert "live records=5" in err
    assert "prefs=(6, 3)" in err
    assert "pool_terminals=4" in err and "undrained=0" in err
    assert "first unmatched record #0" in err
    assert "nearest env row" in err
    assert "largest gap on" in err
    assert "lagging" not in err


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


# ---------------------------------------------------------------------------
# 10. THE SETTLED COST FORM.  Under --cost-form paired-log the reward slot is
#     a log-difference, not a measurement; a ratio built from it is fiction.
# ---------------------------------------------------------------------------
def test_paired_log_latency_comes_from_the_record_not_the_reward_slot():
    rec = _record(REV, 900.0, 0.99, ref_lat=1000.0, ref_temp=1000.0)
    # The slot really does hold the log-difference, positive for a win.
    slot = rec["rewards"][REWARD_NAMES.index("latency_ns")]
    assert slot == pytest.approx(-math.log(0.9))
    assert slot > 0.0                       # NEGATING IT GIVES A NEGATIVE "ns"
    assert gt.record_latency_ns(rec) == pytest.approx(900.0)
    out = gt.episode_fields([rec], head_names=HEADS)
    assert out["paired/lat_ratio_best"] == pytest.approx(0.9)
    assert out["paired/n_with_ref"] == 1


def test_paired_log_without_the_absolute_field_is_nan_never_invented():
    rec = _record(REV, 900.0, 0.99, ref_lat=1000.0, ref_temp=1000.0)
    del rec[gt.CANDIDATE_LATENCY_KEY]
    assert math.isnan(gt.record_latency_ns(rec))
    out = gt.episode_fields([rec], head_names=HEADS)
    assert math.isnan(out["paired/lat_ratio_mean"])
    assert out["paired/n_with_ref"] == 0
    # the temp channel is unaffected: it never rode the reward slot
    assert out["paired/temp_ratio_best"] == pytest.approx(1.0)


def test_absolute_cost_form_still_reads_the_negated_reward_slot():
    rec = _record(REV, 900.0, 0.99, ref_lat=1000.0, ref_temp=1000.0,
                  cost_form="absolute")
    del rec[gt.CANDIDATE_LATENCY_KEY]
    assert gt.record_latency_ns(rec) == pytest.approx(900.0)


def test_reference_temp_key_is_the_one_env_writes():
    """env.py writes ``ref_temp_bytes``.  Reading ``ref_mem_temp_bytes`` --
    a name nothing has ever written -- made paired/temp_ratio_* NaN in every
    run from 2026-09-05 to 2026-09-13 while the number sat on the record."""
    assert gt.REF_TEMP_KEY == "ref_temp_bytes"
    rec = _record(REV, 900.0, 0.99, ref_lat=1000.0, ref_temp=2000.0,
                  temp=1000.0)
    assert set(rec) & set(gt.REF_TEMP_KEY_ALIASES) == {gt.REF_TEMP_KEY}
    out = gt.episode_fields([rec], head_names=HEADS)
    assert out["paired/temp_ratio_best"] == pytest.approx(0.5)
    # a plan log written under the retired name still scores
    old = dict(rec)
    old["ref_mem_temp_bytes"] = old.pop(gt.REF_TEMP_KEY)
    out = gt.episode_fields([old], head_names=HEADS)
    assert out["paired/temp_ratio_best"] == pytest.approx(0.5)


def test_watermark_rides_beside_the_temp_channel():
    """The .45 contract asks for the watermark BESIDE the temp channel; the
    record carries both, so the paired panel carries both."""
    rec = _record(REV, 900.0, 0.99, ref_lat=1000.0, ref_temp=1000.0,
                  temp=800.0, watermark=1600.0, ref_watermark=4000.0)
    out = gt.episode_fields([rec], head_names=HEADS)
    assert out["paired/temp_ratio_best"] == pytest.approx(0.8)
    assert out["paired/watermark_ratio_best"] == pytest.approx(0.4)
    assert out["paired/n_with_watermark"] == 1


# ---------------------------------------------------------------------------
# 11. G5's drift floor must exist on the order the campaign runs.
# ---------------------------------------------------------------------------
MARKOWITZ = [2, 4, 1, 3]          # not descending: no plan is ever rev-exact


def test_drift_floor_comes_from_the_repeated_paired_reference():
    """Under --fixed-order markowitz NO candidate is rev-exact on the wire,
    so the rev-exact-candidate floor is structurally undefined and G5 had
    nothing to compare a spread against.  Ticket .9 re-measures the SAME
    reference once per candidate: its spread within the episode IS the
    instrument drift."""
    recs = [_record(MARKOWITZ, 900.0, 0.9, ref_lat=1000.0, ref_temp=1000.0),
            _record(MARKOWITZ, 910.0, 0.9, ref_lat=1020.0, ref_temp=1000.0),
            _record(MARKOWITZ, 905.0, 0.9, ref_lat=1010.0, ref_temp=1000.0)]
    out = gt.episode_fields(recs, head_names=HEADS)
    assert out["gate/g5/n_rev_exact"] == 0
    assert math.isnan(out["gate/g5/drift_floor_lat_revexact"])
    # (1020 - 1000) / 1010 = 0.0198
    assert out["gate/g5/drift_floor_lat"] == pytest.approx(20.0 / 1010.0)
    assert out["gate/g5/drift_floor_temp"] == pytest.approx(0.0)
    assert out["gate/g5/drift_floor_n"] == 3


def test_g5_present_is_zero_without_a_preference():
    recs = [_record(MARKOWITZ, 900.0, 0.9, ref_lat=1000.0, ref_temp=1000.0)]
    out = gt.episode_fields(recs, head_names=HEADS)
    assert out["gate/g5/present"] == 0
    assert out["gate/g5/n_unmatched"] == 1
    assert gt.episode_fields(**_fake_episode())["gate/g5/present"] == 1


# ---------------------------------------------------------------------------
# 12. TICKET .7: a counter drained in a process that does not own it.
# ---------------------------------------------------------------------------
def test_drain_provenance_flags_counters_read_in_the_wrong_process():
    """The shape of the cancelled canary 65319: the measure actors report 16
    terminal plans and ZERO records reach the trainer, so every gate field
    below is computed over an empty episode while the counters say 16."""
    canary = {"records": [], "terminals": 16, "dropped": 0,
              "actors_seen": 1, "actors_failed": 0}
    out = gt.drain_provenance({"records": []}, canary)
    assert out["measure/drain/pool_terminals"] == 16
    assert out["measure/drain/pool_records"] == 0
    assert out["measure/drain/undrained"] == 16
    assert out["measure/drain/ok"] == 0
    with pytest.raises(gt.DrainProvenanceError) as e:
        gt.drain_provenance({"records": []}, canary, strict=True)
    assert "16" in str(e.value)


def test_drain_provenance_is_ok_when_every_terminal_arrived():
    ep = _fake_episode()
    out = gt.episode_fields(**ep)
    assert out["measure/drain/ok"] == 1
    assert out["measure/drain/undrained"] == 0
    assert out["measure/drain/local_records"] == 2
    assert out["measure/drain/pool_records"] == 4
    assert out["measure/drain/pool_terminals"] == 4
    # a dropped record is accounted for, not undrained
    ok = gt.drain_provenance(None, {"records": [1, 2], "terminals": 3,
                                    "dropped": 1, "actors_seen": 1})
    assert ok["measure/drain/undrained"] == 0 and ok["measure/drain/ok"] == 1


def test_drain_provenance_flags_an_actor_that_could_not_be_polled():
    out = gt.drain_provenance(None, {"records": [1], "terminals": 1,
                                     "actors_seen": 2, "actors_failed": 1})
    assert out["measure/drain/ok"] == 0
    assert out["measure/drain/actors_failed"] == 1
    with pytest.raises(gt.DrainProvenanceError):
        gt.drain_provenance(None, {"records": [1], "terminals": 1,
                                   "actors_seen": 2, "actors_failed": 1},
                            strict=True)


# ---------------------------------------------------------------------------
# 13. The table IS the contract: a documented field nobody emits is a fault.
# ---------------------------------------------------------------------------
def test_a_documented_field_that_is_never_emitted_raises(monkeypatch):
    monkeypatch.setattr(
        gt, "FIELD_TABLE",
        gt.FIELD_TABLE + (("gate/g9/invented", "count", "nothing", "G9"),))
    with pytest.raises(KeyError) as e:
        gt.episode_fields([], head_names=HEADS)
    assert "gate/g9/invented" in str(e.value)
