"""Gate G1-G6 telemetry (ticket dsnn-3qm.45): the wandb fields the convergence
gate of ticket .6 reads, computed once per episode from data the trainer
already holds.

READ-ONLY BY CONSTRUCTION. Every input is a copy the trainer already made
for another purpose -- the plan records the drain delivered
(``env.consume_plan_records`` merged with the measure pool's), the critic
target/prediction pair the value loss compares, the face-head entropy the
loss reports, the per-env preference and terminal reward rows -- and the
only output is a dict of floats for ``wandb.log``. Nothing here is read
by the reward, the advantage, the value target or the sampler, so the
module cannot move a training run; ``docs/GATE_TELEMETRY.md`` lists one
row per field (name, unit, source, gate letter), and
``tests/gate_telemetry_test.py`` pins that every documented name is
emitted and every emitted name is documented.

COUNTERS ARE DRAINED WHERE THEY ARE INCREMENTED (ticket .7). This module
never reads a process-global counter. compile_fallbacks / toolchain_ok
(.21) and the memory-parity summary (.49) ride the plan-record drain and
are logged by ppo.py's plan-log block; they appear in the field table
under their existing names and are not recomputed here.

THE GATE (ticket .6, accepted 2026-09-02):
  G1 recovery of the sweep winners by (vertex, primitive);
  G2 explained variance of the value heads (latency > 0 is the criterion);
  G3 face-head entropy near its uniform floor and decreasing;
  G4 q = 0 fraction -> 0;
  G5 front spread at the simplex corners > drift floor;
  G6 offline contrast >= 1 %  (pre-run number from ticket .42).

PAIRED RATIOS. Every cost claim is a paired ratio against rev-exact
(CONTEXT.md). The per-plan reference latency / temp bytes are ticket .9's
plan-log fields; this module reads them under the names in
``REF_LATENCY_KEY`` / ``REF_TEMP_KEY`` and logs NaN (never raises) while
they are absent, so the paired panels fill in the moment .9 lands.

UNITS. Ratios are dimensionless, candidate / rev-exact, < 1 = cheaper.
Costs read off the reward vector are stored negated in the env and are
made POSITIVE here (ns, bytes). Entropies are nats. Fractions are in
[0, 1]. Explained variance is 1 - Var[target - prediction] / Var[target]
and is NaN, not 0, when the target is constant (an undefined EV must not
plot as "explained nothing").
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

from alphagrad.approx.common.plan_log import decode_wires, kind_of_slot

# ---------------------------------------------------------------------------
# Names shared with other tickets.
# ---------------------------------------------------------------------------
# Ticket .9's per-plan rev-exact reference, positive units, on the plan-log
# record. Absent until .9 lands -> the paired fields read NaN.
REF_LATENCY_KEY = "ref_latency_ns"
# THE NAME THE RECORD ACTUALLY CARRIES.  env.py's plan record writes the
# paired reference's static temp as ``ref_temp_bytes`` (env.py ~1643,
# beside ``ref_latency_ns`` and ``ref_watermark_bytes``).  This module read
# ``ref_mem_temp_bytes`` until 2026-09-13 -- a name nothing has ever
# written -- so ``paired/temp_ratio_*`` read NaN in EVERY run since .45
# landed while the number sat on the record.  The alias list is read in
# order so a plan log written under either name still scores.
REF_TEMP_KEY = "ref_temp_bytes"
REF_TEMP_KEY_ALIASES = (REF_TEMP_KEY, "ref_mem_temp_bytes")
# The runtime watermark of the SAME paired reference (ticket .49): the
# contract asks for the watermark beside the temp channel, so the paired
# panel carries its ratio next to the temp ratio rather than only the
# per-episode means under measure/mem_parity/*.
REF_WATERMARK_KEY = "ref_watermark_bytes"
# THE PLAN'S OWN COST IN ABSOLUTE UNITS.  Under --cost-form paired-log the
# reward slots hold log-differences, so the record carries the ns and the
# bytes separately (env.py ~1646) and every ratio here is built from these.
CANDIDATE_LATENCY_KEY = "candidate_latency_ns"
# Ticket .49's memory fields on the record (always present since .49).
TEMP_KEY = "mem_temp_bytes"
WATERMARK_KEY = "mem_watermark_bytes"
# Ticket .9's quality floor flag on ``args`` (None = no floor set).
QUALITY_FLOOR_ATTR = "quality_floor"

# The env's sentinel for a failed measurement: every cost channel reads
# -1e10. Same 0.99 test the trainer's LIVE mask uses.
SENTINEL_COST = -1e10
_SENTINEL_EDGE = SENTINEL_COST * 0.99

# THE FACE HEAD'S GEOMETRY IS DERIVED, NEVER TYPED (owner ruling
# 2026-09-13).  ``unified_face_head`` is the single source of truth for the
# width, the slot count and every per-slot field size; ``masks`` owns the
# QUANT dtype set, which is now the FOUR floats (float32, bfloat16,
# float8_e5m2, float8_e4m3fn).  The G3 uniform floor counts the leaves
# ``UnifiedFaceHead.score`` can reach, so it MUST move when this table
# moves: a literal here (the old ``FACE_SLOTS = 3`` / one legal cast) makes
# the floor describe a head that is not running, and G3 then compares the
# entropy against the wrong number in a direction nobody notices.
from alphagrad.approx.unified_face_head import (  # noqa: E402
    FACE_SLOTS, MAX_PAIR_IDX, NUM_REDUCE_AXES, NUM_REDUCE_FNS, SLOT_WIDTH,
    head_layout)
from alphagrad.approx.common.masks import (  # noqa: E402
    FACE_QUANT_DTYPES, NUM_FACE_QUANT_DTYPES)

#: The ``--approx-add`` value whose layout the floor is computed under when
#: the caller names none.  ``lossless`` is the one value every 2026-09-13
#: arm runs (owner ruling); it is NOT a fallback for an unknown value --
#: ``head_layout`` raises on those.
APPROX_ADD_DEFAULT = "lossless"

# Order of the (NUM_OPS,) legality override the trainer builds
# (ppo._op_legality_for_variant): DIAG, COMPRESS (= Reduce), QUANT, END.
OP_DIAG, OP_COMPRESS, OP_QUANT = 0, 1, 2

# Action classes as the plan log names them (kind_of_slot): "diag",
# "compress" (CONTEXT.md: Reduce), "quant"; plus the face SKIP.
KINDS = ("skip", "diag", "compress", "quant")
_KIND_ALIASES = {
    "skip": "skip", "diag": "diag", "blockdiag": "diag", "diagonal": "diag",
    "compress": "compress", "reduce": "compress",
    "quant": "quant", "quantize": "quant", "quantise": "quant",
}

# ---------------------------------------------------------------------------
# The field table: one row per field -- (name, unit, source, gate). Names
# with ``{head}`` / ``{corner}`` expand over the run's HEAD_NAMES.
# docs/GATE_TELEMETRY.md is this table rendered; the test pins both ways.
# ---------------------------------------------------------------------------
FIELD_TABLE: tuple[tuple[str, str, str, str], ...] = (
    # paired ratios and quality, every candidate against rev-exact
    ("paired/n", "count", "plan records this episode, live (not sentinelled)", "-"),
    ("paired/n_with_ref", "count", "live records carrying " + REF_LATENCY_KEY + " (ticket .9)", "-"),
    ("paired/lat_ratio_mean", "ratio", "mean over live records of latency_ns / " + REF_LATENCY_KEY, "G5"),
    ("paired/lat_ratio_median", "ratio", "median of the same", "G5"),
    ("paired/lat_ratio_best", "ratio", "min of the same (fastest plan)", "G5"),
    ("paired/temp_ratio_mean", "ratio", "mean of " + TEMP_KEY + " / " + REF_TEMP_KEY, "G5"),
    ("paired/temp_ratio_median", "ratio", "median of the same", "G5"),
    ("paired/temp_ratio_best", "ratio", "min of the same (leanest plan)", "G5"),
    ("paired/grad_cosine_mean", "cosine", "mean quality (reward slot 'quality') over live records", "G4"),
    ("paired/grad_cosine_median", "cosine", "median of the same", "G4"),
    ("paired/grad_cosine_best", "cosine", "max of the same", "G4"),
    ("paired/watermark_ratio_mean", "ratio", "mean of " + WATERMARK_KEY + " / " + REF_WATERMARK_KEY + ": the runtime watermark BESIDE the temp channel (ticket .49)", "G5"),
    ("paired/watermark_ratio_median", "ratio", "median of the same", "G5"),
    ("paired/watermark_ratio_best", "ratio", "min of the same", "G5"),
    ("paired/n_with_watermark", "count", "live records carrying both watermark numbers", "-"),
    ("paired/ref_latency_ns_mean", "ns", "mean of " + REF_LATENCY_KEY + ": the rev-exact reference re-measured once per candidate", "G5"),
    ("paired/ref_temp_bytes_mean", "bytes", "mean of " + REF_TEMP_KEY + ": the same reference's static temp", "G5"),
    # G1
    ("gate/g1/present", "0/1", "1 when a winners table was loaded (--gate-winners-table)", "G1"),
    ("gate/g1/n_winners", "count", "rows in the winners table", "G1"),
    ("gate/g1/n_recovered", "count", "winners whose (vertex, class) some plan this episode applied", "G1"),
    ("gate/g1/recovery", "fraction", "n_recovered / n_winners", "G1"),
    ("gate/g1/recovery_skip", "fraction", "recovery restricted to SKIP winners (NaN if none)", "G1"),
    ("gate/g1/recovery_diag", "fraction", "recovery restricted to Diag winners", "G1"),
    ("gate/g1/recovery_compress", "fraction", "recovery restricted to Reduce winners (code: compress)", "G1"),
    ("gate/g1/recovery_quant", "fraction", "recovery restricted to Quant winners", "G1"),
    ("gate/g1/primitive_mismatch", "count", "winners whose primitive label differs from the run's jaxpr at that vertex", "G1"),
    # G2
    ("gate/g2/n", "count", "(env, step) pairs with a finite target and prediction", "G2"),
    ("gate/g2/ev_{head}", "fraction", "explained variance of value head {head}: 1 - Var[target - pred] / Var[target]; NaN when Var[target] = 0", "G2"),
    # G3
    ("gate/g3/face_entropy_nats", "nats", "the trainer's arity-normalised face-head entropy (entropy/approx_head)", "G3"),
    ("gate/g3/uniform_floor_nats", "nats", "the same quantity if the head were uniform over legal joint outcomes, under the run's legality masks", "G3"),
    ("gate/g3/uniform_floor_per_face_nats", "nats", "mean over faces of log(number of legal joint outcomes)", "G3"),
    ("gate/g3/entropy_over_floor", "ratio", "face_entropy_nats / uniform_floor_nats", "G3"),
    ("gate/g3/n_faces", "count", "faces the floor was computed over", "G3"),
    ("gate/g3/n_outcomes_max", "count", "largest legal joint-outcome count of any face", "G3"),
    ("gate/g3/n_slots", "count", "slots the floor was summed over: head_layout(--approx-add).n_slots, never a literal", "G3"),
    ("gate/g3/head_width", "count", "head_layout(--approx-add).width, the logit count the floor describes", "G3"),
    ("gate/g3/n_quant_dtypes", "count", "len(masks.FACE_QUANT_DTYPES): the QUANT categorical the floor counts leaves of", "G3"),
    ("gate/g3/mask_source", "0/1/2", "where the legality came from: 0 none, 1 the oracle probe, 2 the live per-slot masks", "G3"),
    # G4
    ("gate/g4/n", "count", "live records with a finite quality", "G4"),
    ("gate/g4/q_zero_frac", "fraction", "share of those with quality exactly 0 (destroyed Jacobian)", "G4"),
    ("gate/g4/tau", "cosine", "the quality floor in force (--quality-floor, ticket .9); NaN if none", "G4"),
    ("gate/g4/q_ge_tau_frac", "fraction", "share of live records at or above tau (feasible); NaN if no floor", "G4"),
    # G5
    ("gate/g5/{corner}/n", "count", "live records whose env preference sat at the {corner} corner", "G5"),
    ("gate/g5/{corner}/best_lat_ratio", "ratio", "min paired latency ratio among them", "G5"),
    ("gate/g5/{corner}/best_temp_ratio", "ratio", "min paired temp ratio among them", "G5"),
    ("gate/g5/spread_lat", "ratio", "max - min over corners of best_lat_ratio", "G5"),
    ("gate/g5/spread_temp", "ratio", "max - min over corners of best_temp_ratio", "G5"),
    ("gate/g5/present", "0/1", "1 when at least one live record sat at a named corner (a preference reached the telemetry); 0 on an arm without --preference-conditioned, where a spread is undefined rather than 0", "G5"),
    ("gate/g5/n_rev_exact", "count", "live records whose plan IS rev-exact (reverse order, no approximation)", "G5"),
    ("gate/g5/drift_floor_lat", "ratio", "PAIRED-REFERENCE drift: (max - min) / mean of " + REF_LATENCY_KEY + " over this episode's live records -- the same rev-exact plan re-measured once per candidate, so its spread is pure instrument drift. Defined on every order, including --fixed-order markowitz where no candidate is itself rev-exact", "G5"),
    ("gate/g5/drift_floor_temp", "ratio", "the same over " + REF_TEMP_KEY + " (0 when the static temp is deterministic)", "G5"),
    ("gate/g5/drift_floor_n", "count", "records the drift floor was computed over", "G5"),
    ("gate/g5/drift_floor_lat_revexact", "ratio", "max - min of the latency RATIO over the rev-exact records only; NaN with fewer than two (the pre-2026-09-13 definition, kept for the reverse-order control)", "G5"),
    ("gate/g5/drift_floor_temp_revexact", "ratio", "the same for the temp ratio", "G5"),
    ("gate/g5/n_unmatched", "count", "records that matched no env row (no preference known)", "G5"),
    ("gate/g5/n_joined", "count", "plan records the join tied to an env row this episode. Zero here beside a complete drain means the join failed and not the drain (job 65340)", "G5"),
    ("gate/g5/join_mode", "0/1/2", "which join ran. 0 is none (no env rows, or no shared reward slot), 1 is the float32 reward-vector join, 2 is the env_index identity", "G5"),
    # G6
    ("gate/g6/present", "0/1", "1 when --gate-offline-contrast was given", "G6"),
    ("gate/g6/offline_contrast", "fraction", "ticket .42's offline contrast, copied from the flag; NaN when absent", "G6"),
    # DRAIN PROVENANCE (ticket .7's failure mode, named by the .45 contract)
    ("measure/drain/local_records", "count", "plan records the TRAINER's own env handed over this episode", "drain"),
    ("measure/drain/pool_records", "count", "plan records the MEASURE ACTORS handed over this episode", "drain"),
    ("measure/drain/pool_terminals", "count", "terminal plans the measure actors say they measured", "drain"),
    ("measure/drain/undrained", "count", "pool_terminals not accounted for by pool_records + dropped: a counter incremented in the actor and read in a process that does not own it reads > 0 here", "drain"),
    ("measure/drain/ok", "0/1", "1 when undrained == 0 and every polled actor answered", "drain"),
    ("measure/drain/actors_seen", "count", "measure actors the pool knows about", "drain"),
    ("measure/drain/actors_failed", "count", "actors that could not be polled: their plans are MISSING", "drain"),
)

# Logged by other tickets' code on the same drain; listed so the doc is one table.
FIELDS_LOGGED_ELSEWHERE: tuple[tuple[str, str, str, str], ...] = (
    ("measure/compile_fallbacks_this_ep", "count", "ppo.py plan-log block; env.consume_plan_records (ticket .21)", "toolchain"),
    ("measure/compile_fallbacks_total", "count", "same drain, process lifetime", "toolchain"),
    ("measure/toolchain_ok", "0/1", "same drain; 0 = a measure process failed the toolchain gate", "toolchain"),
    ("measure/mem_parity/gap_mean_bytes", "bytes", "ppo.py plan-log block; env.mem_parity_summary (ticket .49): watermark - temp", "memory"),
    ("measure/mem_parity/gap_max_bytes", "bytes", "same", "memory"),
    ("measure/mem_parity/gap_min_bytes", "bytes", "same", "memory"),
    ("measure/mem_parity/temp_mean_bytes", "bytes", "same: the channel", "memory"),
    ("measure/mem_parity/watermark_mean_bytes", "bytes", "same: the runtime watermark beside the channel", "memory"),
    ("measure/mem_parity/static_fallbacks", "count", "same: readings that were static substitutions", "memory"),
    ("measure/mem_parity/n", "count", "same", "memory"),
    ("measure/mem_parity/n_paired", "count", "same", "memory"),
    ("measure/mem_parity/measured", "count", "same", "memory"),
    ("entropy/approx_head", "nats", "ppo.py host_log; the loss's arity-normalised face-head entropy", "G3"),
    ("explained variance", "fraction", "ppo.py host_log; EV of the SUM over heads (the pre-.45 panel)", "G2"),
)

# Written to wandb.config once per run (toolchain_fingerprint()).
CONFIG_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("toolchain/jax", "version", "jax.__version__", "fingerprint"),
    ("toolchain/jaxlib", "version", "jaxlib.__version__", "fingerprint"),
    ("toolchain/xla_flags", "string", "os.environ['XLA_FLAGS'] ('' if unset)", "fingerprint"),
    ("toolchain/jax_platforms", "string", "os.environ['JAX_PLATFORMS'] ('' if unset)", "fingerprint"),
    ("toolchain/backend", "string", "jax.default_backend()", "fingerprint"),
    ("toolchain/sparse", "0/1/-1", "env.config.sparse (jacve returns SparseTensor); -1 = unknown", "fingerprint"),
    ("toolchain/hostname", "string", "the trainer's host", "fingerprint"),
    ("commit/alphagrad", "sha", "ppo._repo_commits (already there)", "fingerprint"),
    ("commit/graphax", "sha", "ppo._repo_commits (already there)", "fingerprint"),
)


def documented_fields(head_names) -> set[str]:
    """Every per-episode field name this module emits for ``head_names``."""
    out = set()
    for name, _u, _s, _g in FIELD_TABLE:
        if "{head}" in name:
            out.update(name.format(head=h) for h in head_names)
        elif "{corner}" in name:
            out.update(name.format(corner=h) for h in head_names)
        else:
            out.add(name)
    return out


def render_field_table() -> str:
    """The markdown for docs/GATE_TELEMETRY.md, straight from the tables."""
    lines = ["| field | unit | source | gate |", "|---|---|---|---|"]
    for name, unit, src, gate in FIELD_TABLE + FIELDS_LOGGED_ELSEWHERE:
        lines.append(f"| `{name}` | {unit} | {src} | {gate} |")
    lines += ["", "| config key | unit | source | role |", "|---|---|---|---|"]
    for name, unit, src, gate in CONFIG_TABLE:
        lines.append(f"| `{name}` | {unit} | {src} | {gate} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------
NAN = float("nan")


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def summary(x, *, best: str = "min") -> dict:
    """mean / median / best over the finite entries; NaN when there are none."""
    f = _finite(x)
    if f.size == 0:
        return {"mean": NAN, "median": NAN, "best": NAN, "n": 0}
    b = float(f.min()) if best == "min" else float(f.max())
    return {"mean": float(f.mean()), "median": float(np.median(f)),
            "best": b, "n": int(f.size)}


def _idx(names, name) -> int | None:
    try:
        return list(names).index(name)
    except (ValueError, TypeError):
        return None


def _get_float(rec: dict, key: str) -> float:
    v = rec.get(key)
    if v is None:
        return NAN
    try:
        v = float(v)
    except (TypeError, ValueError):
        return NAN
    return v


# ---------------------------------------------------------------------------
# Plan records -> per-record costs, quality, paired ratios.
# ---------------------------------------------------------------------------
def record_is_live(rec: dict) -> bool:
    """False for a sentinelled measurement (any cost channel at -1e10)."""
    if rec.get("sentinelled"):
        return False
    names, rews = rec.get("reward_names"), rec.get("rewards")
    if not names or rews is None:
        return False
    for ch in ("latency_ns", "peak_memory"):
        i = _idx(names, ch)
        if i is not None and i < len(rews):
            try:
                if float(rews[i]) <= _SENTINEL_EDGE:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _first_float(rec: dict, keys) -> float:
    for k in keys:
        v = _get_float(rec, k)
        if math.isfinite(v):
            return v
    return NAN


def record_latency_ns(rec: dict) -> float:
    """POSITIVE measured latency of THIS plan, in ns.

    THE REWARD SLOT IS NOT THE LATENCY under the settled cost form.  With
    ``--cost-form paired-log`` (ticket .9, every 2026-09-13 arm) slot
    ``latency_ns`` holds ``-(log lat_candidate - log lat_ref)`` -- a
    dimensionless log-difference that is POSITIVE for a plan faster than the
    reference.  Negating it gave a negative "latency" and the ``> 0`` guard
    below turned every paired ratio into NaN: that is why
    ``paired/lat_ratio_*`` and ``paired/n_with_ref`` read NaN / 0 in job
    65321 while ``ref_latency_ns`` sat on every record.

    ``candidate_latency_ns`` is the ns env.py records beside the reference
    for exactly this purpose ("so a record can be re-scored in absolute
    units", env.py ~1646).  It is preferred; the negated reward slot is the
    fallback for a record written under ``--cost-form absolute``, where the
    slot really is ``-ns``.
    """
    v = _get_float(rec, CANDIDATE_LATENCY_KEY)
    if math.isfinite(v) and v > 0.0:
        return v
    if str(rec.get("cost_form") or "") == "paired-log":
        # The slot is a log-difference here; there is no ns to recover and
        # inventing one from it would be a fabricated measurement.
        return NAN
    i = _idx(rec.get("reward_names"), "latency_ns")
    if i is None or i >= len(rec.get("rewards") or ()):
        return NAN
    try:
        v = -float(rec["rewards"][i])
    except (TypeError, ValueError):
        return NAN
    return v if v > 0.0 and math.isfinite(v) else NAN


def record_temp_bytes(rec: dict) -> float:
    """XLA static temp bytes of the plan's timed executable (ticket .49)."""
    v = _get_float(rec, TEMP_KEY)
    return v if (math.isfinite(v) and v >= 0.0) else NAN


def record_watermark_bytes(rec: dict) -> float:
    """Runtime peak watermark of the plan's timed executable (ticket .49),
    the number logged BESIDE the temp channel."""
    v = _get_float(rec, WATERMARK_KEY)
    return v if (math.isfinite(v) and v >= 0.0) else NAN


def record_ref_temp_bytes(rec: dict) -> float:
    return _first_float(rec, REF_TEMP_KEY_ALIASES)


def record_quality(rec: dict) -> float:
    i = _idx(rec.get("reward_names"), "quality")
    if i is None or i >= len(rec.get("rewards") or ()):
        return NAN
    try:
        return float(rec["rewards"][i])
    except (TypeError, ValueError):
        return NAN


def record_is_rev_exact(rec: dict) -> bool:
    """Reverse elimination order and no approximation anywhere (CONTEXT.md:
    rev-exact). Decided on the wire, not on a label."""
    order = rec.get("order")
    if not order:
        return False
    if any(int(order[k]) <= int(order[k + 1]) for k in range(len(order) - 1)):
        return False
    return not rec.get("rules") and not rec.get("faces")


def paired_ratios(records) -> dict:
    """Per-record arrays: latency ratio, static-temp ratio, watermark ratio,
    quality, liveness, rev-exactness, and the reference's own measurements.

    Ratios are NaN where the record lacks ticket .9's reference.  The
    ``ref_*`` arrays are kept because they are the SAME rev-exact plan
    re-measured once per candidate, in the same actor, back to back: their
    spread within one episode is pure instrument drift, and that is the
    drift floor G5 compares a front spread against on an order where no
    candidate is itself rev-exact (every ``--fixed-order markowitz`` arm).
    """
    n = len(records)
    lat = np.full(n, NAN)
    temp = np.full(n, NAN)
    wm = np.full(n, NAN)
    q = np.full(n, NAN)
    ref_lat = np.full(n, NAN)
    ref_temp = np.full(n, NAN)
    live = np.zeros(n, dtype=bool)
    rev = np.zeros(n, dtype=bool)
    has_ref = np.zeros(n, dtype=bool)
    for k, rec in enumerate(records):
        live[k] = record_is_live(rec)
        rev[k] = record_is_rev_exact(rec)
        if not live[k]:
            continue
        q[k] = record_quality(rec)
        c_lat, r_lat = record_latency_ns(rec), _get_float(rec, REF_LATENCY_KEY)
        if math.isfinite(r_lat) and r_lat > 0.0:
            ref_lat[k] = r_lat
        if math.isfinite(c_lat) and math.isfinite(r_lat) and r_lat > 0.0:
            lat[k] = c_lat / r_lat
            has_ref[k] = True
        c_t, r_t = record_temp_bytes(rec), record_ref_temp_bytes(rec)
        if math.isfinite(r_t) and r_t > 0.0:
            ref_temp[k] = r_t
        if math.isfinite(c_t) and math.isfinite(r_t) and r_t > 0.0:
            temp[k] = c_t / r_t
        c_w, r_w = record_watermark_bytes(rec), _get_float(rec,
                                                           REF_WATERMARK_KEY)
        if math.isfinite(c_w) and math.isfinite(r_w) and r_w > 0.0:
            wm[k] = c_w / r_w
    return {"lat_ratio": lat, "temp_ratio": temp, "watermark_ratio": wm,
            "quality": q, "live": live, "rev_exact": rev, "has_ref": has_ref,
            "ref_latency_ns": ref_lat, "ref_temp_bytes": ref_temp}


# ---------------------------------------------------------------------------
# G1 -- recovery of the sweep winners by (vertex, primitive).
# ---------------------------------------------------------------------------
def normalize_kind(kind) -> str | None:
    if kind is None:
        return None
    k = str(kind).strip().lower()
    return _KIND_ALIASES.get(k)


def _parse_vertex(v) -> int | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s.startswith("v"):
        s = s[1:]
    s = s.split("/")[0]
    try:
        return int(s)
    except ValueError:
        return None


def load_winners_table(path):
    """The sweep winners (ticket .41) as ``[{vertex, primitive, kind}]``.

    Accepts a CSV with a header or a JSON list of objects. Columns:
    ``vertex`` (``76`` or ``v76`` or ``v76/add``), ``primitive`` (or
    ``prim``; optional), ``kind`` (or ``class``): skip / diag / reduce /
    quant. Returns ``(rows, reason)``; ``rows`` is None when nothing could
    be read, and ``reason`` says why (logged once by the caller).
    """
    if not path:
        return None, "no --gate-winners-table given (sweep .41 has not run)"
    if not os.path.exists(path):
        return None, f"winners table not found: {path!r}"
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError as exc:
        return None, f"winners table unreadable: {exc!r}"
    raw: list[dict] = []
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"winners table is not JSON: {exc}"
        if isinstance(obj, dict):
            obj = obj.get("winners", obj.get("rows", []))
        raw = [r for r in obj if isinstance(r, dict)]
    else:
        raw = list(csv.DictReader(text.splitlines()))
    rows = []
    for r in raw:
        low = {str(k).strip().lower(): v for k, v in r.items()}
        vertex = _parse_vertex(low.get("vertex", low.get("v")))
        kind = normalize_kind(low.get("kind", low.get("class")))
        prim = low.get("primitive", low.get("prim"))
        # "v76/add" carries the primitive after the slash.
        vs = str(low.get("vertex", "")).strip()
        if prim is None and "/" in vs:
            prim = vs.split("/", 1)[1]
        if vertex is None or kind is None:
            continue
        rows.append({"vertex": vertex, "kind": kind,
                     "primitive": None if prim is None else str(prim).strip()})
    if not rows:
        return None, f"winners table {path!r} has no usable rows"
    return rows, ""


def plan_actions(rec: dict) -> set[tuple[int, str]]:
    """The (vertex, kind) pairs a plan record applies: face skips, face-slot
    approximations and per-vertex micro-rules, kinds as kind_of_slot names."""
    out: set[tuple[int, str]] = set()
    try:
        order, rule_specs, face_specs, face_skips = decode_wires(rec)
    except Exception:
        # Not replayable / malformed: fall back to the sparse rows.
        order = [int(v) for v in rec.get("order") or ()]
        for row in rec.get("rules") or ():
            k, b0 = int(row[0]), int(row[2])
            kind = kind_of_slot(b0)
            if kind and k < len(order):
                out.add((order[k], kind))
        for row in rec.get("faces") or ():
            k, skip = int(row[0]), int(row[2])
            if k >= len(order):
                continue
            if skip == 1:
                out.add((order[k], "skip"))
            for s in range((len(row) - 3) // 3):
                kind = kind_of_slot(int(row[3 + 3 * s]))
                if kind:
                    out.add((order[k], kind))
        return out
    n = int(order.shape[0])
    for k in range(n):
        v = int(order[k])
        for r in range(rule_specs.shape[1]):
            kind = kind_of_slot(int(rule_specs[k, r, 0]))
            if kind:
                out.add((v, kind))
        if face_specs.ndim == 4 and face_specs.shape[1]:
            live = face_specs[k, :, :, 0] != -1                  # (F, S)
            for s in range(face_specs.shape[2]):
                for f in np.nonzero(live[:, s])[0]:
                    kind = kind_of_slot(int(face_specs[k, f, s, 0]))
                    if kind:
                        out.add((v, kind))
        if face_skips.ndim == 2 and face_skips.shape[1] and \
                bool((face_skips[k] == 1).any()):
            out.add((v, "skip"))
    return out


def g1_recovery(records, winners, vertex_primitive=None) -> dict:
    """Which winners some plan in this episode re-applied.

    ``winners`` = rows from :func:`load_winners_table` (None = absent).
    ``vertex_primitive`` maps vertex id -> primitive name for THIS run's
    graph; when given, a winner whose label disagrees counts in
    ``primitive_mismatch`` (stable ids drifted, ticket .41) and is not
    recoverable.
    """
    out = {"gate/g1/present": 0, "gate/g1/n_winners": 0,
           "gate/g1/n_recovered": 0, "gate/g1/recovery": NAN,
           "gate/g1/primitive_mismatch": 0}
    for kd in KINDS:
        out[f"gate/g1/recovery_{kd}"] = NAN
    if not winners:
        return out
    applied: set[tuple[int, str]] = set()
    for rec in records:
        if record_is_live(rec):
            applied |= plan_actions(rec)
    out["gate/g1/present"] = 1
    out["gate/g1/n_winners"] = len(winners)
    hit_total = 0
    per_kind_n = {kd: 0 for kd in KINDS}
    per_kind_hit = {kd: 0 for kd in KINDS}
    mismatch = 0
    for w in winners:
        v, kd, prim = int(w["vertex"]), w["kind"], w.get("primitive")
        if vertex_primitive is not None and prim:
            have = vertex_primitive.get(v)
            if have is not None and str(have) != str(prim):
                mismatch += 1
                per_kind_n[kd] += 1
                continue
        per_kind_n[kd] += 1
        if (v, kd) in applied:
            hit_total += 1
            per_kind_hit[kd] += 1
    out["gate/g1/n_recovered"] = hit_total
    out["gate/g1/recovery"] = hit_total / len(winners)
    out["gate/g1/primitive_mismatch"] = mismatch
    for kd in KINDS:
        out[f"gate/g1/recovery_{kd}"] = (
            per_kind_hit[kd] / per_kind_n[kd] if per_kind_n[kd] else NAN)
    return out


# ---------------------------------------------------------------------------
# G2 -- explained variance per value head.
# ---------------------------------------------------------------------------
def explained_variance_per_head(targets, predictions, head_names) -> dict:
    """1 - Var[t - p] / Var[t] per head over every finite (t, p) pair.

    ``targets`` / ``predictions`` are ``(..., H)`` in the SAME space (the
    value loss compares ``_value_target(estim_returns)`` with the head's
    output; feed exactly that pair). NaN when Var[t] = 0 or n < 2: the
    trainer's ``explained_variance`` returns 0 there, which a gate reading
    "EV > 0" cannot tell from a critic that explains nothing.
    """
    out = {"gate/g2/n": 0}
    for h in head_names:
        out[f"gate/g2/ev_{h}"] = NAN
    if targets is None or predictions is None:
        return out
    t = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if t.ndim == 0 or p.shape != t.shape:
        return out
    H = t.shape[-1]
    t = t.reshape(-1, H)
    p = p.reshape(-1, H)
    n_any = 0
    for j, h in enumerate(head_names):
        if j >= H:
            break
        ok = np.isfinite(t[:, j]) & np.isfinite(p[:, j])
        n_any = max(n_any, int(ok.sum()))
        if ok.sum() < 2:
            continue
        var_t = float(np.var(t[ok, j]))
        if var_t <= 0.0:
            continue
        out[f"gate/g2/ev_{h}"] = 1.0 - float(np.var(t[ok, j] - p[ok, j])) / var_t
    out["gate/g2/n"] = n_any
    return out


# ---------------------------------------------------------------------------
# G3 -- face-head entropy against its uniform floor.
# ---------------------------------------------------------------------------
def face_head_geometry(approx_add: str = APPROX_ADD_DEFAULT) -> dict:
    """The per-slot choice arithmetic of the RUNNING face head, DERIVED.

    Every number here comes from ``unified_face_head`` (the layout table
    ``_LAYOUT_SPEC``, the per-slot field offsets) or from
    ``common.masks.FACE_QUANT_DTYPES``.  Nothing is typed, so a width change
    -- a fourth QUANT dtype, a join slot, another reduce fn -- moves the G3
    floor with it instead of leaving G3 comparing the head's entropy against
    the arithmetic of a head that stopped running.

    ``n_quant_default`` is the number of legal QUANT dtypes on a face whose
    mask says only "QUANT is legal here" without saying which casts: the
    operand's own dtype is masked (ticket .40 D4; finding 62 measured
    ``slot_legality.quant == [False, True, True, True]`` over the four
    floats), so it is ``K - 1``, not 1.
    """
    layout = head_layout(approx_add)
    if layout.width != O_SLOT0_OFFSET + SLOT_WIDTH * layout.n_slots + int(
            layout.has_choose):
        raise ValueError(
            f"head_layout({approx_add!r}) reports width {layout.width}, which "
            f"is not 1 + {SLOT_WIDTH}*{layout.n_slots}"
            f"{' + 1' if layout.has_choose else ''}. The G3 floor counts the "
            f"leaves of THAT arithmetic; a width that does not match it would "
            f"make the floor describe a different head.")
    return {
        "mode": layout.mode,
        "width": int(layout.width),
        "n_slots": int(layout.n_slots),
        "slot_width": int(SLOT_WIDTH),
        "n_pair_idx": int(MAX_PAIR_IDX),
        "n_reduce_axes": int(NUM_REDUCE_AXES),
        "n_reduce_fns": int(NUM_REDUCE_FNS),
        "n_quant_dtypes": int(NUM_FACE_QUANT_DTYPES),
        "n_quant_default": int(NUM_FACE_QUANT_DTYPES) - 1,
        "quant_dtypes": tuple(FACE_QUANT_DTYPES),
        # The number of legal joint outcomes per face if EVERY choice of
        # every slot were legal, skip included.  The upper bound the live
        # masks cut down; logged so a run states the head it ran.
        "max_outcomes_per_face": 1 + (
            1 + MAX_PAIR_IDX * (MAX_PAIR_IDX - 1)
            + NUM_REDUCE_AXES * NUM_REDUCE_FNS
            + (NUM_FACE_QUANT_DTYPES - 1)) ** int(layout.n_slots),
    }


#: ``unified_face_head.O_SLOT0`` under a name that says what it is here.
O_SLOT0_OFFSET = 1


def legal_counts_from_slot_masks(pair, comp, quant, n_faces, *,
                                 op_override=None, face_head_on=True,
                                 approx_add: str = APPROX_ADD_DEFAULT):
    """Per-face, PER-SLOT legal choice counts from the LIVE per-slot masks.

    ``LiveFaceStream.face_slot_legality`` returns ``pair (F, S, N, N)``,
    ``comp (F, S, N)`` and ``quant (F, S, K)`` -- the masks the head is
    actually masked by on the ``--live-faces`` path, per slot and per QUANT
    dtype.  This is the source the G3 floor MUST use on that path: the
    oracle probe :func:`legal_counts_from_masks` reads is switched off
    whenever ``--live-faces`` is on (``ppo._NO_ORACLE``), which is every
    campaign arm -- so before 2026-09-13 every ``gate/g3/*`` floor field
    read NaN in every run that mattered.

    ``S`` comes from the array, and the array's ``S`` comes from
    ``env.face_slot_sites()``, which follows ``--approx-add``; ``K`` comes
    from the array too.  Nothing here is a literal.

    Returns ``(n_choices (F_live, S) int64, skip_legal (F_live,) bool)``.
    """
    geom = face_head_geometry(approx_add)
    pair = np.asarray(pair, dtype=np.float64)
    comp = np.asarray(comp, dtype=np.float64)
    quant = np.asarray(quant, dtype=np.float64)
    if pair.ndim != 4 or comp.ndim != 3 or quant.ndim != 3:
        raise ValueError(
            f"per-slot masks must be (F,S,N,N), (F,S,N) and (F,S,K); got "
            f"{pair.shape}, {comp.shape}, {quant.shape}. A silently reshaped "
            f"mask would give a floor for a head nobody ran.")
    nf = int(n_faces)
    if nf < 0 or nf > pair.shape[0]:
        raise ValueError(f"n_faces {nf} outside the mask's face axis "
                         f"{pair.shape[0]}")
    S = int(pair.shape[1])
    if S < geom["n_slots"]:
        raise ValueError(
            f"the per-slot masks carry {S} slots but --approx-add "
            f"{geom['mode']!r} has {geom['n_slots']}: a slot the width HAS "
            f"would go uncounted in the G3 floor.")
    d_ok, r_ok, q_ok = _op_override_flags(op_override)
    n = np.ones((nf, S), dtype=np.int64)
    for f in range(nf):
        for s in range(S):
            per_slot = 1                      # the None leaf
            if d_ok:
                pm = pair[f, s] > 0.5
                np.fill_diagonal(pm, False)   # j_mask_given_i removes i
                per_slot += int(pm.sum())
            if r_ok:
                per_slot += int((comp[f, s] > 0.5).sum()) * geom["n_reduce_fns"]
            if q_ok:
                per_slot += int((quant[f, s] > 0.5).sum())
            n[f, s] = per_slot
    skip_legal = np.full(nf, bool(face_head_on), dtype=bool)
    return n, skip_legal


def _op_override_flags(op_override):
    if op_override is None:
        return True, True, True
    oo = np.asarray(op_override, dtype=np.float64).reshape(-1)
    return (bool(oo[OP_DIAG] > 0.5) if oo.size > OP_DIAG else True,
            bool(oo[OP_COMPRESS] > 0.5) if oo.size > OP_COMPRESS else True,
            bool(oo[OP_QUANT] > 0.5) if oo.size > OP_QUANT else True)


def legal_counts_from_masks(fpair, fcomp, fvalid, fquant=None,
                            op_override=None, *, n_reduce_fns=None,
                            face_head_on=True,
                            approx_add: str = APPROX_ADD_DEFAULT):
    """Per-face legal choice counts for the RUNNING face head, from the
    oracle's per-face masks (the arrays ``face_masks_all`` carries; ticket
    .59's bottom-up rule, ticket .40's profile override).

    THE ORACLE PATH ONLY.  ``ppo._NO_ORACLE`` is true whenever
    ``--live-faces`` is on, so every campaign arm takes
    :func:`legal_counts_from_slot_masks` instead.

    ``fpair`` (F, N, N) pair legality (already gcd-screened by the env),
    ``fcomp`` (F, N) reduce-axis legality, ``fvalid`` (F,) live faces,
    ``fquant`` (F,) per-face non-identity Quant legality (None = the
    hardware default of one legal cast), ``op_override`` the (NUM_OPS,)
    profile mask in (DIAG, COMPRESS, QUANT, END) order (None = all legal).

    Returns ``(n_choices (F_live, FACE_SLOTS) int64, skip_legal (F_live,)
    bool)``. Per slot: 1 (None) + Diag pairs (ordered, i != j) + reduce
    axes x reduce fns + Quant dtypes, each term present only when its op
    is legal -- exactly the leaves ``UnifiedFaceHead.score`` can reach.
    The three slots share the face's legality here (the static oracle
    path); under --face-slot-frames each slot has its own live masks and
    the floor logged is the static one.
    """
    geom = face_head_geometry(approx_add)
    n_reduce_fns = (geom["n_reduce_fns"] if n_reduce_fns is None
                    else int(n_reduce_fns))
    n_slots = geom["n_slots"]
    n_quant_default = geom["n_quant_default"]
    fpair = np.asarray(fpair, dtype=np.float64)
    fcomp = np.asarray(fcomp, dtype=np.float64)
    fvalid = np.asarray(fvalid, dtype=np.float64).reshape(-1)
    live = np.nonzero(fvalid > 0.5)[0]
    d_ok, r_ok, q_ok = _op_override_flags(op_override)
    fq2 = None
    fq1 = None
    if fquant is not None:
        fq = np.asarray(fquant, dtype=np.float64)
        if fq.ndim == 2 and fq.shape[1] == geom["n_quant_dtypes"]:
            fq2 = fq          # (F, K): an exact per-dtype legality row
        else:
            fq1 = fq.reshape(-1)   # (F,): "QUANT is legal on this face"
    n = np.ones((live.size, n_slots), dtype=np.int64)
    for row, f in enumerate(live):
        per_slot = 1
        if d_ok and fpair.ndim == 3 and f < fpair.shape[0]:
            pm = fpair[f] > 0.5
            np.fill_diagonal(pm, False)          # j_mask_given_i removes i
            per_slot += int(pm.sum())
        if r_ok and fcomp.ndim == 2 and f < fcomp.shape[0]:
            per_slot += int((fcomp[f] > 0.5).sum()) * n_reduce_fns
        if q_ok:
            if fq2 is not None:
                per_slot += int((fq2[f] > 0.5).sum()) if f < fq2.shape[0] else 0
            elif fq1 is not None:
                # A LEGALITY BIT IS NOT A COUNT.  The oracle's per-face QUANT
                # array is (F,) -- "some cast is legal here" -- and the head
                # then offers every dtype but the operand's own, i.e. K - 1
                # of the four floats.  Adding 1 here (the pre-2026-09-13
                # code, written when the set was {float32, bfloat16}) states
                # a floor for a two-dtype head.
                per_slot += (n_quant_default
                             if (f < fq1.size and fq1[f] > 0.5) else 0)
            else:
                per_slot += n_quant_default
        n[row, :] = per_slot
    skip_legal = np.full(live.size, bool(face_head_on), dtype=bool)
    return n, skip_legal


def uniform_floor(n_choices, skip_legal) -> dict:
    """The face-head entropy at the uniform distribution over legal joint
    outcomes, in the two normalisations the trainer uses.

    Per face f with per-slot counts n_s and P = prod_s n_s, the legal joint
    outcomes number N_f = [skip legal] + P. The per-face floor is log N_f.
    The trainer's ``entropy/approx_head`` divides the summed face entropy
    by the ARITY (one per valid face plus one per non-None slot), so the
    comparable floor is sum_f log N_f / sum_f E[arity_f] with, under the
    uniform law, E[arity_f] = 1 + (P / N_f) * sum_s (1 - 1 / n_s).
    """
    n = np.asarray(n_choices, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] == 0:
        return {"per_face": NAN, "arity_norm": NAN, "n_faces": 0,
                "n_outcomes_max": 0}
    sk = np.asarray(skip_legal, dtype=bool).reshape(-1)
    n = np.maximum(n, 1.0)
    P = np.prod(n, axis=1)
    N = P + sk.astype(np.float64)
    logN = np.log(N)
    arity = 1.0 + (P / N) * np.sum(1.0 - 1.0 / n, axis=1)
    return {"per_face": float(logN.mean()),
            "arity_norm": float(logN.sum() / arity.sum()),
            "n_faces": int(n.shape[0]),
            "n_outcomes_max": int(N.max())}


#: ``gate/g3/mask_source`` values.
MASK_SOURCE_NONE, MASK_SOURCE_ORACLE, MASK_SOURCE_LIVE_SLOTS = 0, 1, 2


def g3_face_entropy(face_entropy_nats, n_choices=None, skip_legal=None, *,
                    approx_add: str = APPROX_ADD_DEFAULT,
                    mask_source: int = MASK_SOURCE_NONE) -> dict:
    """G3: the head's entropy against the floor of the head that ran.

    The three geometry rows are DERIVED from ``head_layout(approx_add)`` and
    ``masks.FACE_QUANT_DTYPES`` on every call, so a run states the head its
    floor was computed for and a layout change cannot leave a stale floor
    looking valid.
    """
    geom = face_head_geometry(approx_add)
    fl = (uniform_floor(n_choices, skip_legal)
          if n_choices is not None and skip_legal is not None
          else {"per_face": NAN, "arity_norm": NAN, "n_faces": 0,
                "n_outcomes_max": 0})
    h = float(face_entropy_nats) if face_entropy_nats is not None else NAN
    ratio = (h / fl["arity_norm"]
             if math.isfinite(h) and math.isfinite(fl["arity_norm"])
             and fl["arity_norm"] > 0.0 else NAN)
    return {"gate/g3/face_entropy_nats": h,
            "gate/g3/uniform_floor_nats": fl["arity_norm"],
            "gate/g3/uniform_floor_per_face_nats": fl["per_face"],
            "gate/g3/entropy_over_floor": ratio,
            "gate/g3/n_faces": fl["n_faces"],
            "gate/g3/n_outcomes_max": fl["n_outcomes_max"],
            "gate/g3/n_slots": geom["n_slots"],
            "gate/g3/head_width": geom["width"],
            "gate/g3/n_quant_dtypes": geom["n_quant_dtypes"],
            "gate/g3/mask_source": int(mask_source)}


# ---------------------------------------------------------------------------
# G4 -- the q = 0 fraction, and the feasible fraction under a floor.
# ---------------------------------------------------------------------------
def g4_quality_fractions(quality, tau=None) -> dict:
    q = _finite(quality)
    out = {"gate/g4/n": int(q.size), "gate/g4/q_zero_frac": NAN,
           "gate/g4/tau": NAN if tau is None else float(tau),
           "gate/g4/q_ge_tau_frac": NAN}
    if q.size == 0:
        return out
    out["gate/g4/q_zero_frac"] = float(np.mean(q == 0.0))
    if tau is not None:
        out["gate/g4/q_ge_tau_frac"] = float(np.mean(q >= float(tau)))
    return out


# ---------------------------------------------------------------------------
# G5 -- front spread at the simplex corners.
# ---------------------------------------------------------------------------
def preference_corner(pref, head_names, tol: float = 0.9) -> str | None:
    """The head whose weight carries at least ``tol`` of the preference mass
    (a Dirichlet corner draw, or a static one-hot); None in the interior."""
    p = np.asarray(pref, dtype=np.float64).reshape(-1)
    if p.size == 0 or not np.all(np.isfinite(p)):
        return None
    s = float(np.abs(p).sum())
    if s <= 0.0:
        return None
    j = int(np.argmax(np.abs(p)))
    if abs(p[j]) / s < tol or j >= len(head_names):
        return None
    return str(head_names[j])


# ---------------------------------------------------------------------------
# THE RECORD -> ENV JOIN.  G5 needs the PREFERENCE a plan was measured under,
# and the preference is known per ENV while the plan record is written in the
# measure process, which has no env index.  Two joins, in this order.
#
# 1. THE IDENTITY JOIN.  A record that carries ``env_index`` says outright
#    which env row it belongs to.  Nothing writes that key yet; the report
#    of 2026-09-13 states the env.py / ppo.py change that would.  The branch
#    is here so the identity wins the moment the key exists.
#
# 2. THE REWARD-VECTOR JOIN, in float32.  ``env._callback_measured`` builds
#    the reward slots as PYTHON floats (``_reward_slots``) and hands THAT
#    list to the plan record, then returns ``jnp.array(_reward_slots,
#    dtype=jnp.float32)`` to the trainer, because the callback's declared
#    result dtype is float32 (``env._callback_shape``).  So the two copies
#    of one number are a float64 original and its float32 image, and they
#    are bit-equal only when the original is representable in float32.
#
#    Under ``--cost-form absolute`` every slot came from
#    ``float(jnp.median(...))``, i.e. it was ALREADY a float32 value, and
#    the join worked.  Under ``--cost-form paired-log`` (ticket .9, every
#    2026-09-13 arm) slot 2 and slot 5 are ``math.log(candidate) -
#    math.log(reference)`` computed in float64, which is essentially never
#    float32-exact.  That is why job 65340 joined 0 of 4 records while the
#    drain was complete (4 terminals, 4 records) and the preferences were
#    present (prefs=(4, 3)): the gate compared 0.05163000000000000 against
#    0.05163000151515007.
#
#    The fix is not a tolerance.  The join compares the FLOAT32 IMAGE of
#    both sides, which is the same lossy step the transport already took,
#    so the comparison stays EXACT equality in the space the env row lives
#    in.  It also compares every reward name both sides carry, not two of
#    them, so a tie needs eleven equal slots rather than two.
# ---------------------------------------------------------------------------
#: The key a plan record carries when the process that measured it knows
#: which env row it belongs to.  See docs/GATE_TELEMETRY.md, "The G5 join".
ENV_INDEX_KEY = "env_index"

#: The dtype the terminal reward vector is TRANSPORTED in (env._callback_shape).
JOIN_DTYPE = np.float32

#: gate/g5/join_mode.
JOIN_MODE_NONE = 0        # no env rows, or no comparable column
JOIN_MODE_REWARDS = 1     # the float32 reward-vector join
JOIN_MODE_IDENTITY = 2    # every record carried ENV_INDEX_KEY


def record_env_index(rec) -> int | None:
    """The env row this record names, or None when it names none."""
    v = rec.get(ENV_INDEX_KEY) if hasattr(rec, "get") else None
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i >= 0 else None


def _join_columns(env_names, rec_names, n_cols):
    """``[(env column, record column), ...]`` for every reward name BOTH
    sides carry, in the env's order. Name-keyed, so a reordered or extended
    ``reward_names`` on either side narrows the join instead of breaking it.
    """
    env_names = [str(x) for x in (env_names or ())]
    rec_names = [str(x) for x in (rec_names or ())]
    where: dict = {}
    for j, nm in enumerate(rec_names):
        where.setdefault(nm, j)
    return [(i, where[nm]) for i, nm in enumerate(env_names)
            if i < int(n_cols) and nm in where]


def _record_key(rec, cols):
    """This record's reward vector on ``cols``, in the transport dtype."""
    rews = rec.get("rewards")
    if rews is None or not cols:
        return None
    try:
        v = np.asarray(rews, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if any(j >= v.size for _i, j in cols):
        return None
    return v[[j for _i, j in cols]].astype(JOIN_DTYPE)


def _rows_equal(key, block) -> np.ndarray:
    """Exact equality per env row, with NaN counting as equal to NaN (a NaN
    slot is a measurement that did not happen on BOTH copies, not a
    mismatch)."""
    k = key[None, :]
    return ((block == k) | (np.isnan(block) & np.isnan(k))).all(axis=1)


def _nearest_row(key, block):
    """``(env row, per-column absolute difference)`` of the closest row.

    Closest = smallest LARGEST per-column difference. A column that is NaN
    on one side and a number on the other counts as infinitely far, so a
    row that agrees on ten slots and disagrees on one still reads as the
    near miss it is.
    """
    a = key.astype(np.float64)[None, :]
    b = block.astype(np.float64)
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    d = np.abs(np.where(a_nan | b_nan, 0.0, b - a))
    d = np.where(a_nan ^ b_nan, np.inf, d)
    return int(np.argmin(d.max(axis=1))), d


def match_records_to_envs(records, all_rets, reward_names, *,
                          details=None) -> np.ndarray:
    """Env row of each record, or -1. Each env row is used at most once.

    ``details``, when a dict is passed in, is filled with WHAT THE JOIN
    OBSERVED -- the mode it used, the columns it compared, how many records
    joined, and the first unmatched record beside its nearest env row. That
    is what :func:`join_report` turns into the stderr line, so the gate
    reports a measurement instead of a guess.
    """
    records = list(records or ())
    n = len(records)
    out = np.full(n, -1, dtype=np.int64)
    info: dict = {"mode": JOIN_MODE_NONE, "n_records": n, "n_env_rows": 0,
                  "n_env_cols": 0, "columns": [], "n_joined": 0,
                  "why": "", "first_unmatched": None, "nearest": None}

    def _finish():
        info["n_joined"] = int((out >= 0).sum())
        if details is not None:
            details.update(info)
        return out

    if all_rets is None:
        info["why"] = "the gate was handed no terminal reward rows"
        return _finish()
    A = np.asarray(all_rets, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        info["why"] = (f"the terminal reward rows have shape "
                       f"{np.shape(all_rets)}, not (num_envs, NUM_REWARDS)")
        return _finish()
    info["n_env_rows"], info["n_env_cols"] = int(A.shape[0]), int(A.shape[1])
    used = np.zeros(A.shape[0], dtype=bool)

    # 1. THE IDENTITY JOIN -- all or nothing, so a half-stamped drain does
    #    not silently mix two joins with different failure modes.
    ids = [record_env_index(r) for r in records]
    if n and all(i is not None for i in ids):
        info["mode"] = JOIN_MODE_IDENTITY
        for k, e in enumerate(ids):
            if e < A.shape[0] and not used[e]:
                out[k] = e
                used[e] = True
            elif info["first_unmatched"] is None:
                info["first_unmatched"] = {
                    "record": k, "key": {ENV_INDEX_KEY: e}}
                info["nearest"] = None
                info["why"] = (f"record {k} names env {e}, which is "
                               + ("out of range" if e >= A.shape[0]
                                  else "already taken by an earlier record"))
        return _finish()

    # 2. THE FLOAT32 REWARD-VECTOR JOIN.
    info["mode"] = JOIN_MODE_REWARDS
    _cols_cache: dict = {}
    for k, rec in enumerate(records):
        names = rec.get("reward_names")
        if not names:
            if info["first_unmatched"] is None:
                info["first_unmatched"] = {"record": k, "key": {}}
                info["why"] = "the record carries no reward_names"
            continue
        ckey = tuple(str(x) for x in names)
        if ckey not in _cols_cache:
            cols = _join_columns(reward_names, names, A.shape[1])
            block = (A[:, [i for i, _j in cols]].astype(JOIN_DTYPE)
                     if cols else None)
            _cols_cache[ckey] = (cols, block)
        cols, block = _cols_cache[ckey]
        info["columns"] = [str(reward_names[i]) for i, _j in cols]
        if not cols:
            info["why"] = ("the record and the env rows share no reward "
                           "name, so there is nothing to compare")
            continue
        key = _record_key(rec, cols)
        if key is None:
            if info["first_unmatched"] is None:
                info["first_unmatched"] = {"record": k, "key": {}}
                info["why"] = "the record carries no usable reward vector"
            continue
        hit = np.nonzero((~used) & _rows_equal(key, block))[0]
        if hit.size:
            out[k] = int(hit[0])
            used[hit[0]] = True
            continue
        if info["first_unmatched"] is None:
            e, d = _nearest_row(key, block)
            w = int(np.argmax(d[e]))
            names_c = [str(reward_names[i]) for i, _j in cols]
            raw = np.asarray(rec["rewards"], dtype=np.float64).reshape(-1)
            info["first_unmatched"] = {
                "record": k,
                "key": dict(zip(names_c, [float(x) for x in key]))}
            info["nearest"] = {
                "env": e,
                "key": dict(zip(names_c, [float(x) for x in block[e]])),
                "worst_name": names_c[w],
                "worst_record": float(key[w]),
                # The record's UNCAST slot beside its float32 image. A gap
                # that is visible here and not in `worst_record` is a dtype
                # story, which is the one job 65340 turned out to be.
                "worst_record_raw": float(raw[cols[w][1]]),
                "worst_env": float(block[e][w]),
                "worst_diff": float(d[e][w])}
            info["why"] = (
                f"no env row equals record {k} on all {len(cols)} shared "
                f"reward slot(s)")
    return _finish()


def join_report(details, *, prefs=None, drain=None, n_live=None) -> str:
    """The G5 join, as OBSERVED. Numbers only: no cause is named that this
    episode did not measure."""
    d = dict(details or {})
    mode = {JOIN_MODE_NONE: "none", JOIN_MODE_REWARDS: "reward-vector",
            JOIN_MODE_IDENTITY: f"identity ({ENV_INDEX_KEY})"}.get(
                int(d.get("mode", JOIN_MODE_NONE)), "unknown")
    parts = [f"joined {int(d.get('n_joined', 0))} of "
             f"{int(d.get('n_records', 0))} plan record(s) to an env row",
             f"join={mode}"]
    if n_live is not None:
        parts.append(f"live records={int(n_live)}")
    parts.append(f"env rows={int(d.get('n_env_rows', 0))}"
                 f"x{int(d.get('n_env_cols', 0))}")
    cols = list(d.get("columns") or ())
    parts.append(f"slots compared={len(cols)}"
                 + (f" {cols}" if cols else ""))
    parts.append("prefs=" + ("absent" if prefs is None
                             else str(tuple(np.shape(prefs)))))
    if drain:
        parts.append(
            "drain: local={} pool={} pool_terminals={} undrained={} "
            "actors_failed={}".format(
                int(drain.get("measure/drain/local_records", 0)),
                int(drain.get("measure/drain/pool_records", 0)),
                int(drain.get("measure/drain/pool_terminals", 0)),
                int(drain.get("measure/drain/undrained", 0)),
                int(drain.get("measure/drain/actors_failed", 0))))
    if d.get("why"):
        parts.append(str(d["why"]))
    fu, nr = d.get("first_unmatched"), d.get("nearest")
    if fu:
        parts.append("first unmatched record #{}: {}".format(
            fu.get("record"), _fmt_key(fu.get("key"))))
    if nr:
        parts.append("nearest env row {}: {}".format(
            nr.get("env"), _fmt_key(nr.get("key"))))
        parts.append(
            "largest gap on {}: record {!r} (uncast {!r}) vs env {!r}, "
            "difference {:.3e}".format(
                nr.get("worst_name"), nr.get("worst_record"),
                nr.get("worst_record_raw"), nr.get("worst_env"),
                nr.get("worst_diff", NAN)))
    return "; ".join(parts) + "."


def _fmt_key(key) -> str:
    if not key:
        return "(empty)"
    return " ".join(f"{k}={v!r}" for k, v in key.items())


def _relative_spread(x) -> float:
    """(max - min) / |mean| of the finite entries, NaN below two entries.

    Dimensionless so a latency drift in ns and a memory drift in bytes are
    comparable with a RATIO spread, which is what G5 subtracts them from.
    """
    v = _finite(x)
    if v.size < 2:
        return NAN
    m = float(np.abs(v.mean()))
    if not (m > 0.0):
        return NAN
    return float((v.max() - v.min()) / m)


def g5_front_spread(lat_ratio, temp_ratio, corners, head_names,
                    rev_exact=None, live=None,
                    ref_latency_ns=None, ref_temp_bytes=None) -> dict:
    """Per corner: the best paired ratios among plans measured under it;
    the spread across corners; and the DRIFT FLOOR the spread must beat.

    THE DRIFT FLOOR COMES FROM THE PAIRED REFERENCE, not from candidates
    that happen to be rev-exact.  ``record_is_rev_exact`` is decided on the
    wire -- strictly descending elimination order, no rule, no face row --
    and under ``--fixed-order markowitz`` (every 2026-09-13 arm) NO plan
    ever satisfies it, so the old floor was structurally NaN and G5 had
    nothing to compare against.  Ticket .9 re-measures the SAME rev-exact
    reference once per candidate, in the same actor, back to back: the
    spread of ``ref_latency_ns`` / ``ref_temp_bytes`` within one episode is
    exactly the instrument's own noise on the same computation.  The
    rev-exact-candidate definition is kept beside it under
    ``*_revexact`` for the reverse-order control of ticket .60.
    """
    lat = np.asarray(lat_ratio, dtype=np.float64).reshape(-1)
    temp = np.asarray(temp_ratio, dtype=np.float64).reshape(-1)
    n = lat.size
    corners = list(corners) if corners is not None else [None] * n
    live = (np.ones(n, dtype=bool) if live is None
            else np.asarray(live, dtype=bool).reshape(-1))
    rev = (np.zeros(n, dtype=bool) if rev_exact is None
           else np.asarray(rev_exact, dtype=bool).reshape(-1))
    out = {}
    bests_lat, bests_temp = [], []
    for h in head_names:
        sel = np.array([c == h for c in corners], dtype=bool) & live
        out[f"gate/g5/{h}/n"] = int(sel.sum())
        bl = summary(lat[sel])["best"] if sel.any() else NAN
        bt = summary(temp[sel])["best"] if sel.any() else NAN
        out[f"gate/g5/{h}/best_lat_ratio"] = bl
        out[f"gate/g5/{h}/best_temp_ratio"] = bt
        bests_lat.append(bl)
        bests_temp.append(bt)
    fl, ft = _finite(bests_lat), _finite(bests_temp)
    out["gate/g5/spread_lat"] = float(fl.max() - fl.min()) if fl.size >= 2 else NAN
    out["gate/g5/spread_temp"] = float(ft.max() - ft.min()) if ft.size >= 2 else NAN
    # PRESENT: did ANY live record sit at a named corner?  An arm without
    # --preference-conditioned has one implicit weighting and no simplex to
    # spread over; saying so beats emitting NaN with no reason.
    out["gate/g5/present"] = int(any(
        c is not None for c, lv in zip(corners, live) if lv))
    rsel = rev & live
    out["gate/g5/n_rev_exact"] = int(rsel.sum())
    rl, rt = _finite(lat[rsel]), _finite(temp[rsel])
    out["gate/g5/drift_floor_lat_revexact"] = (
        float(rl.max() - rl.min()) if rl.size >= 2 else NAN)
    out["gate/g5/drift_floor_temp_revexact"] = (
        float(rt.max() - rt.min()) if rt.size >= 2 else NAN)
    rlat = (np.full(n, NAN) if ref_latency_ns is None
            else np.asarray(ref_latency_ns, dtype=np.float64).reshape(-1))
    rtmp = (np.full(n, NAN) if ref_temp_bytes is None
            else np.asarray(ref_temp_bytes, dtype=np.float64).reshape(-1))
    out["gate/g5/drift_floor_lat"] = _relative_spread(rlat[live])
    out["gate/g5/drift_floor_temp"] = _relative_spread(rtmp[live])
    out["gate/g5/drift_floor_n"] = int(_finite(rlat[live]).size)
    out["gate/g5/n_unmatched"] = int(sum(1 for c, lv in zip(corners, live)
                                         if lv and c is None))
    return out


# ---------------------------------------------------------------------------
# DRAIN PROVENANCE -- ticket .7's failure mode, which the .45 contract names.
# ---------------------------------------------------------------------------
class DrainProvenanceError(ValueError):
    """A counter was read in a process that does not own it.

    Ticket .7: ``grad_cov/*`` incremented inside the Ray measure actors and
    was drained in the trainer, so it read 0 for all 16,192 plans of wave 1
    and nothing said so.  The same shape reappeared in the cancelled canary
    65319: ``pool_terminals=16`` beside ``pooled=0``.
    """


def drain_provenance(local=None, pool=None, *, strict: bool = False) -> dict:
    """Audit one episode's plan-record drain against what the actors say
    they measured.

    ``local`` is ``env.consume_plan_records()``'s own summary (the trainer's
    process), ``pool`` is ``measure_pool.merge_pool_plan_records``'s.  The
    actors count their terminals themselves; if the records that reached
    this process do not account for them, a counter incremented in the
    actor is being read here and the telemetry built on it is fiction.

    ``strict=True`` raises :class:`DrainProvenanceError` instead of
    reporting, for the test that must FAIL on the .7 shape.
    """
    local = dict(local or {})
    pool = dict(pool or {})
    n_local = len(list(local.get("records") or ()))
    n_pool = len(list(pool.get("records") or ()))
    terminals = int(pool.get("terminals", 0) or 0)
    dropped = int(pool.get("dropped", 0) or 0)
    seen = int(pool.get("actors_seen", 0) or 0)
    failed = int(pool.get("actors_failed", 0) or 0)
    undrained = max(0, terminals - (n_pool + dropped))
    ok = int(undrained == 0 and failed == 0)
    if strict and not ok:
        raise DrainProvenanceError(
            f"the measure actors report {terminals} terminal plan(s) but "
            f"{n_pool} record(s) (+{dropped} dropped) reached this process "
            f"and {failed} actor(s) could not be polled: {undrained} plan(s) "
            f"are counted where they are NOT owned (ticket .7). Every gate "
            f"field computed from this drain would understate by that many.")
    return {"measure/drain/local_records": n_local,
            "measure/drain/pool_records": n_pool,
            "measure/drain/pool_terminals": terminals,
            "measure/drain/undrained": int(undrained),
            "measure/drain/ok": ok,
            "measure/drain/actors_seen": seen,
            "measure/drain/actors_failed": failed}


# ---------------------------------------------------------------------------
# G6 -- the pre-run offline contrast (ticket .42's number, a flag for now).
# ---------------------------------------------------------------------------
def g6_offline_contrast(value) -> dict:
    if value is None:
        return {"gate/g6/present": 0, "gate/g6/offline_contrast": NAN}
    try:
        v = float(value)
    except (TypeError, ValueError):
        return {"gate/g6/present": 0, "gate/g6/offline_contrast": NAN}
    return {"gate/g6/present": 1, "gate/g6/offline_contrast": v}


# ---------------------------------------------------------------------------
# The toolchain fingerprint (wandb.config, once per run).
# ---------------------------------------------------------------------------
def toolchain_fingerprint(*, sparse=None) -> dict:
    out = {"toolchain/xla_flags": os.environ.get("XLA_FLAGS", ""),
           "toolchain/jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
           "toolchain/sparse": (-1 if sparse is None else int(bool(sparse)))}
    try:
        import socket
        out["toolchain/hostname"] = socket.gethostname()
    except Exception:
        out["toolchain/hostname"] = "unknown"
    try:
        import jax
        out["toolchain/jax"] = str(jax.__version__)
        try:
            out["toolchain/backend"] = str(jax.default_backend())
        except Exception:
            out["toolchain/backend"] = "unknown"
    except Exception:
        out["toolchain/jax"] = "unknown"
        out["toolchain/backend"] = "unknown"
    try:
        import jaxlib
        out["toolchain/jaxlib"] = str(jaxlib.__version__)
    except Exception:
        out["toolchain/jaxlib"] = "unknown"
    return out


# ---------------------------------------------------------------------------
# The critic stash: train_episode hands the value target / prediction pair
# (and the per-env preference) to the host through jax.debug.callback; the
# per-episode logging pops it. Read-only copies; nothing reads them back.
# ---------------------------------------------------------------------------
_CRITIC: dict = {}


def stash_critic(targets, predictions, preference) -> None:
    _CRITIC["targets"] = np.array(targets, dtype=np.float32, copy=True)
    _CRITIC["predictions"] = np.array(predictions, dtype=np.float32, copy=True)
    _CRITIC["preference"] = np.array(preference, dtype=np.float32, copy=True)


def pop_critic() -> dict | None:
    if not _CRITIC:
        return None
    out = dict(_CRITIC)
    _CRITIC.clear()
    return out


def env_preferences(critic) -> np.ndarray | None:
    """(num_envs, H) preference per env from the stashed (E, T, H) array
    (constant along T within an episode)."""
    if not critic or critic.get("preference") is None:
        return None
    p = np.asarray(critic["preference"], dtype=np.float64)
    if p.ndim == 3:
        return p[:, 0, :]
    if p.ndim == 2:
        return p
    return None


# ---------------------------------------------------------------------------
# The per-episode entry point.
# ---------------------------------------------------------------------------
def episode_fields(records, *, head_names, all_rets=None, reward_names=None,
                   critic=None, face_entropy_nats=None, legal=None,
                   winners=None, vertex_primitive=None, quality_floor=None,
                   offline_contrast=None, corner_tol: float = 0.9,
                   approx_add: str = APPROX_ADD_DEFAULT,
                   mask_source: int = MASK_SOURCE_NONE,
                   drain_local=None, drain_pool=None) -> dict:
    """Every gate field for one episode. Missing inputs give NaN / 0 /
    absent-flags, never an exception from a missing piece.

    ``records`` -- the drained plan records (trainer + pool) of this episode.
    ``all_rets`` -- (num_envs, NUM_REWARDS) terminal reward rows, to join
    records to envs (and so to preferences). ``critic`` -- the popped stash
    (targets, predictions, preference). ``legal`` -- ``(n_choices,
    skip_legal)`` from :func:`legal_counts_from_masks`. ``winners`` -- rows
    from :func:`load_winners_table`.
    """
    records = list(records or ())
    pr = paired_ratios(records)
    live = pr["live"]
    out: dict = {}
    out["paired/n"] = int(live.sum())
    out["paired/n_with_ref"] = int((pr["has_ref"] & live).sum())
    for key, arr, best in (("lat_ratio", pr["lat_ratio"], "min"),
                           ("temp_ratio", pr["temp_ratio"], "min"),
                           ("watermark_ratio", pr["watermark_ratio"], "min"),
                           ("grad_cosine", pr["quality"], "max")):
        st = summary(arr[live], best=best)
        out[f"paired/{key}_mean"] = st["mean"]
        out[f"paired/{key}_median"] = st["median"]
        out[f"paired/{key}_best"] = st["best"]
    out["paired/n_with_watermark"] = int(
        np.isfinite(pr["watermark_ratio"][live]).sum())
    out["paired/ref_latency_ns_mean"] = summary(
        pr["ref_latency_ns"][live])["mean"]
    out["paired/ref_temp_bytes_mean"] = summary(
        pr["ref_temp_bytes"][live])["mean"]
    out.update(g1_recovery(records, winners, vertex_primitive))
    out.update(explained_variance_per_head(
        None if not critic else critic.get("targets"),
        None if not critic else critic.get("predictions"), head_names))
    n_choices, skip_legal = (legal if legal is not None else (None, None))
    out.update(g3_face_entropy(face_entropy_nats, n_choices, skip_legal,
                               approx_add=approx_add,
                               mask_source=mask_source))
    out.update(g4_quality_fractions(pr["quality"][live], quality_floor))
    prefs = env_preferences(critic)
    drain = drain_provenance(drain_local, drain_pool)
    _join: dict = {}
    env_of = match_records_to_envs(records, all_rets, reward_names,
                                   details=_join)
    out["gate/g5/n_joined"] = int(_join.get("n_joined", 0))
    out["gate/g5/join_mode"] = int(_join.get("mode", JOIN_MODE_NONE))
    corners = []
    for k in range(len(records)):
        e = int(env_of[k])
        if prefs is None or e < 0 or e >= prefs.shape[0]:
            corners.append(None)
        else:
            corners.append(preference_corner(prefs[e], head_names, corner_tol))
    # WHY G5 IS EMPTY, MEASURED RATHER THAN GUESSED.  The line this replaced
    # blamed a lagging measure-actor drain, and job 65340 disproved that on
    # the spot: the drain was complete (4 terminals, 4 records), the
    # preferences were there (prefs=(4, 3)), and the join still returned
    # nothing, because it compared a float64 reward slot against its float32
    # image (see THE RECORD -> ENV JOIN above).  `join_report` prints what
    # this episode actually observed -- the drain counts, the preference
    # shape, how many of how many records joined, and the first unmatched
    # record beside the env row nearest to it -- and names no cause the
    # numbers do not carry.
    _n_live = int(live.sum())
    if _n_live and all(c is None for c, lv in zip(corners, live) if lv):
        print_reason("G5 corners empty",
                     join_report(_join, prefs=prefs, drain=drain,
                                 n_live=_n_live))
    out.update(g5_front_spread(pr["lat_ratio"], pr["temp_ratio"], corners,
                               head_names, rev_exact=pr["rev_exact"],
                               live=live,
                               ref_latency_ns=pr["ref_latency_ns"],
                               ref_temp_bytes=pr["ref_temp_bytes"]))
    out.update(g6_offline_contrast(offline_contrast))
    out.update(drain)
    missing = documented_fields(head_names) - set(out)
    if missing:
        # THE CONTRACT IS THE TABLE.  A field the table names and this
        # function does not emit is a silently absent gate input, which the
        # error policy of tickets .43/.45 forbids outright.
        raise KeyError(
            f"gate telemetry did not emit {len(missing)} documented "
            f"field(s): {sorted(missing)}. Either emit them or take them out "
            f"of FIELD_TABLE -- a gate must not read a name nobody writes.")
    return out


def add_gate_args(p) -> None:
    """The two gate inputs that are files/numbers rather than measurements."""
    p.add_argument(
        "--gate-winners-table", type=str, default=None,
        help="Gate G1 (ticket .45): CSV or JSON of the sweep winners "
        "(ticket .41) with columns vertex (76 / v76 / v76/add), primitive "
        "(optional) and kind (skip/diag/reduce/quant). Per episode the "
        "gate/g1/* panels report which (vertex, class) pairs some plan "
        "re-applied. Absent (the default, the sweep has not run): "
        "gate/g1/present = 0 and the reason is printed once.")
    p.add_argument(
        "--gate-offline-contrast", type=float, default=None,
        help="Gate G6 (ticket .45): ticket .42's offline contrast for this "
        "arm's reward, copied onto gate/g6/offline_contrast every episode "
        "so the run carries the number it is judged against. Absent: "
        "gate/g6/present = 0.")


def print_reason(tag: str, reason: str) -> None:
    print(f"[gate] {tag}: {reason}", file=sys.stderr, flush=True)
