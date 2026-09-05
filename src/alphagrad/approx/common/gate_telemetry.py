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
REF_TEMP_KEY = "ref_mem_temp_bytes"
# Ticket .49's memory fields on the record (always present since .49).
TEMP_KEY = "mem_temp_bytes"
# Ticket .9's quality floor flag on ``args`` (None = no floor set).
QUALITY_FLOOR_ATTR = "quality_floor"

# The env's sentinel for a failed measurement: every cost channel reads
# -1e10. Same 0.99 test the trainer's LIVE mask uses.
SENTINEL_COST = -1e10
_SENTINEL_EDGE = SENTINEL_COST * 0.99

# Head layout of the 94-logit face head (unified_face_head.py): a skip
# Bernoulli, then FACE_SLOTS slots of (op, i, j, axis, reduce_fn, dtype).
FACE_SLOTS = 3
NUM_REDUCE_FNS = 5
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
    ("gate/g5/n_rev_exact", "count", "live records whose plan IS rev-exact (reverse order, no approximation)", "G5"),
    ("gate/g5/drift_floor_lat", "ratio", "max - min of the latency ratio over those rev-exact records (their ratio is pure drift)", "G5"),
    ("gate/g5/drift_floor_temp", "ratio", "the same for the temp ratio (0 when the static temp is deterministic)", "G5"),
    ("gate/g5/n_unmatched", "count", "records that matched no env row (no preference known)", "G5"),
    # G6
    ("gate/g6/present", "0/1", "1 when --gate-offline-contrast was given", "G6"),
    ("gate/g6/offline_contrast", "fraction", "ticket .42's offline contrast, copied from the flag; NaN when absent", "G6"),
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


def record_latency_ns(rec: dict) -> float:
    """POSITIVE measured latency in ns (the reward slot stores -ns)."""
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
    """Per-record arrays: latency ratio, temp ratio, quality, liveness,
    rev-exactness. Ratios are NaN where the record lacks .9's reference."""
    n = len(records)
    lat = np.full(n, NAN)
    temp = np.full(n, NAN)
    q = np.full(n, NAN)
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
        if math.isfinite(c_lat) and math.isfinite(r_lat) and r_lat > 0.0:
            lat[k] = c_lat / r_lat
            has_ref[k] = True
        c_t, r_t = record_temp_bytes(rec), _get_float(rec, REF_TEMP_KEY)
        if math.isfinite(c_t) and math.isfinite(r_t) and r_t > 0.0:
            temp[k] = c_t / r_t
    return {"lat_ratio": lat, "temp_ratio": temp, "quality": q,
            "live": live, "rev_exact": rev, "has_ref": has_ref}


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
def legal_counts_from_masks(fpair, fcomp, fvalid, fquant=None,
                            op_override=None, *, n_reduce_fns=NUM_REDUCE_FNS,
                            face_head_on=True):
    """Per-face legal choice counts for the 94-logit head, from the oracle's
    per-face masks (the arrays ``face_masks_all`` carries; ticket .59's
    bottom-up rule, ticket .40's profile override).

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
    fpair = np.asarray(fpair, dtype=np.float64)
    fcomp = np.asarray(fcomp, dtype=np.float64)
    fvalid = np.asarray(fvalid, dtype=np.float64).reshape(-1)
    live = np.nonzero(fvalid > 0.5)[0]
    if op_override is None:
        d_ok, r_ok, q_ok = True, True, True
    else:
        oo = np.asarray(op_override, dtype=np.float64).reshape(-1)
        d_ok = bool(oo[OP_DIAG] > 0.5) if oo.size > OP_DIAG else True
        r_ok = bool(oo[OP_COMPRESS] > 0.5) if oo.size > OP_COMPRESS else True
        q_ok = bool(oo[OP_QUANT] > 0.5) if oo.size > OP_QUANT else True
    n = np.ones((live.size, FACE_SLOTS), dtype=np.int64)
    for row, f in enumerate(live):
        per_slot = 1
        if d_ok and fpair.ndim == 3 and f < fpair.shape[0]:
            pm = fpair[f] > 0.5
            np.fill_diagonal(pm, False)          # j_mask_given_i removes i
            per_slot += int(pm.sum())
        if r_ok and fcomp.ndim == 2 and f < fcomp.shape[0]:
            per_slot += int((fcomp[f] > 0.5).sum()) * int(n_reduce_fns)
        if q_ok:
            if fquant is None:
                per_slot += 1
            else:
                fq = np.asarray(fquant, dtype=np.float64).reshape(-1)
                per_slot += int(fq[f] > 0.5) if f < fq.size else 0
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


def g3_face_entropy(face_entropy_nats, n_choices=None, skip_legal=None) -> dict:
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
            "gate/g3/n_outcomes_max": fl["n_outcomes_max"]}


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


def match_records_to_envs(records, all_rets, reward_names) -> np.ndarray:
    """Env row of each record, or -1. A record carries the terminal reward
    vector it was measured with; ``all_rets`` (num_envs, NUM_REWARDS) holds
    the same vectors by env, so equality on latency_ns and quality is the
    join (bit-equal floats from one host array). Each env row is used once."""
    n = len(records)
    out = np.full(n, -1, dtype=np.int64)
    if all_rets is None:
        return out
    A = np.asarray(all_rets, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        return out
    il, iq = _idx(reward_names, "latency_ns"), _idx(reward_names, "quality")
    if il is None or iq is None or A.shape[1] <= max(il, iq):
        return out
    used = np.zeros(A.shape[0], dtype=bool)
    for k, rec in enumerate(records):
        names, rews = rec.get("reward_names"), rec.get("rewards")
        if not names or rews is None:
            continue
        jl, jq = _idx(names, "latency_ns"), _idx(names, "quality")
        if jl is None or jq is None or max(jl, jq) >= len(rews):
            continue
        try:
            lat, q = float(rews[jl]), float(rews[jq])
        except (TypeError, ValueError):
            continue
        hit = np.nonzero((~used) & (A[:, il] == lat) & (A[:, iq] == q))[0]
        if hit.size:
            out[k] = int(hit[0])
            used[hit[0]] = True
    return out


def g5_front_spread(lat_ratio, temp_ratio, corners, head_names,
                    rev_exact=None, live=None) -> dict:
    """Per corner: the best paired ratios among plans measured under it;
    the spread across corners; the live drift floor from rev-exact plans."""
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
    rsel = rev & live
    out["gate/g5/n_rev_exact"] = int(rsel.sum())
    rl, rt = _finite(lat[rsel]), _finite(temp[rsel])
    out["gate/g5/drift_floor_lat"] = float(rl.max() - rl.min()) if rl.size >= 2 else NAN
    out["gate/g5/drift_floor_temp"] = float(rt.max() - rt.min()) if rt.size >= 2 else NAN
    out["gate/g5/n_unmatched"] = int(sum(1 for c, lv in zip(corners, live)
                                         if lv and c is None))
    return out


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
                   offline_contrast=None, corner_tol: float = 0.9) -> dict:
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
                           ("grad_cosine", pr["quality"], "max")):
        s = summary(arr[live], best=best)
        out[f"paired/{key}_mean"] = s["mean"]
        out[f"paired/{key}_median"] = s["median"]
        out[f"paired/{key}_best"] = s["best"]
    out.update(g1_recovery(records, winners, vertex_primitive))
    out.update(explained_variance_per_head(
        None if not critic else critic.get("targets"),
        None if not critic else critic.get("predictions"), head_names))
    n_choices, skip_legal = (legal if legal is not None else (None, None))
    out.update(g3_face_entropy(face_entropy_nats, n_choices, skip_legal))
    out.update(g4_quality_fractions(pr["quality"][live], quality_floor))
    prefs = env_preferences(critic)
    env_of = match_records_to_envs(records, all_rets, reward_names)
    corners = []
    for k in range(len(records)):
        e = int(env_of[k])
        if prefs is None or e < 0 or e >= prefs.shape[0]:
            corners.append(None)
        else:
            corners.append(preference_corner(prefs[e], head_names, corner_tol))
    out.update(g5_front_spread(pr["lat_ratio"], pr["temp_ratio"], corners,
                               head_names, rev_exact=pr["rev_exact"],
                               live=live))
    out.update(g6_offline_contrast(offline_contrast))
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
