"""Offline reward scoring (dsnn-3qm.42): choose lambda, tau and eta before PPO.

Scores every plan we already measured under the planned paired reward and
reports, per (lambda_q, tau) grid point, the CONTRAST that gate G6 asks for:

    contrast = R(best feasible plan) - max(R(absorber), R(baseline))

with ``feasible`` = quality >= tau, ``absorber`` = the best-scoring plan whose
quality is at or below ``--absorber-q`` (a plan that kills the gradient and
pays nothing), and ``baseline`` = the exact plan of the order being scored.

The reward is the trainer's (ticket .9, ``--cost-form paired-log``):

    Delta_c = log cost_c(candidate) - log cost_c(reference)
    P0:  R = -lambda_lat * Delta_lat - lambda_mem * Delta_mem + lambda_q * q
    P1:  R = -lambda_lat * Delta_lat - lambda_mem * Delta_mem - lambda_q * max(0, tau - q)

Two kinds of input:

* ``--sweep-summary summary_COMBINED.json`` (landscape_map ``--report-only``):
  one entry per plan and GPU config with ``latency_ratio`` and
  ``static_temp_ratio`` against the PAIRED reverse-exact reference, and
  ``quality``. ``--reference identity`` re-bases the ratios on the same order's
  own exact plan (the ``identity`` entry), which is the baseline a fixed-order
  campaign competes against; ``--reference rev-exact`` keeps them as measured.
* ``--plan-log plan_log_*.jsonl`` (schema 1 or 2): absolute ``latency_ns`` /
  ``peak_memory`` rewards. The reference is the median of the log's own exact
  plans (``requested.total == 0``), so these are same-order ratios.

eta (the RCPO dual-ascent step ``lam <- lam + eta * mean_violation``) cannot
be fixed from static data alone. The report gives the violation statistics
of the plan population at each tau and the eta that moves lambda_q by
``--eta-fraction`` of itself per episode at that mean violation; the choice
is the owner's.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """One measured plan: paired log-costs and quality."""
    plan_id: str
    source: str
    kind: str
    dlat: float        # log(latency / reference latency)
    dmem: float        # log(temp memory / reference temp memory)
    q: float


def _num(x) -> float:
    if isinstance(x, dict):
        return float(x["mean"])
    return float(x)


def _kind_of(plan_id: str) -> str:
    parts = plan_id.split(":")
    if parts[0] in ("pair", "stack"):
        return parts[0]
    if parts[0] == "identity" or parts[0].startswith("noisefloor"):
        return "identity"
    return {"reduce": "compress"}.get(parts[1], parts[1]) if len(parts) > 1 else parts[0]


#: The trainer's floor under log() for the memory channel
#: (``env._MEM_LOG_FLOOR_BYTES``): one byte. A plan that allocates nothing
#: therefore gains ``log(ref_temp / 1 B)`` nats on the memory channel.
TRAINER_MEM_FLOOR_BYTES = 1.0


@dataclass(frozen=True)
class Floors:
    """Absolute cost floors applied to BOTH sides before the log.

    ``lat_ns`` / ``temp_bytes`` of ``None`` mean "no floor" (latency) and the
    trainer's one-byte floor (memory). Passing the reverse-exact costs here
    asks: what does the reward look like if nothing below the cheapest exact
    plan earns credit?"""
    lat_ns: float | None = None
    temp_bytes: float | None = None

    def lat(self, x: float) -> float:
        return x if self.lat_ns is None else max(x, self.lat_ns)

    def mem(self, x: float) -> float:
        return max(x, TRAINER_MEM_FLOOR_BYTES if self.temp_bytes is None else self.temp_bytes)


def points_from_sweep_summary(path: str, *, reference: str = "identity",
                              source: str | None = None,
                              floors: Floors = Floors()) -> list[Point]:
    """Per-plan points from a landscape_map combined summary.

    Works on the ABSOLUTE per-plan costs (``latency_ns``, ``static_temp``)
    so the floors apply; ``reference='identity'`` divides by the order's own
    identity (median over configs), ``'rev-exact'`` by the paired reverse-exact
    cost (identity cost / identity ratio).
    """
    summ = json.load(open(path))["summary"]
    per: dict[str, list] = {}
    for key, v in summ.items():
        pid = re.sub(r"\s*\[.*\]$", "", key)
        per.setdefault(pid, []).append(v)
    if "identity" not in per:
        raise ValueError(f"{path}: no 'identity' entry, cannot re-base")
    ident = per["identity"]
    lat_id = statistics.median(_num(v["latency_ns"]) for v in ident)
    mem_id = statistics.median(_num(v["static_temp"]) for v in ident)
    if reference == "identity":
        lat_ref, mem_ref = lat_id, mem_id
    elif reference == "rev-exact":
        lat_ref = lat_id / statistics.median(_num(v["latency_ratio"]) for v in ident)
        mem_ref = mem_id / statistics.median(_num(v["static_temp_ratio"]) for v in ident)
    else:
        raise ValueError(f"reference must be 'identity' or 'rev-exact', got {reference!r}")
    lat_ref, mem_ref = floors.lat(lat_ref), floors.mem(mem_ref)
    src = source or path
    out = []
    for pid, vals in per.items():
        lat = floors.lat(statistics.median(_num(v["latency_ns"]) for v in vals))
        mem = floors.mem(statistics.median(_num(v["static_temp"]) for v in vals))
        q = statistics.median(_num(v["quality"]) for v in vals)
        if not (lat > 0) or math.isnan(q):
            print(f"[score] drop {pid}: lat={lat:g} mem={mem:g} q={q:g}", file=sys.stderr)
            continue
        out.append(Point(pid, src, _kind_of(pid), math.log(lat / lat_ref), math.log(mem / mem_ref), q))
    return out


def absorber_point(lat_ns: float, temp_bytes: float, q: float, *, ref_lat_ns: float,
                   ref_temp_bytes: float, floors: Floors = Floors(),
                   source: str = "measured") -> Point:
    """The measured skip-everything plan as a point against an explicit
    reference (the order's identity), under the same floors."""
    lat = floors.lat(lat_ns) / floors.lat(ref_lat_ns)
    mem = floors.mem(temp_bytes) / floors.mem(ref_temp_bytes)
    return Point("absorber:skip@all", source, "skip", math.log(lat), math.log(mem), q)


def points_from_plan_log(path: str, *, source: str | None = None,
                         floors: Floors = Floors()) -> list[Point]:
    """Per-plan points from a plan log; the reference is the log's own exact
    plans (median latency / memory of every record with no request)."""
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not recs:
        raise ValueError(f"{path}: no records")
    names = recs[0]["reward_names"]
    i_lat, i_mem, i_q = names.index("latency_ns"), names.index("peak_memory"), names.index("quality")
    exact = [r for r in recs if not r.get("sentinelled") and r["requested"]["total"] == 0]
    if not exact:
        raise ValueError(f"{path}: no exact plan (requested.total == 0) to serve as reference")
    lat0 = floors.lat(statistics.median(-float(r["rewards"][i_lat]) for r in exact))
    mem0 = floors.mem(statistics.median(-float(r["rewards"][i_mem]) for r in exact))
    src = source or path
    out = []
    for k, r in enumerate(recs):
        if r.get("sentinelled"):
            continue
        lat = floors.lat(-float(r["rewards"][i_lat]))
        mem = floors.mem(-float(r["rewards"][i_mem]))
        q = float(r["rewards"][i_q])
        if not (lat > 0):
            continue
        req = r["requested"]
        kind = "identity" if req["total"] == 0 else "plan"
        out.append(Point(f"{src}#{k}", src, kind, math.log(lat / lat0), math.log(mem / mem0), q))
    return out


def reward(p: Point, *, form: str, lam_lat: float, lam_mem: float, lam_q: float, tau: float) -> float:
    cost = -lam_lat * p.dlat - lam_mem * p.dmem
    if form == "P0":
        return cost + lam_q * p.q
    if form == "P1":
        return cost - lam_q * max(0.0, tau - p.q)
    raise ValueError(f"form must be 'P0' or 'P1', got {form!r}")


def score_grid(points: list[Point], *, forms=("P0", "P1"), lam_lat: float = 1.0,
               lam_mem: float = 1.0, lam_qs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
               taus=(0.8, 0.9, 0.95), absorber_q: float = 0.05,
               eta_fraction: float = 0.1) -> list[dict]:
    """One row per (form, lambda_q, tau): the contrast and what made it."""
    baseline = [p for p in points if p.kind == "identity"]
    if not baseline:
        raise ValueError("no identity / exact plan among the points: the baseline is undefined")
    rows = []
    for form in forms:
        for lam_q in lam_qs:
            for tau in taus:
                kw = dict(form=form, lam_lat=lam_lat, lam_mem=lam_mem, lam_q=lam_q, tau=tau)
                r_base = max(reward(p, **kw) for p in baseline)
                absorbers = [p for p in points if p.q <= absorber_q]
                r_abs = max((reward(p, **kw) for p in absorbers), default=float("-inf"))
                feasible = [p for p in points if p.q >= tau and p.kind != "identity"]
                best = max(feasible, key=lambda p: reward(p, **kw), default=None)
                r_best = reward(best, **kw) if best is not None else float("-inf")
                floor = max(r_abs, r_base)
                contrast = r_best - floor
                viol = [max(0.0, tau - p.q) for p in points if p.kind != "identity"]
                mean_viol = statistics.mean(viol) if viol else 0.0
                frac_viol = (sum(1 for v in viol if v > 0) / len(viol)) if viol else 0.0
                eta = (eta_fraction * lam_q / mean_viol) if mean_viol > 0 else float("nan")
                rows.append({
                    "form": form, "lambda_q": lam_q, "tau": tau,
                    "R_baseline": r_base, "R_absorber": r_abs, "R_best": r_best,
                    "best_plan": None if best is None else best.plan_id,
                    "best_q": None if best is None else best.q,
                    "best_dlat": None if best is None else best.dlat,
                    "best_dmem": None if best is None else best.dmem,
                    "contrast": contrast,
                    "contrast_rel": (contrast / abs(floor)) if floor not in (0.0, float("-inf")) else float("nan"),
                    "n_feasible": len(feasible), "n_absorbers": len(absorbers),
                    "mean_violation": mean_viol, "frac_violating": frac_viol,
                    "eta_suggested": eta,
                })
    return rows


def render_markdown(rows: list[dict], points: list[Point], title: str) -> str:
    by_src: dict[str, int] = {}
    for p in points:
        by_src[p.source] = by_src.get(p.source, 0) + 1
    lines = [f"# {title}", "",
             f"{len(points)} points from {len(by_src)} source(s): "
             + ", ".join(f"{k} ({v})" for k, v in by_src.items()), "",
             "contrast = R(best feasible) - max(R(absorber), R(baseline)); feasible = q >= tau; "
             "absorber = best plan with q <= absorber_q; eta_suggested moves lambda_q by the "
             "eta fraction of itself per episode at the population's mean violation.", "",
             "| form | lambda_q | tau | R_base | R_abs | R_best | contrast | rel | feasible | mean viol | frac viol | eta | best plan |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['form']} | {r['lambda_q']:g} | {r['tau']:g} | {r['R_baseline']:+.3f} | "
            f"{r['R_absorber']:+.3f} | {r['R_best']:+.3f} | {r['contrast']:+.3f} | "
            f"{r['contrast_rel']:+.2f} | {r['n_feasible']} | {r['mean_violation']:.3f} | "
            f"{r['frac_violating']:.2f} | {r['eta_suggested']:.2f} | `{r['best_plan']}` |")
    return "\n".join(lines) + "\n"


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-summary", action="append", default=[],
                   help="landscape_map summary_COMBINED.json (repeatable; NAME=PATH labels the source)")
    p.add_argument("--plan-log", action="append", default=[],
                   help="plan_log_*.jsonl (repeatable; NAME=PATH labels the source)")
    p.add_argument("--reference", choices=("identity", "rev-exact"), default="identity",
                   help="reference for sweep summaries: the order's own exact plan, or the paired rev-exact")
    p.add_argument("--lambda-lat", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    p.add_argument("--lambda-q", type=str, default="1,2,3,4,5,6,8")
    p.add_argument("--tau", type=str, default="0.8,0.9,0.95")
    p.add_argument("--forms", type=str, default="P0,P1")
    p.add_argument("--absorber-q", type=float, default=0.05)
    p.add_argument("--floor-lat-ns", type=float, default=None,
                   help="clip every latency at this floor before the log (default: none)")
    p.add_argument("--floor-temp-bytes", type=float, default=None,
                   help="clip every temp memory at this floor before the log "
                        "(default: the trainer's one byte)")
    p.add_argument("--absorber", action="append", default=[],
                   help="a measured skip-everything plan for one source: "
                        "SOURCE=lat_ns,temp_bytes,q,ref_lat_ns,ref_temp_bytes (repeatable)")
    p.add_argument("--eta-fraction", type=float, default=0.1)
    p.add_argument("--separate", action="store_true",
                   help="score each source on its own instead of the union")
    p.add_argument("--out", type=str, default=None, help="write markdown here (and .json beside it)")
    p.add_argument("--title", type=str, default="Offline reward scoring (dsnn-3qm.42)")
    return p


def _split_label(spec: str) -> tuple[str | None, str]:
    if "=" in spec and not spec.split("=", 1)[0].startswith("/"):
        name, path = spec.split("=", 1)
        return name, path
    return None, spec


def main(argv=None) -> int:
    args = make_argparser().parse_args(argv)
    floors = Floors(lat_ns=args.floor_lat_ns, temp_bytes=args.floor_temp_bytes)
    groups: dict[str, list[Point]] = {}
    for spec in args.sweep_summary:
        name, path = _split_label(spec)
        pts = points_from_sweep_summary(path, reference=args.reference, source=name,
                                        floors=floors)
        groups[name or path] = pts
    for spec in args.plan_log:
        name, path = _split_label(spec)
        groups[name or path] = points_from_plan_log(path, source=name, floors=floors)
    for spec in args.absorber:
        name, rest = spec.split("=", 1)
        lat, mem, q, rlat, rmem = (float(x) for x in rest.split(","))
        if name not in groups:
            raise SystemExit(f"--absorber {name}: no such source among {sorted(groups)}")
        groups[name].append(absorber_point(lat, mem, q, ref_lat_ns=rlat, ref_temp_bytes=rmem,
                                           floors=floors, source=name))
    if not groups:
        raise SystemExit("nothing to score: pass --sweep-summary and/or --plan-log")
    lam_qs = tuple(float(x) for x in args.lambda_q.split(","))
    taus = tuple(float(x) for x in args.tau.split(","))
    forms = tuple(args.forms.split(","))
    sections = []
    targets = groups.items() if args.separate else [("union", [p for v in groups.values() for p in v])]
    all_rows = {}
    for label, pts in targets:
        rows = score_grid(pts, forms=forms, lam_lat=args.lambda_lat, lam_mem=args.lambda_mem,
                          lam_qs=lam_qs, taus=taus, absorber_q=args.absorber_q,
                          eta_fraction=args.eta_fraction)
        all_rows[label] = rows
        sections.append(render_markdown(rows, pts, f"{args.title} -- {label} (reference: {args.reference})"))
    text = "\n".join(sections)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        with open(re.sub(r"\.md$", "", args.out) + ".json", "w") as fh:
            json.dump(all_rows, fh, indent=1)
        print(f"[score] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
