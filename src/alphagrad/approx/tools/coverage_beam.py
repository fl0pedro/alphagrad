"""X2 -- the COVERAGE-CONSTRAINED multi-skip frontier, screened on a CPU node.

WHAT THIS IS FOR.  Gradient coverage is computable WITHOUT the Adam walk and
WITHOUT a latency measurement: one compile+exec of the plan's own approximate
gradient function, and one of the SAME-ORDER exact reference, then
``env._grad_coverage`` on the two per-leaf norm vectors.  That is ~0.3 s of
CPU per subset, against ~minutes of Blackwell for a paired latency point.  So
thousands of multi-skip subsets can be screened here and only the survivors --
the subsets with ``frac_leaves_zeroed == 0`` -- ever need a GPU.

WHY THE CONSTRAINT IS THE WHOLE POINT.  A leaf at coverage 0.0 is a FROZEN
PARAMETER.  The campaign's headline single-face skips buy their 0.53-0.58
latency ratios by deleting 11-15 of 16 trainable leaves from the backward pass
while still scoring ~0.9257 on the old quality probe against 0.9260 exact --
the probe cannot see the freeze, and coverage can.  ``frac_leaves_zeroed == 0``
is therefore the constraint, not a tie-break.

AND IT ANSWERS A SECOND QUESTION.  The count of coverage evaluations a beam
needs to first reach the best plan it ever finds is the honest denominator for
"did RL find this".  It is printed as a first-class, labelled result --
``EVALUATIONS TO FIRST REACH THE BEST KNOWN PLAN`` -- because if a few hundred
suffice then RL-as-SEARCH is settled on this target and the live claim becomes
RL-as-AMORTIZATION.  See docs/EXPERIMENT_PLAN.md decision rows 8/9/10.

--------------------------------------------------------------------------
THE INPUT CSV CONTRACT (W0-B P1's ``rows.csv``)
--------------------------------------------------------------------------
One row per measurement.  Rows are paired into (candidate, reference) points
and reduced to one ratio + one quality per FACE.  Required columns::

    role        "candidate" | "reference" | "warmup"   (warmup is DROPPED)
    latency_ns  float, the measured latency of that row
    quality     float, the candidate's quality channel
    trial       pairing key within a plan; candidate and reference of the
                SAME trial are the paired pair (see the measurement protocol:
                candidate and its own exact reference, same process, same
                GPU, back to back, warm)

Plus ONE way to name the face the row is about, tried in this order
(``--face-from`` pins it):

    1. columns ``k`` and ``f``                       -> (k, f)
    2. column  ``face``, as "k/f", "kK/fF" or "K.F"  -> (k, f)
    3. column  ``plan_id`` matching ``skiponly:K.F`` -> (k, f)   [qb_sweep]

``k`` is the POSITION IN THE ELIMINATION ORDER and ``f`` the face slot at that
position -- exactly the pair the wire array ``face_skips[k, f]`` carries and
exactly what the policy emits.  Optional columns are used when present and
never required: ``budget`` (a human label such as ``v74/dot_general``),
``op``, ``config_note`` (checked against this process's effective knobs, and
a mismatch is WARNED about, not silently accepted), ``total_live_faces``.

THE ORDER-ALIGNMENT CHECK IS NOT OPTIONAL.  ``k`` is a POSITION, so it only
names the same face if this process reconstructs the SAME elimination order
the CSV was measured on.  It does not always: ``--seed-vertices`` injects two
extra eliminable vertices (the tangent-seed ``add`` and the adjoint
``reduce_sum``), which lengthens the order from 95 to 97 and SHIFTS every k.
Whenever ``budget`` carries a ``v<N>/`` prefix the tool asserts
``order[k] == N`` for every kept candidate and ABORTS on a mismatch (override
with ``--allow-order-mismatch``, which then screens knowingly-misaligned
indices and says so in every output).  Likewise ``total_live_faces``, when
present, is compared with this process's own live-face census.

    ratio(face)   = agg over trials of  candidate.latency_ns
                                      / reference.latency_ns   (paired)
    quality(face) = agg over trials of  candidate.quality

``agg`` is ``--agg`` (median by default).  Non-finite latencies and qualities
are DROPPED before aggregation and reported as dropped; a bare ``max()`` or
``median()`` over a list containing ``nan`` is order-dependent in Python and
that class of bug has already poisoned this codebase's coverage epsilon once.

--------------------------------------------------------------------------
WHAT IS SCORED, AND WHAT IS NOT
--------------------------------------------------------------------------
Subsets are scored ONLY by ``env._grad_coverage`` against the exact reference
for the SAME order -- no walk, no latency, no quality re-measurement.  The
ranking key is::

    (frac_zeroed ASC, -min_leaf_ratio ASC)          # pure coverage

which for the overwhelmingly common case of many equally-clean children is a
TIE.  ``--tiebreak ratio`` (the default) breaks those ties with the
singleton-derived predicted ratio.  That predicted ratio is A HEURISTIC AND
NOT A MEASUREMENT -- it is the product (or the additive-saving sum) of the
member faces' own singleton ratios, and single-face latency ratios are not
composable in general.  It never enters the coverage verdict, it never makes a
subset survive or die, and every survivor still has to be measured paired on a
GPU before any speedup is quoted.  ``--tiebreak none`` removes it entirely at
the cost of making the beam order arbitrary among ties.

CONVENTIONS INHERITED (checked, not assumed):
  * LEAF SET -- ``_leaf_norms`` reports a 0-d output leaf as ``nan`` and
    ``_grad_coverage`` routes ``nan`` to UNCOUNTED, so the 0-d tangent seed
    that ``--seed-vertices`` appends is NOT counted.  That agrees with
    ``_walk_argnums``, which excludes 0-d argnums from the Adam walk.  The two
    conventions are in AGREEMENT at this HEAD; this tool asserts it (the
    identity plan must read ``min_leaf_ratio == 1.0`` and ``frac_zeroed == 0``
    exactly) and records ``n_uncountable`` in the output so a future
    divergence is visible rather than silent.
  * CLEAN -- ``cov["defined"] and cov["n_zeroed"] == 0``.  An UNDEFINED
    coverage record (no countable leaf at all) is NOT clean; absence of
    evidence is not evidence of coverage.
  * MONOTONICITY -- the beam only expands clean nodes, which is sound iff
    zeroing is monotone under adding skips.  That is not proved, so it is
    CHECKED: every pair of evaluated subsets S subset S' is verified to have
    zeroed(S) subset zeroed(S'), and violations are reported.

PURE ANALYSIS TOOL.  It alters no training path.  Importing this module has NO
side effects: every environment knob is set inside ``main()``, never at import
(``verify_pareto_solution`` sets ``JAX_PLATFORMS=cpu`` at IMPORT and thereby
poisons CHILD processes -- that pattern is deliberately not replicated here),
and ``JAX_PLATFORMS`` is never written at all: the launcher exports it.
``ALPHAGRAD_GRAD_COVERAGE`` / ``ALPHAGRAD_REJECT_FROZEN_GRADS`` are forced to
"0" because coverage is computed DIRECTLY from ``_grad_coverage`` here and the
in-band guard must not also fire.

USAGE (one CPU node, one job)::

    JAX_PLATFORMS=cpu \\
    ALPHAGRAD_SKIP_COUNT_OPS=1 \\
    uv run --no-sync python src/alphagrad/approx/tools/coverage_beam.py \\
      --example TransformerLM --dataset wikitext2 --seed 250197 \\
      --candidates $HOME/dsnn/run_analysis/w0/rows_p1_singleton.csv \\
      --quality-min 0.9 --ratio-max 0.99 \\
      --beam-width 16 --max-depth 8 --greedy-also \\
      --report-eval-counts \\
      --out $HOME/dsnn/run_analysis/w0/x2_screen.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Environment knobs.  APPLIED IN main(), NEVER AT IMPORT.  `setdefault`
# semantics throughout: an exporting launcher always wins.
# ---------------------------------------------------------------------------
ENV_DEFAULTS = {
    "ALPHAGRAD_TLM_SEQ": "32",
    "ALPHAGRAD_TLM_DMODEL": "128",
    "ALPHAGRAD_TLM_VOCAB": "1024",
    "ALPHAGRAD_MAX_FACES": "2538",
    "ALPHAGRAD_MAX_DELTA_TOKENS": "32768",
    "GRAPHAX_PLANNER_EXACT": "1",
    "GRAPHAX_DEMAND_EMIT": "1",
    "ALPHAGRAD_INCREMENTAL_TOKENS": "1",
    "ALPHAGRAD_NEW_SLOT_JOIN": "0",
    "ALPHAGRAD_MAX_EQNS": "512",
    "ALPHAGRAD_SKIP_COUNT_OPS": "1",
    "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
}
# Forced, not defaulted: coverage is computed directly here, so the in-band
# channel and the hard guard must both be inert.
ENV_FORCED = {
    "ALPHAGRAD_GRAD_COVERAGE": "0",
    "ALPHAGRAD_REJECT_FROZEN_GRADS": "0",
}

_PLAN_ID_RE = re.compile(r"^skiponly:(\d+)\.(\d+)$")
_FACE_RE = re.compile(r"^k?(\d+)\s*[/.]\s*f?(\d+)$")
_VERTEX_RE = re.compile(r"^v(\d+)\b")


# ===========================================================================
# CSV -> candidates
# ===========================================================================
@dataclass
class Candidate:
    k: int
    f: int
    label: str = ""
    ratio: float = float("nan")
    quality: float = float("nan")
    n_pairs: int = 0
    ratios: list = field(default_factory=list)
    qualities: list = field(default_factory=list)
    live: bool = True
    # Vertex id parsed out of the CSV's ``budget`` label ("v74/dot_general"),
    # used ONLY to prove this process rebuilt the CSV's elimination order.
    vertex: int = -1

    @property
    def key(self):
        return (self.k, self.f)

    @property
    def name(self):
        return f"k{self.k}/f{self.f}"


def _fnum(s):
    """float(), but every unparseable or non-finite value becomes nan."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _agg(vals, how):
    """Aggregate FINITE values only.  nan in, nan out -- never silently ranked.

    A bare ``max()``/``median()`` over a list holding ``nan`` is
    order-dependent in Python; that is not a hypothetical here (see
    ``_grad_coverage``'s ``_finite_e``).  Non-finite entries are dropped and
    counted by the caller.
    """
    fin = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
    if not fin:
        return float("nan")
    if how == "median":
        return float(statistics.median(fin))
    if how == "mean":
        return float(statistics.fmean(fin))
    if how == "min":
        return float(min(fin))
    if how == "max":
        return float(max(fin))
    raise ValueError(how)


def _face_of_row(row, mode):
    """(k, f) for one CSV row, or None.  See the CSV CONTRACT in the module
    docstring for the precedence."""
    if mode in ("auto", "kf") and row.get("k") not in (None, "") \
            and row.get("f") not in (None, ""):
        try:
            return int(float(row["k"])), int(float(row["f"]))
        except ValueError:
            if mode == "kf":
                raise
    if mode in ("auto", "face") and row.get("face"):
        m = _FACE_RE.match(str(row["face"]).strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        if mode == "face":
            raise ValueError(f"unparseable face column: {row['face']!r}")
    if mode in ("auto", "plan_id") and row.get("plan_id"):
        m = _PLAN_ID_RE.match(str(row["plan_id"]).strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        if mode == "plan_id":
            return None
    return None


def parse_candidates(path, *, face_from="auto", agg="median",
                     quality_min=0.9, ratio_max=0.99):
    """Read P1's ``rows.csv`` and return (kept, all_faces, report)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    cols = set(rows[0].keys())
    for need in ("role", "latency_ns", "quality"):
        if need not in cols:
            raise SystemExit(
                f"{path}: missing required column {need!r}; the CSV contract "
                f"is documented at the top of {os.path.basename(__file__)}. "
                f"Columns present: {sorted(cols)}")

    n_warmup = n_unnamed = n_nonfinite = 0
    # (k, f) -> trial -> role -> row
    points = {}
    labels = {}
    notes = set()
    live_face_totals = set()
    for r in rows:
        role = (r.get("role") or "").strip()
        if role == "warmup":
            n_warmup += 1
            continue
        if role not in ("candidate", "reference"):
            continue
        if r.get("config_note"):
            notes.add(r["config_note"])
        # Collected BEFORE the face-name test on purpose: it is the identity
        # row, which names no face, that carries the graph's live-face total.
        tlf = _fnum(r.get("total_live_faces"))
        if math.isfinite(tlf) and tlf > 0:
            live_face_totals.add(int(tlf))
        kf = _face_of_row(r, face_from)
        if kf is None:
            n_unnamed += 1
            continue
        if r.get("budget") and kf not in labels:
            labels[kf] = str(r["budget"])
        points.setdefault(kf, {}).setdefault(str(r.get("trial", "")), {})[role] = r

    cands = []
    for kf, trials in sorted(points.items()):
        c = Candidate(k=kf[0], f=kf[1], label=labels.get(kf, ""))
        mv = _VERTEX_RE.match(c.label)
        c.vertex = int(mv.group(1)) if mv else -1
        for _t, d in sorted(trials.items()):
            cand, ref = d.get("candidate"), d.get("reference")
            if cand is None or ref is None:
                continue                      # unpaired: never comparable
            lc, lr = _fnum(cand["latency_ns"]), _fnum(ref["latency_ns"])
            q = _fnum(cand["quality"])
            if not (math.isfinite(lc) and math.isfinite(lr) and lr > 0.0):
                n_nonfinite += 1
            else:
                c.ratios.append(lc / lr)
            if math.isfinite(q):
                c.qualities.append(q)
            else:
                n_nonfinite += 1
        c.n_pairs = len(c.ratios)
        c.ratio = _agg(c.ratios, agg)
        c.quality = _agg(c.qualities, agg)
        cands.append(c)

    kept = [c for c in cands
            if math.isfinite(c.ratio) and math.isfinite(c.quality)
            and c.quality > quality_min and c.ratio < ratio_max]
    kept.sort(key=lambda c: c.ratio)
    report = {
        "csv": os.path.abspath(path),
        "rows_total": len(rows),
        "rows_warmup_dropped": n_warmup,
        "rows_unnamed_face_dropped": n_unnamed,
        "values_nonfinite_dropped": n_nonfinite,
        "faces_seen": len(cands),
        "faces_kept": len(kept),
        "filter": {"quality_min": quality_min, "ratio_max": ratio_max,
                   "agg": agg, "face_from": face_from},
        "config_notes": sorted(notes),
        "csv_total_live_faces": sorted(live_face_totals),
        "csv_max_k": max((c.k for c in cands), default=-1),
    }
    return kept, cands, report


def check_config_note(notes, warn):
    """Warn (never fail) when the CSV was produced under knobs this process
    does not have.  Face INDICES are graph-state dependent, so a plan-join or
    scale mismatch means the (k, f) pairs may not denote the same faces."""
    eff = {"newslotjoin": os.environ.get("ALPHAGRAD_NEW_SLOT_JOIN", "0")}
    bad = []
    for note in notes:
        for m in re.finditer(r"(\w+)=([\w.-]+)", note):
            key, val = m.group(1).lower(), m.group(2)
            if key in eff and val != eff[key]:
                bad.append(f"CSV says {key}={val}, this process has "
                           f"{key}={eff[key]}")
    for b in sorted(set(bad)):
        warn(f"CONFIG MISMATCH: {b} -- face indices are graph-state "
             f"dependent, so (k,f) may not denote the same face")
    return sorted(set(bad))


# ===========================================================================
# The coverage engine
# ===========================================================================
class CoverageEngine:
    """One TransformerLM-shaped env; ``evaluate(frozenset_of_faces) -> cov``.

    ONE compile+exec per DISTINCT subset (memoised on the frozenset), plus one
    for the same-order exact reference.  ``n_evals`` counts the compile+execs
    -- the honest denominator; ``n_lookups`` counts every request.
    """

    def __init__(self, args, log):
        self.log = log
        self.n_evals = 0
        self.n_lookups = 0
        self.n_hits = 0
        self._cache = {}
        self._t_eval = 0.0
        self._build(args)

    # -- construction ------------------------------------------------------
    def _build(self, args):
        import numpy as np
        import jax
        import jax.random as jrand
        import alphagrad.approx.env as envmod
        from alphagrad.approx.env import (
            VertexEliminationEnv, MAX_RULES_PER_VERTEX, FACE_SLOTS,
            _grad_coverage, _leaf_norms)
        from alphagrad.approx.common.examples import (
            get_fn, get_args, data_gen, grad_target_setup)
        from graphax import inline_call_primitives, jacve
        from types import SimpleNamespace as NS

        self.np, self.jax = np, jax
        self.envmod = envmod
        self._grad_coverage = _grad_coverage
        self._leaf_norms = _leaf_norms
        self.MAX_RULES_PER_VERTEX = MAX_RULES_PER_VERTEX
        self.FACE_SLOTS = FACE_SLOTS
        self.jacve = jacve

        # --seed-vertices is DEPRECATED (A4) and injects TWO extra eliminable
        # vertices, which lengthens the order and SHIFTS every k in the CSV.
        # Default OFF; the order-alignment check below is what proves it.
        a = NS(measure_grad=True, seed_vertices=bool(args.seed_vertices))
        key = jrand.PRNGKey(args.seed)
        key, ak = jrand.split(key)
        fn0 = get_fn(args.example)
        xs = get_args(args.example, ak, dataset=args.dataset)
        gen = data_gen(args.example, dataset=args.dataset, dataset_size=-1)
        fn0, xs, argnums = grad_target_setup(a, fn0, xs, args.example)

        cj = jax.make_jaxpr(fn0)(*xs)
        jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
        try:
            from jax.extend.core import ClosedJaxpr
        except ImportError:                                # pragma: no cover
            from jax._src.core import ClosedJaxpr
        cj = ClosedJaxpr(jx, consts) if jx is not cj.jaxpr else cj

        self.env = VertexEliminationEnv.from_jaxpr(
            cj, args=xs, argnums=argnums, num_envs=0, data_gen=gen,
            target_fun=fn0, cmp_type="latency", mem_type="peak_memory",
            measure_latency=True, per_face=True, measure_grad=True,
            delta_obs=True)
        envmod.configure_max_faces(
            envmod.derived_max_faces(cj.jaxpr, argnums, cj.literals, xs))
        self.cfg = self.env.config
        # THE ORDER.  Reverse mode -- the pin every W0 number is quoted
        # against, and the order the singleton sweep itself ran on.
        self.o_list = [int(v) for v in
                       sorted((int(v) for v in self.env.valid_vertices),
                              reverse=True)]
        self.MF = envmod.MAX_FACES
        self.log(f"order len {len(self.o_list)}  MAX_FACES {self.MF}")

    # -- live-face census --------------------------------------------------
    def live_faces_per_step(self):
        """``n_faces[k]`` on the BASELINE (no-transform) replay -- the basis
        the singleton sweep enumerated against.  A candidate (k, f) is live
        iff ``f < n_faces[k]``."""
        from graphax import faces_of
        from graphax.incremental import IncrementalJaxpr
        ij = IncrementalJaxpr(self.cfg.jaxpr, tuple(self.cfg.argnums),
                              list(self.env.consts), list(self.env.args),
                              track_faces=False)
        out = []
        for v in self.o_list:
            out.append(len(faces_of(ij.graph, ij.tgraph, int(v),
                                    self.cfg.jaxpr)))
            ij.eliminate(int(v), (), None)
        return out

    # -- the plan wire -----------------------------------------------------
    def _wire(self, skips):
        np = self.np
        n = len(self.o_list)
        specs = np.full((n, self.MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
        specs[:, :, 2] = 0
        face_specs = np.full((n, self.MF, self.FACE_SLOTS, 3), -1,
                             dtype=np.int32)
        face_skips = np.zeros((n, self.MF), dtype=np.int32)
        for (k, f) in skips:
            face_skips[k, f] = 1
        return specs, face_specs, face_skips

    def _build_fn(self, skips):
        """The MEASUREMENT PATH's own approx gradient fn for these skips --
        the same construction ``grad_coverage_probe`` and
        ``ls_face_forensics`` use, so a disagreement is about COVERAGE and
        never about which function was measured."""
        specs, face_specs, face_skips = self._wire(skips)
        ft = None
        if skips:
            ft = self.envmod._face_transforms_for_order(
                self.cfg, list(self.env.consts), list(self.env.args),
                self.o_list, specs.tolist(), face_specs, face_skips)
        return self.jacve(self.cfg.target_fun, list(self.o_list),
                          argnums=self.cfg.argnums, has_aux=self.cfg.has_aux,
                          sparse_representation=self.cfg.sparse,
                          transforms=[], face_transforms=ft)

    def _norms(self, skips):
        out = self.jax.jit(self._build_fn(skips))(*self.env.args)
        return self._leaf_norms(out, self.cfg.has_aux)

    # -- the reference and the evaluations ---------------------------------
    def prepare_exact(self):
        t0 = time.perf_counter()
        self.exact_norms = self._norms([])
        self.t_exact = time.perf_counter() - t0
        cov = self._grad_coverage(self.exact_norms, self.exact_norms)
        # The identity plan against itself MUST be exactly clean.  If it is
        # not, the leaf-set convention has moved and every verdict below is
        # meaningless -- die here rather than screen against a broken
        # reference.
        if not (cov["min_leaf_ratio"] == 1.0 and cov["frac_zeroed"] == 0.0):
            raise SystemExit(
                "IDENTITY REFERENCE IS NOT CLEAN -- min_leaf_ratio="
                f"{cov['min_leaf_ratio']!r} frac_zeroed="
                f"{cov['frac_zeroed']!r}.  The leaf-set convention "
                "(_leaf_norms 0-d -> nan -> uncounted) has changed; fix that "
                "before screening anything.")
        self.identity_cov = cov
        self.log(f"exact same-order reference: {len(self.exact_norms)} leaves"
                 f"  counted {cov['n_counted']}  uncounted "
                 f"{cov['n_uncounted']}  uncountable "
                 f"{cov.get('n_uncountable', 'n/a')}"
                 f"  compile+exec {self.t_exact:.2f}s")
        self.log("identity check PASS: min_leaf_ratio == 1.0 and "
                 "frac_zeroed == 0 exactly")
        return cov

    def evaluate(self, subset):
        """Coverage of one subset.  Memoised; ``n_evals`` counts real work."""
        subset = frozenset(subset)
        self.n_lookups += 1
        hit = self._cache.get(subset)
        if hit is not None:
            self.n_hits += 1
            return hit
        t0 = time.perf_counter()
        norms = self._norms(sorted(subset))
        cov = self._grad_coverage(norms, self.exact_norms)
        self._t_eval += time.perf_counter() - t0
        self.n_evals += 1
        cov["_eval_index"] = self.n_evals
        cov["_subset"] = sorted(subset)
        self._cache[subset] = cov
        return cov

    @property
    def cache(self):
        return self._cache


def is_clean(cov):
    """``frac_leaves_zeroed == 0`` -- the constraint.  An UNDEFINED record is
    not clean: no countable leaf is absence of evidence, not evidence."""
    return bool(cov.get("defined")) and int(cov.get("n_zeroed", 1)) == 0


# ===========================================================================
# Predicted ratio -- A HEURISTIC, NEVER A MEASUREMENT
# ===========================================================================
def predict_ratio(subset, by_key, how):
    rs = [by_key[kf].ratio for kf in subset if kf in by_key]
    rs = [r for r in rs if math.isfinite(r)]
    if not rs:
        return 1.0
    if how == "prod":
        p = 1.0
        for r in rs:
            p *= r
        return p
    if how == "addsave":
        return max(0.0, 1.0 - sum(1.0 - r for r in rs))
    raise ValueError(how)


def rank_key(cov, subset, by_key, predict, tiebreak):
    """PURE COVERAGE first.  ``tiebreak`` only orders otherwise-equal nodes and
    never decides survival."""
    frac = float(cov.get("frac_zeroed", 1.0)) if cov.get("defined") else 1.0
    mlr = float(cov.get("min_leaf_ratio", 0.0)) if cov.get("defined") else 0.0
    tb = (predict_ratio(subset, by_key, predict)
          if tiebreak == "ratio" else 0.0)
    return (frac, -mlr, tb, tuple(sorted(subset)))


# ===========================================================================
# Searches
# ===========================================================================
def greedy_search(eng, cands, args, log):
    by_key = {c.key: c for c in cands}
    cur = frozenset()
    trace = []
    log("\n=== GREEDY (one face added per depth, best coverage wins) ===")
    for depth in range(1, args.max_depth + 1):
        best = None
        n_here = 0
        for c in cands:
            if c.key in cur:
                continue
            if eng.n_evals >= args.max_evals:
                log(f"  [greedy] eval budget {args.max_evals} exhausted")
                break
            cand = cur | {c.key}
            cov = eng.evaluate(cand)
            n_here += 1
            if not is_clean(cov) and args.prune == "clean":
                continue
            k = rank_key(cov, cand, by_key, args.predict, args.tiebreak)
            if best is None or k < best[0]:
                best = (k, cand, cov, c)
        if best is None:
            log(f"  depth {depth}: no clean addition exists -- greedy stops "
                f"at |S|={len(cur)}")
            break
        _, cur, cov, c = best
        trace.append({"depth": depth, "added": c.name, "label": c.label,
                      "subset": [f"k{a}/f{b}" for a, b in sorted(cur)],
                      "clean": is_clean(cov),
                      "min_leaf_ratio": cov.get("min_leaf_ratio"),
                      "frac_zeroed": cov.get("frac_zeroed"),
                      "n_zeroed": cov.get("n_zeroed"),
                      "predicted_ratio": predict_ratio(cur, by_key,
                                                       args.predict),
                      "evals_so_far": eng.n_evals,
                      "evals_this_depth": n_here})
        log(f"  depth {depth}: +{c.name:<10s} ({c.label:<22s}) "
            f"clean={is_clean(cov)!s:<5s} min_leaf_ratio="
            f"{cov.get('min_leaf_ratio'):.6g} zeroed={cov.get('n_zeroed')}"
            f"  pred_ratio={trace[-1]['predicted_ratio']:.4f}"
            f"  evals={eng.n_evals}")
        if eng.n_evals >= args.max_evals:
            break
    return {"trace": trace,
            "final_subset": [f"k{a}/f{b}" for a, b in sorted(cur)],
            "final_size": len(cur)}


def beam_search(eng, cands, args, log):
    by_key = {c.key: c for c in cands}
    beam = [frozenset()]
    seen = {frozenset()}
    levels = []
    log(f"\n=== BEAM (width {args.beam_width}, depth {args.max_depth}, "
        f"prune={args.prune}, tiebreak={args.tiebreak}) ===")
    for depth in range(1, args.max_depth + 1):
        scored = []
        n_here = 0
        stop = False
        for node in beam:
            for c in cands:
                if c.key in node:
                    continue
                child = node | {c.key}
                if child in seen:
                    continue
                seen.add(child)
                if eng.n_evals >= args.max_evals:
                    log(f"  [beam] eval budget {args.max_evals} exhausted")
                    stop = True
                    break
                cov = eng.evaluate(child)
                n_here += 1
                if args.prune == "clean" and not is_clean(cov):
                    continue
                scored.append((rank_key(cov, child, by_key, args.predict,
                                        args.tiebreak), child, cov))
            if stop:
                break
        if not scored:
            log(f"  depth {depth}: no surviving child "
                f"({n_here} evaluated here) -- beam stops")
            break
        scored.sort(key=lambda t: t[0])
        beam = [t[1] for t in scored[:args.beam_width]]
        top = scored[0]
        levels.append({
            "depth": depth,
            "children_evaluated": n_here,
            "children_surviving": len(scored),
            "beam_kept": len(beam),
            "evals_so_far": eng.n_evals,
            "best_min_leaf_ratio": top[2].get("min_leaf_ratio"),
            "best_predicted_ratio": predict_ratio(top[1], by_key,
                                                  args.predict),
            "beam": [[f"k{a}/f{b}" for a, b in sorted(s)] for s in beam],
        })
        log(f"  depth {depth}: evaluated {n_here:>5d}  survived "
            f"{len(scored):>5d}  kept {len(beam):>3d}  "
            f"best min_leaf_ratio={top[2].get('min_leaf_ratio'):.6g}  "
            f"best pred_ratio={levels[-1]['best_predicted_ratio']:.4f}  "
            f"evals={eng.n_evals}")
        if stop:
            break
    return {"levels": levels,
            "final_beam": [[f"k{a}/f{b}" for a, b in sorted(s)]
                           for s in beam]}


# ===========================================================================
# Monotonicity audit
# ===========================================================================
def monotonicity_violations(cache, limit=20, max_items=4000):
    """zeroed(S) must be a subset of zeroed(S') whenever S subset S'.  The
    beam's clean-only pruning is sound exactly when this holds."""
    items = [(s, set(c.get("zeroed", []))) for s, c in cache.items()
             if c.get("defined")]
    if len(items) > max_items:
        # O(n^2); above this the audit costs more than the screen.
        items = items[:max_items]
    items.sort(key=lambda t: len(t[0]))
    out = []
    for i, (s, zs) in enumerate(items):
        for (t, zt) in items[i + 1:]:
            if len(t) <= len(s) or not s <= t:
                continue
            if not zs <= zt:
                out.append({
                    "sub": [f"k{a}/f{b}" for a, b in sorted(s)],
                    "sup": [f"k{a}/f{b}" for a, b in sorted(t)],
                    "lost": sorted(zs - zt)})
                if len(out) >= limit:
                    return out
    return out


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="coverage_beam",
        description="X2: coverage-constrained multi-skip frontier + the "
                    "evaluation count to first reach the best known plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True,
                   help="W0-B P1's rows.csv (contract: module docstring)")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--example", default="TransformerLM")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument("--quality-min", type=float, default=0.9)
    p.add_argument("--ratio-max", type=float, default=0.99)
    p.add_argument("--agg", default="median",
                   choices=["median", "mean", "min", "max"])
    p.add_argument("--face-from", default="auto",
                   choices=["auto", "kf", "face", "plan_id"])
    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--greedy-also", action="store_true",
                   help="run greedy in addition to the beam")
    p.add_argument("--no-beam", action="store_true",
                   help="greedy only (the beam is on by default)")
    p.add_argument("--prune", default="clean", choices=["clean", "none"],
                   help="'clean' expands only frac_zeroed==0 nodes")
    p.add_argument("--tiebreak", default="ratio", choices=["ratio", "none"],
                   help="orders coverage-equal nodes by the PREDICTED "
                        "(heuristic, unmeasured) latency ratio")
    p.add_argument("--predict", default="prod", choices=["prod", "addsave"])
    p.add_argument("--max-evals", type=int, default=20000)
    p.add_argument("--max-candidates", type=int, default=0,
                   help="0 = all; otherwise the N best-ratio kept faces")
    p.add_argument("--target", default="",
                   help="comma-separated 'k/f' naming a reference plan; the "
                        "eval index at which the search first reaches it is "
                        "reported alongside the discovered best")
    p.add_argument("--report-eval-counts", action="store_true",
                   help="accepted for launcher compatibility; the counts are "
                        "ALWAYS reported")
    p.add_argument("--dry-run", action="store_true",
                   help="parse the CSV and print the candidate list; import "
                        "no JAX and evaluate no coverage")
    p.add_argument("--check-live-faces", action="store_true", default=True)
    p.add_argument("--no-check-live-faces", dest="check_live_faces",
                   action="store_false")
    p.add_argument("--seed-vertices", action="store_true",
                   help="DEPRECATED (A4). Injects 2 eliminable vertices and "
                        "SHIFTS every k; default off. Only for reproducing "
                        "an archived run's graph.")
    p.add_argument("--allow-order-mismatch", action="store_true",
                   help="screen even when k no longer names the CSV's "
                        "vertex. Results are then NOT comparable to the CSV.")
    return p


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                      # pragma: no cover
        return ""


def main(argv=None):
    args = build_parser().parse_args(argv)
    warnings = []

    def log(msg):
        print(msg, flush=True)

    def warn(msg):
        warnings.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)

    kept, all_faces, report = parse_candidates(
        args.candidates, face_from=args.face_from, agg=args.agg,
        quality_min=args.quality_min, ratio_max=args.ratio_max)
    log(f"CSV {report['csv']}")
    log(f"  rows {report['rows_total']}  warmup dropped "
        f"{report['rows_warmup_dropped']}  unnamed-face dropped "
        f"{report['rows_unnamed_face_dropped']}  non-finite values dropped "
        f"{report['values_nonfinite_dropped']}")
    log(f"  faces seen {report['faces_seen']}  KEPT "
        f"{report['faces_kept']} at quality > {args.quality_min} and "
        f"ratio < {args.ratio_max} ({args.agg} over paired trials)")
    report["config_mismatch"] = check_config_note(report["config_notes"], warn)

    if args.max_candidates and len(kept) > args.max_candidates:
        warn(f"--max-candidates {args.max_candidates}: screening only the "
             f"{args.max_candidates} best-ratio of {len(kept)} kept faces")
        kept = kept[:args.max_candidates]

    log("\n  kept candidates (ratio is the CSV's measured singleton ratio):")
    for c in kept:
        log(f"    {c.name:<10s} {c.label:<24s} ratio={c.ratio:.4f} "
            f"quality={c.quality:.6f} pairs={c.n_pairs}")

    if not kept:
        log("\nNO CANDIDATES SURVIVE THE FILTER -- X2 has nothing to search. "
            "This is a result, not a failure: report it.")
    if args.dry_run:
        log("\n--dry-run: no coverage evaluated.")
        if args.out:
            _write(args.out, {"meta": _meta(args, warnings),
                              "csv_report": report,
                              "candidates": [asdict(c) for c in kept]}, log)
        return 0

    # ------- environment, applied HERE and not at import -------------------
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    for k, v in ENV_FORCED.items():
        os.environ[k] = v
    if "JAX_PLATFORMS" not in os.environ:
        warn("JAX_PLATFORMS is unset; this tool does NOT set it (that is the "
             "import side-effect that poisons child processes). Export "
             "JAX_PLATFORMS=cpu for a CPU-node screen.")

    t_start = time.perf_counter()
    eng = CoverageEngine(args, log)
    eng.prepare_exact()

    by_key = {c.key: c for c in kept}
    nf = []
    dropped_dead = []
    if args.check_live_faces:
        nf = eng.live_faces_per_step()
        dead = []
        for c in kept:
            c.live = (0 <= c.k < len(nf)) and (c.f < nf[c.k])
            if not c.live:
                dead.append(f"{c.name} (step {c.k} has "
                            f"{nf[c.k] if 0 <= c.k < len(nf) else '?'} faces)")
        if dead:
            warn(f"{len(dead)} of {len(kept)} CSV faces are NOT LIVE in this "
                 f"graph and are dropped: " + ", ".join(dead))
            dropped_dead = dead
            kept = [c for c in kept if c.live]
            by_key = {c.key: c for c in kept}
        log(f"live-face census: {sum(nf)} live faces over {len(nf)} steps; "
            f"{len(kept)} candidates live")

    # ------- ORDER ALIGNMENT: does k still name the CSV's vertex? ---------
    align = {"checked": 0, "mismatched": [], "order_len": len(eng.o_list),
             "csv_max_k": report["csv_max_k"],
             "csv_total_live_faces": report["csv_total_live_faces"],
             "this_total_live_faces": (sum(nf) if args.check_live_faces
                                       else None),
             "csv_faces_not_live_here": dropped_dead}
    for c in kept:
        if c.vertex < 0:
            continue
        align["checked"] += 1
        got = eng.o_list[c.k] if 0 <= c.k < len(eng.o_list) else None
        if got != c.vertex:
            align["mismatched"].append(
                {"face": c.name, "csv_label": c.label,
                 "csv_vertex": c.vertex, "order_vertex": got})
    if align["mismatched"]:
        msg = (f"ORDER MISALIGNED: {len(align['mismatched'])} of "
               f"{align['checked']} kept faces name a different vertex than "
               f"this process's order position. e.g. "
               f"{align['mismatched'][0]}. Order length here is "
               f"{len(eng.o_list)}; the CSV's largest k is "
               f"{report['csv_max_k']}. k is a POSITION, so every (k,f) "
               f"below would denote the WRONG face.")
        if not args.allow_order_mismatch:
            raise SystemExit(
                msg + "  Refusing to screen. Most likely cause: "
                "--seed-vertices (it injects 2 eliminable vertices and "
                "shifts every k). Re-run with the setting the CSV was "
                "measured under, or pass --allow-order-mismatch to screen "
                "knowingly-misaligned indices.")
        warn(msg + "  --allow-order-mismatch given: SCREENING ANYWAY. "
             "Nothing below is comparable to the CSV.")
    else:
        log(f"order alignment OK: {align['checked']} kept faces all match "
            f"their CSV vertex label at their own k")
    if (args.check_live_faces and report["csv_total_live_faces"]
            and sum(nf) not in report["csv_total_live_faces"]):
        warn(f"LIVE-FACE COUNT DIFFERS: CSV says "
             f"{report['csv_total_live_faces']}, this process counts "
             f"{sum(nf)} -- the graph is not the one the CSV was measured on")

    target = []
    if args.target:
        for tok in args.target.split(","):
            m = _FACE_RE.match(tok.strip())
            if not m:
                raise SystemExit(f"--target: unparseable {tok!r}")
            target.append((int(m.group(1)), int(m.group(2))))
        target = frozenset(target)

    # THE BEAM RUNS FIRST, ON PURPOSE.  Both searches share one memo (that is
    # what makes the screen cheap), so whichever runs second inherits the
    # other's cache and its evaluation count stops being quotable.  The
    # headline number this experiment exists to produce is the BEAM's, so the
    # beam gets the pristine counter and greedy is charged only the
    # ADDITIONAL evaluations it needs.
    greedy = beam = None
    beam_evals = 0
    if not args.no_beam:
        beam = beam_search(eng, kept, args, log)
        beam_evals = eng.n_evals
    if args.greedy_also or args.no_beam:
        greedy = greedy_search(eng, kept, args, log)
        if greedy is not None:
            greedy["additional_evaluations"] = eng.n_evals - beam_evals

    # ------- survivors ----------------------------------------------------
    survivors = []
    for s, cov in eng.cache.items():
        if not s or not is_clean(cov):
            continue
        survivors.append({
            "subset": [f"k{a}/f{b}" for a, b in sorted(s)],
            "size": len(s),
            "min_leaf_ratio": cov["min_leaf_ratio"],
            "frac_zeroed": cov["frac_zeroed"],
            "n_counted": cov["n_counted"],
            "n_uncountable": cov.get("n_uncountable"),
            "predicted_ratio": predict_ratio(s, by_key, args.predict),
            "first_eval_index": cov["_eval_index"],
        })
    survivors.sort(key=lambda d: (d["predicted_ratio"], -d["size"]))

    # ------- THE EVALUATION COUNT ----------------------------------------
    best = survivors[0] if survivors else None
    evals_to_best = best["first_eval_index"] if best else None
    tgt = None
    if target:
        cov = eng.cache.get(target)
        tgt = {"subset": [f"k{a}/f{b}" for a, b in sorted(target)],
               "reached": cov is not None,
               "eval_index": cov["_eval_index"] if cov else None,
               "clean": is_clean(cov) if cov else None}

    viol = monotonicity_violations(eng.cache)
    if viol:
        warn(f"{len(viol)} MONOTONICITY VIOLATION(S): a superset un-zeroed a "
             f"leaf its subset zeroed. clean-only pruning is NOT sound here; "
             f"re-run with --prune none before trusting the frontier.")

    wall = time.perf_counter() - t_start
    log("\n" + "=" * 72)
    log("EVALUATION COUNT  (the honest denominator for \"did RL find this\")")
    log("=" * 72)
    log(f"  coverage evaluations performed          : {eng.n_evals}")
    log(f"  subset lookups (incl. memo hits)        : {eng.n_lookups} "
        f"({eng.n_hits} hits)")
    log(f"  exact same-order references             : 1")
    log(f"  wall in coverage evaluation             : {eng._t_eval:.1f}s "
        f"of {wall:.1f}s total  "
        f"({eng._t_eval / max(eng.n_evals, 1):.3f}s/eval)")
    log(f"  of which the BEAM spent                 : {beam_evals}"
        + ("" if beam is not None else "  (beam not run)"))
    if greedy is not None:
        log(f"  greedy's ADDITIONAL evaluations         : "
            f"{greedy.get('additional_evaluations')}  (it reuses the beam's "
            f"memo; the beam ran first so its count stands alone)")
    if best:
        found_by = ("the BEAM" if beam is not None
                    and best["first_eval_index"] <= beam_evals
                    else "GREEDY, after the beam")
        log(f"  best coverage-clean subset found        : "
            f"{' + '.join(best['subset'])}  (|S|={best['size']})")
        log(f"    its PREDICTED ratio ({args.predict}, HEURISTIC, "
            f"NOT MEASURED) : {best['predicted_ratio']:.4f}")
        log(f"    min_leaf_ratio                        : "
            f"{best['min_leaf_ratio']:.6g}   frac_zeroed 0.0")
        log(f"    first reached by                      : {found_by}")
        log(f"  >> EVALUATIONS TO FIRST REACH THE BEST KNOWN PLAN: "
            f"{evals_to_best} <<")
        log("     (self-referential: 'best known' is the best this run "
            "found. Pass --target to count against an externally named "
            "plan instead.)")
    else:
        log("  best coverage-clean subset found        : NONE")
        log("  >> EVALUATIONS TO FIRST REACH THE BEST KNOWN PLAN: "
            "N/A (no multi-skip subset is coverage-clean) <<")
    if tgt:
        log(f"  --target {' + '.join(tgt['subset'])}: reached="
            f"{tgt['reached']} at eval {tgt['eval_index']} clean="
            f"{tgt['clean']}")
    log("=" * 72)

    log(f"\ncoverage-clean survivors: {len(survivors)}")
    for d in survivors[:40]:
        log(f"  pred_ratio={d['predicted_ratio']:.4f}  |S|={d['size']}  "
            f"min_leaf_ratio={d['min_leaf_ratio']:.6g}  "
            f"eval#{d['first_eval_index']}  {' + '.join(d['subset'])}")
    if len(survivors) > 40:
        log(f"  ... {len(survivors) - 40} more (full list in the JSON)")

    log("\nCAVEAT, stated every run: predicted_ratio is a HEURISTIC "
        "composition of singleton ratios, not a measurement. Every survivor "
        "must be measured paired -- candidate and its own same-order exact "
        "reference, same process, same GPU, back to back, warm, "
        "--latency-inner-reps 50 -- before any speedup is quoted.")
    for w in warnings:
        log(f"WARNING: {w}")

    payload = {
        "meta": _meta(args, warnings),
        "csv_report": report,
        "order_alignment": align,
        "live_faces_per_step": nf,
        "order": eng.o_list,
        "candidates": [asdict(c) for c in kept],
        "identity_coverage": {k: v for k, v in eng.identity_cov.items()
                              if k not in ("approx_norms", "exact_norms")},
        "exact_norms": eng.exact_norms,
        "greedy": greedy,
        "beam": beam,
        "survivors": survivors,
        "monotonicity_violations": viol,
        "eval_counts": {
            "coverage_evaluations": eng.n_evals,
            "beam_evaluations": beam_evals,
            "greedy_additional_evaluations": (
                greedy.get("additional_evaluations") if greedy else None),
            "best_first_reached_by": (
                None if not best else
                ("beam" if beam is not None
                 and best["first_eval_index"] <= beam_evals else "greedy")),
            "subset_lookups": eng.n_lookups,
            "memo_hits": eng.n_hits,
            "exact_references": 1,
            "seconds_in_evaluation": eng._t_eval,
            "seconds_total": wall,
            "evals_to_best_known_plan": evals_to_best,
            "best_known_plan": best,
            "target": tgt,
        },
        "warnings": warnings,
    }
    if args.out:
        _write(args.out, payload, log)
    return 0


def _meta(args, warnings):
    return {
        "tool": "coverage_beam.py",
        "git": _git_sha(),
        "argv": sys.argv,
        "args": vars(args),
        "env": {k: os.environ.get(k) for k in
                sorted(set(list(ENV_DEFAULTS) + list(ENV_FORCED) +
                           ["JAX_PLATFORMS", "ALPHAGRAD_FORCE_REV_ORDER",
                            "JAX_COMPILATION_CACHE_DIR"]))},
        "timestamp": time.time(),
        "n_warnings": len(warnings),
    }


def _write(path, payload, log):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log(f"\nwrote {os.path.abspath(path)}")


if __name__ == "__main__":
    raise SystemExit(main())
