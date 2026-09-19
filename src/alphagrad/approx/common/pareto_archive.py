"""Shared Pareto-front archive over the reward-vector objective channels.

Generalises the front/hypervolume bookkeeping that lived inline in
``cmorl_ray.py`` so single-objective drivers (e.g. ``ppo_ray``) can also persist
the multi-objective frontier they sweep through. Objectives are MAXIMISED — the
env's ``reward_vec`` is already sign-oriented (``-cost`` for cost channels,
``+cosine`` for quality), so a chosen subset of channel indices is used as-is.

Two artefacts:
  * ``<name>_pareto_front.json``      — the live non-dominated front (obj + seq).
  * ``<name>_all_front_candidates.json`` — EVERY point ever admitted to the
    front (incl. later-pruned), with the FULL reward_vec so any objective set
    can be re-scored offline. Append-only, deduped by sequence.
"""
from __future__ import annotations

import json
import functools
import numpy as np

# Exact-match sentinel signature shared with the reward plumbing
# (common.compile_cache.SENTINEL_REWARD_VALUE). A failed/skipped measure stamps this
# on cost channels; a zeroed-reward no-op reads as all-zeros. Neither is a
# real Pareto point, so the archive rejects both (exact match, not <=).
_SENTINEL_OBJ_VALUE = -1e10


def pareto_mask(points) -> np.ndarray:
    """Boolean mask of non-dominated rows (maximization)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)
    ge = np.all(pts[:, None, :] >= pts[None, :, :], axis=-1)
    gt = np.any(pts[:, None, :] > pts[None, :, :], axis=-1)
    return ~np.any(ge & gt, axis=0)


def _hv2d_min(pts, ref):
    pts = pts[np.argsort(pts[:, 0])]
    rx, ry = ref
    hv = (ry - pts[0, 1]) * (rx - pts[0, 0])
    for i in range(1, pts.shape[0]):
        hv += (pts[i - 1, 1] - pts[i, 1]) * (rx - pts[i, 0])
    return float(hv)


def _hv3d_min(pts, ref):
    pts = pts[np.argsort(pts[:, 2])]
    rx, ry, rz = ref
    vol, active, i, n = 0.0, [], 0, pts.shape[0]
    while i < n:
        z = pts[i, 2]
        while i < n and pts[i, 2] == z:
            active.append(pts[i, :2])
            i += 1
        z_next = pts[i, 2] if i < n else rz
        vol += _hv2d_min(np.asarray(active), (rx, ry)) * (z_next - z)
    return float(vol)


def _hv_mc_normalized(pts, ref, *, n_samples: int = 100_000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume for D>=4 objectives (no cheap exact formula).

    MAXIMIZATION front ``pts`` (P x D) above nadir ``ref`` (D,). Because our
    channels span wildly different magnitudes (peak_memory ~1e9, latency ~1e5,
    flops ~1e9, cosine ~1), the volume is computed in per-axis MIN-MAX
    NORMALIZED space over ``front union ref`` so no single 1e9 axis dominates and
    the result is a scale-free fraction in ~[0, 1] comparable across
    episodes/runs. Deterministic (fixed ``seed``); degenerate zero-width axes are
    skipped (they contribute no volume).

    HV_frac = mean_s [ some front point dominates sample s (>= in every dim) ]
    over N uniform samples in the normalized box [0, 1]^D_eff; the reported value
    IS that fraction (already the box-volume-normalized dominated volume, since
    the normalized box has unit volume on the kept axes).
    """
    pts = np.asarray(pts, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    # per-axis min/max over front + nadir -> normalization span
    lo = np.minimum(pts.min(axis=0), ref)
    hi = pts.max(axis=0)
    span = hi - lo
    keep = span > 0.0            # drop zero-width axes (no volume along them)
    if not np.any(keep):
        return 0.0
    ptsn = (pts[:, keep] - lo[keep]) / span[keep]     # front in [0,1]^Dk
    refn = (ref[keep] - lo[keep]) / span[keep]        # nadir in [0,1]^Dk (>=0)
    rng = np.random.default_rng(seed)
    # sample uniformly in the box [refn, 1]^Dk (region that CAN be dominated
    # above the nadir); scale the fraction back by that box's volume so the
    # result is the dominated volume as a fraction of the FULL [0,1]^Dk unit box.
    hicorner = np.ones_like(refn)
    box_span = hicorner - refn
    if np.any(box_span <= 0.0):
        return 0.0
    S = rng.random((int(n_samples), ptsn.shape[1]))
    samp = refn + S * box_span
    # dominated iff some front point >= sample in ALL kept dims
    dom = np.any(np.all(ptsn[:, None, :] >= samp[None, :, :], axis=2), axis=0)
    box_vol = float(np.prod(box_span))
    return float(dom.mean() * box_vol)


def hypervolume(points, ref) -> float:
    """Hypervolume dominated by ``points`` (maximization) above nadir ``ref``.
    Exact for 2/3 objectives; a finite normalized Monte-Carlo estimate for
    >=4 objectives (see _hv_mc_normalized)."""
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    m = pts.shape[1]
    if m not in (2, 3):
        # D>=4: no cheap exact HV -> deterministic normalized Monte-Carlo
        # estimate (finite, scale-free ~[0,1]). <=3 keeps the exact sweep.
        return _hv_mc_normalized(pts[pareto_mask(pts)], ref)
    pts = pts[pareto_mask(pts)]
    pts = pts[np.all(pts > ref, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    # _hv2d_min/_hv3d_min are MIN-oriented sweeps; negate points + ref to use
    # them for a MAXIMIZATION front (matches cmorl_ray._hypervolume).
    if m == 2:
        return _hv2d_min(-pts, (-ref[0], -ref[1]))
    return _hv3d_min(-pts, (-ref[0], -ref[1], -ref[2]))


class ParetoArchive:
    """Append-only non-dominated front over ``obj_idx`` slices of reward vectors.

    Args:
        obj_names: display names of the objective channels (len = #objectives).
        obj_idx:   indices into the full reward_vec for those channels.
    """

    def __init__(self, obj_names, obj_idx, quality_floor=None):
        self.obj_names = list(obj_names)
        self.obj_idx = [int(i) for i in obj_idx]
        # THE QUALITY FLOOR tau (ticket dsnn-3qm.9, --quality-floor): a
        # candidate below it is INFEASIBLE and never a usable Pareto point
        # -- its cost was obtained by not computing the gradient. None (the
        # default) admits every non-dominated candidate.
        self.quality_floor = (None if quality_floor is None
                              else float(quality_floor))
        self.pts: list[np.ndarray] = []          # live front objective vectors
        self.seqs: list = []                     # parallel sequences
        # Episode at which each LIVE front point was admitted. Parallel to
        # pts/seqs and pruned with them. Without this the plotting side had
        # nothing to read and stamped the current episode onto every point,
        # so the whole front looked like it was re-measured every episode.
        self.eps: list[int] = []
        self.all_candidates: list[dict] = []     # every admitted point
        self._seen: set = set()
        # Fixed HV reference captured on first non-empty call; class has
        # no save/load, so process lifetime bounds the reference lifetime.
        self._hv_ref: np.ndarray | None = None

    def add(self, reward_vec, seq, episode: int) -> bool:
        """Admit one (reward_vec, seq) to the front. Returns True if it was
        non-dominated on arrival (and thus added + logged). Incremental O(front):
        reject if dominated by / equal to an existing front point, else drop the
        points it dominates. ``obj_idx`` is trusted (a bad index raises rather
        than being silently swallowed, so channel/width drift surfaces loudly)."""
        g = np.array([float(reward_vec[i]) for i in self.obj_idx], dtype=np.float64)
        if not np.all(np.isfinite(g)):
            return False
        # Reject sentinel / failed-measure candidates (never real Pareto
        # points): any objective at the exact sentinel value, OR the
        # all-objective-exactly-zero no-op (zeroed reward reads as best-cost
        # and would spuriously dominate the front + inflate the HV box).
        if np.any(g == _SENTINEL_OBJ_VALUE) or not np.any(g != 0.0):
            return False
        # QUALITY FLOOR AT INSERTION (sibling audit 2026-08-10; the env var
        # ALPHAGRAD_QUALITY_GATE_MIN it read until 2026-09-04 is deleted,
        # ticket .9 -- the floor is the --quality-floor argument now): the
        # reward never guarded the archive, so destroyed plans (quality 0.0)
        # sat on the front as fake best-cost points on BOTH arms. An
        # infeasible candidate is not a usable Pareto point -- its cost was
        # obtained by not computing the gradient. None keeps the historical
        # behaviour.
        if self.quality_floor is not None:
            for _k, _nm in enumerate(self.obj_names):
                if "cos" in _nm or "quality" in _nm:
                    if g[_k] < self.quality_floor:
                        return False
                    break
        for p in self.pts:
            # dominated by, or objective-identical to, an existing front point
            if np.allclose(g, p) or (np.all(p >= g) and np.any(p > g)):
                return False
        # g is non-dominated → keep it, drop any existing points it dominates
        survivors = [
            (p, s, e) for p, s, e in zip(self.pts, self.seqs, self.eps)
            if not (np.all(g >= p) and np.any(g > p))
        ]
        self.pts = [p for p, _, _ in survivors] + [g]
        self.seqs = [s for _, s, _ in survivors] + [seq]
        # stamp the admitting episode ONCE; never rewritten afterwards
        self.eps = [e for _, _, e in survivors] + [int(episode)]
        key = repr(seq)
        if key not in self._seen:
            self._seen.add(key)
            full = reward_vec.tolist() if hasattr(reward_vec, "tolist") else list(reward_vec)
            self.all_candidates.append({
                "episode": int(episode),
                "reward_vec": [float(x) for x in full],
                "obj": {nm: float(g[k]) for k, nm in enumerate(self.obj_names)},
                "seq": seq,
            })
        return True

    def add_many(self, solutions, episode: int) -> int:
        """``solutions`` = iterable of (reward_vec, seq). Returns #added."""
        return sum(int(self.add(vec, seq, episode)) for vec, seq in solutions)

    def hypervolume(self) -> float:
        if not self.pts:
            return 0.0
        pts = np.stack(self.pts)
        # Capture the reference point once, on the first non-empty call.
        # Recomputing ``pts.min(axis=0) - 1.0`` per call is a moving
        # reference: a 1-point front then always scores 1.0 regardless of
        # where the point sits, and HV values are incomparable across
        # calls. A fixed reference keeps the metric monotone over time.
        if self._hv_ref is None:
            self._hv_ref = pts.min(axis=0) - 1.0
        return hypervolume(pts, self._hv_ref)

    def dump_front(self, path: str, extra: dict | None = None) -> None:
        _hv = self.hypervolume()
        payload = {
            "objectives": list(self.obj_names),
            # null (not bare NaN) for the >3-objective case so the file is
            # strict-JSON parseable (jq / JS / json.loads(strict)).
            "hypervolume": _hv if np.isfinite(_hv) else None,
            "num_points": len(self.pts),
            "front": [
                {"obj": {nm: float(v) for nm, v in zip(self.obj_names, pt)}, "seq": seq}
                for pt, seq in zip(self.pts, self.seqs)
            ],
        }
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def dump_all_candidates(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "objectives": list(self.obj_names),
            "num_candidates": len(self.all_candidates),
            "note": (
                "every solution admitted to the non-dominated front at any point "
                "during training (including later-pruned ones); the full reward_vec "
                "is stored so any objective set can be re-scored offline."
            ),
            "candidates": self.all_candidates,
        }
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


# TICKET dsnn-dfw.44, owner rulings 2026-09-19 (second design). Points live in
# LOG-RATIO space against the paired rev-exact reference: 0 is parity and LOWER
# IS BETTER. The coordinate is the UNFLOORED log ratio, because
# `env.paired_log_costs` floors the REWARD at the reference and that maps every
# plan at or below parity onto exactly 0. A point is a BAND: a 90 percent
# distribution-free interval for the MEDIAN of the plan's per-window paired log
# ratios. The first design used the q05..q95 of the ratios themselves, widened
# to the instrument drift floor; that band is the spread of ONE window, it was
# ten times wider than the differences between orders, and the whole run
# archived one point. This band is the uncertainty of the MEDIAN, it shrinks as
# a point absorbs measurements, and there is no drift floor.
_POOL_CAP = 512          # pooled windows kept per point per objective


@functools.lru_cache(maxsize=4096)
def _median_order_stat_lo(n: int) -> int:
    """1-based lower order statistic of the 90 percent median interval.

    The largest ``l`` with ``P(Bin(n, 1/2) <= l - 1) <= 0.05``, decided in
    exact integers (``20 * sum_k C(n, k) <= 2**n``) so the band carries no
    floating-point tie and no RNG. 0 when the sample is too small to exclude
    any order statistic.
    """
    if n < 1:
        raise ValueError(f"a median interval needs at least one sample, got {n}")
    total = 1 << n
    term = 1                       # C(n, 0), stepped by C(n,k+1)=C(n,k)(n-k)/(k+1)
    cum = 0
    lo = 0
    for k in range(0, n):
        cum += term
        if cum * 20 > total:
            break
        lo = k + 1
        term = term * (n - k) // (k + 1)
    return lo


def median_band(x) -> tuple:
    """``(median, lo, hi)`` of a sample, the 90 percent median interval."""
    a = np.sort(np.asarray(x, dtype=np.float64))
    if a.size == 0:
        raise ValueError("median_band got an empty sample")
    med = float(np.median(a))
    l = _median_order_stat_lo(int(a.size))
    if l < 1:
        # Too few windows to exclude any order statistic: the whole range.
        return med, float(a[0]), float(a[-1])
    return med, float(a[l - 1]), float(a[a.size - l])


class RatioBandArchive:

    def __init__(self, obj_names, cap: int = 64, quality_floor=None,
                 pool_cap: int = _POOL_CAP):
        self.obj_names = list(obj_names)
        if not self.obj_names:
            raise ValueError("RatioBandArchive needs at least one objective")
        self.cap = int(cap)
        if self.cap < 1:
            raise ValueError(f"RatioBandArchive cap must be >= 1, got {cap!r}")
        self.pool_cap = int(pool_cap)
        if self.pool_cap < 1:
            raise ValueError(
                f"RatioBandArchive pool_cap must be >= 1, got {pool_cap!r}")
        self.quality_floor = (None if quality_floor is None
                              else float(quality_floor))
        self.pts: list[np.ndarray] = []       # the medians ARE the coordinate
        self.lo: list[np.ndarray] = []
        self.hi: list[np.ndarray] = []
        self.samples: list[list] = []         # pooled windows, per objective
        self.counts: list[int] = []           # measurements pooled into a point
        self.seqs: list = []
        self.eps: list[int] = []
        self.all_candidates: list[dict] = []
        self._seen: set = set()
        self._hv_ref: np.ndarray | None = None
        self.n_merged = 0
        self.n_dropped_cap = 0

    def _windows(self, dist) -> list:
        out = []
        for nm in self.obj_names:
            w = np.asarray(dist[nm], dtype=np.float64).reshape(-1)
            if w.size == 0:
                raise ValueError(
                    f"RatioBandArchive: objective {nm!r} has no per-window "
                    f"ratio; a channel with no windows must still hand its one "
                    f"reading as a length-1 sample")
            if not np.all(np.isfinite(w)):
                raise ValueError(
                    f"RatioBandArchive: objective {nm!r} has a non-finite "
                    f"window ratio: {w.tolist()[:8]}")
            out.append(w)
        return out

    def _fit(self, sample) -> tuple:
        med = np.empty((len(self.obj_names),), dtype=np.float64)
        lo = np.empty_like(med)
        hi = np.empty_like(med)
        for k, w in enumerate(sample):
            med[k], lo[k], hi[k] = median_band(w)
        return med, lo, hi

    def band(self, i: int) -> tuple:
        return self.lo[i], self.hi[i]

    def band_width(self, i: int) -> float:
        return float(np.sum(self.hi[i] - self.lo[i]))

    def add(self, dist, seq, episode: int, quality=None) -> bool:
        if dist is None:
            raise ValueError(
                "RatioBandArchive.add got no distribution: a plan with no "
                "per-window ratios has no point (needs --cost-form "
                "paired-log and a plan log)")
        w = self._windows(dist)
        med, _lo, _hi = self._fit(w)
        if (self.quality_floor is not None and quality is not None
                and float(quality) < self.quality_floor):
            return False
        dominates: list[int] = []
        for i in range(len(self.pts)):
            better = med < self.lo[i]
            worse = med > self.hi[i]
            if not better.any() and not worse.any():
                # Inside the band everywhere: POOL the windows into the point
                # and re-fit, so a point that is measured again knows more.
                self.samples[i] = [
                    np.concatenate([self.samples[i][k], w[k]])[:self.pool_cap]
                    for k in range(len(self.obj_names))]
                self.pts[i], self.lo[i], self.hi[i] = self._fit(self.samples[i])
                self.counts[i] += 1
                self.n_merged += 1
                return False
            if worse.any() and not better.any():
                return False
            if better.any() and not worse.any():
                dominates.append(i)
        _drop = set(dominates)
        keep = [i for i in range(len(self.pts)) if i not in _drop]
        self.pts = [self.pts[i] for i in keep] + [med]
        self.lo = [self.lo[i] for i in keep] + [_lo]
        self.hi = [self.hi[i] for i in keep] + [_hi]
        self.samples = [self.samples[i] for i in keep] + [
            [x[:self.pool_cap] for x in w]]
        self.counts = [self.counts[i] for i in keep] + [1]
        self.seqs = [self.seqs[i] for i in keep] + [seq]
        self.eps = [self.eps[i] for i in keep] + [int(episode)]
        while len(self.pts) > self.cap:
            widths = [self.band_width(i) for i in range(len(self.pts))]
            drop = int(np.argmax(np.asarray(widths)))
            for lst in (self.pts, self.lo, self.hi, self.samples,
                        self.counts, self.seqs, self.eps):
                del lst[drop]
            self.n_dropped_cap += 1
        key = repr(seq)
        if key not in self._seen:
            self._seen.add(key)
            self.all_candidates.append({
                "episode": int(episode),
                "obj": self._named(med),
                "band_lo": self._named(_lo),
                "band_hi": self._named(_hi),
                "seq": seq,
            })
        return True

    def _named(self, vec) -> dict:
        return {nm: float(vec[k]) for k, nm in enumerate(self.obj_names)}

    def add_many(self, solutions, episode: int) -> int:
        return sum(int(self.add(d, s, episode, quality=q))
                   for d, s, q in solutions)

    def hypervolume(self) -> float:
        if not self.pts:
            return 0.0
        # The sweep MAXIMISES, so minimisation medians enter negated.
        pts = -np.stack(self.pts)
        if self._hv_ref is None:
            self._hv_ref = pts.min(axis=0) - 1.0
        return hypervolume(pts, self._hv_ref)

    def front(self) -> list:
        out = []
        for i in range(len(self.pts)):
            out.append({
                "obj": self._named(self.pts[i]),
                "band_lo": self._named(self.lo[i]),
                "band_hi": self._named(self.hi[i]),
                "n": int(self.counts[i]),
                "windows": {nm: int(self.samples[i][k].size)
                            for k, nm in enumerate(self.obj_names)},
                "episode": int(self.eps[i]),
                "seq": self.seqs[i],
            })
        return out

    def dump_front(self, path: str, extra: dict | None = None) -> None:
        _hv = self.hypervolume()
        payload = {
            "objectives": list(self.obj_names),
            "space": "log ratio against the paired rev-exact reference; "
                     "0 is parity and lower is better",
            "band": "90 percent distribution-free interval for the median of "
                    "the pooled per-window paired log ratios",
            "hypervolume": _hv if np.isfinite(_hv) else None,
            "num_points": len(self.pts),
            "cap": int(self.cap),
            "pool_cap": int(self.pool_cap),
            "merged_measurements": int(self.n_merged),
            "dropped_at_cap": int(self.n_dropped_cap),
            "front": self.front(),
        }
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def dump_all_candidates(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "objectives": list(self.obj_names),
            "num_candidates": len(self.all_candidates),
            "candidates": self.all_candidates,
        }
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
