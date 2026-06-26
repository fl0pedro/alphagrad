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
import numpy as np


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


def hypervolume(points, ref) -> float:
    """Hypervolume dominated by ``points`` (maximization) above nadir ``ref``.
    Returns NaN for >3 objectives (no cheap exact formula wired)."""
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    m = pts.shape[1]
    if m not in (2, 3):
        return float("nan")
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

    def __init__(self, obj_names, obj_idx):
        self.obj_names = list(obj_names)
        self.obj_idx = [int(i) for i in obj_idx]
        self.pts: list[np.ndarray] = []          # live front objective vectors
        self.seqs: list = []                     # parallel sequences
        self.all_candidates: list[dict] = []     # every admitted point
        self._seen: set = set()

    def add(self, reward_vec, seq, episode: int) -> bool:
        """Admit one (reward_vec, seq) to the front. Returns True if it was
        non-dominated on arrival (and thus added + logged). Incremental O(front):
        reject if dominated by / equal to an existing front point, else drop the
        points it dominates. ``obj_idx`` is trusted (a bad index raises rather
        than being silently swallowed, so channel/width drift surfaces loudly)."""
        g = np.array([float(reward_vec[i]) for i in self.obj_idx], dtype=np.float64)
        if not np.all(np.isfinite(g)):
            return False
        for p in self.pts:
            # dominated by, or objective-identical to, an existing front point
            if np.allclose(g, p) or (np.all(p >= g) and np.any(p > g)):
                return False
        # g is non-dominated → keep it, drop any existing points it dominates
        survivors = [
            (p, s) for p, s in zip(self.pts, self.seqs)
            if not (np.all(g >= p) and np.any(g > p))
        ]
        self.pts = [p for p, _ in survivors] + [g]
        self.seqs = [s for _, s in survivors] + [seq]
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
        return hypervolume(pts, pts.min(axis=0) - 1.0)

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
