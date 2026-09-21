"""The swept front against the training archive's front (dsnn-dfw.86).

Reads the two `pareto_front.json` documents and the sweep's plan records,
and writes three things:

* `preference_front_comparison.json` -- set coverage in BOTH directions and
  the hypervolume of both fronts under ONE nadir, computed here from both
  fronts together and recorded in the file. Neither archive's own
  `hypervolume` field is read: both classes freeze their own nadir on the
  first non-empty call, so those two numbers are not in one space.
* `preference_front_table.md` -- one row per preference point: the median
  latency and memory log ratio with the 90 percent band, and the feasible
  fraction.
* `preference_front.png` -- two panels. A: the fronts in ratio space, every
  generated point coloured by the preference that produced it, the archive
  front in grey. B: the achieved ratio against the preference weight, one
  line per channel, unsmoothed.

Pure host arithmetic over JSON. CPU node, no JAX.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from alphagrad.approx.common import preference_sweep as psweep


# The validated reference palette (dataviz skill, references/palette.md),
# light surface. Slot 1 blue and slot 2 orange carry the two CHANNELS in
# panel B; the blue 100->700 ramp carries the preference WEIGHT in panel A,
# which is a magnitude and so is one hue, light to dark.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_LAT = "#2a78d6"
SERIES_MEM = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swept", required=True,
                   help="preference_sweep_front.json")
    p.add_argument("--plans", required=True,
                   help="preference_sweep_plans.jsonl")
    p.add_argument("--archive", required=True,
                   help="the training run's pareto_front.json")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--quality-floor", type=float, default=0.9)
    p.add_argument("--archive-window", type=int, default=200,
                   help="Compare against the archive points admitted in the "
                        "last N episodes of the run as well. The archive is "
                        "a lifetime union and the swept front is one "
                        "checkpoint, so this is the like-for-like set.")
    p.add_argument("--objectives", default="latency,peak_memory",
                   help="the two objective names, compute first")
    return p


def _weight_of(plans, objectives):
    """order tuple -> the latency weight that produced that plan."""
    out = {}
    for p in plans:
        key = tuple(int(v) for v in p["order"])
        out.setdefault(key, float(p["w"][0]))
    return out


def _order_of(point) -> tuple:
    seq = point.get("seq")
    if isinstance(seq, dict):
        seq = seq.get("seq")
    if not seq:
        return ()
    return tuple(int(v) for v, _calls in seq)


def _band_arrays(doc, objectives):
    """(P, 2) medians and the (2, 2, P) error-bar offsets of one front."""
    med = psweep.front_points(doc, objectives)
    lo = np.asarray([[float(p["band_lo"][nm]) for nm in objectives]
                     for p in doc.get("front") or ()], dtype=np.float64)
    hi = np.asarray([[float(p["band_hi"][nm]) for nm in objectives]
                     for p in doc.get("front") or ()], dtype=np.float64)
    lo = lo.reshape(med.shape)
    hi = hi.reshape(med.shape)
    return med, np.abs(med - lo), np.abs(hi - med)


def _view(front_pts, rollouts, axis: int, pad: float = 0.08):
    """The window the bulk of the data lives in, on one axis.

    Every front point is kept -- a front point outside the view would be the
    comparison hiding its own evidence. The ROLLOUT cloud enters by its 2nd
    to 98th percentile: a handful of plans measure a log ratio of +5 (a
    memory blow-up, which is a real measurement and is in the table and the
    file), and letting those three points set the axis collapses the other
    173 onto one pixel.
    """
    keep = [np.asarray(x, dtype=np.float64).reshape(-1)
            for x in front_pts if np.size(x)]
    r = np.asarray(rollouts, dtype=np.float64).reshape(-1)
    if r.size:
        keep.append(np.percentile(r, [2.0, 98.0]))
    allv = np.concatenate(keep) if keep else np.zeros(1)
    lo, hi = float(np.min(allv)), float(np.max(allv))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    if span <= 0.0:
        span = 1.0
    return lo - pad * span, hi + pad * span


def figure(path, swept_doc, archive_doc, archive_late_doc, plans, table,
           objectives, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    cmap = LinearSegmentedColormap.from_list("w_lat", BLUE_RAMP)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.6, 5.2), dpi=200,
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.30})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_a, ax_b):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)

    # ---- PANEL A: the two fronts, with the preference as the colour key ---
    a_med, a_lo, a_hi = _band_arrays(archive_doc, objectives)
    if a_med.size:
        ax_a.errorbar(a_med[:, 0], a_med[:, 1],
                      xerr=np.stack([a_lo[:, 0], a_hi[:, 0]]),
                      yerr=np.stack([a_lo[:, 1], a_hi[:, 1]]),
                      fmt="o", ms=6.5, mfc="none", mec=MUTED, mew=1.4,
                      ecolor=GRID, elinewidth=1.2, capsize=0, zorder=2,
                      label=f"training archive ({a_med.shape[0]})")
    l_med = psweep.front_points(archive_late_doc, objectives)
    if l_med.size:
        ax_a.scatter(l_med[:, 0], l_med[:, 1], s=42, marker="o",
                     facecolors=MUTED, edgecolors=SURFACE, linewidths=1.4,
                     zorder=3)
    w_of = _weight_of(plans, objectives)
    ok = [p for p in plans if p.get("refused") is None]
    if ok:
        ax_a.scatter(
            [p["band"]["latency"]["median"] for p in ok],
            [p["band"]["memory"]["median"] for p in ok],
            c=[float(p["w"][0]) for p in ok], cmap=cmap, norm=norm,
            s=16, alpha=0.45, linewidths=0.0, zorder=3,
            label=f"swept rollouts ({len(ok)})")
    s_med, s_lo, s_hi = _band_arrays(swept_doc, objectives)
    if s_med.size:
        s_w = np.asarray([w_of.get(_order_of(p), np.nan)
                          for p in swept_doc["front"]], dtype=np.float64)
        ax_a.errorbar(s_med[:, 0], s_med[:, 1],
                      xerr=np.stack([s_lo[:, 0], s_hi[:, 0]]),
                      yerr=np.stack([s_lo[:, 1], s_hi[:, 1]]),
                      fmt="none", ecolor=AXIS, elinewidth=1.2, capsize=0,
                      zorder=4)
        ax_a.scatter(s_med[:, 0], s_med[:, 1], c=s_w, cmap=cmap, norm=norm,
                     s=90, edgecolors=SURFACE, linewidths=1.6, zorder=5,
                     label=f"swept front ({s_med.shape[0]})")
    ax_a.axhline(0.0, color=AXIS, lw=1.0, zorder=1)
    ax_a.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
    roll_lat = np.asarray([p["band"]["latency"]["median"] for p in ok])
    roll_mem = np.asarray([p["band"]["memory"]["median"] for p in ok])
    xlim = _view([a_med[:, 0] if a_med.size else [],
                  s_med[:, 0] if s_med.size else []], roll_lat, 0)
    ylim = _view([a_med[:, 1] if a_med.size else [],
                  s_med[:, 1] if s_med.size else []], roll_mem, 1)
    outside = int(np.sum(~((roll_lat >= xlim[0]) & (roll_lat <= xlim[1])
                           & (roll_mem >= ylim[0]) & (roll_mem <= ylim[1]))))
    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(*ylim)
    ax_a.set_xlabel("latency, log ratio against the paired rev-exact reference",
                    color=INK_2, fontsize=10)
    ax_a.set_ylabel("memory, log ratio", color=INK_2, fontsize=10)
    ax_a.set_title("A  the front, coloured by the preference that produced it",
                   color=INK, fontsize=11, loc="left", pad=10)
    ax_a.text(0.015, 0.075,
              f"coverage: swept of archive "
              f"{comparison['coverage_swept_of_archive']:.2f}, archive of "
              f"swept {comparison['coverage_archive_of_swept']:.2f}\n"
              f"against the last "
              f"{comparison['archive_window_episodes']} episodes: "
              f"{comparison['coverage_swept_of_archive_late']:.2f} and "
              f"{comparison['coverage_archive_late_of_swept']:.2f}\n"
              f"hypervolume: swept {comparison['hypervolume_swept']:.4g}, "
              f"archive {comparison['hypervolume_archive']:.4g}, late "
              f"{comparison['hypervolume_archive_late']:.4g}"
              + (f"\n{outside} of {len(ok)} rollouts lie outside this view"
                 if outside else ""),
              transform=ax_a.transAxes, fontsize=8.5, color=INK_2,
              va="bottom", ha="left", linespacing=1.5)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=6.5, mfc="none",
               mec=MUTED, mew=1.4,
               label=f"training archive front ({a_med.shape[0]})"),
        Line2D([], [], marker="o", ls="none", ms=6.5, color=MUTED,
               mec=SURFACE, mew=1.4,
               label=f"admitted in the last "
                     f"{comparison['archive_window_episodes']} episodes "
                     f"({comparison['num_archive_late']})"),
        Line2D([], [], marker="o", ls="none", ms=4.5, color=BLUE_RAMP[6],
               alpha=0.6, label=f"swept rollouts ({len(ok)})"),
        Line2D([], [], marker="o", ls="none", ms=9, color=BLUE_RAMP[7],
               mec=SURFACE, mew=1.6,
               label=f"swept front ({s_med.shape[0] if s_med.size else 0})"),
    ]
    ax_a.legend(handles=handles, loc="center right", frameon=False,
                fontsize=9, labelcolor=INK_2, handletextpad=0.6)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_a,
        fraction=0.045, pad=0.02)
    cb.set_label("latency weight of w (memory weight = 1 - it)",
                 color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    # ---- PANEL B: monotonicity, unsmoothed -------------------------------
    rows = [r for r in table if r["n"]]
    t = np.asarray([float(r["w"][0]) for r in rows])
    for name, key, colour in (("latency", "latency", SERIES_LAT),
                              ("memory", "memory", SERIES_MEM)):
        med = np.asarray([float(r[f"{key}_median"]) for r in rows])
        lo = np.asarray([float(r[f"{key}_lo"]) for r in rows])
        hi = np.asarray([float(r[f"{key}_hi"]) for r in rows])
        ax_b.errorbar(t, med, yerr=np.stack([np.abs(med - lo),
                                             np.abs(hi - med)]),
                      fmt="-o", color=colour, ecolor=colour, ms=6,
                      mew=1.4, mec=SURFACE, lw=2.0, elinewidth=1.2,
                      alpha=0.95, capsize=0, label=name, zorder=3)
        if med.size:
            ax_b.annotate(name, (t[-1], med[-1]), textcoords="offset points",
                          xytext=(8, 0), color=colour, fontsize=9,
                          va="center", ha="left", annotation_clip=False)
    ax_b.axhline(0.0, color=AXIS, lw=1.0, zorder=1)
    if t.size:
        ax_b.set_xlim(float(t.min()) - 0.04, float(t.max()) + 0.16)
    ax_b.set_xlabel("latency weight of the pinned w", color=INK_2, fontsize=10)
    ax_b.set_ylabel("achieved median log ratio, 90 percent band",
                    color=INK_2, fontsize=10)
    ax_b.set_title("B  does the policy honour its conditioning",
                   color=INK, fontsize=11, loc="left", pad=10)
    ax_b.legend(loc="upper left", frameon=False, fontsize=9,
                labelcolor=INK_2)

    fig.text(0.008, 0.012,
             f"Lower is better on both axes; 0 is parity with the paired "
             f"reverse-exact reference.\n"
             f"Panel A's bars are each point's own 90 percent "
             f"distribution-free interval for the median of its pooled "
             f"per-window paired log ratios; panel B's is the same interval "
             f"over the "
             f"{max((r['n'] for r in table if r['n']), default=0)} plans of "
             f"that weight.\n"
             f"Hypervolume for both fronts under ONE nadir, "
             f"{np.array2string(np.asarray(comparison['nadir']), precision=4)}"
             f", in {comparison['nadir_space']}. Quality floor "
             f"{comparison['quality_floor']:g} on grad-cosine.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom",
             linespacing=1.6)
    fig.subplots_adjust(left=0.065, right=0.965, top=0.91, bottom=0.20)
    fig.savefig(path, facecolor=SURFACE)
    return path


def table_markdown(table, objectives) -> str:
    lines = [
        "| w (latency, memory) | plans | refused | "
        f"{objectives[0]} median [90% band] | {objectives[1]} median "
        f"[90% band] | quality median | feasible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table:
        if not r["n"]:
            lines.append(f"| {r['w'][0]:.2f}, {r['w'][1]:.2f} | 0 | "
                         f"{r['refused']} | - | - | - | - |")
            continue
        lines.append(
            f"| {r['w'][0]:.2f}, {r['w'][1]:.2f} | {r['n']} | {r['refused']} | "
            f"{r['latency_median']:+.4f} [{r['latency_lo']:+.4f}, "
            f"{r['latency_hi']:+.4f}] | "
            f"{r['memory_median']:+.4f} [{r['memory_lo']:+.4f}, "
            f"{r['memory_hi']:+.4f}] | {r['quality_median']:.4f} | "
            f"{r['feasible_fraction']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    a = make_argparser().parse_args()
    objectives = tuple(x.strip() for x in a.objectives.split(","))
    if len(objectives) != 2:
        raise ValueError(f"--objectives wants two names, got {a.objectives!r}")
    with open(a.swept) as fh:
        swept_doc = json.load(fh)
    with open(a.archive) as fh:
        archive_doc = json.load(fh)
    plans = psweep.load_plan_records(a.plans)
    os.makedirs(a.out, exist_ok=True)

    swept = psweep.front_points(swept_doc, objectives)
    archive = psweep.front_points(archive_doc, objectives)
    comparison = psweep.compare_fronts(swept, archive, objectives)
    comparison["swept_front"] = a.swept
    comparison["archive_front"] = a.archive
    comparison["plans"] = a.plans
    comparison["rollouts"] = len(plans)
    comparison["rollouts_measured"] = sum(
        1 for p in plans if p.get("refused") is None)
    comparison["quality_floor"] = float(a.quality_floor)
    for name in ("seed", "checkpoint", "checkpoint_episode", "lag_lambda",
                 "run_name", "plans_per_weight"):
        if name in swept_doc:
            comparison[name] = swept_doc[name]
    if "seed" not in comparison:
        # A front written before the sweep put the seed on the file. The
        # checkpoint it names is the authority, so read it there rather than
        # parsing a run name: five seeds are compared as a set and a set
        # keyed on a guess is a set that is wrong quietly.
        from alphagrad.approx.common.checkpoint import read_ppo_meta
        comparison["seed"] = int(
            read_ppo_meta(swept_doc["checkpoint"])["args"]["seed"])
    # ---- THE LIKE-FOR-LIKE ARCHIVE ------------------------------------
    # The swept front is ONE checkpoint. The archive is the union over the
    # whole run. The window keeps the points that entered the front in its
    # last `--archive-window` episodes, and the coverage against that set is
    # the one the two sides can both be held to. The nadir does not move:
    # the window is a subset of the archive the nadir was taken over, so all
    # three hypervolumes stay in one space.
    archive_late_doc = psweep.front_window(archive_doc, a.archive_window)
    archive_late = psweep.front_points(archive_late_doc, objectives)
    ref = np.asarray(comparison["nadir"], dtype=np.float64)
    comparison["archive_window_episodes"] = int(a.archive_window)
    comparison["archive_window_since_episode"] = int(
        archive_late_doc["window_since_episode"])
    comparison["num_archive_late"] = int(archive_late.shape[0])
    comparison["coverage_swept_of_archive_late"] = psweep.set_coverage(
        swept, archive_late)
    comparison["coverage_archive_late_of_swept"] = psweep.set_coverage(
        archive_late, swept)
    comparison["hypervolume_archive_late"] = psweep.hypervolume_of(
        archive_late, ref)
    table = psweep.per_weight_table(plans, a.quality_floor)
    comparison["per_weight"] = table

    out_json = os.path.join(a.out, "preference_front_comparison.json")
    with open(out_json, "w") as fh:
        json.dump(comparison, fh, indent=2)
    out_md = os.path.join(a.out, "preference_front_table.md")
    with open(out_md, "w") as fh:
        fh.write(table_markdown(table, objectives))
    out_png = os.path.join(a.out, "preference_front.png")
    figure(out_png, swept_doc, archive_doc, archive_late_doc, plans, table,
           objectives, comparison)
    print(f"[compare] swept {comparison['num_swept']} points, archive "
          f"{comparison['num_archive']}, archive since episode "
          f"{comparison['archive_window_since_episode']} "
          f"{comparison['num_archive_late']}", flush=True)
    print(f"[compare] coverage swept-of-archive "
          f"{comparison['coverage_swept_of_archive']:.3f}, archive-of-swept "
          f"{comparison['coverage_archive_of_swept']:.3f}, swept-of-late "
          f"{comparison['coverage_swept_of_archive_late']:.3f}, "
          f"late-of-swept {comparison['coverage_archive_late_of_swept']:.3f}",
          flush=True)
    print(f"[compare] hypervolume swept {comparison['hypervolume_swept']:.6g}, "
          f"archive {comparison['hypervolume_archive']:.6g}, archive-late "
          f"{comparison['hypervolume_archive_late']:.6g}, under nadir "
          f"{comparison['nadir']}", flush=True)
    print(f"[compare] {out_json}\n[compare] {out_md}\n[compare] {out_png}",
          flush=True)
    print(table_markdown(table, objectives))
    return 0


if __name__ == "__main__":
    sys.exit(main())
