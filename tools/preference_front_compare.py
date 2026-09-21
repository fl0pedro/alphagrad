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


def figure(path, swept_doc, archive_doc, plans, table, objectives, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("w_lat", BLUE_RAMP)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.2, 5.0), dpi=200,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26})
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
    ax_a.set_xlabel("latency, log ratio against the paired rev-exact reference",
                    color=INK_2, fontsize=10)
    ax_a.set_ylabel("memory, log ratio", color=INK_2, fontsize=10)
    ax_a.set_title("A  the front, coloured by the preference that produced it",
                   color=INK, fontsize=11, loc="left", pad=10)
    ax_a.text(0.02, 0.03,
              f"coverage swept of archive {comparison['coverage_swept_of_archive']:.2f}"
              f"   archive of swept {comparison['coverage_archive_of_swept']:.2f}\n"
              f"hypervolume swept {comparison['hypervolume_swept']:.4g}"
              f"   archive {comparison['hypervolume_archive']:.4g}",
              transform=ax_a.transAxes, fontsize=8.5, color=INK_2,
              va="bottom", ha="left")
    leg = ax_a.legend(loc="upper right", frameon=False, fontsize=9,
                      labelcolor=INK_2)
    for handle in leg.legend_handles:
        try:
            handle.set_color(MUTED)
        except Exception:
            pass
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
                      mew=1.4, mec=SURFACE, lw=2.0, elinewidth=1.0,
                      alpha=0.95, capsize=0, label=name, zorder=3)
        if med.size:
            ax_b.annotate(name, (t[-1], med[-1]), textcoords="offset points",
                          xytext=(6, 0), color=colour, fontsize=9,
                          va="center")
    ax_b.axhline(0.0, color=AXIS, lw=1.0, zorder=1)
    ax_b.set_xlabel("latency weight of the pinned w", color=INK_2, fontsize=10)
    ax_b.set_ylabel("achieved median log ratio", color=INK_2, fontsize=10)
    ax_b.set_title("B  does the policy honour its conditioning",
                   color=INK, fontsize=11, loc="left", pad=10)
    ax_b.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK_2)

    fig.text(0.008, 0.015,
             f"Lower is better on both axes; 0 is parity with the paired "
             f"reverse-exact reference. Bars are the 90 percent "
             f"distribution-free interval for the median of the pooled "
             f"per-window paired log ratios. Hypervolume for both fronts "
             f"under ONE nadir, "
             f"{np.array2string(np.asarray(comparison['nadir']), precision=4)}"
             f" in {comparison['nadir_space']}.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.17)
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
    table = psweep.per_weight_table(plans, a.quality_floor)
    comparison["per_weight"] = table

    out_json = os.path.join(a.out, "preference_front_comparison.json")
    with open(out_json, "w") as fh:
        json.dump(comparison, fh, indent=2)
    out_md = os.path.join(a.out, "preference_front_table.md")
    with open(out_md, "w") as fh:
        fh.write(table_markdown(table, objectives))
    out_png = os.path.join(a.out, "preference_front.png")
    figure(out_png, swept_doc, archive_doc, plans, table, objectives,
           comparison)
    print(f"[compare] swept {comparison['num_swept']} points, archive "
          f"{comparison['num_archive']}; coverage swept-of-archive "
          f"{comparison['coverage_swept_of_archive']:.3f}, archive-of-swept "
          f"{comparison['coverage_archive_of_swept']:.3f}; hypervolume "
          f"{comparison['hypervolume_swept']:.6g} against "
          f"{comparison['hypervolume_archive']:.6g} under nadir "
          f"{comparison['nadir']}", flush=True)
    print(f"[compare] {out_json}\n[compare] {out_md}\n[compare] {out_png}",
          flush=True)
    print(table_markdown(table, objectives))
    return 0


if __name__ == "__main__":
    sys.exit(main())
