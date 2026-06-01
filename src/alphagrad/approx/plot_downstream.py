"""Plot downstream training curves from CSVs produced by ``downstream_train.py``.

Reads every ``*_seed*.csv`` matching ``--csv-glob`` and groups them by
the prefix before ``_seed`` (the variant label set in
``run_downstream_train.sh``). For each group it averages across seeds
and renders four PNGs into ``--output-dir``:

* ``loss.png``       — train loss vs step (log-y).
* ``acc.png``        — test accuracy vs step.
* ``step_time.png``  — wall-time per step (the actual cost we care about).
* ``cossim.png``     — cosine similarity of the approximate gradient
  against the exact reference, vs step (sanity for the replay).

Also emits ``summary.csv`` with final-step values per variant and the
``steps_to_90pct_acc`` if reached.

Conventions match ``alphagrad/eval/MLP/evaluation_MLP.ipynb`` (font dict,
fill_between for ±2σ across seeds).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _font():
    return {"family": "DejaVu Sans", "weight": "normal", "size": 13}


def _parse_label(path: str) -> tuple[str, int]:
    """Split ``<variant>_seed<N>.csv`` into ``(variant, seed)``."""
    stem = Path(path).stem
    m = re.match(r"^(?P<v>.*?)_seed(?P<s>\d+)$", stem)
    if not m:
        return stem, 0
    return m.group("v"), int(m.group("s"))


def _load_csv(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline="")))
    out: dict[str, np.ndarray] = {}
    if not rows:
        return out
    for k in rows[0].keys():
        out[k] = np.array([
            float(r[k]) if r[k] not in ("", "nan", "None") else np.nan
            for r in rows
        ])
    return out


def _aggregate(variant_runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack runs along a new leading axis, return per-step mean/std."""
    if not variant_runs:
        return {}
    base = variant_runs[0]
    out: dict[str, np.ndarray] = {"step": base["step"]}
    for k in (
        "train_loss", "test_acc", "step_wall_ms",
        "peak_mem_bytes", "grad_cossim_vs_exact",
    ):
        if k not in base:
            continue
        stacked = np.stack(
            [r[k][: len(base["step"])] for r in variant_runs], axis=0,
        )
        # Skip the NaN strides (test_acc and cossim are sparse).
        with np.errstate(invalid="ignore"):
            out[f"{k}_mean"] = np.nanmean(stacked, axis=0)
            out[f"{k}_std"] = np.nanstd(stacked, axis=0)
    return out


def _pick_color(label: str, idx: int) -> str:
    palette = plt.cm.tab10.colors
    return palette[idx % len(palette)]


def _plot_series(
    by_variant: dict[str, dict[str, np.ndarray]],
    *,
    y_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    log_y: bool = False,
    reference: str | None = None,
):
    plt.rc("font", **_font())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, agg) in enumerate(sorted(by_variant.items())):
        steps = agg.get("step")
        mean = agg.get(f"{y_key}_mean")
        std = agg.get(f"{y_key}_std")
        if steps is None or mean is None:
            continue
        # Drop NaN steps (sparse measurements like test_acc).
        valid = ~np.isnan(mean)
        if not valid.any():
            continue
        is_ref = (label == reference)
        ax.plot(
            steps[valid], mean[valid],
            label=label,
            color="black" if is_ref else _pick_color(label, i),
            linestyle="--" if is_ref else "-",
            linewidth=1.5,
        )
        if std is not None:
            valid_std = valid & ~np.isnan(std)
            if valid_std.any():
                ax.fill_between(
                    steps[valid_std],
                    mean[valid_std] - std[valid_std],
                    mean[valid_std] + std[valid_std],
                    color=_pick_color(label, i),
                    alpha=0.15,
                )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


def _steps_to_acc(
    steps: np.ndarray, acc: np.ndarray, target: float
) -> float:
    valid = ~np.isnan(acc)
    if not valid.any():
        return float("nan")
    sv, av = steps[valid], acc[valid]
    reached = av >= target
    if not reached.any():
        return float("nan")
    return float(sv[np.argmax(reached)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv-glob", required=True,
        help="Glob over per-source CSVs, e.g. 'out/downstream/*_seed*.csv'.",
    )
    p.add_argument(
        "--output-dir", default="out/downstream/plots",
        help="Directory for the four PNGs and summary.csv.",
    )
    p.add_argument(
        "--reference", default="jax_grad",
        help="Variant label to draw as dashed black (the exact reference).",
    )
    p.add_argument(
        "--acc-target", type=float, default=0.9,
        help="Target accuracy for `steps_to_<target>_acc` in summary.csv.",
    )
    args = p.parse_args()

    csv_paths = sorted(glob.glob(args.csv_glob))
    if not csv_paths:
        raise SystemExit(f"No CSVs matched {args.csv_glob!r}")

    by_variant_runs: dict[str, list[dict]] = defaultdict(list)
    for path in csv_paths:
        variant, seed = _parse_label(path)
        runs = _load_csv(path)
        if runs:
            by_variant_runs[variant].append(runs)
            print(f"loaded {variant} seed={seed} (n_steps={len(runs['step'])})")

    by_variant = {v: _aggregate(rs) for v, rs in by_variant_runs.items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_series(
        by_variant, y_key="train_loss",
        title="Train loss", ylabel="train loss",
        out_path=out_dir / "loss.png", log_y=True,
        reference=args.reference,
    )
    _plot_series(
        by_variant, y_key="test_acc",
        title="Test accuracy", ylabel="test accuracy",
        out_path=out_dir / "acc.png",
        reference=args.reference,
    )
    _plot_series(
        by_variant, y_key="step_wall_ms",
        title="Wall time per step (ms)", ylabel="ms / step",
        out_path=out_dir / "step_time.png", log_y=True,
        reference=args.reference,
    )
    _plot_series(
        by_variant, y_key="peak_mem_bytes",
        title="Peak HBM per step", ylabel="bytes",
        out_path=out_dir / "peak_mem.png", log_y=True,
        reference=args.reference,
    )
    _plot_series(
        by_variant, y_key="grad_cossim_vs_exact",
        title="Gradient cossim vs exact (sanity)",
        ylabel="cossim(approx, exact)",
        out_path=out_dir / "cossim.png",
        reference=args.reference,
    )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "final_loss", "final_test_acc",
            "median_step_ms", "median_cossim",
            f"steps_to_{int(args.acc_target * 100)}pct_acc",
        ])
        for variant, agg in sorted(by_variant.items()):
            steps = agg.get("step", np.array([]))
            loss = agg.get("train_loss_mean", np.array([np.nan]))
            acc = agg.get("test_acc_mean", np.array([np.nan]))
            tms = agg.get("step_wall_ms_mean", np.array([np.nan]))
            css = agg.get("grad_cossim_vs_exact_mean", np.array([np.nan]))
            w.writerow([
                variant,
                f"{float(loss[-1]) if len(loss) else float('nan'):+.5g}",
                f"{float(np.nanmax(acc)) if len(acc) else float('nan'):.4f}",
                f"{float(np.nanmedian(tms)) if len(tms) else float('nan'):.3f}",
                f"{float(np.nanmedian(css)) if len(css) else float('nan'):.4f}",
                f"{_steps_to_acc(steps, acc, args.acc_target):.0f}",
            ])
    print(f"  -> {summary_path}")


if __name__ == "__main__":
    main()
