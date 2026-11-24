"""Figures for the write-up, from committed result CSVs.

    python plots.py sweep workspace/results/sweep_20261124.csv     # model fit + break-even maps
Writes PNGs to workspace/figures/.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

FIG_DIR = os.path.join("workspace", "figures")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GAIN_LOSS = LinearSegmentedColormap.from_list("gain_loss", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def _load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("T", "s", "p", "seconds", "base_seconds", "saved", "predicted", "wasted_tool_s"):
            r[k] = float(r[k])
        r["slots"] = int(r["slots"])
    return rows


def _cells(rows, key):
    cells = defaultdict(list)
    for r in rows:
        cells[key(r)].append(r)
    return cells


def sweep_fit(rows, out):
    """Measured vs predicted saving, one dot per (backend, s, p, T) cell."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for slots, color, label in ((0, BLUE, "unlimited backend"), (1, ORANGE, "one backend slot")):
        cells = _cells([r for r in rows if r["slots"] == slots], lambda r: (r["s"], r["p"], r["T"]))
        xs = [sum(r["predicted"] for r in rs) / len(rs) for rs in cells.values()]
        ys = [sum(r["saved"] for r in rs) / len(rs) for rs in cells.values()]
        ax.scatter(xs, ys, s=22, color=color, edgecolor=SURFACE, linewidth=1, label=label, zorder=3)
    lo = min(min(r["predicted"] for r in rows), min(r["saved"] for r in rows)) * 0.6
    hi = max(r["predicted"] for r in rows) * 1.05
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle="--", zorder=2)
    ax.text(hi, hi, "  measured = predicted", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("predicted saving per run (model units)")
    ax.set_ylabel("measured saving per run (model units)")
    ax.set_title("Latency model vs measurement", loc="left")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def sweep_breakeven(rows, out, s=0.5):
    """Speedup (geomean base/spec) over guess accuracy p x think time T, per backend, at guesser latency s."""
    rows = [r for r in rows if r["s"] == s]
    Ts = sorted({r["T"] for r in rows})
    ps = sorted({r["p"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.5)
    for ax, slots, title in zip(axes, (0, 1), ("Unlimited backend", "One backend slot (rate-limited API)")):
        cells = _cells([r for r in rows if r["slots"] == slots], lambda r: (r["p"], r["T"]))
        grid = [[math.exp(sum(math.log(r["base_seconds"] / r["seconds"]) for r in cells[(p, T)]) /
                          len(cells[(p, T)])) for T in Ts] for p in ps]
        im = ax.imshow(grid, origin="lower", cmap=GAIN_LOSS, norm=norm, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(Ts)), [f"{T:g}" for T in Ts])
        ax.set_yticks(range(len(ps)), [f"{p:g}" for p in ps])
        ax.set_xlabel("agent think time T (units; tools take 0.7-2.0)")
        ax.set_title(title, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("guess accuracy p")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("speedup vs no speculation (x)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Where speculation pays: guesser latency s = {s:g}", x=0.01, ha="left", color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "sweep":
        sys.exit(__doc__)
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = _load(sys.argv[2])
    sweep_fit(rows, os.path.join(FIG_DIR, "sweep_model_fit.png"))
    for s in sorted({r["s"] for r in rows}):
        sweep_breakeven(rows, os.path.join(FIG_DIR, f"sweep_breakeven_s{s:g}.png"), s=s)
    print(f"wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
