"""Plot the held-out benchmark by version from results/benchmark.tsv.

  python plot_benchmark.py            # -> ../results/benchmark_versions.png

Versions are ordered, so they take steps of one blue ramp (light = oldest),
validated with the dataviz ordinal checks. Bars are the mean over seeds; dots
are the individual seeds.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # oldest -> newest
STAGES = ["reach", "close", "grasp", "rise", "lift", "hold"]
METRICS = [("stage_score", "Stage score"), ("success_1", "Success@1"),
           ("success_k", "Success@3"), ("lifted_1", "Lifted@1\n(old metric)")]


def load():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "benchmark.tsv")), delimiter="\t"))
    # Chronological order (v1.0 < v1.1 < v2.0 ...), not the order runs finished.
    versions = sorted({r["label"] for r in rows},
                      key=lambda v: [int(x) for x in v.lstrip("v").split(".")])
    by = {v: [r for r in rows if r["label"] == v] for v in versions}
    return versions, by


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    versions, by = load()
    colors = {v: RAMP[i % len(RAMP)] for i, v in enumerate(versions)}
    n = len(versions)

    fig = plt.figure(figsize=(14, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.7, 0.9], wspace=0.42)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) headline metrics: grouped bars (mean) + seed dots
    width = 0.8 / n
    for i, v in enumerate(versions):
        for j, (key, _) in enumerate(METRICS):
            vals = np.array([float(r[key]) for r in by[v]])
            x = j + (i - (n - 1) / 2) * width
            ax1.bar(x, vals.mean(), width * 0.9, color=colors[v], edgecolor=SURFACE,
                    linewidth=1, label=v if j == 0 else None, zorder=2)
            ax1.scatter(np.full(len(vals), x), vals, s=22, color=INK, edgecolor=SURFACE,
                        linewidth=0.8, zorder=3)
    ax1.set_xticks(range(len(METRICS)), [m[1] for m in METRICS], color=INK)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("% of 200 held-out positions", color=INK2, fontsize=9)
    ax1.set_title("Headline metrics (bar = mean, dot = seed)", loc="left",
                  color=INK, fontsize=11)
    style(ax1)

    # (b) attempt-1 stage funnel
    xs = np.arange(len(STAGES))
    ends = []
    for v in versions:
        rates = np.mean([[float(r[f"rate_{s}"]) for s in STAGES] for r in by[v]], axis=0)
        ax2.plot(xs, rates, color=colors[v], linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ends.append([rates[-1], rates[-1], v])
    # Direct end labels, pushed apart so close values don't collide.
    ends.sort(key=lambda e: -e[0])
    for k in range(1, len(ends)):
        ends[k][1] = min(ends[k][1], ends[k - 1][1] - 6)
    for val, y, v in ends:
        ax2.annotate(f"{v}  {val:.0f}%", (xs[-1], val), xytext=(xs[-1] + 0.25, y),
                     textcoords="data", va="center", fontsize=9, color=INK)
    ax2.set_xticks(xs, STAGES, color=INK)
    ax2.set_xlim(-0.3, len(STAGES) + 0.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% reaching stage on attempt 1", color=INK2, fontsize=9)
    ax2.set_title("Stage funnel (mean over seeds)", loc="left", color=INK, fontsize=11)
    style(ax2)

    # (c) mean attempts among successes: dots, not bars (the scale starts at 1)
    for i, v in enumerate(versions):
        vals = np.array([float(r["mean_attempts"]) for r in by[v]])
        ax3.scatter(np.full(len(vals), i), vals, s=22, color=INK, edgecolor=SURFACE,
                    linewidth=0.8, zorder=3)
        ax3.scatter([i], [vals.mean()], s=90, color=colors[v], edgecolor=SURFACE,
                    linewidth=1.5, zorder=4)
    ax3.set_xticks(range(n), versions, color=INK)
    ax3.set_xlim(-0.5, n - 0.5)
    ax3.set_ylim(0.95, max(1.35, ax3.get_ylim()[1]))
    ax3.set_ylabel("attempts (of 3), successful positions", color=INK2, fontsize=9)
    ax3.set_title("Mean attempts", loc="left", color=INK, fontsize=11)
    style(ax3)

    fig.legend(*ax1.get_legend_handles_labels(), loc="upper right", ncol=n,
               frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Cube-lift BC: held-out benchmark by version "
                 "(200 positions, 3 seeds, up to 3 attempts)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    out = os.path.join(RESULTS, "benchmark_versions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
