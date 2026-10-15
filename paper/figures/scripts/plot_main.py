#!/usr/bin/env python3
"""Paper figures for the PAP main comparison and goodput/SLO analysis.

Input : ../data/summary.json and ../data/requests.csv (built by make_data.sh
        from PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30 + aligned DP/PD runs).
Output: ../*.pdf (vector, Type-42 fonts) and ../png/*.png (preview).

Setup: Qwen3-8B FP16, 8x L20, native 32K, 60 sessions x 3 turns, closed-loop
concurrency 30, two independent service starts per architecture (360 requests).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PNG = ROOT / "png"

# --- style -----------------------------------------------------------------
COL1 = 3.35  # single-column width (in), ACM/IEEE two-column templates
COL2 = 7.0   # full text width

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Encoding: PAP = orange (the one accent); DP8 = neutral gray; PD xPyD = one
# blue ramp, light -> dark as more GPUs go to Decode. Hatching gives a second,
# print/CVD-safe channel for the three families.
ARCH = {
    "pap7pa1p_kv_migration": dict(label="PAP 7PA1P", color="#eb6834", hatch=""),
    "dp8":    dict(label="DP8",     color="#8a8984", hatch=""),
    "pd7p1d": dict(label="PD 7P1D", color="#a9c8ee", hatch="////"),
    "pd6p2d": dict(label="PD 6P2D", color="#5d9be3", hatch="////"),
    "pd4p4d": dict(label="PD 4P4D", color="#2a78d6", hatch="////"),
    "pd2p6d": dict(label="PD 2P6D", color="#184a8a", hatch="////"),
}
MAIN = ["pap7pa1p_kv_migration", "pd6p2d", "pd7p1d", "dp8"]
ALL = ["pap7pa1p_kv_migration", "dp8", "pd7p1d", "pd6p2d", "pd4p4d", "pd2p6d"]
INK2 = "#52514e"


def load():
    summary = json.loads((DATA / "summary.json").read_text())
    rows = defaultdict(list)
    with (DATA / "requests.csv").open() as f:
        for r in csv.DictReader(f):
            for k in ("ttft_ms", "itl_ms", "e2e_ms"):
                r[k] = float(r[k])
            r["turn"] = int(r["turn"])
            rows[r["architecture"]].append(r)
    return summary["architectures"], rows, summary["slos"]


def save(fig, name):
    PNG.mkdir(exist_ok=True)
    fig.savefig(ROOT / f"{name}.pdf")
    fig.savefig(PNG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def per_run_means(rows, arch, metric):
    by = defaultdict(list)
    for r in rows[arch]:
        by[r["run"]].append(r[metric])
    return [np.mean(v) for _, v in sorted(by.items())]


# --- Fig. 1: main comparison bars ---------------------------------------------
def fig_main(summ, rows, archs, name, width):
    panels = [
        ("ttft_ms", "Mean TTFT (s)", 1e-3, "lower"),
        ("itl_ms", "Mean ITL (ms)", 1.0, "lower"),
        ("thr", "Throughput (req/s)", 1.0, "higher"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(width, 1.55 if width < 5 else 1.75))
    y = np.arange(len(archs))[::-1]
    clip = {"ttft_ms": 16.0}  # keep PD 2P6D (34.7 s) from flattening the rest
    for ax, (metric, title, scale, better) in zip(axes, panels):
        vals = []
        for a in archs:
            v = summ[a]["request_throughput_rps"] if metric == "thr" else summ[a][metric]["mean"] * scale
            vals.append(v)
        lim = clip.get(metric) if (metric in clip and max(vals) > clip[metric]) else None
        for yi, a, v in zip(y, archs, vals):
            st = ARCH[a]
            ax.barh(yi, min(v, lim) if lim else v, height=0.66, color=st["color"],
                    hatch=st["hatch"], edgecolor="white", linewidth=0.0)
            right = v
            if metric != "thr":
                for m in per_run_means(rows, a, metric):
                    m *= scale
                    if not lim or m <= lim:
                        ax.plot(m, yi, "|", color="#0b0b0b", ms=6, mew=0.8)
                        right = max(right, m)
            txt = f"{v:.2f}" if metric != "itl_ms" else f"{v:.1f}"
            if lim and v > lim:
                ax.text(lim * 0.97, yi, f"{txt} →", va="center", ha="right", fontsize=6.5,
                        color="#0b0b0b", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            else:
                ax.text(right, yi, " " + txt, va="center", ha="left", fontsize=6.5,
                        color="#0b0b0b", fontweight="bold" if a.startswith("pap") else None)
        if lim:
            ax.set_xlim(0, lim)
        else:
            ax.set_xlim(0, max(vals) * 1.28)
        ax.set_yticks(y, [ARCH[a]["label"] for a in archs] if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{title}\n({better} is better)", loc="left", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    for lab in axes[0].get_yticklabels():
        if lab.get_text().startswith("PAP"):
            lab.set_fontweight("bold")
    fig.text(0.0, -0.06, "Bars: pooled mean over 2 runs (360 requests); ticks: each run's mean.",
             fontsize=6, color=INK2, transform=fig.transFigure)
    fig.tight_layout(w_pad=0.6)
    save(fig, name)


# --- Fig. 2: tail latency CDFs ----------------------------------------------
def fig_cdf(rows, archs, name):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.8))
    specs = [("ttft_ms", "TTFT (s)", 1e-3, (0, 20)), ("itl_ms", "ITL (ms)", 1, (0, 180)),
             ("e2e_ms", "E2E latency (s)", 1e-3, (0, 30))]
    for ax, (m, xl, sc, xlim) in zip(axes, specs):
        for a in archs[::-1]:
            v = np.sort([r[m] * sc for r in rows[a]])
            p = np.arange(1, len(v) + 1) / len(v)
            pap = a.startswith("pap")
            ax.step(v, p, where="post", color=ARCH[a]["color"], lw=1.8 if pap else 1.2,
                    ls="-" if pap or a == "dp8" else (0, (4, 1.5)), zorder=3 if pap else 2,
                    label=ARCH[a]["label"])
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(xl)
        ax.axhline(0.99, color=INK2, lw=0.5, ls=":")
    axes[0].set_ylabel("CDF of requests")
    axes[0].text(19.7, 0.975, "P99", ha="right", fontsize=6, color=INK2, va="top")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(ARCH[a]["label"]) for a in archs]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center",
               ncol=len(archs), bbox_to_anchor=(0.5, 1.08), handlelength=2.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


# --- Fig. 3: TTFT / ITL / throughput tradeoff -----------------------------------
def fig_tradeoff(summ, archs, name):
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    offsets = {"pap7pa1p_kv_migration": (-9, -8), "dp8": (8, -2), "pd7p1d": (8, 2),
               "pd6p2d": (9, 6), "pd4p4d": (0, -15), "pd2p6d": (0, 13)}
    for a in archs:
        s = summ[a]
        x, yv, t = s["ttft_ms"]["mean"] / 1e3, s["itl_ms"]["mean"], s["request_throughput_rps"]
        pap = a.startswith("pap")
        ax.scatter(x, yv, s=t * 45 * (2.2 if pap else 1), marker="*" if pap else "o",
                   color=ARCH[a]["color"], edgecolor="white", linewidth=1.0, zorder=3)
        dx, dy = offsets[a]
        ax.annotate(f"{ARCH[a]['label']}\n{t:.2f} req/s", (x, yv), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5,
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"),
                    va="center", fontweight="bold" if pap else None, linespacing=1.0)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlim(1.5, 70)
    ax.set_ylim(15, 90)
    ax.set_xlabel("Mean TTFT (s, log scale)")
    ax.set_ylabel("Mean ITL (ms)")
    ax.text(0.98, 0.97, "marker area ∝ throughput\nlower-left is better", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=INK2)
    fig.tight_layout()
    save(fig, name)


# --- Fig. 4: per-turn TTFT ------------------------------------------------------
def fig_turns(summ, archs, name):
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    ends = {}
    for a in archs[::-1]:
        bt = summ[a]["by_turn"]
        v = [bt[str(t)]["ttft_ms"]["mean"] / 1e3 for t in range(3)]
        pap = a.startswith("pap")
        ax.plot(range(3), v, marker="*" if pap else "o", ms=7 if pap else 4.5,
                color=ARCH[a]["color"], lw=1.8 if pap else 1.2,
                ls="-" if pap or a == "dp8" else (0, (4, 1.5)), mec="white", mew=0.6,
                zorder=3 if pap else 2)
        ends[a] = v[2]
    ax.set_xticks(range(3), ["Turn 1", "Turn 2", "Turn 3"])
    ax.set_xlim(-0.15, 2.15)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1.2, 70)
    ax.set_ylabel("Mean TTFT (s, log scale)")
    # place end labels in log space, at least `gap` apart
    gap, placed = 0.1, []
    for a, v in sorted(ends.items(), key=lambda kv: kv[1]):
        y = np.log10(v)
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(ARCH[a]["label"], (2, v), xytext=(2.1, 10 ** y), textcoords="data",
                    va="center", fontsize=6.5, fontweight="bold" if a.startswith("pap") else None)
    fig.tight_layout()
    save(fig, name)


# --- Fig. 5: goodput at diagnostic SLO points -------------------------------------
def fig_goodput(summ, slos, archs, name):
    keys = list(slos)
    fig, ax = plt.subplots(figsize=(COL2, 1.9))
    n = len(archs)
    w = 0.8 / n
    x = np.arange(len(keys))
    for i, a in enumerate(archs):
        st = ARCH[a]
        for j, k in enumerate(keys):
            g = summ[a]["goodput"][k]
            xi = x[j] - 0.4 + w * (i + 0.5)
            feas = g["feasible_at_95pct"]
            ax.bar(xi, g["goodput_rps"], width=w * 0.92,
                   color=st["color"] if feas else "white",
                   edgecolor=st["color"], linewidth=1.0)
            ax.text(xi, g["goodput_rps"] + 0.05, f"{g['attainment_pct']:.0f}",
                    ha="center", va="bottom", fontsize=5.5, color=INK2)
    labels = []
    for k in keys:
        s = slos[k]
        labels.append(f"{k}: TTFT≤{s['ttft_ms']/1e3:.0f} s\nITL≤{s['itl_ms']} ms, E2E≤{s['e2e_ms']/1e3:.0f} s")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Goodput (req/s)")
    ax.set_ylim(0, 3.6)
    h = [Patch(facecolor=ARCH[a]["color"], edgecolor=ARCH[a]["color"]) for a in archs]
    l = [ARCH[a]["label"] for a in archs]
    h += [Patch(facecolor="#8a8984", edgecolor="#8a8984"),
          Patch(facecolor="white", edgecolor="#8a8984")]
    l += ["≥95% attainment (feasible)", "<95% (infeasible)"]
    ax.legend(h, l, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32), columnspacing=1.2)
    ax.text(1.0, 1.02, "numbers above bars: SLO attainment (%)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=INK2)
    fig.tight_layout()
    save(fig, name)


# --- Fig. 6: SLO sensitivity (attainment vs TTFT SLO, two ITL SLOs) -----------------
def fig_slo_sweep(rows, archs, name):
    ttft_grid = np.linspace(0, 20, 401)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.9), sharey=True)
    for ax, itl in zip(axes, (50, 75)):
        for a in archs[::-1]:
            t = np.array([r["ttft_ms"] for r in rows[a]]) / 1e3
            ok_itl = np.array([r["itl_ms"] <= itl for r in rows[a]])
            att = [(ok_itl & (t <= g)).mean() * 100 for g in ttft_grid]
            pap = a.startswith("pap")
            ax.plot(ttft_grid, att, color=ARCH[a]["color"], lw=1.8 if pap else 1.2,
                    ls="-" if pap or a == "dp8" else (0, (4, 1.5)), zorder=3 if pap else 2,
                    label=ARCH[a]["label"])
        ax.axhline(95, color=INK2, lw=0.6, ls=":")
        ax.text(0.3, 93, "95%", ha="left", va="top", fontsize=6, color=INK2)
        ax.set_title(f"ITL SLO ≤ {itl} ms", loc="left")
        ax.set_xlabel("TTFT SLO (s)")
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("SLO attainment (%)")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(ARCH[a]["label"]) for a in archs]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center",
               ncol=len(archs), bbox_to_anchor=(0.5, 1.09), handlelength=2.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def main():
    summ, rows, slos = load()
    fig_main(summ, rows, MAIN, "fig_main_comparison", COL1 * 1.35)
    fig_main(summ, rows, ALL, "fig_main_comparison_all", COL2)
    fig_cdf(rows, ALL, "fig_latency_cdf")
    fig_tradeoff(summ, ALL, "fig_tradeoff")
    fig_turns(summ, ALL, "fig_ttft_per_turn")
    fig_goodput(summ, slos, ALL, "fig_goodput_slo")
    fig_slo_sweep(rows, ALL, "fig_slo_attainment_sweep")


if __name__ == "__main__":
    main()
