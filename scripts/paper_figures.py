"""Generate the ACL paper figures from results/ JSONs.

Consistent visual language across the paper:
  hyperbolic = teal  (#0F766E)   euclidean = vermilion (#C2410C)
  neutral    = gray  (#6B7280)   accents kept light, no chartjunk

Outputs single-column-width PDFs into the Overleaf repo's figures/ dir and
prints the seed-averaged P1 numbers used in the main results table.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEAL = "#0F766E"
TEAL_LIGHT = "#5EEAD4"
VERM = "#C2410C"
VERM_LIGHT = "#FDBA74"
GRAY = "#6B7280"

OUT = Path("/vol/fob-vol7/mi18/goldejon/acl2026-hyperbolic-ner/figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

COL_W = 3.35  # ACL single column width in inches


# ── Fig 1: long-tail rank-frequency ─────────────────────────────────────
def fig_longtail():
    counts = []
    for line in open("results/e0/type_counts.tsv"):
        counts.append(int(line.rsplit("\t", 1)[1]))
    counts = np.array(sorted(counts, reverse=True))
    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    tail_mask = counts <= 5
    ax.fill_between(ranks[tail_mask], counts[tail_mask], 0.5,
                    color=VERM_LIGHT, alpha=0.45, lw=0,
                    label=r"tail ($\leq$5 mentions): 76% of types")
    ax.loglog(ranks, counts, color=TEAL, lw=1.8)
    ax.axhline(5, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("type rank")
    ax.set_ylabel("mentions")
    ax.set_ylim(0.5, None)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "longtail.pdf")
    plt.close(fig)
    print("[fig] longtail.pdf")


# ── Fig 2: C1 false negatives vs batch size ─────────────────────────────
def fig_c1():
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    styles = {
        "ufet": dict(color=TEAL, marker="o", label="UFET (multi-label)"),
        "finerweb": dict(color=VERM, marker="s", label="FiNERweb (single-tag)"),
    }
    for name, st in styles.items():
        r = json.load(open(f"results/c1/{name}.json"))
        Bs, pct = [], []
        for B, v in r["infonce_view"].items():
            Bs.append(int(B))
            pct.append(v["pct_examples_with_false_negative"])
        ax.plot(Bs, pct, lw=1.8, ms=4.5, **st)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax.set_xlabel("batch size")
    ax.set_ylabel("% examples with a\nfalse in-batch negative")
    ax.set_ylim(0, 102)
    ax.axhline(100, color=GRAY, lw=0.6, ls=":")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "false_negatives.pdf")
    plt.close(fig)
    print("[fig] false_negatives.pdf")


# ── Fig 3: C2 gradient conflict by bucket ───────────────────────────────
def fig_c2():
    runs = defaultdict(list)
    for f in glob.glob("results/c2/*-d64-s*.json"):
        r = json.load(open(f))
        runs[r["config"]["geometry"]].append(r)

    buckets = ["head", "mid", "tail"]
    means, stds = {}, {}
    for geom in ("euclidean", "hyperbolic"):
        m, s = [], []
        for b in buckets:
            v = [r["buckets"][b]["mean_cos_pos_fneg"] for r in runs[geom]]
            m.append(np.mean(v)); s.append(np.std(v))
        means[geom], stds[geom] = m, s

    x = np.arange(len(buckets))
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.bar(x - w / 2, means["euclidean"], w, yerr=stds["euclidean"],
           color=VERM, label="Euclidean", capsize=2,
           error_kw=dict(lw=0.8))
    ax.bar(x + w / 2, means["hyperbolic"], w, yerr=stds["hyperbolic"],
           color=TEAL, label="hyperbolic", capsize=2,
           error_kw=dict(lw=0.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-1, color=GRAY, lw=0.6, ls=":")
    ax.text(2.35, -0.985, "perfect\ntug-of-war", fontsize=7, color=GRAY,
            va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([r"head ($\geq$100)", "mid (6–99)", r"tail ($\leq$5)"])
    ax.set_xlabel("label frequency bucket")
    ax.set_ylabel(r"$\cos(g_{\mathrm{pos}},\, g_{\mathrm{false\text{-}neg}})$")
    ax.set_ylim(-1.05, 0.08)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_conflict.pdf")
    plt.close(fig)
    print("[fig] gradient_conflict.pdf")


# ── P1 table numbers: seed-averaged, incl head/mid/tail ─────────────────
def p1_numbers():
    cells = defaultdict(list)
    for f in glob.glob("results/p1_seeds/*fla*.json") + glob.glob("results/p1/*fla*.json"):
        r = json.load(open(f))
        c = r["config"]
        cells[(c["geometry"], c["dim"])].append(r)
    print("\n=== P1 flat-supervision numbers (mean±std) for the main table ===")
    for key in sorted(cells, key=lambda k: (k[0], k[1])):
        rs = cells[key]
        if len(rs) < 2:
            continue
        def agg(fn):
            v = [fn(r) for r in rs]
            return f"{np.mean(v):.3f}±{np.std(v):.3f}"
        print(f"{key[0]:<11} d{key[1]:<4} n={len(rs)} | "
              f"micro {agg(lambda r: r['test']['micro_f1'])} | "
              f"macro {agg(lambda r: r['test']['macro_f1'])} | "
              f"mAP {agg(lambda r: r['test_map'])} | "
              f"head {agg(lambda r: r['test_tail']['head_f1'])} | "
              f"mid {agg(lambda r: r['test_tail']['mid_f1'])} | "
              f"tail {agg(lambda r: r['test_tail']['tail_f1'])}")


if __name__ == "__main__":
    fig_longtail()
    fig_c1()
    fig_c2()
    p1_numbers()
