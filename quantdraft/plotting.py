"""Figures from a results JSONL written by ``quantdraft bench`` (or the toy
validation)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#6b7280", "#db2777"]


def load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _spec_rows(rows):
    return [r for r in rows if r.get("kind") == "speculative"]


def _label(spec: str) -> str:
    return Path(spec).name if "/" in spec else spec


def plot_alpha(rows, out, title=None):
    """Acceptance probability per draft, at each temperature (rejection rule)."""
    spec = [r for r in _spec_rows(rows) if r.get("verify", "rejection") == "rejection"]
    drafts = list(dict.fromkeys(r["draft"] for r in spec))
    temps = sorted({r["temperature"] for r in spec})
    alpha = defaultdict(list)
    for r in spec:
        alpha[(r["draft"], r["temperature"])].append(r["alpha"])
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(drafts)), 4), dpi=150)
    w = 0.8 / max(len(temps), 1)
    for j, T in enumerate(temps):
        ys = [sum(alpha[(d, T)]) / len(alpha[(d, T)]) if alpha[(d, T)] else 0 for d in drafts]
        xs = [i + (j - (len(temps) - 1) / 2) * w for i in range(len(drafts))]
        ax.bar(xs, ys, width=w, color=PALETTE[j % len(PALETTE)], label=f"T={T:g}")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(drafts)), [_label(d) for d in drafts], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("acceptance probability α")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_speedup(rows, out, temperature=0.0, title=None):
    """Speedup vs fixed gamma per draft. Measured if available, else the
    bandwidth-model projection (dashed)."""
    spec = [r for r in _spec_rows(rows) if r["temperature"] == temperature
            and str(r["gamma"]).isdigit() and r.get("verify", "rejection") == "rejection"]
    drafts = list(dict.fromkeys(r["draft"] for r in spec))
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    projected_only = True
    for i, d in enumerate(drafts):
        rs = sorted((r for r in spec if r["draft"] == d), key=lambda r: int(r["gamma"]))
        xs = [int(r["gamma"]) for r in rs]
        c = PALETTE[i % len(PALETTE)]
        meas = [r.get("speedup") for r in rs]
        if all(m is not None for m in meas):
            projected_only = False
            ax.plot(xs, meas, marker="o", color=c, label=_label(d))
            pred = [r.get("predicted_speedup") for r in rs]
            if all(p is not None for p in pred):
                ax.plot(xs, pred, ls=":", color=c, lw=1)
        else:
            ys = [r.get("projected_speedup") for r in rs]
            ax.plot(xs, ys, marker="o", ls="--", color=c, label=_label(d))
        for r in (r for r in _spec_rows(rows) if r["draft"] == d and r["gamma"] == "auto"
                  and r["temperature"] == temperature):
            y = r.get("speedup") if r.get("speedup") is not None else r.get("projected_speedup")
            if y is not None:
                ax.axhline(y, color=c, lw=0.8, alpha=0.5)
    ax.axhline(1.0, color="#111827", lw=0.8)
    ax.set_xlabel("draft length γ")
    ax.set_ylabel("projected speedup (bandwidth model)" if projected_only else
                  "decode speedup vs target alone")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rule_comparison(rows, out, title=None):
    """Rejection sampling vs mlx-lm-style token matching at T > 0."""
    spec = [r for r in _spec_rows(rows) if r["temperature"] > 0]
    keys = sorted({(r["draft"], r["temperature"]) for r in spec})
    vals = defaultdict(list)
    for r in spec:
        vals[(r["draft"], r["temperature"], r["verify"])].append(r["tokens_per_round"])
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(keys)), 4), dpi=150)
    w = 0.38
    for j, (v, c) in enumerate((("rejection", PALETTE[0]), ("match", PALETTE[6]))):
        ys = [(sum(vals[(d, T, v)]) / len(vals[(d, T, v)])) if vals[(d, T, v)] else 0
              for d, T in keys]
        ax.bar([i + (j - 0.5) * w for i in range(len(keys))], ys, width=w, color=c,
               label={"rejection": "rejection sampling (paper)",
                      "match": "token matching (mlx-lm)"}[v])
    ax.set_xticks(range(len(keys)), [f"{_label(d)}\nT={T:g}" for d, T in keys], fontsize=7)
    ax.set_ylabel("tokens per target forward (mean over γ)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_all(results, out_dir, title=None) -> list[Path]:
    rows = load(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = [plot_alpha(rows, out / "alpha_by_draft.png", title)]
    for T in sorted({r["temperature"] for r in _spec_rows(rows)}):
        made.append(plot_speedup(rows, out / f"speedup_T{T:g}.png", T, title))
    rc = plot_rule_comparison(rows, out / "rejection_vs_match.png", title)
    if rc:
        made.append(rc)
    return made
