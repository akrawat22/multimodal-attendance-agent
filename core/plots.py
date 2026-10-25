"""Matplotlib charts used by the dashboard (light theme, one palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_COLORS = {
    "Silence": "#B8C0CC",
    "Instructor Lecture": "#3B6FD4",
    "Student Discussion": "#2E9E6B",
    "Ambient Distraction": "#E07A3F",
}
INK, MUTED, GRID = "#1F2933", "#6B7785", "#E6E9EE"
C_VIS, C_AUD, C_ENG = "#7C5CBF", "#2E9E6B", "#1F2933"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _time_axis(ax, total_s):
    step = 60 if total_s <= 1800 else 300
    if total_s <= 240:
        step = 30
    ticks = np.arange(0, total_s + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 60)}:{int(t % 60):02d}" for t in ticks])
    ax.set_xlim(0, total_s)
    ax.set_xlabel("Session time (min:sec)")


def presence_chart(att: pd.DataFrame, threshold_pct: float):
    d = att.sort_values("Presence %")
    h = max(2.4, 0.26 * len(d) + 0.8)
    fig, ax = plt.subplots(figsize=(6.2, h))
    cols = ["#2E9E6B" if s == "Present" else "#E07A3F" for s in d["Status"]]
    ax.barh(d["Student ID"], d["Presence %"], color=cols, height=0.7)
    ax.axvline(threshold_pct, color=INK, linestyle="--", linewidth=1)
    ax.text(threshold_pct + 1, -0.9, f"{threshold_pct:.0f}% threshold", fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time present (% of session)")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def headcount_chart(vis_win: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    t = (vis_win["t_start"] + vis_win["t_end"]) / 2
    ax.fill_between(t, vis_win["n_faces"], color="#3B6FD4", alpha=0.15)
    ax.plot(t, vis_win["n_faces"], color="#3B6FD4", linewidth=2)
    ax.set_ylabel("Faces visible")
    ax.set_ylim(0, max(1, vis_win["n_faces"].max()) * 1.15)
    _time_axis(ax, vis_win["t_end"].max())
    _style(ax)
    fig.tight_layout()
    return fig


def audio_timeline(sec_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    for lab, col in CLASS_COLORS.items():
        m = (sec_df["label"] == lab).to_numpy()
        ax.bar(sec_df["sec"][m] + 0.5, np.ones(m.sum()), width=1.0, color=col, label=lab, linewidth=0)
    ax.set_yticks([])
    _time_axis(ax, len(sec_df))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS.values()]
    fig.legend(handles, list(CLASS_COLORS), loc="upper center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.78, bottom=0.34, left=0.03, right=0.97)
    return fig


def audio_level_chart(sec_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.plot(sec_df["sec"], sec_df["rel_db"], color=MUTED, linewidth=0.9)
    for lab, col in CLASS_COLORS.items():
        m = (sec_df["label"] == lab).to_numpy()
        ax.scatter(sec_df["sec"][m], sec_df["rel_db"][m], s=6, color=col, zorder=3)
    ax.set_ylabel("Level above noise floor (dB)")
    _time_axis(ax, len(sec_df))
    _style(ax)
    fig.tight_layout()
    return fig


def class_share_chart(sec_df: pd.DataFrame):
    share = sec_df["label"].value_counts(normalize=True).reindex(list(CLASS_COLORS)).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.barh(list(share.index)[::-1], share.values[::-1], color=[CLASS_COLORS[k] for k in list(share.index)[::-1]], height=0.6)
    for i, v in enumerate(share.values[::-1]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(100, share.max() + 15) if share.max() > 85 else 100)
    ax.set_xlabel("Share of session")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def engagement_chart(fused: pd.DataFrame, low_thr: float, suggestions=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    t = (fused["t_start"] + fused["t_end"]) / 2
    total = fused["t_end"].max()
    low = fused["engagement"].to_numpy() < low_thr
    ax.axhspan(0, low_thr, color="#E07A3F", alpha=0.07)
    ax.axhline(low_thr, color="#E07A3F", linestyle="--", linewidth=1)
    ax.text(total, low_thr + 1.5, f"low engagement < {low_thr:.0f}", ha="right", fontsize=8, color="#E07A3F")
    if fused["visual"].notna().any():
        ax.plot(t, fused["visual"], color=C_VIS, linewidth=1.3, alpha=0.75, label="Visual attentiveness")
    if fused["audio"].notna().any():
        ax.plot(t, fused["audio"], color=C_AUD, linewidth=1.3, alpha=0.75, label="Audio interaction")
    ax.plot(t, fused["engagement"], color=C_ENG, linewidth=2.6, label="Engagement Index")
    i_hi, i_lo = int(fused["engagement"].idxmax()), int(fused["engagement"].idxmin())
    for i, lab, dy in ((i_hi, "peak", 6), (i_lo, "dip", -10)):
        ax.scatter([t[i]], [fused["engagement"].iloc[i]], color=C_ENG, zorder=5, s=28)
        ax.annotate(f"{lab} {fused['engagement'].iloc[i]:.0f}", (t[i], fused["engagement"].iloc[i]),
                    textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (0-100)")
    _time_axis(ax, total)
    _style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
