"""Teaching-suggestion agent (rule-based, transparent and fully offline).

It looks at *collective* trends only and returns non-intrusive pedagogical
prompts. It is decision support: the teacher stays in control.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd


@dataclass
class AdvisorConfig:
    low_engagement: float = 45.0
    high_engagement: float = 75.0
    min_low_s: float = 90.0         # sustained dip needed before advising
    noise_frac: float = 0.35        # share of 'Ambient Distraction' that counts as noisy
    long_lecture_s: float = 360.0   # continuous lecture longer than this => break it up
    headcount_drop: float = 0.30    # >30% fewer faces than the session peak


@dataclass
class Suggestion:
    start_s: float
    end_s: float
    severity: str   # "info" | "attention" | "positive"
    title: str
    message: str

    def to_dict(self):
        return asdict(self)


def fmt(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _runs(mask: np.ndarray):
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def generate_suggestions(fused: pd.DataFrame, cfg: AdvisorConfig = None) -> List[Suggestion]:
    cfg = cfg or AdvisorConfig()
    out: List[Suggestion] = []
    if fused is None or fused.empty:
        return out
    t0, t1 = fused["t_start"].to_numpy(), fused["t_end"].to_numpy()
    eng = fused["engagement"].to_numpy()
    wlen = float((t1 - t0).mean())

    def span(i, j):
        return float(t0[i]), float(t1[j - 1])

    # 1) sustained low engagement -> change the activity
    for i, j in _runs(eng < cfg.low_engagement):
        a, b = span(i, j)
        if b - a >= cfg.min_low_s:
            mins = (b - a) / 60
            out.append(Suggestion(a, b, "attention", "Engagement dip",
                f"Collective engagement stayed below {cfg.low_engagement:.0f}% for about {mins:.1f} min "
                f"({fmt(a)}-{fmt(b)}). Consider a quick poll, a think-pair-share question, "
                "an interactive discussion, or a 2-minute stretch break."))

    # 2) noisy classroom
    if "Ambient Distraction" in fused:
        noisy = fused["Ambient Distraction"].to_numpy() >= cfg.noise_frac
        for i, j in _runs(noisy):
            a, b = span(i, j)
            if b - a >= max(2 * wlen, 20):
                out.append(Suggestion(a, b, "attention", "High background noise",
                    f"Ambient noise was high from {fmt(a)} to {fmt(b)}. Re-gather attention with a "
                    "clear signal (pause, countdown or board prompt) and restate the task before continuing."))

    # 3) long monologue with falling attention
    if "Instructor Lecture" in fused:
        lec = fused["Instructor Lecture"].to_numpy() >= 0.5
        for i, j in _runs(lec):
            a, b = span(i, j)
            if b - a >= cfg.long_lecture_s and j - i >= 4:
                slope = np.polyfit(np.arange(j - i), eng[i:j], 1)[0]
                if slope < -0.3:
                    out.append(Suggestion(a, b, "attention", "Long lecture, attention falling",
                        f"Continuous lecturing for {(b - a) / 60:.0f} min ({fmt(a)}-{fmt(b)}) while engagement "
                        "trended down. Insert a short activity, example or question to reset attention."))

    # 4) quiet and visually disengaged -> ask something
    if "Silence" in fused and "forward_frac" in fused:
        q = (fused["Silence"].to_numpy() >= 0.7) & (fused["forward_frac"].fillna(1).to_numpy() < 0.5)
        for i, j in _runs(q):
            a, b = span(i, j)
            if b - a >= max(2 * wlen, 30):
                out.append(Suggestion(a, b, "info", "Quiet and looking away",
                    f"Low interaction and many students looking away ({fmt(a)}-{fmt(b)}). "
                    "A direct question to the class or a quick check-for-understanding may help."))

    # 5) headcount drop
    if "n_faces" in fused and fused["n_faces"].max() > 0:
        peak = fused["n_faces"].max()
        low = fused["n_faces"].to_numpy() < (1 - cfg.headcount_drop) * peak
        for i, j in _runs(low):
            a, b = span(i, j)
            if b - a >= max(3 * wlen, 60):
                out.append(Suggestion(a, b, "info", "Fewer students visible",
                    f"Visible headcount dropped more than {cfg.headcount_drop:.0%} below the peak "
                    f"({fmt(a)}-{fmt(b)}). Check whether students left, or whether the camera view changed."))

    # 6) what worked
    for i, j in _runs(eng >= cfg.high_engagement):
        a, b = span(i, j)
        if b - a >= max(2 * wlen, 30):
            out.append(Suggestion(a, b, "positive", "Engagement peak",
                f"Engagement was high between {fmt(a)} and {fmt(b)}. Whatever you did here "
                "(activity, question style, pacing) is worth repeating."))

    if not out:
        m = float(np.nanmean(eng))
        out.append(Suggestion(float(t0[0]), float(t1[-1]), "positive" if m >= 55 else "info",
                              "Steady session",
                              f"No sustained dips detected (average engagement {m:.0f}%). Keep the current pacing."))
    out.sort(key=lambda s: s.start_s)
    return out
