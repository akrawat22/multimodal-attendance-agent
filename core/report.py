"""Session summary: KPIs and a downloadable educator report."""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .suggestions import Suggestion, fmt


def summarise(attendance: Optional[pd.DataFrame], fused: pd.DataFrame,
              audio_sec: Optional[pd.DataFrame], duration_s: float) -> dict:
    s = {"session_length": fmt(duration_s)}
    if attendance is not None and len(attendance):
        n = len(attendance)
        p = int((attendance["Status"] == "Present").sum())
        s.update(students_tracked=n, present=p, absent=n - p,
                 attendance_rate=round(100 * p / n, 1))
    eng = fused["engagement"]
    s["avg_engagement"] = round(float(eng.mean()), 1)
    i_hi, i_lo = int(eng.idxmax()), int(eng.idxmin())
    s["peak_engagement"] = {"value": round(float(eng.iloc[i_hi]), 1), "at": fmt(fused["t_start"].iloc[i_hi])}
    s["lowest_engagement"] = {"value": round(float(eng.iloc[i_lo]), 1), "at": fmt(fused["t_start"].iloc[i_lo])}
    s["time_below_45"] = f"{100 * float((eng < 45).mean()):.0f}%"
    if audio_sec is not None and len(audio_sec):
        share = audio_sec["label"].value_counts(normalize=True)
        s["audio_share_pct"] = {k: round(100 * float(v), 1) for k, v in share.items()}
        s["participation_rate_pct"] = round(100 * float((audio_sec["label"] == "Student Discussion").mean()), 1)
    if "forward_frac" in fused:
        s["avg_forward_facing_pct"] = round(100 * float(fused["forward_frac"].mean(skipna=True)), 1)
    return s


def to_markdown(summary: dict, attendance: Optional[pd.DataFrame],
                suggestions: List[Suggestion], title: str = "Session Report") -> str:
    L = [f"# {title}", f"_Generated {datetime.now():%Y-%m-%d %H:%M} - processed locally, no images or audio stored._", ""]
    L += ["## Key numbers", f"- Session length: **{summary['session_length']}**"]
    if "attendance_rate" in summary:
        L.append(f"- Attendance: **{summary['present']}/{summary['students_tracked']}** present "
                 f"({summary['attendance_rate']}%), {summary['absent']} below the sustained-presence threshold")
    L += [f"- Average engagement: **{summary['avg_engagement']}%**",
          f"- Peak: {summary['peak_engagement']['value']}% at {summary['peak_engagement']['at']}",
          f"- Lowest: {summary['lowest_engagement']['value']}% at {summary['lowest_engagement']['at']}",
          f"- Time below 45% engagement: {summary['time_below_45']}"]
    if "participation_rate_pct" in summary:
        L.append(f"- Student verbal participation: {summary['participation_rate_pct']}% of the session")
        L.append("- Audio mix: " + ", ".join(f"{k} {v}%" for k, v in summary["audio_share_pct"].items()))
    L += ["", "## Teaching suggestions"]
    for sg in suggestions:
        L.append(f"- **{fmt(sg.start_s)}-{fmt(sg.end_s)} - {sg.title}.** {sg.message}")
    if attendance is not None and len(attendance):
        L += ["", "## Attendance register", "",
              attendance[["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %", "Status"]]
              .to_markdown(index=False)]
    L += ["", "---", "Engagement figures are collective trends meant to support teaching decisions, "
          "not to evaluate or discipline individual students."]
    return "\n".join(L)


def to_json(summary: dict, suggestions: List[Suggestion]) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError
    return json.dumps({"summary": summary, "suggestions": [s.to_dict() for s in suggestions]},
                      indent=2, default=conv)
