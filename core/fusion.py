"""Multimodal fusion: combine visual attentiveness and audio interaction into
one Engagement Index (0-100) per window, smoothed over time."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def fuse(
    vis_win: Optional[pd.DataFrame],
    aud_win: Optional[pd.DataFrame],
    w_visual: float = 0.6,
    w_audio: float = 0.4,
    smooth: int = 3,
) -> pd.DataFrame:
    """Weighted fusion. If one modality is missing/NaN for a window, the other is
    used on its own (weights are re-normalised per window)."""
    base = vis_win if vis_win is not None else aud_win
    if base is None:
        raise ValueError("Need at least one of video or audio.")
    df = base[["window", "t_start", "t_end"]].copy()
    n = len(df)
    v = vis_win["visual"].to_numpy(dtype=float) if vis_win is not None else np.full(n, np.nan)
    a = aud_win["audio"].to_numpy(dtype=float) if aud_win is not None else np.full(n, np.nan)
    m = min(len(v), len(a)) if (vis_win is not None and aud_win is not None) else n
    if vis_win is not None and aud_win is not None:
        # align on the shorter of the two recordings
        df = df.iloc[:m].copy()
        v, a = v[:m], a[:m]
    wv = np.where(np.isnan(v), 0.0, w_visual)
    wa = np.where(np.isnan(a), 0.0, w_audio)
    tot = wv + wa
    raw = np.where(tot > 0, (np.nan_to_num(v) * wv + np.nan_to_num(a) * wa) / np.where(tot == 0, 1, tot), np.nan)
    df["visual"] = v * 100
    df["audio"] = a * 100
    df["engagement_raw"] = raw * 100
    s = pd.Series(df["engagement_raw"].to_numpy())
    df["engagement"] = s.interpolate(limit_direction="both").rolling(max(1, smooth), center=True, min_periods=1).mean().to_numpy()
    if vis_win is not None:
        for c in ["n_faces", "forward_frac", "restless_frac", "inactive_frac"]:
            df[c] = vis_win[c].to_numpy()[:m]
    if aud_win is not None:
        for c in ["Silence", "Instructor Lecture", "Student Discussion", "Ambient Distraction"]:
            df[c] = aud_win[c].to_numpy()[:m]
    return df.reset_index(drop=True)
