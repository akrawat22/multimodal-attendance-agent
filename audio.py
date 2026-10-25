"""Audio module: voice-activity detection and classroom sound classification.

Every 1-second window is labelled as one of
    Silence | Instructor Lecture | Student Discussion | Ambient Distraction
using spectral energy, speech-band ratio, spectral flatness and syllabic
modulation. Thresholds are *relative to the recording's own noise floor*, so
they adapt to different microphones and room volumes.

Limitation (be upfront in demos): without speaker diarisation, "Instructor"
vs "Student" is inferred from the length of continuous speech runs: long
monologues => lecture, short alternating bursts => discussion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CLASSES = ["Silence", "Instructor Lecture", "Student Discussion", "Ambient Distraction"]
# how much each acoustic state contributes to an "interaction" score (0..1)
AUDIO_SCORE = {
    "Silence": 0.40,              # quiet seat-work: neutral, slightly below active
    "Instructor Lecture": 0.70,
    "Student Discussion": 1.00,
    "Ambient Distraction": 0.05,
}


@dataclass
class AudioConfig:
    sr: int = 16000
    n_fft: int = 512
    hop: int = 256
    active_db: float = 8.0       # dB above noise floor => something is happening
    speech_band: Tuple[int, int] = (300, 3400)
    band_ratio_min: float = 0.55
    flatness_max: float = 0.35   # tonal/voiced => low flatness
    cv_min: float = 0.45         # syllabic modulation of the energy envelope
    lecture_min_run_s: int = 8   # continuous speech >= this => instructor lecture
    min_spread_db: float = 10.0  # p90-p10 level spread below which no noise floor can be learned
    assumed_floor_dbfs: float = -55.0


def _ffmpeg_exe() -> Optional[str]:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # pip wheel that bundles a static ffmpeg binary
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def load_audio(path: str, sr: int = 16000, max_seconds: Optional[float] = None) -> np.ndarray:
    """Load an audio file (wav/flac/ogg/mp3) or the audio track of a video (mp4/mov/mkv...).

    Plain audio goes through librosa/soundfile; anything else is decoded by ffmpeg.
    """
    import subprocess

    import librosa

    y = None
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    except Exception:  # noqa: BLE001 - container not understood by libsndfile
        exe = _ffmpeg_exe()
        if exe is None:
            raise RuntimeError("Could not decode this file: ffmpeg is not available.")
        cmd = [exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-f", "f32le", "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError("No audio track could be read from this file.")
        y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y is None or y.size == 0:
        raise ValueError("The audio file contains no samples.")
    return y.astype(np.float32)


def analyse_audio(y: np.ndarray, cfg: Optional[AudioConfig] = None) -> pd.DataFrame:
    """Return one row per second with features and a class label."""
    import librosa

    cfg = cfg or AudioConfig()
    sr = cfg.sr
    n_sec = int(len(y) // sr)
    if n_sec < 1:
        raise ValueError("Audio is shorter than 1 second.")
    y = y[: n_sec * sr]

    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop, center=False)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
    band = (freqs >= cfg.speech_band[0]) & (freqs <= cfg.speech_band[1])
    frame_rms = np.sqrt(S.sum(axis=0) + 1e-12)
    frame_band = S[band].sum(axis=0) / (S.sum(axis=0) + 1e-12)
    sb = S[band] + 1e-12
    frame_flat = np.exp(np.log(sb).mean(axis=0)) / sb.mean(axis=0)

    fps = (S.shape[1]) / n_sec
    per = int(np.floor(fps))
    rows = []
    for s in range(n_sec):
        a, b = int(round(s * fps)), int(round((s + 1) * fps))
        b = max(b, a + 2)
        r = frame_rms[a:b]
        rows.append(dict(
            sec=s,
            rms_db=20 * np.log10(np.sqrt(np.mean(y[s * sr:(s + 1) * sr] ** 2)) + 1e-9),  # dBFS
            band_ratio=float(frame_band[a:b].mean()),
            flatness=float(frame_flat[a:b].mean()),
            cv=float(np.std(r) / (np.mean(r) + 1e-9)),
        ))
    df = pd.DataFrame(rows)

    floor = float(np.percentile(df["rms_db"], 10))
    spread = float(np.percentile(df["rms_db"], 90)) - floor
    if spread < cfg.min_spread_db:
        # no quiet moments to learn from (e.g. a clip of continuous talking):
        # assume an ordinary quiet room instead of treating speech as the floor
        floor = min(floor, cfg.assumed_floor_dbfs)
    df["rel_db"] = df["rms_db"] - floor
    active = df["rel_db"] > cfg.active_db
    speechlike = (active & (df["band_ratio"] > cfg.band_ratio_min)
                  & (df["flatness"] < cfg.flatness_max) & (df["cv"] > cfg.cv_min))
    noise = active & ~speechlike

    sp = speechlike.to_numpy()

    labels = np.array(["Silence"] * n_sec, dtype=object)
    labels[noise.to_numpy()] = "Ambient Distraction"
    i = 0
    while i < n_sec:
        if sp[i]:
            j = i
            while j < n_sec and sp[j]:
                j += 1
            labels[i:j] = "Instructor Lecture" if (j - i) >= cfg.lecture_min_run_s else "Student Discussion"
            i = j
        else:
            i += 1
    df["label"] = labels
    df["noise_floor_db"] = floor
    return df


def audio_windows(sec_df: pd.DataFrame, duration_s: float, window_s: float) -> pd.DataFrame:
    """Aggregate per-second labels into the shared analysis windows."""
    n_win = int(np.ceil(duration_s / window_s))
    out = pd.DataFrame({
        "window": np.arange(n_win),
        "t_start": np.arange(n_win) * window_s,
        "t_end": np.minimum((np.arange(n_win) + 1) * window_s, duration_s),
    })
    for c in CLASSES:
        out[c] = 0.0
    if sec_df is None or sec_df.empty:
        out["audio"] = np.nan
        return out
    d = sec_df[sec_df["sec"] < duration_s].copy()
    d["win"] = (d["sec"] // window_s).astype(int).clip(upper=n_win - 1)
    frac = pd.crosstab(d["win"], d["label"], normalize="index")
    for c in CLASSES:
        if c in frac.columns:
            out.loc[frac.index, c] = frac[c].to_numpy()
    covered = d["win"].unique()
    out["audio"] = sum(out[c] * AUDIO_SCORE[c] for c in CLASSES)
    out.loc[~out["window"].isin(covered), "audio"] = np.nan
    out["dominant"] = out[CLASSES].idxmax(axis=1)
    return out
