"""Synthetic demo session so the app can be explored (and graded) without
uploading any real classroom footage. It produces the *same data structures*
as the real pipeline, and the synthetic waveform goes through the real audio
analyser, so the demo exercises the genuine fusion / suggestion code."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from .vision import OBS_COLUMNS, VisionResult

DEMO_DURATION_S = 900          # 15-minute lesson
DEMO_SR = 8000

# (start, end, forward-probability at start, at end) -> classroom story
ATTENTION_STORY = [
    (0, 60, 0.92, 0.90),      # warm-up
    (60, 300, 0.88, 0.62),    # lecture, attention decays
    (300, 420, 0.55, 0.45),   # side chatter begins
    (420, 540, 0.35, 0.30),   # disengaged + noisy
    (540, 660, 0.94, 0.95),   # poll / discussion revives the room
    (660, 780, 0.72, 0.68),   # quiet seat-work
    (780, 900, 0.82, 0.88),   # wrap-up discussion
]


def _attn(t: float) -> float:
    for a, b, p0, p1 in ATTENTION_STORY:
        if a <= t < b:
            return p0 + (p1 - p0) * (t - a) / (b - a)
    return 0.8


# ----------------------------------------------------------------------------- #
# Vision (synthetic tracks)
# ----------------------------------------------------------------------------- #
def make_demo_vision(seed: int = 7, sample_fps: float = 2.0) -> VisionResult:
    rng = np.random.default_rng(seed)
    T = DEMO_DURATION_S
    ts = np.arange(0, T, 1.0 / sample_fps)
    seats = [(80 + 95 * c, 70 + 75 * r) for r in range(4) for c in range(6)]  # 24 seats
    spans = {i: (0.0, T) for i in range(1, 25)}
    spans[21] = (0.35 * T, T)        # arrived very late  -> ~65% => Absent
    spans[22] = (0.0, 0.50 * T)      # left early         -> ~50% => Absent
    spans[23] = (200.0, 212.0)       # walked past camera -> ~1%  => Absent
    spans[24] = (0.10 * T, T)        # a bit late         -> ~90% => Present

    rows = []
    for tid in range(1, 25):
        cx0, cy0 = seats[tid - 1]
        bias = rng.normal(0, 0.06)
        drop_left = 0
        for t in ts:
            a, b = spans[tid]
            if not (a <= t <= b):
                continue
            if drop_left > 0:                       # short detector dropout
                drop_left -= 1
                continue
            if rng.random() < 0.03:
                drop_left = int(rng.integers(2, 7))
                continue
            if rng.random() < 0.06:                 # random single-frame miss
                continue
            p = float(np.clip(_attn(t) + bias + rng.normal(0, 0.05), 0.02, 0.99))
            if rng.random() < p:
                pose, move = "forward", abs(rng.normal(0.06, 0.04))
            else:
                pose = "turned" if rng.random() < 0.5 else "slumped"
                move = abs(rng.normal(0.9, 0.4)) if rng.random() < 0.45 else abs(rng.normal(0.02, 0.01))
            rows.append(dict(t=float(t), track_id=tid, cx=cx0 + rng.normal(0, 1.2),
                             cy=cy0 + (14 if pose == "slumped" else 0) + rng.normal(0, 1.2),
                             w=44.0, h=44.0, pose=pose, move=float(move)))
    obs = pd.DataFrame(rows, columns=OBS_COLUMNS)

    # illustrative preview frame (clearly synthetic)
    frame = np.full((360, 640, 3), (28, 32, 40), np.uint8)
    peak_t = 600.0
    snap = obs[(obs["t"] > peak_t - 1) & (obs["t"] < peak_t + 1)]
    col = {"forward": (76, 175, 80), "turned": (255, 165, 0), "slumped": (220, 60, 60)}
    for _, r in snap.iterrows():
        x, y = int(r["cx"] - r["w"] / 2), int(r["cy"] - r["h"] / 2)
        cv2.rectangle(frame, (x, y), (x + int(r["w"]), y + int(r["h"])), col[r["pose"]], 2)
        cv2.putText(frame, f"#{int(r['track_id'])}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col[r["pose"]], 1, cv2.LINE_AA)
    cv2.putText(frame, "SYNTHETIC DEMO - no real students", (14, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return VisionResult(obs=obs, duration_s=float(T), n_samples=len(ts), noise_tracks=0,
                        preview_rgb=frame, frame_size=(640, 360))


# ----------------------------------------------------------------------------- #
# Audio (synthetic waveform)
# ----------------------------------------------------------------------------- #
def _speech(dur: float, f0: float, amp: float, rng, sr: int = DEMO_SR, syl_rate: float = 4.0, depth: float = 0.95):
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28)) + 0.03 * np.sin(2 * np.pi * 2.1 * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros(n, np.float32)
    k = 1
    while k * f0 < 3800:
        fk = k * f0
        g = (np.exp(-((fk - 500) / 400) ** 2) + 0.6 * np.exp(-((fk - 1500) / 600) ** 2)
             + 0.3 * np.exp(-((fk - 2500) / 700) ** 2))
        y += (g / k ** 0.3) * np.sin(k * phase).astype(np.float32)
        k += 1
    syll = np.abs(np.sin(np.pi * syl_rate * t + rng.uniform(0, 3))) ** 1.5
    env = (1 - depth) + depth * syll
    # occasional breath pauses between phrases
    pause = np.ones(n, np.float32)
    pos = rng.uniform(3.5, 6.5)
    while pos < dur - 1:
        a, b = int(pos * sr), int((pos + rng.uniform(0.25, 0.55)) * sr)
        pause[a:b] = 0.05
        pos += rng.uniform(4.0, 7.0)
    y = y * env * pause
    return amp * y / (np.max(np.abs(y)) + 1e-9)


def _babble(dur: float, amp: float, rng, sr: int = DEMO_SR):
    n = int(dur * sr)
    mix = np.zeros(n, np.float32)
    for f0 in rng.uniform(95, 260, 6):
        mix += _speech(dur, f0, 1.0, rng, sr, syl_rate=rng.uniform(2.5, 5.5), depth=0.5)
    mix += 1.2 * rng.normal(0, 1, n).astype(np.float32)          # chairs, shuffling, whispers
    return amp * mix / (np.max(np.abs(mix)) + 1e-9)


def make_demo_audio(seed: int = 11) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    sr, T = DEMO_SR, DEMO_DURATION_S
    y = rng.normal(0, 0.0015, T * sr).astype(np.float32)      # quiet room tone

    def put(a, b, sig):
        y[int(a * sr): int(a * sr) + len(sig)] += sig[: len(y) - int(a * sr)]

    put(0, 300, _speech(300, 120, 0.30, rng))                 # instructor lecture
    put(300, 420, _speech(120, 120, 0.28, rng))               # lecture with light chatter
    put(300, 420, _babble(120, 0.04, rng))
    put(420, 540, _babble(120, 0.30, rng))                    # classroom noise
    # discussion: short alternating bursts from different voices
    pos, k = 540.0, 0
    voices = [190, 240, 160, 210]
    while pos < 660:
        d = float(rng.uniform(3.0, 6.0))
        put(pos, pos + d, _speech(d, voices[k % 4], 0.22, rng))
        pos += d + float(rng.uniform(1.5, 3.0))
        k += 1
    # quiet seat-work (mostly silence + one short teacher hint)
    put(720, 733, _speech(13, 120, 0.25, rng))
    put(780, 840, _speech(60, 120, 0.30, rng))                # wrap-up explanation
    pos = 840.0
    while pos < 895:
        d = float(rng.uniform(3.0, 5.5))
        put(pos, pos + d, _speech(d, voices[k % 4], 0.22, rng))
        pos += d + float(rng.uniform(1.5, 2.5))
        k += 1
    return np.clip(y, -1, 1), sr
