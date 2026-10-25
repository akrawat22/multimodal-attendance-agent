"""Vision module: face detection, tracking, sustained-presence attendance and
per-window visual attentiveness (head pose, posture, movement, inactivity).

Detector: OpenCV Haar cascades (frontal + profile). They ship inside the
opencv wheel, need no model download and run fully offline on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class VisionConfig:
    sample_fps: float = 2.0          # frames analysed per second of video
    scale_width: int = 640           # frames are down-scaled to this width
    max_gap_s: float = 5.0           # gaps shorter than this still count as "present"
    reid_gap_s: float = 90.0         # a lost seat can be re-claimed within this time
    min_hits: int = 3                # tracks with fewer detections are treated as noise
    presence_threshold: float = 0.70  # fraction of session needed to be "Present"
    min_face_frac: float = 0.035     # min face width as fraction of frame width
    slump_thr: float = 0.60          # drop (in face heights) below baseline => slumped
    restless_thr: float = 0.45       # face widths / second => restless movement
    inactive_eps: float = 0.04       # face widths / second => practically motionless
    window_s: float = 10.0           # analysis window for visual attentiveness


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    kind: str  # "frontal" | "profile"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class FaceDetector:
    """Frontal faces => 'looking forward'; profile-only faces => 'turned away'."""

    def __init__(self, cfg: Optional[VisionConfig] = None):
        self.cfg = cfg or VisionConfig()
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.frontal.empty() or self.profile.empty():
            raise RuntimeError(
                "OpenCV Haar cascades not found. Install 'opencv-python-headless<5' "
                "(OpenCV 5 removed the cascade classifier)."
            )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        h0, w0 = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        min_side = max(24, int(self.cfg.min_face_frac * w0))
        min_size = (min_side, min_side)

        dets: List[Detection] = []
        for (x, y, w, h) in self.frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
        ):
            dets.append(Detection(int(x), int(y), int(w), int(h), "frontal"))

        profile_boxes = []
        for (x, y, w, h) in self.profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
        ):
            profile_boxes.append((int(x), int(y), int(w), int(h)))
        flipped = cv2.flip(gray, 1)
        for (x, y, w, h) in self.profile.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=5, minSize=min_size
        ):
            profile_boxes.append((int(w0 - x - w), int(y), int(w), int(h)))

        for pb in profile_boxes:
            if all(_iou(pb, (d.x, d.y, d.w, d.h)) < 0.25 for d in dets):
                dets.append(Detection(*pb, "profile"))
        return dets


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    w: float
    h: float
    last_t: float
    hits: int = 0
    base_cy: List[float] = field(default_factory=list)
    prev: Optional[Tuple[float, float, float]] = None  # (t, cx, cy)


class FaceTracker:
    """Greedy nearest-centre tracker tuned for seated students.

    Students barely move, so a track that disappears can be re-claimed by a new
    detection at (almost) the same spot for up to ``reid_gap_s`` seconds. This
    keeps one identity per seat even if the detector misses someone for a while.
    """

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self._next = 1

    def update(self, t: float, dets: List[Detection]) -> List[Tuple[Track, Detection]]:
        cands = []
        for tr in self.tracks:
            gap = t - tr.last_t
            if gap > self.cfg.reid_gap_s:
                continue
            reach = 1.2 if gap <= self.cfg.max_gap_s else 0.7
            for di, d in enumerate(dets):
                dist = np.hypot(d.cx - tr.cx, d.cy - tr.cy) / max(tr.w, 1.0)
                if dist <= reach:
                    cands.append((dist, tr.tid, di))
        cands.sort()
        used_t, used_d, matches = set(), set(), []
        by_id = {tr.tid: tr for tr in self.tracks}
        for dist, tid, di in cands:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            matches.append((by_id[tid], dets[di]))
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(self._next, d.cx, d.cy, d.w, d.h, t)
            self._next += 1
            self.tracks.append(tr)
            matches.append((tr, d))
        return matches


# --------------------------------------------------------------------------- #
# Video processing
# --------------------------------------------------------------------------- #
@dataclass
class VisionResult:
    obs: pd.DataFrame                 # one row per (sample time, tracked face)
    duration_s: float
    n_samples: int
    noise_tracks: int
    preview_rgb: Optional[np.ndarray] = None
    frame_size: Tuple[int, int] = (0, 0)


OBS_COLUMNS = ["t", "track_id", "cx", "cy", "w", "h", "pose", "move"]


def _annotate(bgr: np.ndarray, rows: List[dict]) -> np.ndarray:
    out = bgr.copy()
    colors = {"forward": (80, 175, 76), "turned": (0, 165, 255),
              "slumped": (0, 0, 220), "profile": (0, 165, 255)}
    for r in rows:
        x, y, w, h = int(r["cx"] - r["w"] / 2), int(r["cy"] - r["h"] / 2), int(r["w"]), int(r["h"])
        col = colors.get(r["pose"], (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, f"#{r['track_id']}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def process_video(
    path: str,
    cfg: Optional[VisionConfig] = None,
    max_seconds: Optional[float] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> VisionResult:
    """Run detection + tracking over a recorded classroom video (processed locally)."""
    cfg = cfg or VisionConfig()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try an .mp4 (H.264) recording.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_s = n_frames / fps if n_frames > 0 else None
    limit_s = min([v for v in (total_s, max_seconds) if v] or [None]) if (total_s or max_seconds) else None

    detector = FaceDetector(cfg)
    tracker = FaceTracker(cfg)
    step = max(1, int(round(fps / cfg.sample_fps)))
    rows: List[dict] = []
    best_preview, best_count = None, -1
    frame_size = (0, 0)
    idx, n_samples, last_t = 0, 0, 0.0

    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / fps
        if limit_s and t > limit_s:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                idx += 1
                continue
            h0, w0 = frame.shape[:2]
            if w0 > cfg.scale_width:
                s = cfg.scale_width / w0
                frame = cv2.resize(frame, (cfg.scale_width, int(h0 * s)), interpolation=cv2.INTER_AREA)
            frame_size = (frame.shape[1], frame.shape[0])
            dets = detector.detect(frame)
            frame_rows = []
            for tr, d in tracker.update(t, dets):
                # movement in face-widths per second
                move = np.nan
                if tr.prev is not None:
                    dt = t - tr.prev[0]
                    if 0 < dt <= cfg.max_gap_s:
                        move = np.hypot(d.cx - tr.prev[1], d.cy - tr.prev[2]) / max(d.w, 1) / dt
                pose = "forward" if d.kind == "frontal" else "turned"
                if pose == "forward":
                    tr.base_cy.append(d.cy)
                    tr.base_cy = tr.base_cy[-60:]
                    base = float(np.median(tr.base_cy))
                    if len(tr.base_cy) >= 5 and (d.cy - base) > cfg.slump_thr * d.h:
                        pose = "slumped"
                tr.cx, tr.cy, tr.w, tr.h = d.cx, d.cy, d.w, d.h
                tr.last_t = t
                tr.hits += 1
                tr.prev = (t, d.cx, d.cy)
                r = dict(t=t, track_id=tr.tid, cx=d.cx, cy=d.cy, w=d.w, h=d.h, pose=pose, move=move)
                rows.append(r)
                frame_rows.append(r)
            if len(frame_rows) > best_count:
                best_count = len(frame_rows)
                best_preview = _annotate(frame, frame_rows)
            n_samples += 1
            last_t = t
            if progress and limit_s:
                progress(min(1.0, t / limit_s))
        idx += 1
    cap.release()

    obs = pd.DataFrame(rows, columns=OBS_COLUMNS)
    duration = max(last_t + 1.0 / cfg.sample_fps, 1e-6)
    if limit_s:
        duration = min(duration, limit_s + 1.0 / cfg.sample_fps)
    # drop noise tracks (a handful of false-positive detections)
    hits = obs.groupby("track_id").size() if len(obs) else pd.Series(dtype=int)
    keep = set(hits[hits >= cfg.min_hits].index)
    noise = int(len(hits) - len(keep))
    obs = obs[obs["track_id"].isin(keep)].reset_index(drop=True)
    if progress:
        progress(1.0)
    return VisionResult(obs, duration, n_samples, noise, best_preview, frame_size)


def detect_snapshot(bgr: np.ndarray, cfg: Optional[VisionConfig] = None) -> Tuple[np.ndarray, int, int]:
    """Single photo / webcam snapshot: returns (annotated RGB, forward, turned)."""
    cfg = cfg or VisionConfig()
    h0, w0 = bgr.shape[:2]
    if w0 > 1280:
        s = 1280 / w0
        bgr = cv2.resize(bgr, (1280, int(h0 * s)), interpolation=cv2.INTER_AREA)
    dets = FaceDetector(cfg).detect(bgr)
    rows = [dict(cx=d.cx, cy=d.cy, w=d.w, h=d.h, track_id=i + 1,
                 pose="forward" if d.kind == "frontal" else "turned") for i, d in enumerate(dets)]
    fwd = sum(1 for d in dets if d.kind == "frontal")
    return _annotate(bgr, rows), fwd, len(dets) - fwd


# --------------------------------------------------------------------------- #
# Attendance (sustained presence)
# --------------------------------------------------------------------------- #
def _mmss(sec: float) -> str:
    sec = int(round(max(sec, 0)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def compute_attendance(obs: pd.DataFrame, duration_s: float, cfg: VisionConfig) -> pd.DataFrame:
    """Sustained-presence attendance register.

    A student is *Present* only if their face is seen for at least
    ``presence_threshold`` of the session. Short detection gaps (<= max_gap_s)
    are bridged, so a missed frame never breaks presence, while someone merely
    walking past the camera never reaches the threshold.
    """
    cols = ["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %",
            "Status", "track_id", "first_s", "last_s", "present_s"]
    if obs.empty or duration_s <= 0:
        return pd.DataFrame(columns=cols)
    dwell = 1.0 / cfg.sample_fps
    recs = []
    for tid, g in obs.groupby("track_id"):
        ts = np.sort(g["t"].to_numpy())
        gaps = np.diff(ts)
        bridged = float(np.where(gaps <= cfg.max_gap_s, gaps, dwell).sum() + dwell)
        bridged = min(bridged, duration_s)
        recs.append(dict(track_id=int(tid), first_s=float(ts[0]), last_s=float(ts[-1]),
                         present_s=bridged, x=float(g["cx"].median())))
    df = pd.DataFrame(recs).sort_values(["first_s", "x"]).reset_index(drop=True)
    df["Student ID"] = [f"S{i + 1:02d}" for i in range(len(df))]
    df["Presence %"] = (100 * df["present_s"] / duration_s).clip(upper=100).round(1)
    df["Status"] = np.where(df["present_s"] / duration_s >= cfg.presence_threshold, "Present", "Absent")
    df["First Seen"] = df["first_s"].map(_mmss)
    df["Last Seen"] = df["last_s"].map(_mmss)
    df["Total Duration"] = df["present_s"].map(_mmss)
    return df[cols]


# --------------------------------------------------------------------------- #
# Visual attentiveness per window
# --------------------------------------------------------------------------- #
POSE_SCORE = {"forward": 1.0, "slumped": 0.4, "turned": 0.2}


def visual_windows(obs: pd.DataFrame, duration_s: float, cfg: VisionConfig) -> pd.DataFrame:
    """Aggregate head pose, posture, movement and inactivity into windows.

    Only *collective* numbers are produced (no per-student engagement score).
    """
    n_win = int(np.ceil(duration_s / cfg.window_s))
    out = pd.DataFrame({
        "window": np.arange(n_win),
        "t_start": np.arange(n_win) * cfg.window_s,
        "t_end": np.minimum((np.arange(n_win) + 1) * cfg.window_s, duration_s),
    })
    cols = dict(n_faces=0, forward_frac=np.nan, restless_frac=np.nan,
                inactive_frac=np.nan, visual=np.nan)
    for c, v in cols.items():
        out[c] = v
    if obs.empty:
        return out
    o = obs.copy()
    o["win"] = (o["t"] // cfg.window_s).astype(int).clip(upper=n_win - 1)
    o["restless"] = (o["move"] > cfg.restless_thr).astype(float)
    o["still"] = ((o["move"] < cfg.inactive_eps) & (o["pose"] != "forward")).astype(float)
    base = o["pose"].map(POSE_SCORE).astype(float)
    o["score"] = base * np.where(o["restless"] > 0, 0.7, 1.0) * np.where(o["still"] > 0, 0.6, 1.0)
    o["fwd"] = (o["pose"] == "forward").astype(float)
    g = o.groupby("win")
    agg = pd.DataFrame({
        "n_faces": g["track_id"].nunique(),
        "forward_frac": g["fwd"].mean(),
        "restless_frac": g["restless"].mean(),
        "inactive_frac": g["still"].mean(),
        "visual": g["score"].mean(),
    })
    for c in agg.columns:
        out.loc[agg.index, c] = agg[c].to_numpy()
    out["n_faces"] = out["n_faces"].fillna(0).astype(int)
    out["visual"] = out["visual"].interpolate(limit_direction="both") if out["visual"].notna().any() else out["visual"]
    return out
