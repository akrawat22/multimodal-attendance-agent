import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio import AudioConfig, analyse_audio, audio_windows
from core.demo import make_demo_audio, make_demo_vision
from core.fusion import fuse
from core.suggestions import generate_suggestions
from core.vision import VisionConfig, compute_attendance, process_video, visual_windows


# ----------------------------------------------------------------- demo data
@pytest.fixture(scope="module")
def demo():
    vis = make_demo_vision()
    y, sr = make_demo_audio()
    sec = analyse_audio(y, AudioConfig(sr=sr))
    return vis, sec, y, sr


def test_sustained_presence_rejects_passers_by(demo):
    vis, *_ = demo
    att = compute_attendance(vis.obs, vis.duration_s, VisionConfig())
    status = dict(zip(att["track_id"], att["Status"]))
    assert status[23] == "Absent"            # walked past for 12 s
    assert status[22] == "Absent"            # left halfway
    assert status[21] == "Absent"            # arrived at 35%
    assert status[24] == "Present"           # 10% late is still >= 70%
    assert (att["Status"] == "Present").sum() == 21


def test_audio_classes_match_story(demo):
    _, sec, *_ = demo
    share = lambda a, b: sec.iloc[a:b]["label"].value_counts(normalize=True)
    assert share(0, 300).get("Instructor Lecture", 0) > 0.95
    assert share(420, 540).get("Ambient Distraction", 0) > 0.9
    assert share(540, 660).get("Student Discussion", 0) > 0.5
    assert share(660, 780).get("Silence", 0) > 0.7


def test_fusion_and_advisor(demo):
    vis, sec, *_ = demo
    cfg = VisionConfig()
    f = fuse(visual_windows(vis.obs, vis.duration_s, cfg), audio_windows(sec, vis.duration_s, cfg.window_s))
    assert f["engagement"].between(0, 100).all()
    assert f["engagement"].iloc[45:50].mean() < 45 < f["engagement"].iloc[0:5].mean()
    titles = [s.title for s in generate_suggestions(f)]
    assert "High background noise" in titles
    assert "Engagement dip" in titles


def test_fusion_handles_single_modality(demo):
    vis, sec, *_ = demo
    cfg = VisionConfig()
    v, a = visual_windows(vis.obs, vis.duration_s, cfg), audio_windows(sec, vis.duration_s, cfg.window_s)
    assert fuse(v, None)["engagement"].notna().all()
    assert fuse(None, a)["engagement"].notna().all()


def test_audio_longer_than_video_is_clipped(demo):
    _, sec, *_ = demo
    w = audio_windows(sec, 300.0, 10.0)
    assert len(w) == 30


# --------------------------------------------------------------- real video
def test_real_video_pipeline(tmp_path):
    pytest.importorskip("skimage")
    from tests.make_test_video import make_video

    path = make_video(tmp_path / "class.mp4", seconds=60, fps=10)
    res = process_video(str(path), VisionConfig(sample_fps=2.0))
    att = compute_attendance(res.obs, res.duration_s, VisionConfig(sample_fps=2.0))
    print(att)
    assert len(att) >= 5
    present = att[att["Status"] == "Present"]
    assert len(present) == 4                     # four always-there students (occlusion bridged)
    assert (att["Status"] == "Absent").sum() >= 1   # late arrival / passer-by never reach 70%
    assert res.preview_rgb is not None
