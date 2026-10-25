"""Multimodal Attendance & Engagement Agent  -  Streamlit dashboard.

Privacy-first: video and audio are analysed locally in memory. Uploaded files
are deleted right after processing; nothing is sent to any external service.
"""
from __future__ import annotations

import os
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.audio import AUDIO_SCORE, AudioConfig, analyse_audio, audio_windows, load_audio
from core.demo import DEMO_SR, make_demo_audio, make_demo_vision
from core.fusion import fuse
from core.plots import (audio_level_chart, audio_timeline, class_share_chart,
                        engagement_chart, headcount_chart, presence_chart)
from core.report import summarise, to_json, to_markdown
from core.suggestions import AdvisorConfig, fmt, generate_suggestions
from core.vision import VisionConfig, compute_attendance, detect_snapshot, process_video, visual_windows

st.set_page_config(page_title="Attendance & Engagement Agent", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1180px;}
h1 {font-size: 1.75rem !important; margin-bottom: .1rem;}
.sub {color:#6B7785; margin-bottom:.8rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
.card {border:1px solid #E6E9EE; border-left:5px solid #3B6FD4; border-radius:10px; padding:.7rem 1rem; margin:.5rem 0; background:#fff;}
.card.attention {border-left-color:#E07A3F;}
.card.positive {border-left-color:#2E9E6B;}
.card.info {border-left-color:#3B6FD4;}
.card .t {font-weight:700; color:#1F2933;}
.card .when {color:#6B7785; font-size:.82rem; margin-left:.4rem;}
.card .m {color:#3E4C59; margin-top:.15rem;}
.note {color:#6B7785; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached demo data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _demo_inputs():
    vis = make_demo_vision()
    y, sr = make_demo_audio()
    sec = analyse_audio(y, AudioConfig(sr=sr))
    return vis, sec


# --------------------------------------------------------------------------- #
# Heavy processing (runs only when "Analyze session" is pressed)
# --------------------------------------------------------------------------- #
def _save_temp(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def run_analysis(mode, video_file, audio_file, use_video_audio, fps, max_minutes, window_s, gap_s):
    S = dict(mode=mode, vision=None, audio_sec=None, duration_s=0.0, fps=fps, notes=[])
    if mode == "demo":
        vis, sec = _demo_inputs()
        S.update(vision=vis, audio_sec=sec, duration_s=vis.duration_s, fps=2.0)
        S["notes"].append("Synthetic demo session: 24 simulated seats and a generated classroom soundtrack.")
        return S

    cfg = VisionConfig(sample_fps=fps, window_s=window_s, max_gap_s=gap_s)
    paths = []
    try:
        if video_file is not None:
            vpath = _save_temp(video_file)
            paths.append(vpath)
            bar = st.progress(0.0, text="Detecting and tracking faces (in memory, nothing is stored)...")
            vis = process_video(vpath, cfg, max_seconds=max_minutes * 60,
                                progress=lambda p: bar.progress(min(p, 1.0), text=f"Detecting and tracking faces... {p:.0%}"))
            bar.empty()
            S.update(vision=vis, duration_s=vis.duration_s)
            if vis.noise_tracks:
                S["notes"].append(f"Ignored {vis.noise_tracks} very short false-positive detections.")
            if vis.obs.empty:
                S["notes"].append("No faces were detected. Check lighting, camera angle and that faces are large enough.")
        apath = None
        if audio_file is not None:
            apath = _save_temp(audio_file)
            paths.append(apath)
        elif video_file is not None and use_video_audio:
            apath = paths[0]
        if apath:
            try:
                with st.spinner("Analysing audio..."):
                    lim = max_minutes * 60 if video_file is not None else None
                    y = load_audio(apath, sr=16000, max_seconds=lim)
                    sec = analyse_audio(y, AudioConfig(sr=16000))
                S["audio_sec"] = sec
                if S["duration_s"] == 0:
                    S["duration_s"] = float(len(sec))
            except Exception as e:  # noqa: BLE001
                if audio_file is not None:
                    raise
                S["notes"].append("This video has no readable audio track. "
                                  "Upload a separate audio file to include audio analysis.")
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
    return S


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🎓 Attendance & Engagement Agent")
    st.markdown('<span class="pill" style="background:#E3F4EC;color:#1B6B49">🔒 100% local processing</span>', unsafe_allow_html=True)
    st.caption("No frames, faces or audio are saved or sent anywhere.")
    mode_label = st.radio("Data source", ["Demo session (synthetic)", "Upload my own recording"], index=0)
    mode = "demo" if mode_label.startswith("Demo") else "upload"

    video_file = audio_file = None
    use_video_audio = True
    if mode == "upload":
        video_file = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv", "webm"])
        audio_file = st.file_uploader("Audio (optional)", type=["wav", "mp3", "ogg", "flac", "m4a"])
        use_video_audio = st.checkbox("Use the video's own audio track if no audio file", value=True)

    with st.expander("Processing settings", expanded=False):
        fps = st.slider("Frames analysed per second", 1.0, 5.0, 2.0, 0.5, disabled=(mode == "demo"))
        max_minutes = st.slider("Max minutes to process", 1, 30, 10, disabled=(mode == "demo"))
    with st.expander("Attendance & analysis rules", expanded=False):
        presence_pct = st.slider("Sustained presence needed (% of session)", 40, 95, 70, 5)
        gap_s = st.slider("Bridge detection gaps up to (s)", 1.0, 15.0, 5.0, 0.5)
        window_s = st.slider("Analysis window (s)", 5, 60, 10, 5)
    with st.expander("Engagement fusion & advisor", expanded=False):
        w_vis = st.slider("Weight of visual signal (%)", 0, 100, 60, 5)
        low_thr = st.slider("Low-engagement threshold", 20, 70, 45, 5)
        min_low = st.slider("Advise after a dip lasting (s)", 30, 600, 90, 10)
        smooth = st.slider("Smoothing (windows)", 1, 7, 3, 2)

    go = st.button("▶ Analyze session", type="primary", width="stretch",
                   disabled=(mode == "upload" and video_file is None and audio_file is None))
    if mode == "upload":
        st.caption("Heavy steps (face detection, audio analysis) run when you press Analyze. "
                   "Threshold, window, weight and advisor sliders update the results instantly.")


# --------------------------------------------------------------------------- #
# Run / restore analysis
# --------------------------------------------------------------------------- #
if go or "S" not in st.session_state:
    if go or mode == "demo":
        try:
            with st.spinner("Preparing session..."):
                st.session_state["S"] = run_analysis(mode, video_file, audio_file, use_video_audio,
                                                     fps, max_minutes, window_s, gap_s)
        except Exception as e:  # noqa: BLE001
            st.error(f"Analysis failed: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

S = st.session_state.get("S")

st.markdown("# Multimodal Attendance & Engagement Agent")
st.markdown('<div class="sub">Presence-based attendance, collective engagement trends and teaching suggestions from classroom video and audio.</div>', unsafe_allow_html=True)

if S is None:
    st.info("Upload a classroom recording in the sidebar and press **Analyze session**, or switch to the demo session.")
    st.stop()

# --------------------------------------------------------------------------- #
# Derived results (cheap; recomputed whenever a slider changes)
# --------------------------------------------------------------------------- #
vision, sec_df, duration = S["vision"], S["audio_sec"], S["duration_s"]
vcfg = VisionConfig(sample_fps=S["fps"], presence_threshold=presence_pct / 100, max_gap_s=gap_s, window_s=float(window_s))

attendance = vis_win = aud_win = None
if vision is not None:
    attendance = compute_attendance(vision.obs, duration, vcfg)
    vis_win = visual_windows(vision.obs, duration, vcfg)
if sec_df is not None:
    aud_win = audio_windows(sec_df, duration, float(window_s))
if vis_win is None and aud_win is None:
    st.warning("Nothing to analyse yet. Upload a video and/or an audio file.")
    st.stop()
if vision is not None and vision.obs.empty and sec_df is None:
    st.warning("No faces were detected in this video, so there is nothing to score. Try better lighting, a camera "
               "that sees faces front-on, or lower the face-size limit by using a higher-resolution recording.")
    st.stop()

fused = fuse(vis_win if (vis_win is not None and vision is not None and not vision.obs.empty) else None,
             aud_win, w_visual=w_vis / 100, w_audio=1 - w_vis / 100, smooth=smooth)
suggestions = generate_suggestions(fused, AdvisorConfig(low_engagement=float(low_thr), min_low_s=float(min_low)))
summary = summarise(attendance, fused, sec_df, duration)

for n in S["notes"]:
    st.caption("ℹ️ " + n)

tabs = st.tabs(["1 · Attendance", "2 · Audio interaction", "3 · Engagement fusion", "4 · Teaching suggestions", "5 · Session report"])

# ---- Tab 1: attendance ------------------------------------------------------
with tabs[0]:
    if attendance is None or attendance.empty:
        st.info("No video analysed, so there is no attendance register. Upload a classroom video to track presence.")
    else:
        present = int((attendance["Status"] == "Present").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faces tracked", len(attendance))
        c2.metric("Present", present)
        c3.metric("Absent / passing by", len(attendance) - present)
        c4.metric("Attendance rate", f"{100 * present / len(attendance):.0f}%")
        st.markdown(f'<div class="note">A student is marked <b>Present</b> only after being seen for at least '
                    f'<b>{presence_pct}%</b> of the session ({fmt(duration)}). Short detection gaps up to {gap_s:.0f}s are bridged, '
                    f'so a missed frame does not break attendance and someone walking past never counts.</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1])
        with left:
            show = attendance[["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %", "Status"]]
            st.dataframe(show, hide_index=True, height=min(520, 38 * (len(show) + 1)))
            st.download_button("⬇ Download attendance CSV", show.to_csv(index=False).encode(),
                               "attendance_register.csv", "text/csv")
        with right:
            st.pyplot(presence_chart(attendance, presence_pct))
        if vision.preview_rgb is not None:
            st.markdown("**Sample frame with tracked faces** (green = facing forward, orange = turned, red = slumped)")
            st.image(vision.preview_rgb, width=640, caption="Annotated in memory only; never saved.")
        if vis_win is not None:
            st.markdown("**Headcount over time**")
            st.pyplot(headcount_chart(vis_win))

    st.divider()
    with st.expander("📷 Quick headcount from a photo or webcam snapshot"):
        st.caption("Works in the browser camera too. The image is processed in memory and discarded.")
        cam = st.camera_input("Take a snapshot")
        img = st.file_uploader("...or upload a classroom photo", type=["jpg", "jpeg", "png"], key="photo")
        src = cam or img
        if src is not None:
            arr = cv2.imdecode(np.frombuffer(src.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                st.error("Could not read that image.")
            else:
                ann, fwd, turned = detect_snapshot(arr)
                st.image(ann, width=640)
                st.write(f"**{fwd + turned} faces detected** - {fwd} facing forward, {turned} turned away.")

# ---- Tab 2: audio -----------------------------------------------------------
with tabs[1]:
    if sec_df is None:
        st.info("No audio was provided. Upload an audio file (or tick *Use the video's own audio track*) to see speech and noise analysis.")
    else:
        a, b = st.columns([1.5, 1])
        with a:
            st.markdown("**Classroom sound timeline**")
            st.pyplot(audio_timeline(sec_df))
            st.markdown("**Sound level above the room's noise floor**")
            st.pyplot(audio_level_chart(sec_df))
        with b:
            st.markdown("**Share of session**")
            st.pyplot(class_share_chart(sec_df))
            st.markdown(f'<div class="note">Noise floor of this recording: {sec_df["noise_floor_db"].iloc[0]:.0f} dB (thresholds are relative to it).</div>', unsafe_allow_html=True)
        with st.expander("How the sound classes are decided"):
            st.markdown(
                "- **Silence** - nothing noticeably above the room's own noise floor.\n"
                "- **Instructor Lecture** - speech-like sound (voice band, tonal, syllabic rhythm) in long continuous runs (8s+).\n"
                "- **Student Discussion** - speech-like sound in short alternating bursts: questions, answers, group talk.\n"
                "- **Ambient Distraction** - loud but not speech-like, or overlapping chatter (babble, shuffling, noise).\n\n"
                "Without speaker identification, *instructor* vs *student* is inferred from how long each speech run lasts.")

# ---- Tab 3: fusion ----------------------------------------------------------
with tabs[2]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average engagement", f"{summary['avg_engagement']:.0f}%")
    m2.metric("Peak", f"{summary['peak_engagement']['value']:.0f}%", f"at {summary['peak_engagement']['at']}", delta_color="off")
    m3.metric("Lowest", f"{summary['lowest_engagement']['value']:.0f}%", f"at {summary['lowest_engagement']['at']}", delta_color="off")
    m4.metric("Time below threshold", summary["time_below_45"] if low_thr == 45 else f"{100 * float((fused['engagement'] < low_thr).mean()):.0f}%")
    st.pyplot(engagement_chart(fused, float(low_thr)))
    only = []
    if vis_win is None or (vision is not None and vision.obs.empty):
        only.append("audio only")
    if aud_win is None:
        only.append("video only")
    st.markdown(f'<div class="note">Engagement Index = {w_vis}% visual attentiveness (head pose, posture, movement, stillness) + '
                f'{100 - w_vis}% audio interaction (discussion scores highest, noise lowest), smoothed over {smooth} window(s) of {window_s}s.'
                f'{" Using " + ", ".join(only) + "." if only else ""} Scores describe the <b>group</b>, never an individual student.</div>', unsafe_allow_html=True)
    with st.expander("Window-by-window data"):
        tbl = fused.copy()
        tbl.insert(1, "time", tbl["t_start"].map(fmt))
        st.dataframe(tbl.round(2), hide_index=True)
        st.download_button("⬇ Download engagement CSV", tbl.round(3).to_csv(index=False).encode(), "engagement_timeline.csv", "text/csv")

# ---- Tab 4: suggestions -----------------------------------------------------
with tabs[3]:
    st.markdown("Non-intrusive prompts for the teacher. The agent only advises; **you decide**.")
    for sg in suggestions:
        st.markdown(f'<div class="card {sg.severity}"><span class="t">{sg.title}</span>'
                    f'<span class="when">{fmt(sg.start_s)} – {fmt(sg.end_s)}</span>'
                    f'<div class="m">{sg.message}</div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Replay as a live advisor**")
    t_now = st.slider("Session time", 0, int(max(duration - 1, 1)), 0, 10, format="%d s")
    i = int(min(t_now // window_s, len(fused) - 1))
    row = fused.iloc[i]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Engagement at {fmt(t_now)}", f"{row['engagement']:.0f}%")
    k2.metric("Visual attentiveness", "–" if pd.isna(row["visual"]) else f"{row['visual']:.0f}%")
    k3.metric("Audio interaction", "–" if pd.isna(row["audio"]) else f"{row['audio']:.0f}%")
    active = [s for s in suggestions if s.start_s <= t_now < s.end_s]
    if active:
        for sg in active:
            st.info(f"**{sg.title}** – {sg.message}")
    else:
        st.success("No advisory at this moment.")

# ---- Tab 5: report ----------------------------------------------------------
with tabs[4]:
    md = to_markdown(summary, attendance, suggestions)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Report (Markdown)", md.encode(), "session_report.md", "text/markdown", width="stretch")
    d2.download_button("⬇ Summary (JSON)", to_json(summary, suggestions).encode(), "session_summary.json", "application/json", width="stretch")
    if attendance is not None and not attendance.empty:
        d3.download_button("⬇ Attendance (CSV)", attendance.iloc[:, :6].to_csv(index=False).encode(), "attendance_register.csv", "text/csv", width="stretch")
    st.divider()
    st.markdown(md)
