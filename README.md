# Multimodal Attendance & Engagement Agent

A privacy-first classroom intelligence dashboard. From a classroom video and/or audio recording it
1. marks **attendance by sustained presence** (not a single lucky frame),
2. measures **collective engagement** by fusing what cameras see (head pose, posture, movement) with what microphones hear (lecture, discussion, noise), and
3. gives the teacher **non-intrusive teaching suggestions** and a downloadable session report.

> Skill Wallet Capstone Project 5 · Generative AI · Privacy-first classroom analytics

**All processing happens locally in memory.** Uploaded files are deleted right after analysis; no frame, face crop or audio clip is stored or sent to any cloud API. Scores describe the *group*, never an individual student.

## Try it in 10 seconds
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens with a **synthetic 15-minute demo lesson** (24 simulated seats + generated classroom audio), so every tab is populated without uploading anything. Switch to *Upload my own recording* in the sidebar to analyse a real video / audio file.

## Features by epic
| Epic | What it does | Where |
|---|---|---|
| 1 Requirements & offline env | Local-only pipeline, no external APIs, Python 3.9+ | `requirements.txt`, this README |
| 2 Face tracking & sustained presence | Haar-cascade face detection (frontal + profile), seat-aware tracker, presence must reach **70 %** of the session (gaps ≤ 5 s are bridged) | `core/vision.py` → *Attendance* tab |
| 3 Audio speech & noise | Voice-activity detection + 4-way classification: Silence / Instructor Lecture / Student Discussion / Ambient Distraction | `core/audio.py` → *Audio* tab |
| 4 Multimodal fusion & scoring | Visual attentiveness + audio interaction → Engagement Index 0-100 per window | `core/fusion.py` → *Fusion* tab |
| 5 Suggestion engine & reports | Transparent rule-based advisor, live-replay slider, Markdown / JSON / CSV reports | `core/suggestions.py`, `core/report.py` |
| 6 Dashboard, testing, deployment | Streamlit app, pytest suite, Streamlit Community Cloud ready | `app.py`, `tests/` |

## How it works
```
video ─► frame sampling (2 fps) ─► face detection ─► seat-aware tracker ─┬─► sustained-presence attendance register (CSV)
                                                                          └─► head pose / posture / movement ─► visual attentiveness ─┐
audio ─► 1-second windows ─► energy + speech-band + flatness + syllabic modulation ─► class labels ─► audio interaction ───────────────┤
                                                                                                                                     ▼
                                                              weighted fusion (default 60/40) ─► Engagement Index ─► advisor rules ─► dashboard & report
```
* **Head pose:** frontal face = looking forward, profile-only = turned away. **Posture:** a face that sinks well below that student's own baseline = slumped.
* **Movement / inactivity:** face displacement per second (in face-widths) flags restlessness; a motionless, non-forward face counts as inactive.
* **Audio thresholds are relative to the recording's own noise floor**, so they adapt to different microphones and gain.
* **Advisor rules:** sustained engagement dip, high background noise, long lecture with falling attention, quiet + looking away, headcount drop, and engagement peaks worth repeating.

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a **public** GitHub repo (`app.py`, `requirements.txt`, `core/`, `.streamlit/`, `README.md`).
2. Go to <https://share.streamlit.io> → **New app** → pick the repo, branch `main`, main file `app.py`.
3. *Advanced settings* → choose **Python 3.12**. Deploy.
4. Paste the repo URL and the `https://<name>.streamlit.app` URL into Skill Wallet (Add Github Link / Add Demo Link).

On the cloud there is no server-side webcam. Use *Upload my own recording*, or the browser camera snapshot in the Attendance tab.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```
Covers: sustained-presence logic (passers-by and late arrivals stay *Absent*, short occlusions are bridged), audio classification, gain robustness, fusion with one or both modalities, advisor output, and an end-to-end run on a synthetic video.

## Known limitations (honest notes)
* Haar cascades are fast and fully offline but less accurate than deep detectors: expect missed faces in poor light, strong angles, masks or very small/far faces. Use a front-facing camera and 720p+ footage.
* Seat-aware tracking assumes students stay roughly in their seat.
* Instructor vs student speech is inferred from speech-run length (no speaker identification).
* Audio thresholds were tuned on synthetic audio plus basic robustness checks; calibrate on a real recording from your classroom before drawing conclusions.
* Engagement scores are heuristics for supporting teaching decisions, **not** for grading or disciplining students.
* Face *detection* is used for counting and tracking only; there is no face recognition or identity storage. Students appear as anonymous IDs (S01, S02, ...).

## Project structure
```
app.py                 Streamlit dashboard (5 tabs)
core/vision.py         face detection, tracking, attendance, visual windows
core/audio.py          loading, voice-activity detection, class labels
core/fusion.py         engagement index
core/suggestions.py    teaching-suggestion agent
core/report.py         KPIs + report export
core/demo.py           synthetic demo session
core/plots.py          charts
tests/                 pytest suite
```
