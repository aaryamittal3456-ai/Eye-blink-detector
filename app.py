"""
app.py  —  Eye Blink Monitor  (Flask)
======================================
Features:
  - Real-time blink detection
  - Break reminder after 20 min low blinking
  - Daily report saved to reports/ folder
  - Sound alert trigger via API
  - Desktop notifications via API

Run:  python app.py
Open: http://localhost:5000
"""

import cv2, time, threading, json, os
from datetime import datetime
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, request

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)
from blink_engine import (
    Settings, BlinkDetector, FrameResult,
    LEFT_EAR_PTS, RIGHT_EAR_PTS, LEFT_CONTOUR, RIGHT_CONTOUR,
    get_eye_landmarks, compute_ear_8pt, classify_health,
)
from model_utils import ensure_model

app = Flask(__name__)
os.makedirs("reports", exist_ok=True)

DEFAULT_SETTINGS = Settings(
    ear_threshold    = 0.21,
    consec_frames    = 1,
    alert_low_bpm    = 8.0,
    alert_high_bpm   = 25.0,
    smoothing_window = 150,
    adaptive_mode    = True,
    calib_frames     = 60,
)

# ── Shared state ───────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.running         = False
        self.detector        = None
        self.last_result     = FrameResult()
        self.last_jpeg       = None
        self.lock            = threading.Lock()
        # Alert tracking
        self.low_blink_since = None   # timestamp when low blinking started
        self.break_alert_sent= False  # True once 20-min alert fired this stretch
        self.pending_alerts  = []     # list of alert dicts for browser to pop
        self.alert_lock      = threading.Lock()
        # Session log (for daily report)
        self.session_start   = None
        self.session_log     = []     # list of {time, blinks, bpm, status}

state = AppState()


# ══════════════════════════════════════════════════════════════════════
#  Alert helpers
# ══════════════════════════════════════════════════════════════════════

def push_alert(kind: str, title: str, message: str):
    """Queue an alert for the browser to display."""
    with state.alert_lock:
        state.pending_alerts.append({
            "kind":    kind,       # "break" | "low" | "high"
            "title":   title,
            "message": message,
            "ts":      time.time(),
        })

def check_alerts(r: FrameResult):
    """Called every frame to decide whether to fire alerts."""
    if r.calibrating or r.bpm == 0:
        return

    now = time.time()

    # ── Low-blink tracking ─────────────────────────────────────────────
    if r.bpm < DEFAULT_SETTINGS.alert_low_bpm and r.face_found:
        if state.low_blink_since is None:
            state.low_blink_since    = now
            state.break_alert_sent   = False
        else:
            low_duration = now - state.low_blink_since
            # 20-minute break reminder
            if low_duration >= 1200 and not state.break_alert_sent:
                push_alert(
                    "break",
                    "⏰ Take a Break!",
                    "You've had low blinking for over 20 minutes. "
                    "Look away from the screen and blink slowly 10 times."
                )
                state.break_alert_sent = True
            # Immediate low-blink notification (fires once per low stretch)
            elif low_duration >= 30 and not state.break_alert_sent:
                push_alert(
                    "low",
                    "👁️ Blink More!",
                    f"Your blink rate is {r.bpm:.1f} bpm — too low. "
                    "Try to blink consciously every few seconds."
                )
    else:
        # Reset low-blink tracker when rate recovers
        state.low_blink_since  = None
        state.break_alert_sent = False


# ══════════════════════════════════════════════════════════════════════
#  Report saving
# ══════════════════════════════════════════════════════════════════════

def save_report(det: BlinkDetector):
    """Write a JSON + simple text report for this session."""
    if not det or not det.history:
        return

    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    time_str  = now.strftime("%H-%M-%S")
    fname     = f"reports/session_{date_str}_{time_str}"

    # Compute summary
    bpms     = [h.bpm for h in det.history if h.bpm > 0]
    avg_bpm  = round(sum(bpms) / len(bpms), 1) if bpms else 0
    elapsed  = det.elapsed()
    m, s     = divmod(int(elapsed), 60)
    _, _, _  = classify_health(avg_bpm, DEFAULT_SETTINGS)
    status_lbl, _, _ = classify_health(avg_bpm, DEFAULT_SETTINGS)

    summary = {
        "date":          date_str,
        "start_time":    now.strftime("%H:%M:%S"),
        "duration":      f"{m:02d}:{s:02d}",
        "total_blinks":  det.blink_count,
        "avg_bpm":       avg_bpm,
        "status":        status_lbl,
        "history":       [
            {"t": round(h.timestamp,1), "bpm": round(h.bpm,1),
             "ear": round(h.ear_avg,3), "status": h.status}
            for h in det.history
        ],
    }

    # Save JSON
    with open(fname + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save readable text report
    lines = [
        "=" * 48,
        "  EYE BLINK MONITOR — Session Report",
        "=" * 48,
        f"  Date          : {date_str}",
        f"  Duration      : {m:02d}:{s:02d}",
        f"  Total blinks  : {det.blink_count}",
        f"  Avg blink rate: {avg_bpm} bpm",
        f"  Eye health    : {status_lbl}",
        "=" * 48,
        "",
        "  Minute-by-minute bpm:",
    ]
    # Bucket history into minutes
    buckets = {}
    for h in det.history:
        minute = int(h.timestamp // 60)
        buckets.setdefault(minute, []).append(h.bpm)
    for min_idx in sorted(buckets):
        avg = sum(buckets[min_idx]) / len(buckets[min_idx])
        lines.append(f"    Min {min_idx+1:02d}  →  {avg:.1f} bpm")

    with open(fname + ".txt", "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Report saved: {fname}.txt")
    return fname


# ══════════════════════════════════════════════════════════════════════
#  Frame drawing
# ══════════════════════════════════════════════════════════════════════

def draw_frame(frame, r: FrameResult, det: BlinkDetector):
    h, w = frame.shape[:2]

    if r.face_found and det._lms_cache is not None:
        lms    = det._lms_cache
        cont_l = get_eye_landmarks(lms, LEFT_CONTOUR,  w, h)
        cont_r = get_eye_landmarks(lms, RIGHT_CONTOUR, w, h)
        col = ((0,200,255) if det.calibrating()
               else (40,40,230) if r.eye_closed
               else (0,230,90))
        for pts in (cont_l, cont_r):
            p = pts.astype(np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [p], True, col, 2, cv2.LINE_AA)
            for pt in pts.astype(np.int32):
                cv2.circle(frame, tuple(pt), 2, col, -1, cv2.LINE_AA)

    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,44), (10,14,26), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    m, s = divmod(int(r.elapsed), 60)
    cv2.putText(frame,
        f"Blinks: {r.blink_count}   BPM: {r.bpm:.1f}   "
        f"{'Calibrating...' if r.calibrating else r.status}",
        (12,16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,220,255), 1, cv2.LINE_AA)
    cv2.putText(frame,
        f"Session: {m:02d}:{s:02d}   EAR L:{r.ear_left:.3f} R:{r.ear_right:.3f}",
        (12,36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140,160,200), 1, cv2.LINE_AA)

    if r.calibrating:
        bw = int(w * r.calib_progress)
        cv2.rectangle(frame, (0,h-6), (w,h), (15,20,40), -1)
        cv2.rectangle(frame, (0,h-6), (bw,h), (56,189,248), -1)
        cv2.putText(frame,
            f"Setting up for your eyes... {int(r.calib_progress*100)}%",
            (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,190,255), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════
#  Camera thread
# ══════════════════════════════════════════════════════════════════════

def camera_loop():
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode = RunningMode.VIDEO,
        num_faces    = 1,
        min_face_detection_confidence        = 0.45,
        min_face_presence_confidence         = 0.45,
        min_tracking_confidence              = 0.45,
        output_face_blendshapes              = False,
        output_facial_transformation_matrixes = False,
    )

    det = BlinkDetector(DEFAULT_SETTINGS)
    state.detector      = det
    state.session_start = datetime.now().strftime("%H:%M:%S")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while state.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            elap  = det.elapsed()

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = landmarker.detect_for_video(mp_img, int(elap * 1000))

            r = FrameResult(elapsed=elap)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                det._lms_cache = lms
                r.face_found   = True
                lp = get_eye_landmarks(lms, LEFT_EAR_PTS,  w, h)
                rp = get_eye_landmarks(lms, RIGHT_EAR_PTS, w, h)
                r.ear_left  = compute_ear_8pt(lp)
                r.ear_right = compute_ear_8pt(rp)
                r.ear_avg   = (r.ear_left + r.ear_right) / 2.0
                det.update(r.ear_avg)
                r.eye_closed  = det.eye_closed
                r.blink_count = det.blink_count
            else:
                det._counter   = 0
                det.eye_closed = False
                det._lms_cache = None

            r.bpm            = det.compute_bpm()
            r.calibrating    = det.calibrating()
            r.calib_progress = det.calib_progress()
            r.baseline_ear   = det.baseline_ear()
            r.adaptive_thresh= det.adaptive_threshold()

            lbl, col, warn = classify_health(r.bpm, DEFAULT_SETTINGS)
            r.status       = lbl
            r.status_color = col
            r.warning      = warn if not r.calibrating else ""

            det.maybe_record_snapshot(r.bpm, r.ear_avg, r.status)
            check_alerts(r)

            frame = draw_frame(frame, r, det)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with state.lock:
                state.last_result = r
                state.last_jpeg   = jpeg.tobytes()

    # Session ended — save report
    save_report(state.detector)
    cap.release()
    state.detector  = None
    state.last_jpeg = None


# ══════════════════════════════════════════════════════════════════════
#  Flask routes
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    if not state.running:
        state.running = True
        state.low_blink_since  = None
        state.break_alert_sent = False
        with state.alert_lock:
            state.pending_alerts = []
        threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    state.running = False
    return jsonify({"ok": True})

@app.route("/reset", methods=["POST"])
def reset():
    if state.detector:
        state.detector.reset(DEFAULT_SETTINGS)
    state.last_result      = FrameResult()
    state.low_blink_since  = None
    state.break_alert_sent = False
    with state.alert_lock:
        state.pending_alerts = []
    return jsonify({"ok": True})

@app.route("/stats")
def stats():
    r   = state.last_result
    det = state.detector
    history = []
    if det and det.history:
        history = [
            {"t": round(h.timestamp,1), "bpm": round(h.bpm,1)}
            for h in det.history[-120:]
        ]

    # Pop any pending alerts
    with state.alert_lock:
        alerts = state.pending_alerts[:]
        state.pending_alerts = []

    # Low-blink duration (seconds) for the progress ring in UI
    low_dur = 0
    if state.low_blink_since:
        low_dur = int(time.time() - state.low_blink_since)

    return jsonify({
        "running":       state.running,
        "blink_count":   r.blink_count,
        "bpm":           round(r.bpm, 1),
        "ear_avg":       round(r.ear_avg, 3),
        "ear_left":      round(r.ear_left, 3),
        "ear_right":     round(r.ear_right, 3),
        "face_found":    r.face_found,
        "eye_closed":    r.eye_closed,
        "calibrating":   r.calibrating,
        "calib_pct":     int(r.calib_progress * 100),
        "status":        r.status,
        "status_color":  r.status_color,
        "warning":       r.warning,
        "elapsed":       round(r.elapsed, 0),
        "history":       history,
        "alerts":        alerts,
        "low_dur_sec":   low_dur,
    })

@app.route("/reports")
def list_reports():
    files = sorted(
        [f for f in os.listdir("reports") if f.endswith(".json")],
        reverse=True
    )[:20]
    reports = []
    for f in files:
        try:
            with open(f"reports/{f}") as fp:
                d = json.load(fp)
            reports.append({
                "file":    f,
                "date":    d.get("date",""),
                "time":    d.get("start_time",""),
                "duration":d.get("duration",""),
                "blinks":  d.get("total_blinks",0),
                "avg_bpm": d.get("avg_bpm",0),
                "status":  d.get("status",""),
            })
        except:
            pass
    return jsonify(reports)

def gen_frames():
    while True:
        with state.lock:
            jpeg = state.last_jpeg
        if jpeg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
        else:
            time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("\n👁️  Eye Blink Monitor")
    print("─" * 36)
    print("  Open →  http://localhost:5000")
    print("  Press Ctrl+C to quit\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
