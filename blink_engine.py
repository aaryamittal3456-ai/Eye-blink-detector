"""
blink_engine.py  —  v2  (glasses-aware, adaptive calibration)
=============================================================

Key improvements over v1:
  1. EXPANDED landmark sets  — uses 8 points per eye (full upper/lower
     eyelid) instead of 6, giving a more robust EAR that is less thrown
     off by glasses frames cutting across the 6-point contour.

  2. ADAPTIVE BASELINE CALIBRATION  — spends the first N frames
     measuring the user's personal open-eye EAR, then derives the
     threshold as:
         threshold = baseline_ear * RATIO
     This makes detection work regardless of eye shape, face distance,
     or whether the user wears glasses.

  3. RELATIVE-DROP DETECTION  — instead of a hard absolute cutoff, a
     blink is confirmed when the EAR drops to ≤ threshold AND the drop
     from the rolling baseline is ≥ MIN_DROP_RATIO.  This rejects the
     slow "squint drift" that glasses reflections cause.

  4. REFRACTORY PERIOD  — after each blink a short lockout (~300 ms)
     prevents double-counting the same blink as the eye reopens.

  5. DEBOUNCE SMOOTHING  — EAR values are smoothed with a tiny rolling
     window (3 frames) to reduce single-frame noise from reflections.
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────
#  Expanded eye landmark sets  (MediaPipe 478-point mesh)
#
#  We use 8 points per eye arranged as:
#    p0 = left corner       p4 = right corner
#    p1, p2 = upper lid     p3 = upper-mid
#    p5, p6 = lower lid     p7 = lower-mid
#
#  This gives 3 vertical distances instead of 2, yielding a more
#  stable EAR when part of the iris is occluded by a glasses frame.
# ──────────────────────────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 398, 384, 385, 386, 387, 388, 263,
                 373, 374, 380, 381, 382, 362]   # 8 unique pts used below
RIGHT_EYE_IDX = [33,  246, 161, 160, 159, 158, 157, 133,
                 153, 145, 144, 163, 7,   33]

# The 8 points actually used for EAR (indices into the face mesh):
#   left  eye: corner_L, top1, top2, corner_R, bot1, bot2  + top_mid, bot_mid
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380, 386, 374]
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144, 159, 145]

# Contour points just for drawing (more complete ring)
LEFT_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_CONTOUR = [33,  7,   163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]

# Calibration / detection parameters
CALIB_FRAMES     = 60        # ~2 s at 30 fps to build personal baseline
EAR_CLOSE_RATIO  = 0.88      # threshold = baseline * this ratio
                              # 0.88 means: if open EAR=0.50, threshold=0.44
                              # glasses users need a HIGH ratio because glass
                              # frames prevent the eye from appearing fully closed
MIN_DROP_RATIO   = 0.08      # blink must drop at least 8% from baseline
                              # lowered so partial blinks through glasses count
EAR_SMOOTH_WIN   = 2         # frames for noise smoothing (less = more responsive)
REFRACTORY_SEC   = 0.20      # seconds to ignore after a blink fires
MAX_BLINK_FRAMES = 30        # eye closed > this = not a blink (e.g. looking away)


# ══════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Settings:
    # These are now *starting* values; adaptive calibration can override
    ear_threshold:    float = 0.21   # used before calibration completes
    consec_frames:    int   = 2      # min closed frames to register blink
    alert_low_bpm:    float = 8.0
    alert_high_bpm:   float = 25.0
    smoothing_window: int   = 180    # frames for rolling bpm average
    # Adaptive calibration toggles
    adaptive_mode:    bool  = True   # False → use fixed ear_threshold
    calib_frames:     int   = CALIB_FRAMES


@dataclass
class SessionSnapshot:
    timestamp: float
    blinks:    int
    bpm:       float
    ear_avg:   float
    status:    str


@dataclass
class FrameResult:
    blink_count:    int   = 0
    bpm:            float = 0.0
    ear_left:       float = 0.0
    ear_right:      float = 0.0
    ear_avg:        float = 0.0
    eye_closed:     bool  = False
    face_found:     bool  = False
    status:         str   = "Measuring..."
    status_color:   str   = "#94a3b8"
    warning:        str   = ""
    elapsed:        float = 0.0
    calibrating:    bool  = False
    calib_progress: float = 0.0   # 0.0 – 1.0
    baseline_ear:   float = 0.0
    adaptive_thresh: float = 0.0


# ══════════════════════════════════════════════════════════════════════
#  Landmark helpers
# ══════════════════════════════════════════════════════════════════════

def get_eye_landmarks(face_landmarks, indices: List[int],
                      img_w: int, img_h: int) -> np.ndarray:
    """Convert normalised landmarks → pixel-space ndarray (N, 2)."""
    return np.array(
        [(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h)
         for i in indices],
        dtype=np.float32,
    )


def compute_ear_8pt(pts: np.ndarray) -> float:
    """
    8-point EAR using 3 vertical distances and 1 horizontal.

    Layout expected (indices into pts array):
      pts[0] = left corner    pts[4] = right corner  (horizontal)
      pts[1] = upper-left     pts[5] = lower-left    (vertical pair 1)
      pts[2] = upper-mid      pts[6] = lower-mid     (vertical pair 2)
      pts[3] = upper-right    pts[7] = lower-right   (vertical pair 3)

    EAR = (||p1-p5|| + ||p2-p6|| + ||p3-p7||) / (3 * ||p0-p4||)
    """
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[6])
    C = np.linalg.norm(pts[3] - pts[7])
    D = np.linalg.norm(pts[0] - pts[4])
    return float((A + B + C) / (3.0 * D + 1e-6))


# ══════════════════════════════════════════════════════════════════════
#  Health classification
# ══════════════════════════════════════════════════════════════════════

def classify_health(bpm: float, settings: Settings):
    """Returns (label, css_color, warning_text)."""
    if bpm == 0.0:
        return "Measuring...", "#94a3b8", ""
    if bpm < settings.alert_low_bpm:
        return (
            "Low blinking — risk of eye strain",
            "#ef4444",
            "⚠️  Blink more often & take a short break!",
        )
    if bpm <= 20:
        return "Healthy blinking ✓", "#22c55e", ""
    if bpm <= settings.alert_high_bpm:
        return "Slightly elevated blinking", "#f59e0b", ""
    return (
        "Frequent blinking — possible irritation",
        "#f59e0b",
        "⚠️  Your eyes may be irritated. Rest them.",
    )


# ══════════════════════════════════════════════════════════════════════
#  Adaptive Calibrator
# ══════════════════════════════════════════════════════════════════════

class EARCalibrator:
    """
    Collects open-eye EAR samples for the first N frames,
    then computes a personal baseline and detection threshold.
    """

    def __init__(self, n_frames: int = CALIB_FRAMES):
        self.n_frames    = n_frames
        self._samples: deque = deque(maxlen=n_frames)
        self.baseline    = 0.0
        self.threshold   = 0.0
        self.done        = False

    def feed(self, ear: float) -> bool:
        """
        Add one sample. Only called when eye is likely open (ear > 0.15).
        Returns True when calibration just completed.
        """
        if self.done:
            return False
        if ear > 0.15:                    # ignore frames where eye is shut
            self._samples.append(ear)
        if len(self._samples) >= self.n_frames:
            self._finish()
            return True
        return False

    def _finish(self):
        arr = np.array(self._samples)
        # 75th percentile = true open-eye EAR, works well for glasses
        self.baseline  = float(np.percentile(arr, 75))
        self.threshold = self.baseline * EAR_CLOSE_RATIO
        self.done      = True
    @property
    def progress(self) -> float:
        return min(len(self._samples) / self.n_frames, 1.0)

    def reset(self, n_frames: Optional[int] = None):
        if n_frames:
            self.n_frames = n_frames
        self._samples.clear()
        self.baseline  = 0.0
        self.threshold = 0.0
        self.done      = False


# ══════════════════════════════════════════════════════════════════════
#  Blink Detector  (state machine with adaptive threshold)
# ══════════════════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Counts blinks robustly using:
      - adaptive per-user EAR baseline
      - relative drop confirmation
      - refractory period
      - max-closed-frame cap
      - 3-frame EAR smoothing
    """

    def __init__(self, settings: Settings):
        self.settings      = settings
        self._calibrator   = EARCalibrator(settings.calib_frames)
        self._ear_buf: deque = deque(maxlen=EAR_SMOOTH_WIN)

        self._counter      = 0       # consecutive below-threshold frames
        self.blink_count   = 0
        self.eye_closed    = False
        self._last_blink_t = 0.0     # timestamp of last confirmed blink
        self._start        = time.time()

        # history for charts
        self.history: List[SessionSnapshot] = []
        self._last_snap    = 0.0

        # BPM smoothing
        self._bpm_hist: deque = deque(maxlen=settings.smoothing_window)

        # cache landmarks for draw_overlay
        self._lms_cache    = None

    # ── public API ────────────────────────────────────────────────────

    def update(self, raw_ear: float) -> bool:
        """
        Feed one frame's average raw EAR.
        Returns True when a new blink is confirmed.
        """
        # 1. Smooth EAR
        self._ear_buf.append(raw_ear)
        ear = float(np.mean(self._ear_buf))

        # 2. Calibration phase
        if self.settings.adaptive_mode and not self._calibrator.done:
            self._calibrator.feed(ear)
            # During calibration use the fixed threshold
            threshold = self.settings.ear_threshold
            baseline  = threshold / EAR_CLOSE_RATIO   # approximate
        else:
            if self.settings.adaptive_mode:
                threshold = self._calibrator.threshold
                baseline  = self._calibrator.baseline
            else:
                threshold = self.settings.ear_threshold
                baseline  = threshold / EAR_CLOSE_RATIO

        # 3. Relative drop check — must fall meaningfully from baseline
        drop_ratio = (baseline - ear) / (baseline + 1e-6)
        eye_is_closed = (ear < threshold) and (drop_ratio >= MIN_DROP_RATIO)

        # 4. State machine
        blinked = False
        if eye_is_closed:
            self._counter  += 1
            self.eye_closed = True
        else:
            if (self.eye_closed
                    and self._counter >= self.settings.consec_frames
                    and self._counter <= MAX_BLINK_FRAMES):
                # Refractory check
                now = time.time()
                if now - self._last_blink_t >= REFRACTORY_SEC:
                    self.blink_count  += 1
                    self._last_blink_t = now
                    blinked            = True
            self._counter   = 0
            self.eye_closed = False

        return blinked

    def compute_bpm(self) -> float:
        elapsed = self.elapsed()
        if elapsed < 3.0:
            return 0.0
        raw = (self.blink_count / elapsed) * 60.0
        self._bpm_hist.append(raw)
        return float(np.mean(self._bpm_hist))

    def elapsed(self) -> float:
        return time.time() - self._start

    def calibrating(self) -> bool:
        return self.settings.adaptive_mode and not self._calibrator.done

    def calib_progress(self) -> float:
        return self._calibrator.progress

    def baseline_ear(self) -> float:
        return self._calibrator.baseline

    def adaptive_threshold(self) -> float:
        if self.settings.adaptive_mode and self._calibrator.done:
            return self._calibrator.threshold
        return self.settings.ear_threshold

    def maybe_record_snapshot(self, bpm: float, ear_avg: float, status: str):
        now = self.elapsed()
        if now - self._last_snap >= 1.0:
            self.history.append(SessionSnapshot(
                timestamp = now,
                blinks    = self.blink_count,
                bpm       = bpm,
                ear_avg   = ear_avg,
                status    = status,
            ))
            self._last_snap = now

    def reset(self, settings: Settings = None):
        if settings:
            self.settings = settings
        self._calibrator.reset(self.settings.calib_frames)
        self._ear_buf.clear()
        self._counter      = 0
        self.blink_count   = 0
        self.eye_closed    = False
        self._last_blink_t = 0.0
        self._bpm_hist.clear()
        self._start        = time.time()
        self.history       = []
        self._last_snap    = 0.0
        self._lms_cache    = None
