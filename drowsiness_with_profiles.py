"""
Original drowsiness detection script enhanced with:
  1. Per-eye independent calibration & thresholds
  2. Driver profile save/load (JSON)

Controls:
  [S] Save current calibration as a driver profile
  [R] Recalibrate (new driver or lighting changed)
  [Q] Quit
"""

from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import json
import os
import argparse

# ─── Sound ────────────────────────────────────────────────────────
mixer.init()
mixer.music.load("music.wav")

# ─── Constants ────────────────────────────────────────────────────
CALIB_FRAMES  = 100     # frames to collect during calibration (~3 sec)
THRESH_RATIO  = 0.75    # auto threshold = baseline × this
FRAME_CHECK   = 20      # consecutive drowsy frames to trigger alert
BLINK_FILTER  = 0.15    # ignore EAR below this during calibration (blinks)
PROFILES_FILE = "driver_profiles.json"

# ─── dlib setup (unchanged from original) ─────────────────────────
detect  = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


# ══════════════════════════════════════════════════════════════════
# 1. EAR FUNCTION (unchanged from original)
# ══════════════════════════════════════════════════════════════════

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# ══════════════════════════════════════════════════════════════════
# 2. PROFILE MANAGER  ← NEW
#    Saves/loads per-driver calibration data to a JSON file
# ══════════════════════════════════════════════════════════════════

def load_profiles():
    """Load all driver profiles from JSON file."""
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_profile(driver_name, left_thresh, right_thresh,
                 left_baseline, right_baseline,
                 left_eye_size, right_eye_size):
    """Save calibration data for one driver into the JSON file."""
    profiles = load_profiles()
    profiles[driver_name] = {
        "left_thresh":      round(left_thresh,    4),
        "right_thresh":     round(right_thresh,   4),
        "left_baseline":    round(left_baseline,  4),
        "right_baseline":   round(right_baseline, 4),
        "left_eye_size":    left_eye_size,   # [avg_w, avg_h]
        "right_eye_size":   right_eye_size,
    }
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"[SAVED] Profile '{driver_name}' → {PROFILES_FILE}")


def load_driver_profile(driver_name):
    """Return saved profile dict for driver_name, or None if not found."""
    profiles = load_profiles()
    return profiles.get(driver_name, None)


# ══════════════════════════════════════════════════════════════════
# 3. CALIBRATION HELPER  ← NEW
#    Collects open-eye EAR samples and computes per-eye thresholds
# ══════════════════════════════════════════════════════════════════

def clamp_threshold(value):
    """Keep threshold in a safe range regardless of calibration result."""
    return float(np.clip(value, 0.18, 0.32))


def run_calibration(cap, frame_count=CALIB_FRAMES):
    """
    Show a progress bar while collecting EAR samples.
    Returns (left_thresh, right_thresh,
             left_baseline, right_baseline,
             left_eye_size, right_eye_size)
    """
    left_ears,  right_ears  = [], []
    left_sizes, right_sizes = [], []

    print("[CALIBRATION] Keep eyes open and look at the camera...")

    while True:
        ret, frame = cap.read()
        frame  = imutils.resize(frame, width=450)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        progress = min(len(left_ears), len(right_ears))
        bar_fill = int((progress / frame_count) * 300)
        pct      = int(progress / frame_count * 100)

        # ── Instruction text ──────────────────────────────────
        cv2.putText(frame, "CALIBRATION: Keep eyes OPEN",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # ── Progress bar ──────────────────────────────────────
        cv2.rectangle(frame, (10, 38), (310, 55), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 38), (10 + bar_fill, 55), (0, 200, 255), -1)
        cv2.putText(frame, f"{pct}%",
                    (315, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for subject in subjects:
            shape    = predict(gray, subject)
            shape    = face_utils.shape_to_np(shape)
            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            l_ear = eye_aspect_ratio(leftEye)
            r_ear = eye_aspect_ratio(rightEye)

            # Measure eye bounding box size
            lx, ly, lw, lh = cv2.boundingRect(leftEye)
            rx, ry, rw, rh = cv2.boundingRect(rightEye)

            # Draw eye contours in cyan during calibration
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 200, 255), 1)

            # Show live per-eye values
            cv2.putText(frame, f"Left  EAR: {l_ear:.3f}  size: {lw}x{lh}px",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 255, 150), 1)
            cv2.putText(frame, f"Right EAR: {r_ear:.3f}  size: {rw}x{rh}px",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 200, 255), 1)

            # Only collect samples while eyes are open (filter blinks)
            if l_ear > BLINK_FILTER:
                left_ears.append(l_ear)
                left_sizes.append((lw, lh))
            if r_ear > BLINK_FILTER:
                right_ears.append(r_ear)
                right_sizes.append((rw, rh))

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        # Done collecting enough samples?
        if min(len(left_ears), len(right_ears)) >= frame_count:
            break

    # ── Compute baselines and thresholds ──────────────────────────
    left_baseline  = float(np.mean(left_ears))
    right_baseline = float(np.mean(right_ears))
    left_thresh    = clamp_threshold(left_baseline  * THRESH_RATIO)
    right_thresh   = clamp_threshold(right_baseline * THRESH_RATIO)

    ls = np.array(left_sizes)
    rs = np.array(right_sizes)
    left_eye_size  = [round(float(np.mean(ls[:, 0])), 1),
                      round(float(np.mean(ls[:, 1])), 1)]
    right_eye_size = [round(float(np.mean(rs[:, 0])), 1),
                      round(float(np.mean(rs[:, 1])), 1)]

    print(f"[DONE] Left  — baseline: {left_baseline:.4f}  thresh: {left_thresh:.4f}"
          f"  eye size: {left_eye_size[0]}x{left_eye_size[1]}px")
    print(f"[DONE] Right — baseline: {right_baseline:.4f}  thresh: {right_thresh:.4f}"
          f"  eye size: {right_eye_size[0]}x{right_eye_size[1]}px")

    return (left_thresh, right_thresh,
            left_baseline, right_baseline,
            left_eye_size, right_eye_size)


# ══════════════════════════════════════════════════════════════════
# 4. MAIN DETECTION LOOP
#    Identical structure to your original, with per-eye thresholds
# ══════════════════════════════════════════════════════════════════

def main(driver_name="Driver"):

    cap  = cv2.VideoCapture(0)
    flag = 0

    # ── Try to load saved profile ──────────────────────────────────
    profile = load_driver_profile(driver_name)

    if profile:
        # Use saved values — no calibration needed
        print(f"[PROFILE] Loaded saved profile for '{driver_name}'")
        left_thresh    = profile["left_thresh"]
        right_thresh   = profile["right_thresh"]
        left_baseline  = profile["left_baseline"]
        right_baseline = profile["right_baseline"]
        left_eye_size  = profile["left_eye_size"]
        right_eye_size = profile["right_eye_size"]
    else:
        # No profile found → run calibration
        print(f"[INFO] No saved profile for '{driver_name}'. Running calibration...")
        (left_thresh, right_thresh,
         left_baseline, right_baseline,
         left_eye_size, right_eye_size) = run_calibration(cap)

    # ── Detection loop ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape    = predict(gray, subject)
            shape    = face_utils.shape_to_np(shape)
            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # ── Per-eye EAR ───────────────────────────────────────
            leftEAR  = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # ── Eye contours: green = open, red = closed ──────────
            lColor = (0, 255, 0) if leftEAR  >= left_thresh  else (0, 0, 255)
            rColor = (0, 255, 0) if rightEAR >= right_thresh else (0, 0, 255)
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, lColor, 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, rColor, 1)

            # ── HUD: per-eye values + thresholds ──────────────────
            cv2.putText(frame, f"L-EAR: {leftEAR:.3f}  thr:{left_thresh:.3f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if leftEAR >= left_thresh else (0, 0, 255), 2)

            cv2.putText(frame, f"R-EAR: {rightEAR:.3f}  thr:{right_thresh:.3f}",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if rightEAR >= right_thresh else (0, 0, 255), 2)

            cv2.putText(frame, f"Driver: {driver_name}",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            cv2.putText(frame,
                        f"EyeL:{left_eye_size[0]:.0f}x{left_eye_size[1]:.0f}  "
                        f"EyeR:{right_eye_size[0]:.0f}x{right_eye_size[1]:.0f} px",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

            # ── Drowsiness check (either eye below its threshold) ──
            # Using per-eye threshold instead of single global thresh
            if leftEAR < left_thresh or rightEAR < right_thresh:
                flag += 1
                print(flag)
                if flag >= FRAME_CHECK:
                    cv2.putText(frame, "*** ALERT! ***",
                                (120, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if not mixer.music.get_busy():   # prevent restarter every frame
                        mixer.music.play()
            else:
                flag = 0
                mixer.music.stop()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # ── [S] Save current calibration as a profile ─────────────
        elif key == ord("s"):
            save_profile(driver_name,
                         left_thresh, right_thresh,
                         left_baseline, right_baseline,
                         left_eye_size, right_eye_size)

        # ── [R] Recalibrate (new driver or lighting changed) ───────
        elif key == ord("r"):
            print("[RECALIB] Starting recalibration...")
            (left_thresh, right_thresh,
             left_baseline, right_baseline,
             left_eye_size, right_eye_size) = run_calibration(cap)
            flag = 0
            mixer.music.stop()

    cv2.destroyAllWindows()
    cap.release()


# ─── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Get driver name from command line if provided
    # Usage: python drowsiness_with_profiles.py Alice
    if len(sys.argv) > 1:
        driver = sys.argv[1]
    else:
        driver = "Driver"   # default name

    print(f"Starting for driver: {driver}")
    main(driver_name=driver)
