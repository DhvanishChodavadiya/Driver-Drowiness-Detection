import cv2
import numpy as np
import json
import os
import sys
import time
import urllib.request
from scipy.spatial import distance
from pygame import mixer
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

# ─── Sound ────────────────────────────────────────────────────────
mixer.init()
ALARM_FILE = "music.wav"
alarm_loaded = os.path.exists(ALARM_FILE)
if alarm_loaded:
    mixer.music.load(ALARM_FILE)
else:
    print("[WARNING] 'music.wav' not found — alarm disabled.")

def play_alarm():
    if alarm_loaded and not mixer.music.get_busy():
        mixer.music.play()

def stop_alarm():
    if alarm_loaded:
        mixer.music.stop()

# ─── Constants ────────────────────────────────────────────────────
CALIB_FRAMES    = 80
THRESH_RATIO    = 0.75
FRAME_CHECK     = 20
BLINK_FILTER    = 0.16
YAW_THRESHOLD   = 30
DISTRACT_FRAMES = 120
PROFILES_FILE   = "driver_profiles.json"
MODEL_PATH      = "face_landmarker.task"

# ── Yawn constants ────────────────────────────────────────────────
MAR_THRESHOLD   = 0.6
YAWN_FRAMES     = 30        # raised to ~1 sec to reduce false positives
YAWN_COOLDOWN   = 90        # frames before another yawn alert can fire

# ── Phone detection constants ─────────────────────────────────────
PHONE_TIME_LIMIT     = 10.0        # seconds before alert triggers
PHONE_CONF_THRESHOLD = 0.45        # YOLO confidence threshold for 'cell phone'
HAND_FACE_DIST_RATIO = 0.35        # hand centroid within this fraction of frame width from face
PHONE_COOLDOWN       = 5.0         # seconds between repeated phone alerts

# ─── Eye landmark indices ─────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ─── Mouth landmark indices ───────────────────────────────────────
MOUTH_IDX = [13, 14, 78, 308, 82, 312]

POSE_IDS = [1, 152, 226, 446, 57, 287]
FACE_3D  = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0,  -63.6,  -12.5],
    [-43.3,   32.7,  -26.0],
    [ 43.3,   32.7,  -26.0],
    [-28.9,  -28.9,  -24.1],
    [ 28.9,  -28.9,  -24.1],
], dtype=np.float64)

HAND_MODEL_PATH = "hand_landmarker.task"

# ─── Download models if missing ───────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading face_landmarker.task (~5 MB)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[INFO] Face model download complete.")

def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("[INFO] Downloading hand_landmarker.task (~9 MB)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, HAND_MODEL_PATH)
        print("[INFO] Hand model download complete.")

# ─── Build face landmarker ────────────────────────────────────────
def build_landmarker():
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)

# ─── Build hand landmarker (new Tasks API) ───────────────────────
def build_hands():
    ensure_hand_model()
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)

# ─── Load YOLO model ──────────────────────────────────────────────
def build_yolo():
    """Load YOLOv8n — downloads automatically on first run (~6 MB)."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("[INFO] YOLOv8n loaded for phone detection.")
        return model
    except ImportError:
        print("[WARNING] ultralytics not installed — YOLO phone detection disabled.")
        print("          Run: pip install ultralytics")
        return None
    except Exception as e:
        print(f"[WARNING] YOLO load failed: {e} — YOLO phone detection disabled.")
        return None

# ─── Helpers ──────────────────────────────────────────────────────
def get_landmarks_px(result, w, h):
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    return {i: (int(lms[i].x * w), int(lms[i].y * h)) for i in range(len(lms))}

def ear(lms, indices):
    pts = np.array([lms[i] for i in indices], dtype=np.float64)
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mar(lms):
    pts = np.array([lms[i] for i in MOUTH_IDX], dtype=np.float64)
    A = distance.euclidean(pts[0], pts[1])
    B = distance.euclidean(pts[4], pts[5])
    C = distance.euclidean(pts[2], pts[3])
    return (A + B) / (2.0 * C)

def draw_mouth(frame, lms, color):
    pts = np.array([lms[i] for i in MOUTH_IDX], dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)

def cam_matrix(w, h):
    f = w
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)

def get_yaw(lms, K):
    pts2d = np.array([lms[i] for i in POSE_IDS], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, pts2d, K, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    return float(np.degrees(np.arctan2(-R[2,0], sy if sy > 1e-6 else 1e-6)))

def clamp_thr(v):
    return float(np.clip(v, 0.16, 0.32))

def draw_eye(frame, lms, indices, color):
    pts = np.array([lms[i] for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)

# ─── Phone detection helpers ──────────────────────────────────────
def face_center(lms, w, h):
    """Return pixel center of face using nose tip landmark."""
    nose = lms.get(1)
    if nose:
        return nose
    # fallback: average of all landmarks
    xs = [v[0] for v in lms.values()]
    ys = [v[1] for v in lms.values()]
    return (int(np.mean(xs)), int(np.mean(ys)))

def hand_near_face(hand_result, face_lms, w, h):
    """
    Returns True if any detected hand centroid is close to the face center.
    'Close' = within HAND_FACE_DIST_RATIO * frame_width pixels.
    Also returns hand centroid for drawing.
    Uses the new mediapipe.tasks HandLandmarker result format.
    """
    if not hand_result.hand_landmarks or face_lms is None:
        return False, None

    fc = face_center(face_lms, w, h)
    threshold = w * HAND_FACE_DIST_RATIO

    for hand_lms in hand_result.hand_landmarks:
        hx = int(np.mean([lm.x * w for lm in hand_lms]))
        hy = int(np.mean([lm.y * h for lm in hand_lms]))
        dist = distance.euclidean((hx, hy), fc)
        if dist < threshold:
            return True, (hx, hy)
    return False, None

def yolo_phone_detected(yolo_model, frame):
    """
    Run YOLOv8 on frame and return True if 'cell phone' (class 67) is detected
    with confidence above threshold. Also returns bounding box or None.
    """
    if yolo_model is None:
        return False, None
    results = yolo_model(frame, verbose=False)[0]
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 67 and conf >= PHONE_CONF_THRESHOLD:   # 67 = cell phone in COCO
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return True, (x1, y1, x2, y2)
    return False, None

def draw_phone_overlay(frame, hand_centroid, phone_box, elapsed, limit):
    """Draw phone detection visuals on frame."""
    # Hand-near-face indicator
    if hand_centroid:
        cv2.circle(frame, hand_centroid, 18, (0, 100, 255), 2)
        cv2.putText(frame, "Hand near face", (hand_centroid[0]+22, hand_centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

    # YOLO phone bounding box
    if phone_box:
        x1, y1, x2, y2 = phone_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "PHONE", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Timer bar — fills red as time on phone increases
    if elapsed > 0:
        bar_w = int(min(elapsed / limit, 1.0) * 300)
        color = (0, 200, 255) if elapsed < limit * 0.6 else (0, 80, 255)
        cv2.rectangle(frame, (10, 125), (310, 140), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 125), (10 + bar_w, 140), color, -1)
        cv2.putText(frame, f"Phone: {elapsed:.1f}s / {limit:.0f}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

# ─── Profiles ─────────────────────────────────────────────────────
def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}

def save_profile(name, lt, rt, lb, rb):
    p = load_profiles()
    p[name] = {"left_thresh": round(lt,4), "right_thresh": round(rt,4),
               "left_baseline": round(lb,4), "right_baseline": round(rb,4)}
    with open(PROFILES_FILE, "w") as f:
        json.dump(p, f, indent=2)
    print(f"[SAVED] Profile '{name}'")

def load_profile(name):
    return load_profiles().get(name)

# ─── Calibration ──────────────────────────────────────────────────
def calibrate(cap, landmarker, K, n=CALIB_FRAMES):
    le, re = [], []
    print("[CALIBRATION] Keep eyes OPEN, face the camera...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame  = cv2.resize(frame, (640, 480))
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        res    = landmarker.detect(mp_img)
        lms    = get_landmarks_px(res, w, h)

        prog = min(len(le), len(re))
        pct  = int(prog / n * 100)
        fill = int(pct / 100 * 400)

        cv2.putText(frame, "CALIBRATION - Keep eyes OPEN",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.rectangle(frame, (10,45), (410,62), (50,50,50), -1)
        cv2.rectangle(frame, (10,45), (10+fill,62), (0,200,255), -1)
        cv2.putText(frame, f"{pct}%", (415,59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if lms:
            l = ear(lms, LEFT_EYE_IDX)
            r = ear(lms, RIGHT_EYE_IDX)
            draw_eye(frame, lms, LEFT_EYE_IDX,  (0,255,255))
            draw_eye(frame, lms, RIGHT_EYE_IDX, (0,200,255))
            cv2.putText(frame, f"Left EAR:  {l:.3f}", (10,85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,255,150), 1)
            cv2.putText(frame, f"Right EAR: {r:.3f}", (10,108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,200,255), 1)
            if l > BLINK_FILTER: le.append(l)
            if r > BLINK_FILTER: re.append(r)

        cv2.imshow("Drowsiness Monitor", frame)
        cv2.waitKey(1)
        if min(len(le), len(re)) >= n:
            break

    lb, rb = float(np.mean(le)), float(np.mean(re))
    lt, rt = clamp_thr(lb*THRESH_RATIO), clamp_thr(rb*THRESH_RATIO)
    print(f"[DONE] Left  baseline={lb:.4f} thresh={lt:.4f}")
    print(f"       Right baseline={rb:.4f} thresh={rt:.4f}")
    return lt, rt, lb, rb

# ─── Main ─────────────────────────────────────────────────────────
def main(driver_name="Driver"):
    landmarker  = build_landmarker()
    hands_model = build_hands()
    yolo_model  = build_yolo()

    cap = None
    for cam_index in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"[INFO] Camera found at index {cam_index}")
            break
        cap.release()

    if not cap or not cap.isOpened():
        print("[ERROR] No camera found!")
        return

    for _ in range(5):
        ret, sample = cap.read()

    if not ret or sample is None:
        print("[ERROR] Camera opened but can't read frames.")
        cap.release()
        return

    sample = cv2.resize(sample, (640, 480))
    h, w = sample.shape[:2]
    K = cam_matrix(w, h)

    # ── State counters ─────────────────────────────────────────────
    flag          = 0
    distract_flag = 0
    yawn_flag     = 0
    yawn_cooldown = 0

    # ── Phone detection state ──────────────────────────────────────
    phone_start_time   = None   # when continuous phone use began
    phone_alert_shown  = False  # is alert currently on screen
    phone_last_alert   = 0.0    # timestamp of last alert (for cooldown)

    # ── Load or calibrate profile ──────────────────────────────────
    profile = load_profile(driver_name)
    if profile:
        print(f"[PROFILE] Loaded '{driver_name}'")
        lt = profile["left_thresh"]
        rt = profile["right_thresh"]
        lb = profile["left_baseline"]
        rb = profile["right_baseline"]
    else:
        print(f"[INFO] No profile for '{driver_name}' — calibrating...")
        lt, rt, lb, rb = calibrate(cap, landmarker, K)

    print("\n[RUNNING] Q=quit  S=save  R=recalibrate\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv2.resize(frame, (640, 480))
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        res    = landmarker.detect(mp_img)
        lms    = get_landmarks_px(res, w, h)

        # ── MediaPipe Hands (Tasks API) ────────────────────────────
        hand_result = hands_model.detect(mp_img)

        drowsy = distracted = yawning = False
        yaw = 0.0
        direction = "FORWARD"
        mar_val = 0.0
        now = time.time()

        if lms:
            # ── EAR ───────────────────────────────────────────────
            l_ear = ear(lms, LEFT_EYE_IDX)
            r_ear = ear(lms, RIGHT_EYE_IDX)

            lc = (0,255,0) if l_ear >= lt else (0,0,255)
            rc = (0,255,0) if r_ear >= rt else (0,0,255)
            draw_eye(frame, lms, LEFT_EYE_IDX,  lc)
            draw_eye(frame, lms, RIGHT_EYE_IDX, rc)

            if l_ear < lt or r_ear < rt:
                drowsy = True

            # ── MAR (Yawn) ────────────────────────────────────────
            mar_val = mar(lms)
            if mar_val > MAR_THRESHOLD:
                yawning = True
                draw_mouth(frame, lms, (0, 165, 255))
            else:
                draw_mouth(frame, lms, (0, 255, 0))

            # ── Head yaw ──────────────────────────────────────────
            yaw = get_yaw(lms, K)
            if   yaw >  YAW_THRESHOLD:
                direction  = "LOOKING RIGHT >>"
                distracted = True
            elif yaw < -YAW_THRESHOLD:
                direction  = "<< LOOKING LEFT"
                distracted = True

            # ── HUD ───────────────────────────────────────────────
            cv2.putText(frame, f"L-EAR:{l_ear:.3f} thr:{lt:.3f}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)
            cv2.putText(frame, f"R-EAR:{r_ear:.3f} thr:{rt:.3f}",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
            cv2.putText(frame, f"MAR:{mar_val:.3f} thr:{MAR_THRESHOLD:.2f}",
                        (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,165,255) if yawning else (0,255,0), 2)
            cv2.putText(frame, f"Yaw:{yaw:+.1f}deg  {direction}",
                        (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,0,255) if distracted else (0,220,0), 2)

        # ── Phone detection (dual method) ──────────────────────────
        hand_close, hand_centroid = hand_near_face(hand_result, lms, w, h)
        yolo_phone,  phone_box    = yolo_phone_detected(yolo_model, frame)

        # Combine signals: either method alone can trigger the timer
        phone_in_use = hand_close or yolo_phone

        if phone_in_use:
            if phone_start_time is None:
                phone_start_time = now            # start the clock
            phone_elapsed = now - phone_start_time
            draw_phone_overlay(frame, hand_centroid, phone_box, phone_elapsed, PHONE_TIME_LIMIT)

            # Trigger alert after PHONE_TIME_LIMIT seconds
            if phone_elapsed >= PHONE_TIME_LIMIT:
                phone_alert_shown = True
                if now - phone_last_alert >= PHONE_COOLDOWN:
                    phone_last_alert = now
                    play_alarm()
        else:
            # Reset as soon as phone use stops
            phone_start_time  = None
            phone_alert_shown = False

        # Phone alert overlay
        if phone_alert_shown:
            cv2.rectangle(frame, (0, 240), (640, 310), (0, 0, 0), -1)
            cv2.putText(frame, "!!! PUT DOWN YOUR PHONE !!!",
                        (55, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # ── Drowsiness counter ─────────────────────────────────────
        if drowsy:
            flag += 1
            if flag >= FRAME_CHECK:
                cv2.putText(frame, "*** DROWSY ALERT! ***",
                            (140,340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                play_alarm()
        else:
            flag = 0
            if not distracted and not yawning and not phone_alert_shown:
                stop_alarm()

        # ── Distraction counter ────────────────────────────────────
        if distracted:
            distract_flag += 1
            if distract_flag >= DISTRACT_FRAMES:
                cv2.putText(frame, f"EYES ON ROAD! {direction}",
                            (60,380), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,100,255), 3)
                play_alarm()
        else:
            distract_flag = 0
            if not drowsy and not yawning and not phone_alert_shown:
                stop_alarm()

        # ── Yawn counter ───────────────────────────────────────────
        if yawn_cooldown > 0:
            yawn_cooldown -= 1

        if yawning:
            yawn_flag += 1
            if yawn_flag >= YAWN_FRAMES and yawn_cooldown == 0:
                cv2.putText(frame, "*** YAWN DETECTED! ***",
                            (120,300), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,165,255), 3)
                play_alarm()
                yawn_cooldown = YAWN_COOLDOWN   # prevent repeated alerts
        else:
            yawn_flag = 0
            if not drowsy and not distracted and not phone_alert_shown:
                stop_alarm()

        cv2.putText(frame, "S=save  R=recalib  Q=quit",
                    (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
        cv2.imshow("Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_profile(driver_name, lt, rt, lb, rb)
        elif key == ord("r"):
            lt, rt, lb, rb = calibrate(cap, landmarker, K)
            flag = distract_flag = yawn_flag = yawn_cooldown = 0
            phone_start_time = None
            phone_alert_shown = False
            stop_alarm()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    del hands_model

# ─── Run ──────────────────────────────────────────────────────────
main(driver_name="Driver")
