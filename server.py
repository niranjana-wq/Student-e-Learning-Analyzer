"""
server.py – Flask backend for the Student Attentiveness Analyzer Dashboard.
Runs the ML analyzer in a background thread and exposes:
  GET /                → serves the dashboard HTML
  GET /video_feed      → MJPEG live video stream
  GET /data_stream     → Server-Sent Events with live JSON metrics
  POST /control        → {"action": "start"|"stop"}
  GET /session_report  → current session summary JSON
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import sys
import os
import threading
import time
import json
import queue
from collections import deque
from typing import Optional
from flask import Flask, Response, send_from_directory, jsonify, request
from flask_cors import CORS

# ── Try to load TF + joblib (graceful degradation if not present) ──────────
try:
    import tensorflow as tf
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] TensorFlow or joblib not found – running without ML model.")

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    try:
        import winsound
    except ImportError:
        IS_WINDOWS = False

# ── MediaPipe Landmark Indices ────────────────────────────────────────────
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]
LEFT_EYE_EAR  = [160, 144, 158, 153, 33, 133]
RIGHT_EYE_EAR = [387, 373, 385, 380, 362, 263]
LEFT_EYE_GAZE  = [33, 133]
LEFT_PUPIL     = 473
RIGHT_EYE_GAZE = [362, 263]
RIGHT_PUPIL    = 468
MOUTH_MAR      = [13, 14, 61, 291]
LEFT_EYE_TOP   = 159
LEFT_EYEBROW   = 105
RIGHT_EYE_TOP  = 386
RIGHT_EYEBROW  = 334
NOSE_BRIDGE    = 6
CHIN           = 199

# ── Thresholds ────────────────────────────────────────────────────────────
YAWN_THRESHOLD       = 0.45
EYEBROW_THRESHOLD    = 0.11
BLINK_THRESHOLD      = 0.22
GAZE_THRESHOLD_LEFT  = 0.7
GAZE_THRESHOLD_RIGHT = 1.3
HISTORY_LEN          = 20
NOD_THRESHOLD        = 9.0
SHAKE_THRESHOLD      = 13.0
MOVEMENT_COOLDOWN    = 30
INATTENTIVE_THRESHOLD = 45
BEEP_COOLDOWN        = 45
PITCH_THRESHOLD      = -20
MODEL_FILE  = "attentiveness_model.keras"
SCALER_FILE = "scaler.joblib"
TIME_STEPS  = 30
N_FEATURES  = 7

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────
class AnalyzerState:
    def __init__(self):
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[bytes] = None
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=60)
        self.metrics: dict = {
            "status": "INITIALIZING",
            "confidence": 0.0,
            "ear": 0.0,
            "mar": 0.0,
            "gaze_ratio": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "blink": "---",
            "gaze_dir": "---",
            "yawn": "---",
            "eyebrow": "---",
            "head_move": "---",
            "face_detected": False,
            "inattentive_counter": 0,
            "inattentive_threshold": INATTENTIVE_THRESHOLD,
        }
        # Session stats
        self.session_start: Optional[float] = None
        self.total_frames   = 0
        self.attentive_frames = 0
        self.alert_count    = 0
        self.metrics_lock   = threading.Lock()

state = AnalyzerState()

# ── Helper functions ──────────────────────────────────────────────────────
def calculate_ear(eye_landmarks, lm2d):
    p1 = lm2d[eye_landmarks[0]]; p2 = lm2d[eye_landmarks[1]]
    p3 = lm2d[eye_landmarks[2]]; p4 = lm2d[eye_landmarks[3]]
    p5 = lm2d[eye_landmarks[4]]; p6 = lm2d[eye_landmarks[5]]
    v1 = np.linalg.norm(p1 - p3); v2 = np.linalg.norm(p2 - p4)
    h  = np.linalg.norm(p5 - p6)
    return (v1 + v2) / (2.0 * h) if h else 0.0

def get_gaze_ratio(eye_landmarks, lm2d, pupil_lm):
    lc = lm2d[eye_landmarks[0]]; rc = lm2d[eye_landmarks[1]]
    p  = lm2d[pupil_lm]
    dl = np.linalg.norm(p - lc); dr = np.linalg.norm(p - rc)
    return dr / dl if dl else 0.0

def calculate_mar(mouth_landmarks, lm2d):
    p1 = lm2d[mouth_landmarks[0]]; p2 = lm2d[mouth_landmarks[1]]
    p3 = lm2d[mouth_landmarks[2]]; p4 = lm2d[mouth_landmarks[3]]
    v = np.linalg.norm(p1 - p2); h = np.linalg.norm(p3 - p4)
    return v / h if h else 0.0

def calculate_eyebrow_ratio(lm2d):
    let_top = lm2d[LEFT_EYE_TOP];  let_brow = lm2d[LEFT_EYEBROW]
    ret_top = lm2d[RIGHT_EYE_TOP]; ret_brow = lm2d[RIGHT_EYEBROW]
    nb = lm2d[NOSE_BRIDGE];        ch = lm2d[CHIN]
    ld = np.linalg.norm(let_top - let_brow)
    rd = np.linalg.norm(ret_top - ret_brow)
    nd = np.linalg.norm(nb - ch)
    return ((ld + rd) / 2.0) / nd if nd else 0.0

def detect_nod(pitch_hist):
    if len(pitch_hist) < HISTORY_LEN // 2: return False
    rec = list(pitch_hist)[-10:]
    return (any(p < -NOD_THRESHOLD for p in rec) and
            any(p >  NOD_THRESHOLD / 2 for p in rec) and
            (max(rec) - min(rec)) > NOD_THRESHOLD)

def detect_shake(yaw_hist):
    if len(yaw_hist) < HISTORY_LEN // 2: return False
    rec = list(yaw_hist)[-10:]
    return (any(y < -SHAKE_THRESHOLD for y in rec) and
            any(y >  SHAKE_THRESHOLD for y in rec) and
            (max(rec) - min(rec)) > SHAKE_THRESHOLD)

# ── Main analyzer thread ──────────────────────────────────────────────────
def analyzer_thread():
    global state

    # Load ML model
    model = scaler = None
    if ML_AVAILABLE and os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model  = tf.keras.models.load_model(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            print("[INFO] Model and scaler loaded.")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")

    # MediaPipe
    mp_face_mesh      = mp.solutions.face_mesh
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        state.running = False
        return

    # Per-session state
    pitch_history   = deque(maxlen=HISTORY_LEN)
    yaw_history     = deque(maxlen=HISTORY_LEN)
    feature_history = deque(maxlen=TIME_STEPS)
    movement_counter     = 0
    head_move_status     = "---"
    inattentive_counter  = 0
    beep_cooldown_counter = 0
    model_prediction  = "---"
    model_confidence  = 0.0

    state.session_start    = time.time()
    state.total_frames     = 0
    state.attentive_frames = 0
    state.alert_count      = 0

    while state.running:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        state.total_frames += 1

        face_detected = False
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Defaults
            x_angle = y_angle = z_angle = 0.0
            avg_ear = avg_gaze_ratio = mar = avg_eb_ratio = 0.0
            blink_status = gaze_direction = yawn_status = eyebrow_status = "---"

            if results.multi_face_landmarks:
                face_detected = True
                fl = results.multi_face_landmarks[0]

                lm2d = np.array([[int(l.x * img_w), int(l.y * img_h)]
                                  for l in fl.landmark], dtype=np.int32)

                # Head pose
                face_2d, face_3d = [], []
                _pts3d = {1: [0,0,0], 199: [0,-330,-65], 33: [-225,170,-135],
                          263: [225,170,-135], 61: [-150,-150,-125], 291: [150,-150,-125]}
                for idx in HEAD_POSE_LANDMARKS:
                    face_3d.append(_pts3d[idx]); face_2d.append(lm2d[idx])
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                fl_cam  = 1 * img_w
                cam_m   = np.array([[fl_cam, 0, img_w/2],[0, fl_cam, img_h/2],[0,0,1]])
                dc      = np.zeros((4,1), dtype=np.float64)
                ok_pose, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_m, dc)
                if ok_pose:
                    rm, _ = cv2.Rodrigues(rvec)
                    sy = math.sqrt(rm[0,0]**2 + rm[1,0]**2)
                    if sy > 1e-6:
                        x_angle = math.degrees(math.atan2(rm[2,1], rm[2,2]))
                        y_angle = math.degrees(math.atan2(-rm[2,0], sy))
                        z_angle = math.degrees(math.atan2(rm[1,0], rm[0,0]))
                    else:
                        x_angle = math.degrees(math.atan2(-rm[1,2], rm[1,1]))
                        y_angle = math.degrees(math.atan2(-rm[2,0], sy))
                    pitch_history.append(x_angle); yaw_history.append(y_angle)
                    # Draw axes
                    ax3 = np.array([[300,0,0],[0,300,0],[0,0,300]], dtype=np.float64)
                    ax2, _ = cv2.projectPoints(ax3, rvec, tvec, cam_m, dc)
                    nt = (lm2d[1][0], lm2d[1][1])
                    cv2.line(frame, nt, (int(ax2[0][0][0]), int(ax2[0][0][1])), (0,0,220), 2)
                    cv2.line(frame, nt, (int(ax2[1][0][0]), int(ax2[1][0][1])), (0,220,0), 2)
                    cv2.line(frame, nt, (int(ax2[2][0][0]), int(ax2[2][0][1])), (220,0,0), 2)

                # Head move detection
                if movement_counter > 0: movement_counter -= 1
                if movement_counter == 0:
                    head_move_status = "---"
                    if detect_nod(pitch_history):
                        head_move_status = "NODDING"; movement_counter = MOVEMENT_COOLDOWN
                        pitch_history.clear()
                    elif detect_shake(yaw_history):
                        head_move_status = "SHAKING"; movement_counter = MOVEMENT_COOLDOWN
                        yaw_history.clear()

                # EAR / blink
                avg_ear = (calculate_ear(LEFT_EYE_EAR, lm2d) +
                           calculate_ear(RIGHT_EYE_EAR, lm2d)) / 2.0
                blink_status = "BLINKING" if avg_ear < BLINK_THRESHOLD else "Open"

                # Gaze
                lg = get_gaze_ratio(LEFT_EYE_GAZE, lm2d, LEFT_PUPIL)
                rg = get_gaze_ratio(RIGHT_EYE_GAZE, lm2d, RIGHT_PUPIL)
                avg_gaze_ratio = (lg + rg) / 2.0
                if avg_gaze_ratio < GAZE_THRESHOLD_LEFT:   gaze_direction = "RIGHT"
                elif avg_gaze_ratio > GAZE_THRESHOLD_RIGHT: gaze_direction = "LEFT"
                else:                                        gaze_direction = "CENTER"

                # MAR / yawn
                mar = calculate_mar(MOUTH_MAR, lm2d)
                yawn_status = "YES" if mar > YAWN_THRESHOLD else "NO"

                # Eyebrow
                avg_eb_ratio  = calculate_eyebrow_ratio(lm2d)
                eyebrow_status = "RAISED" if avg_eb_ratio > EYEBROW_THRESHOLD else "Normal"

                # ML prediction
                if model and scaler:
                    feature_history.append([y_angle, x_angle, z_angle,
                                            avg_ear, mar, avg_eb_ratio, avg_gaze_ratio])
                    if len(feature_history) == TIME_STEPS:
                        fs = scaler.transform(np.array(list(feature_history)))
                        fr = np.reshape(fs, (1, TIME_STEPS, N_FEATURES))
                        prob = model.predict(fr, verbose=0)[0][0]
                        if prob > 0.5:
                            model_prediction = "ATTENTIVE"
                            model_confidence = float(prob * 100)
                        else:
                            model_prediction = "INATTENTIVE"
                            model_confidence = float((1 - prob) * 100)
                else:
                    # Heuristic fallback
                    is_bad = (gaze_direction != "CENTER" or
                              yawn_status == "YES" or
                              x_angle < PITCH_THRESHOLD)
                    model_prediction = "INATTENTIVE" if is_bad else "ATTENTIVE"
                    model_confidence = 75.0

                # Inattentive counter
                if model_prediction == "INATTENTIVE":
                    inattentive_counter += 1
                else:
                    inattentive_counter = 0
                    state.attentive_frames += 1

                if beep_cooldown_counter > 0: beep_cooldown_counter -= 1
                if inattentive_counter > INATTENTIVE_THRESHOLD and beep_cooldown_counter == 0:
                    if IS_WINDOWS:
                        threading.Thread(target=lambda: winsound.Beep(1000, 600),
                                         daemon=True).start()
                    inattentive_counter = 0
                    beep_cooldown_counter = BEEP_COOLDOWN
                    state.alert_count += 1

                # Draw mesh
                mp_drawing.draw_landmarks(
                    image=frame, landmark_list=fl,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Status overlay
                s_color = (34, 197, 94) if model_prediction == "ATTENTIVE" else (239, 68, 68)
                if model_prediction == "---": s_color = (234, 179, 8)
                cv2.rectangle(frame, (0, 0), (img_w, 72), (10, 10, 20), -1)
                cv2.putText(frame, f"STATUS: {model_prediction}",
                            (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, s_color, 2)
                if model_prediction != "---":
                    cv2.putText(frame, f"{model_confidence:.1f}%",
                                (img_w - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, s_color, 2)

            else:
                # No face
                model_prediction = "NO FACE"
                model_confidence = 0.0
                inattentive_counter += 1
                if beep_cooldown_counter > 0: beep_cooldown_counter -= 1
                if inattentive_counter > INATTENTIVE_THRESHOLD and beep_cooldown_counter == 0:
                    if IS_WINDOWS:
                        threading.Thread(target=lambda: winsound.Beep(700, 500),
                                         daemon=True).start()
                    inattentive_counter = 0
                    beep_cooldown_counter = BEEP_COOLDOWN
                    state.alert_count += 1
                cv2.rectangle(frame, (0, 0), (img_w, 72), (10, 10, 20), -1)
                cv2.putText(frame, "NO FACE DETECTED",
                            (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (239, 68, 68), 2)

        except Exception as e:
            print(f"[WARN] Frame error: {e}")

        # Encode JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with state.frame_lock:
            state.latest_frame = buf.tobytes()

        # Push metrics
        m = {
            "status":               model_prediction,
            "confidence":           round(model_confidence, 1),
            "ear":                  round(float(avg_ear), 3),
            "mar":                  round(float(mar), 3),
            "eb_ratio":             round(float(avg_eb_ratio), 3),
            "gaze_ratio":           round(float(avg_gaze_ratio), 3),
            "yaw":                  round(float(y_angle), 1),
            "pitch":                round(float(x_angle), 1),
            "roll":                 round(float(z_angle), 1),
            "blink":                blink_status,
            "gaze_dir":             gaze_direction,
            "yawn":                 yawn_status,
            "eyebrow":              eyebrow_status,
            "head_move":            head_move_status,
            "face_detected":        face_detected,
            "inattentive_counter":  inattentive_counter,
            "inattentive_threshold": INATTENTIVE_THRESHOLD,
            "session_elapsed":      round(time.time() - state.session_start, 1) if state.session_start else 0,
            "attentive_pct":        round(state.attentive_frames / max(state.total_frames, 1) * 100, 1),
            "alert_count":          state.alert_count,
            "total_frames":         state.total_frames,
            "ts":                   time.time(),
        }
        with state.metrics_lock:
            state.metrics = m
        try:
            state.metrics_queue.put_nowait(m)
        except queue.Full:
            try: state.metrics_queue.get_nowait()
            except: pass
            try: state.metrics_queue.put_nowait(m)
            except: pass

    cap.release()
    face_mesh.close()
    state.running = False
    print("[INFO] Analyzer stopped.")

# ── Flask routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory("static", p)

def gen_frames():
    while True:
        with state.frame_lock:
            frame = state.latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)   # ~30 fps ceiling

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data_stream")
def data_stream():
    def event_stream():
        last_ts = 0.0
        while True:
            with state.metrics_lock:
                m = dict(state.metrics)
            if m.get("ts", 0) != last_ts:
                last_ts = m.get("ts", 0)
                yield f"data: {json.dumps(m)}\n\n"
            time.sleep(0.1)
    return Response(event_stream(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

@app.route("/control", methods=["POST"])
def control():
    action = request.json.get("action", "")
    if action == "start" and not state.running:
        state.running = True
        state.metrics["status"] = "INITIALIZING"
        t = threading.Thread(target=analyzer_thread, daemon=True)
        t.start()
        return jsonify({"ok": True, "state": "started"})
    elif action == "stop" and state.running:
        state.running = False
        with state.frame_lock:
            state.latest_frame = None
        return jsonify({"ok": True, "state": "stopped"})
    return jsonify({"ok": False, "state": "no_change"})

@app.route("/session_report")
def session_report():
    elapsed = round(time.time() - state.session_start, 1) if state.session_start else 0
    with state.metrics_lock:
        m = dict(state.metrics)
    return jsonify({
        "elapsed":        elapsed,
        "total_frames":   state.total_frames,
        "attentive_pct":  m.get("attentive_pct", 0),
        "alert_count":    state.alert_count,
        "running":        state.running,
    })

if __name__ == "__main__":
    print("═" * 55)
    print("  Student Attentiveness Analyzer  –  Web Dashboard")
    print("  Open  http://127.0.0.1:5000  in your browser")
    print("═" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
