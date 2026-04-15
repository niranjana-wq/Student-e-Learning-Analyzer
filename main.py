import cv2
import numpy as np
import mediapipe as mp  # Import mediapipe
import math  # Import math for calculations
from collections import deque # For temporal analysis
import sys # --- *** NEW: To check OS for sound *** ---

# --- *** NEW: Import winsound for Windows beep *** ---
IS_WINDOWS = sys.platform.startswith('win')
if IS_WINDOWS:
    import winsound
# --- *** END NEW *** ---


# --- MediaPipe Landmark Indices ---
# --- *** MOVED BLOCK: All landmarks defined before use *** ---
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291] # 6 points for solvePnP
LEFT_EYE_EAR = [160, 144, 158, 153, 33, 133] # Vertical P1,P2,P3,P4; Horizontal P5,P6
RIGHT_EYE_EAR = [387, 373, 385, 380, 362, 263] # Vertical P1,P2,P3,P4; Horizontal P5,P6
LEFT_EYE_GAZE = [33, 133] # [left_corner, right_corner]
LEFT_PUPIL = 473
RIGHT_EYE_GAZE = [362, 263] # [left_corner, right_corner]
RIGHT_PUPIL = 468
MOUTH_MAR = [13, 14, 61, 291] # Vertical P1,P2; Horizontal P3,P4
LEFT_EYE_TOP = 159
LEFT_EYEBROW = 105
RIGHT_EYE_TOP = 386
RIGHT_EYEBROW = 334
NOSE_BRIDGE = 6
CHIN = 199 # Using landmark 199 for chin (matches pose)
# --- *** END MOVED BLOCK *** ---


# --- *** MODIFIED: More Sensitive Threshold Constants *** ---
YAWN_THRESHOLD = 0.40 # Lowered: Easier to trigger
EYEBROW_THRESHOLD = 0.09 # Lowered: Easier to trigger
BLINK_THRESHOLD = 0.37 # Raised: Easier to trigger
GAZE_THRESHOLD_LEFT = 0.8 # Narrowed "CENTER" window
GAZE_THRESHOLD_RIGHT = 1.12 # Narrowed "CENTER" window
HISTORY_LEN = 20 # How many frames of history to keep
NOD_THRESHOLD = 7.0 # Lowered: Easier to trigger
SHAKE_THRESHOLD = 10.0 # Lowered: Easier to trigger
MOVEMENT_COOLDOWN = 20 # Frames to wait after a move
INATTENTIVE_THRESHOLD = 45 # Lowered: Beeps faster (~1 second)
BEEP_COOLDOWN = 45 # Wait 5 seconds before beeping again
PITCH_THRESHOLD = -20 # Threshold for looking down
# --- End Thresholds ---


# --- Helper Functions ---
# (No changes to helper functions: calculate_ear, get_gaze_ratio, 
# calculate_mar, calculate_eyebrow_ratio, detect_nod, detect_shake)

def calculate_ear(eye_landmarks, landmarks_2d):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    # Get 2D coordinates for vertical landmarks
    p1 = landmarks_2d[eye_landmarks[0]] # 160
    p2 = landmarks_2d[eye_landmarks[1]] # 144
    p3 = landmarks_2d[eye_landmarks[2]] # 158
    p4 = landmarks_2d[eye_landmarks[3]] # 153
    # Get 2D coordinates for horizontal landmarks
    p5 = landmarks_2d[eye_landmarks[4]] # 33
    p6 = landmarks_2d[eye_landmarks[5]] # 133

    # Calculate vertical distances
    v_dist1 = np.linalg.norm(p1 - p3)
    v_dist2 = np.linalg.norm(p2 - p4)
    # Calculate horizontal distance
    h_dist = np.linalg.norm(p5 - p6)
    
    # Calculate EAR
    if h_dist == 0:
        return 0
    ear = (v_dist1 + v_dist2) / (2.0 * h_dist)
    
    # --- *** NEW: Debug Print *** ---
    # print(f"v1:{v_dist1:.2f}, v2:{v_dist2:.2f}, h:{h_dist:.2f}, EAR:{ear:.2f}")
    # --- *** END NEW *** ---

    return ear


def get_gaze_ratio(eye_landmarks, landmarks_2d, pupil_landmark):
    """Calculates the gaze ratio (horizontal) for a single eye."""
    # Get 2D coordinates for eye corners
    eye_left_corner = landmarks_2d[eye_landmarks[0]]
    eye_right_corner = landmarks_2d[eye_landmarks[1]]
    # Get 2D coordinates for pupil
    pupil = landmarks_2d[pupil_landmark]

    # Calculate horizontal distances
    dist_to_left = np.linalg.norm(pupil - eye_left_corner)
    dist_to_right = np.linalg.norm(pupil - eye_right_corner)
    
    # Calculate gaze ratio
    if dist_to_left == 0:
        return 0
    ratio = dist_to_right / dist_to_left
    return ratio

def calculate_mar(mouth_landmarks, landmarks_2d):
    """Calculates the Mouth Aspect Ratio (MAR) for yawning."""
    # Get 2D coordinates for vertical landmarks (inner lip)
    p1 = landmarks_2d[mouth_landmarks[0]] # 13
    p2 = landmarks_2d[mouth_landmarks[1]] # 14
    # Get 2D coordinates for horizontal landmarks (mouth corners)
    p3 = landmarks_2d[mouth_landmarks[2]] # 61
    p4 = landmarks_2d[mouth_landmarks[3]] # 291

    # Calculate vertical distance
    v_dist = np.linalg.norm(p1 - p2)
    # Calculate horizontal distance
    h_dist = np.linalg.norm(p3 - p4)
    
    # Calculate MAR
    if h_dist == 0:
        return 0
    mar = v_dist / h_dist
    return mar

def calculate_eyebrow_ratio(landmarks_2d):
    """Calculates the normalized eyebrow raisal ratio."""
    # Get 2D coordinates
    left_eye_top = landmarks_2d[LEFT_EYE_TOP]
    left_eyebrow = landmarks_2d[LEFT_EYEBROW]
    right_eye_top = landmarks_2d[RIGHT_EYE_TOP]
    right_eyebrow = landmarks_2d[RIGHT_EYEBROW]
    nose_bridge = landmarks_2d[NOSE_BRIDGE]
    chin = landmarks_2d[CHIN]

    # Calculate vertical distances
    left_dist = np.linalg.norm(left_eye_top - left_eyebrow)
    right_dist = np.linalg.norm(right_eye_top - right_eyebrow)
    
    avg_eyebrow_dist = (left_dist + right_dist) / 2.0
    
    # Calculate normalization distance
    norm_dist = np.linalg.norm(nose_bridge - chin)

    if norm_dist == 0:
        return 0
    
    # Calculate ratio
    ratio = avg_eyebrow_dist / norm_dist
    return ratio

def detect_nod(pitch_history, min_angle, max_angle):
    """Detects a nod (down-up) motion from pitch history."""
    if len(pitch_history) < HISTORY_LEN // 2:
        return False
    
    # Check a shorter, recent window
    recent_pitches = list(pitch_history)[-10:] # Look at last 10 frames
    
    has_down = any(p < -min_angle for p in recent_pitches)
    has_up = any(p > max_angle for p in recent_pitches) # Look for return to neutral/up
    
    # Check if the total range of motion is significant
    if has_down and has_up:
        if max(recent_pitches) - min(recent_pitches) > NOD_THRESHOLD:
            return True
    return False

def detect_shake(yaw_history, min_angle, max_angle):
    """Detects a shake (left-right) motion from yaw history."""
    if len(yaw_history) < HISTORY_LEN // 2:
        return False
        
    recent_yaws = list(yaw_history)[-10:] # Look at last 10 frames
    
    has_left = any(y < -min_angle for y in recent_yaws)
    has_right = any(y > max_angle for y in recent_yaws)
    
    if has_left and has_right:
        if max(recent_yaws) - min(recent_yaws) > SHAKE_THRESHOLD:
            return True
    return False
# --- *** END HELPER FUNCTIONS *** ---


# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # We only care about one student
    refine_landmarks=True,  # This gives us landmarks for pupils and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# ------------------------------

# --- History & Counter Initialization ---
pitch_history = deque(maxlen=HISTORY_LEN)
yaw_history = deque(maxlen=HISTORY_LEN)
movement_counter = 0 # Cooldown for nod/shake
head_move_status = "---"
inattentive_counter = 0
beep_cooldown_counter = 0
# --- *** END NEW *** ---


print("Initializing Video Capture...")
# 1. Initialize Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream with index 0 (DSHOW).")
    
    # Let's try index 1 as a fallback
    print("Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open video stream with index 1 (DSHOW) either.")
        print("Please check camera permissions and drivers.")
        exit()

print("Successfully opened video stream.")

# 2. Loop to read frames
while True:
    success, frame = cap.read()
    
    if not success:
        print("Error: Can't receive frame. Exiting ...")
        break
        
    frame = cv2.flip(frame, 1)

    # Store frame shape for calculations
    img_h, img_w, img_c = frame.shape
    
    # --- *** MODIFIED: Entire processing block is now in try/except *** ---
    try:
        # --- MediaPipe Processing ---
        # 1. Convert BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Process the image and find face landmarks
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        # 3. Convert the color space from RGB to BGR
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- *** MODIFIED: Added ELSE block for no face detection *** ---
        # 4. Process landmarks and estimate pose
        if results.multi_face_landmarks:
            
            # --- Initialize all feature variables ---
            x_angle, y_angle, z_angle = 0.0, 0.0, 0.0
            avg_ear, avg_gaze_ratio, mar = 0.0, 0.0, 0.0
            avg_eb_ratio = 0.0 
            blink_status, gaze_direction, yawn_status = "---", "---", "---"
            eyebrow_status = "---" 
            pose_status = "---" 
            # --- END INIT ---

            # We'll use the first face found (max_num_faces=1)
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- Create 2D Landmark Array ---
            # Get 2D coordinates for all 478 landmarks
            landmarks_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)
                landmarks_2d.append([x, y])
            landmarks_2d = np.array(landmarks_2d, dtype=np.int32)
            
            # --- Head Pose Estimation ---
            # 1. Get 2D/3D points for solvePnP
            face_2d = [] # 2D image points
            face_3d = [] # 3D model points
            
            for idx in HEAD_POSE_LANDMARKS:
                if idx == 1: # Nose tip
                    face_3d.append([0.0, 0.0, 0.0])
                elif idx == 199: # Chin
                    face_3d.append([0.0, -330.0, -65.0])
                elif idx == 33: # Left eye left corner
                    face_3d.append([-225.0, 170.0, -135.0])
                elif idx == 263: # Right eye right corner
                    face_3d.append([225.0, 170.0, -135.0])
                elif idx == 61: # Left mouth corner
                    face_3d.append([-150.0, -150.0, -125.0])
                elif idx == 291: # Right mouth corner
                    face_3d.append([150.0, -150.0, -125.0])
                face_2d.append(landmarks_2d[idx])
            
            # Convert to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # 2. Setup Camera Matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1] ])

            # 3. Setup Distortion Coefficients
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # 4. Calculate pose with solvePnP
            (pose_success, rot_vec, trans_vec) = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_coeffs
            )
            
            if pose_success:
                pose_status = "DETECTED"
                # 5. Project 3D axes
                axis_3d = np.array([
                    [500, 0, 0],   # X-axis (Red)
                    [0, 500, 0],   # Y-axis (Green)
                    [0, 0, 500]    # Z-axis (Blue)
                ], dtype=np.float64)
                
                (axis_2d, _) = cv2.projectPoints(
                    axis_3d, rot_vec, trans_vec, cam_matrix, dist_coeffs
                )
                
                # 6. Draw the pose axes
                nose_tip = (landmarks_2d[1][0], landmarks_2d[1][1])
                p_x = (int(axis_2d[0][0][0]), int(axis_2d[0][0][1]))
                cv2.line(frame, nose_tip, p_x, (0, 0, 255), 3) # Red (X)
                p_y = (int(axis_2d[1][0][0]), int(axis_2d[1][0][1]))
                cv2.line(frame, nose_tip, p_y, (0, 255, 0), 3) # Green (Y)
                p_z = (int(axis_2d[2][0][0]), int(axis_2d[2][0][1]))
                cv2.line(frame, nose_tip, p_z, (255, 0, 0), 3) # Blue (Z)

                # 7. Get Euler angles for display
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    x_angle = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2])) # Pitch
                    y_angle = math.degrees(math.atan2(-rot_mat[2, 0], sy))           # Yaw
                    z_angle = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0])) # Roll
                else:
                    x_angle = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1])) # Pitch
                    y_angle = math.degrees(math.atan2(-rot_mat[2, 0], sy))            # Yaw
                    z_angle = 0
            else:
                pose_status = "NOT DETECTED"
            # --- *** END OF HEAD POSE *** ---
            
            
            # --- Update History & Detect Moves ---
            if pose_success:
                pitch_history.append(x_angle)
                yaw_history.append(y_angle)
            
            # Cooldown
            if movement_counter > 0:
                movement_counter -= 1
            
            if movement_counter == 0:
                head_move_status = "---" # Reset status
                # Check for nod
                if detect_nod(pitch_history, NOD_THRESHOLD, NOD_THRESHOLD / 2):
                    head_move_status = "NODDING"
                    movement_counter = MOVEMENT_COOLDOWN # Start cooldown
                    pitch_history.clear() # Clear history
                # Check for shake
                elif detect_shake(yaw_history, SHAKE_THRESHOLD, SHAKE_THRESHOLD):
                    head_move_status = "SHAKING"
                    movement_counter = MOVEMENT_COOLDOWN # Start cooldown
                    yaw_history.clear() # Clear history
            # --- *** END NEW *** ---


            # --- Eye-Gaze & Blink Detection ---
            left_ear = calculate_ear(LEFT_EYE_EAR, landmarks_2d)
            right_ear = calculate_ear(RIGHT_EYE_EAR, landmarks_2d)
            avg_ear = (left_ear + right_ear) / 2.0
            blink_status = "BLINKING" if avg_ear < BLINK_THRESHOLD else "Open"
            
            left_gaze_ratio = get_gaze_ratio(LEFT_EYE_GAZE, landmarks_2d, LEFT_PUPIL)
            right_gaze_ratio = get_gaze_ratio(RIGHT_EYE_GAZE, landmarks_2d, RIGHT_PUPIL)
            avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0

            if avg_gaze_ratio < GAZE_THRESHOLD_LEFT:
                gaze_direction = "RIGHT" # Looking right
            elif avg_gaze_ratio > GAZE_THRESHOLD_RIGHT:
                gaze_direction = "LEFT" # Looking left
            else:
                gaze_direction = "CENTER"

            # --- Yawn Detection ---
            mar = calculate_mar(MOUTH_MAR, landmarks_2d)
            yawn_status = "YAWN: YES!" if mar > YAWN_THRESHOLD else "YAWN: NO"
            
            # --- Eyebrow Raisal Detection ---
            avg_eb_ratio = calculate_eyebrow_ratio(landmarks_2d)
            eyebrow_status = "RAISED" if avg_eb_ratio > EYEBROW_THRESHOLD else "Normal"
            

            # --- *** MODIFIED: Inattentive Beep Logic (Face Detected) *** ---
            is_inattentive = False
            if yawn_status == "YAWN: YES!": is_inattentive = True
            if gaze_direction != "CENTER": is_inattentive = True
            if pose_status == "NOT DETECTED": is_inattentive = True # Pose failed
            if x_angle < PITCH_THRESHOLD: is_inattentive = True # Looking down
            if blink_status == "BLINKING": is_inattentive = True # <-- *** NEWLY ADDED ***

            if is_inattentive:
                inattentive_counter += 1
            else:
                inattentive_counter = 0 # Reset if attentive
                
            # Manage cooldown
            if beep_cooldown_counter > 0:
                beep_cooldown_counter -= 1

            # Beep if threshold crossed and not on cooldown
            if inattentive_counter > INATTENTIVE_THRESHOLD and beep_cooldown_counter == 0:
                if IS_WINDOWS:
                    # --- *** MODIFIED: Increased beep duration *** ---
                    winsound.Beep(1000, 1000) # 1000Hz for 1000ms (1 second)
                else:
                    print('\a') # Fallback for non-Windows
                inattentive_counter = 0 # Reset counter after beeping
                beep_cooldown_counter = BEEP_COOLDOWN # Start cooldown
            # --- *** END BEEP LOGIC *** ---


            # --- Draw the face mesh (optional) ---
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            
            # --- *** MODIFIED: Commented out crashing pupil/iris drawing *** ---
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES, # <-- This was the typo
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())
            # --- *** END MODIFIED *** ---
                
            # --- Consolidated Feature Display ---
            # Re-arranged to be more compact and fit on screen
            
            # Pose
            pose_color = (0, 0, 255) if pose_status != "DETECTED" else (0, 255, 0)
            cv2.putText(frame, f"Pose (YPR): {y_angle:.1f}, {x_angle:.1f}, {z_angle:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pose_color, 2)
            
            # Blink
            blink_color = (0, 0, 255) if blink_status == "BLINKING" else (0, 255, 0)
            cv2.putText(frame, f"Blink: {blink_status} (EAR: {avg_ear:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blink_color, 2)
            
            # Gaze
            gaze_color = (0, 0, 255) if gaze_direction != "CENTER" else (0, 255, 0)
            cv2.putText(frame, f"Gaze: {gaze_direction} (Ratio: {avg_gaze_ratio:.2f})", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, gaze_color, 2)
            
            # Yawn
            yawn_color = (0, 0, 255) if yawn_status == "YAWN: YES!" else (0, 255, 0)
            cv2.putText(frame, f"Yawn: {yawn_status} (MAR: {mar:.2f})", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, yawn_color, 2)
            
            # Eyebrow Display
            eb_color = (0, 0, 255) if eyebrow_status == "RAISED" else (0, 255, 0)
            cv2.putText(frame, f"Eyebrows: {eyebrow_status} (Ratio: {avg_eb_ratio:.2f})", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, eb_color, 2)

            # Head Move Display
            move_color = (0, 255, 255) # Yellow
            cv2.putText(frame, f"Head Move: {head_move_status}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, move_color, 2)
            
            # Pose Status (less important)
            cv2.putText(frame, f"Pose Status: {pose_status}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pose_color, 2)

            # Inattentive Timer Display (Positioned at bottom of frame)
            inattentive_color = (0, 0, 255) if inattentive_counter > 0 else (0, 255, 0)
            cv2.putText(frame, f"Inattentive Timer: {inattentive_counter}/{INATTENTIVE_THRESHOLD}", (20, img_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, inattentive_color, 2)
            # --- *** END MODIFIED *** ---

        # --- *** NEW: ELSE BLOCK FOR NO FACE DETECTED *** ---
        else:
            # 1. Increment inattentive counter
            inattentive_counter += 1
            
            # 2. Manage beep cooldown
            if beep_cooldown_counter > 0:
                beep_cooldown_counter -= 1

            # 3. Check if we should beep
            if inattentive_counter > INATTENTIVE_THRESHOLD and beep_cooldown_counter == 0:
                if IS_WINDOWS:
                    # --- *** MODIFIED: Increased beep duration *** ---
                    winsound.Beep(700, 800) # 700Hz for 800ms
                else:
                    print('\a')
                inattentive_counter = 0 # Reset counter
                beep_cooldown_counter = BEEP_COOLDOWN # Start cooldown

            # 4. Display Warning Message
            warn_color = (0, 0, 255) # Red
            cv2.putText(frame, "WARNING: FACE NOT DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, warn_color, 2)
            
            # 5. Display Inattentive Timer
            inattentive_color = (0, 0, 255) if inattentive_counter > 0 else (0, 255, 0)
            cv2.putText(frame, f"Inattentive Timer: {inattentive_counter}/{INATTENTIVE_THRESHOLD}", (20, img_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, inattentive_color, 2)
        # --- *** END NEW *** ---

    except Exception as e:
        # This robust try/except block keeps the app from crashing
        # If you see errors, they will print here instead of crashing
        print(f"Error processing frame: {e}")
        pass
    # --- *** END OF MODIFIED BLOCK *** ---

    cv2.imshow('Student e-Learning Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Release everything
print("Releasing resources...")
face_mesh.close() 
cap.release()
cv2.destroyAllWindows()