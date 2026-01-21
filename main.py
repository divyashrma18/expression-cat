import cv2
import time
import numpy as np
import mediapipe as mp

# --------------------
# Setup
# --------------------
cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --------------------
# Images
# --------------------
IMAGES = {
    "happy": "animals/smile.jpeg",
    "surprise": "animals/surprise.jpeg",
    "thumbs_up": "animals/thumbsup.jpeg",
    "wink": "animals/wink.jpeg"
}

locked_action = None
action_start = None
HOLD_TIME = 0.6

# --------------------
# Helpers
# --------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --------------------
# Loop
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.time()

    detected = None

    # =====================
    # 1️⃣ HAND: THUMBS UP
    # =====================
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0].landmark

        thumb_tip = hand[4]
        thumb_ip = hand[3]
        index_mcp = hand[5]

        if thumb_tip.y < thumb_ip.y < index_mcp.y:
            detected = "thumbs_up"

    # =====================
    # 2️⃣ FACE
    # =====================
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks and not detected:
        lm = face_results.multi_face_landmarks[0].landmark

        # ---- Mouth (Smile / Surprise)
        top_lip = lm[13]
        bottom_lip = lm[14]
        mouth_dist = distance(
            (top_lip.x, top_lip.y),
            (bottom_lip.x, bottom_lip.y)
        )

        # ---- Eyes (Wink)
        left_eye = distance(
            (lm[159].x, lm[159].y),
            (lm[145].x, lm[145].y)
        )
        right_eye = distance(
            (lm[386].x, lm[386].y),
            (lm[374].x, lm[374].y)
        )

        if mouth_dist > 0.05:
            detected = "surprise"
        elif mouth_dist > 0.025:
            detected = "happy"
        elif (left_eye < 0.008 and right_eye > 0.012) or (right_eye < 0.008 and left_eye > 0.012):
            detected = "wink"

    # =====================
    # 3️⃣ HOLD LOGIC
    # =====================
    if detected:
        if detected != locked_action:
            locked_action = detected
            action_start = now
        elif now - action_start >= HOLD_TIME:
            pass
    else:
        locked_action = None

    # =====================
    # UI
    # =====================
    cv2.putText(
        frame,
        f"Detected: {locked_action}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if locked_action:
        img = cv2.imread(IMAGES[locked_action])
        if img is not None:
            img = cv2.resize(img, (400, 400))
            frame[20:420, 20:420] = img

    cv2.imshow("Expression → Animal", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
