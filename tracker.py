import cv2
import mediapipe as mp
import numpy as np

neutral_img = cv2.imread("neutral.jpg")
thinking_img = cv2.imread("thinking.jpg")
thumbs_up_img = cv2.imread("thumbs_up.jpg")
pointing_img = cv2.imread("pointing.jpg")
shocked_img = cv2.imread("shocked.jpg")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    wrist = lm[mp_hands.HandLandmark.WRIST]

    thumb_pt = np.array([thumb_tip.x, thumb_tip.y])
    index_pt = np.array([index_tip.x, index_tip.y])
    distance = np.linalg.norm(thumb_pt - index_pt)

    # Thumbs up 
    if thumb_tip.y < thumb_ip.y and all([
        index_tip.y > index_mcp.y,
        middle_tip.y > index_mcp.y,
        ring_tip.y > index_mcp.y,
        pinky_tip.y > index_mcp.y
    ]):
        return "thumbs_up"

    if index_tip.y < index_mcp.y and all([
        middle_tip.y > index_mcp.y,
        ring_tip.y > index_mcp.y,
        pinky_tip.y > index_mcp.y
    ]):
        vertical_diff = wrist.y - index_tip.y

        # If finger is high above wrist → pointing
        if vertical_diff > 0.25:
            return "pointing"
        # If finger is near mouth/chin level → thinking
        elif vertical_diff < 0.25:
            return "thinking"

    # Shocked 
    if all([
        index_tip.y < index_mcp.y,
        middle_tip.y < index_mcp.y,
        ring_tip.y < index_mcp.y,
        pinky_tip.y < index_mcp.y
    ]):
        return "shocked"

    return "neutral"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "neutral"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)

    if gesture == "thumbs_up":
        img = thumbs_up_img
    elif gesture == "thinking":
        img = thinking_img
    elif gesture == "pointing":
        img = pointing_img
    elif gesture == "shocked":
        img = shocked_img
    else:
        img = neutral_img

    img_resized = cv2.resize(img, (400, 400))
    cv2.imshow("Webcam", frame)
    cv2.imshow("Reaction", img_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
