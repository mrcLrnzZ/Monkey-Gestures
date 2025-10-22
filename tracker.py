import cv2
import mediapipe as mp
import numpy as np

neutral_img = cv2.imread("neutral.jpg")
thinking_img = cv2.imread("thinking.jpg")
thumbs_up_img = cv2.imread("thumbs_up.jpg")
pointing_img = cv2.imread("pointing.jpg")
shocked_img = cv2.imread("shocked.jpg")

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks, face_boxes, frame_w, frame_h):
    lm = hand_landmarks.landmark

    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

    index_x, index_y = int(index_tip.x * frame_w), int(index_tip.y * frame_h)

    if thumb_tip.y < thumb_ip.y and all([
        index_tip.y > index_mcp.y,
        middle_tip.y > index_mcp.y,
        ring_tip.y > index_mcp.y,
        pinky_tip.y > index_mcp.y
    ]):
        return "thumbs_up"

    if all([
        index_tip.y < index_mcp.y,
        middle_tip.y < index_mcp.y,
        ring_tip.y < index_mcp.y,
        pinky_tip.y < index_mcp.y
    ]):
        return "shocked"

    for (x1, y1, x2, y2) in face_boxes:
        if (x1 - 30) < index_x < (x2 + 30) and (y1 - 30) < index_y < (y2 + 30):
            return "thinking"

    if index_tip.y < index_mcp.y and all([
        middle_tip.y > index_mcp.y,
        ring_tip.y > index_mcp.y,
        pinky_tip.y > index_mcp.y
    ]):
        return "pointing"

    return "neutral"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    results_hands = hands.process(rgb)
    results_face = face_detection.process(rgb)

    face_boxes = []

    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * frame_w)
            y1 = int(bboxC.ymin * frame_h)
            w = int(bboxC.width * frame_w)
            h = int(bboxC.height * frame_h)
            face_boxes.append((x1, y1, x1 + w, y1 + h))
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

    gesture = "neutral"

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks, face_boxes, frame_w, frame_h)

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
    #cv2.putText(frame, f"Gesture: {gesture}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Reaction", img_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
