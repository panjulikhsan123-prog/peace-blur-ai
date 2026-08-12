import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)

    peace = False

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            fingers = []

            # Thumb
            fingers.append(1 if lm[4].x < lm[3].x else 0)

            # Index
            fingers.append(1 if lm[8].y < lm[6].y else 0)

            # Middle
            fingers.append(1 if lm[12].y < lm[10].y else 0)

            # Ring
            fingers.append(1 if lm[16].y < lm[14].y else 0)

            # Pinky
            fingers.append(1 if lm[20].y < lm[18].y else 0)

            if fingers == [0, 1, 1, 0, 0]:
                peace = True

    if peace:
        frame = cv2.GaussianBlur(frame, (55, 55), 0)

        cv2.putText(
            frame,
            "PEACE DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Peace Blur", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()