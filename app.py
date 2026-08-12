from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils


def generate_frames():
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        blur = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark

                fingers = []

                # Thumb
                if landmarks[4].x < landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Index, Middle, Ring, Pinky
                for tip in [8, 12, 16, 20]:
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Peace gesture
                if fingers == [0, 1, 1, 0, 0]:
                    blur = True

        if blur:
            frame = cv2.GaussianBlur(frame, (41, 41), 0)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)