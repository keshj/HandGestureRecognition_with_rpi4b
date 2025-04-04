import cv2
import mediapipe as mp
from flask import Flask, Response
from collections import Counter
from helper_module import findnameoflandmark, findpostion

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hand Tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# DroidCam URL
url = "http://10.164.97.232:4747/video"
cap = cv2.VideoCapture(url)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Hand tracking model
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6, 
                       min_tracking_confidence=0.6, max_num_hands=1)

tip_ids = [8, 12, 16, 20]

def generate_frames():
    """Generator function to process frames and stream them."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        fingers = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                a = findpostion(frame)
                b = findnameoflandmark(frame)
                
                if len(a) != 0 and len(b) != 0:
                    finger = [1 if a[0][1] < a[4][1] else 0]  # Thumb detection
                    fingers = [1 if a[tip][2] < a[tip - 2][2] else 0 for tip in tip_ids]
                    
                    x = fingers + finger
                    c = Counter(x)
                    up, down = c[1], c[0]
                    print(f'Fingers Up: {up}, Fingers Down: {down}')
        
        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Flask route to serve video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

# Open http://raspberrypi.local:5000/video_feed in a browser to view the video
