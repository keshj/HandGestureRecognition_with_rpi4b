# Import required packages
import cv2
import mediapipe as mp
from flask import Flask, Response
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# DroidCam URL
url = "http://10.92.128.232:4747/video"
cap = cv2.VideoCapture(url)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Hand tracking model
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6, 
                       min_tracking_confidence=0.6, max_num_hands=1)

def generate_frames():
    """Generator function to process frames and stream them."""
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (320, 240))

        # Convert BGR to RGB for MediaPipe processing
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        elapsed_time = time.time() - start_time
        #print(f"Time taken: {elapsed_time:.2f} seconds")
        #time.sleep(max(0, 0.07 - elapsed_time)) # fps ~14
        

@app.route('/video_feed')
def video_feed():
    """Flask route to serve video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


# open http://raspberrypi.local:5000/video_feed in a browser to view the video
