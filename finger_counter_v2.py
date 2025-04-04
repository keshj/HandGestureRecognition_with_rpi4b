import cv2
import mediapipe as mp
from flask import Flask, Response
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hand Tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# DroidCam URL
ip_addr = input("Enter the IP address: ")
url = f"http://{ip_addr}:4747/video"
cap = cv2.VideoCapture(url)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Hand tracking model with optimized parameters
hands = mp_hands.Hands(static_image_mode=False, 
                       min_detection_confidence=0.6, 
                       min_tracking_confidence=0.6, 
                       max_num_hands=1)

# Indices of finger tip landmarks
tip_ids = [4, 8, 12, 16, 20]  # Include thumb (4)
# Indices of finger pip (middle) joints
pip_ids = [2, 6, 10, 14, 18]  # Include thumb IP joint (2)

def generate_frames():
    """Generator function to process frames and stream them."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Convert image to RGB only once
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions
                landmark_positions = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmark_positions.append((idx, x, y))
                
                # Count fingers only if we have enough landmarks
                if len(landmark_positions) >= 21:  # MediaPipe hand has 21 landmarks
                    fingers_up = []
                    
                    # Special case for thumb: compare with wrist point horizontally
                    # Check if thumb is left or right of wrist (depends on hand orientation)
                    wrist_x = landmark_positions[0][1]
                    thumb_tip_x = landmark_positions[tip_ids[0]][1]
                    thumb_mcp_x = landmark_positions[5][1]  # MCP of index finger
                    
                    # Determine if it's left or right hand based on landmark positions
                    is_right_hand = thumb_mcp_x < wrist_x
                    
                    # Thumb is up if tip is to the left of IP for right hand,
                    # or to the right of IP for left hand
                    thumb_ip_x = landmark_positions[pip_ids[0]][1]
                    if (is_right_hand and thumb_tip_x < thumb_ip_x) or \
                       (not is_right_hand and thumb_tip_x > thumb_ip_x):
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)
                    
                    # For other fingers, check if tip is above pip (vertically)
                    for i in range(1, 5):  # Index, middle, ring, pinky
                        if landmark_positions[tip_ids[i]][2] < landmark_positions[pip_ids[i]][2]:
                            fingers_up.append(1)  # Finger is up
                        else:
                            fingers_up.append(0)  # Finger is down
                    
                    # Count fingers
                    up_count = sum(fingers_up)
                    down_count = 5 - up_count
                    
                    # Print finger counts
                    print(f'Fingers Up: {up_count}, Fingers Down: {down_count}')
                    
                    # Display finger count on frame
                    cv2.putText(frame, f'Up: {up_count}, Down: {down_count}', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Flask route to serve video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Serve a simple HTML page with the video stream."""
    return """
    <html>
    <head>
        <title>Hand Gesture Detection</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            h1 { color: #333; }
            .video-container { margin: 20px auto; }
        </style>
    </head>
    <body>
        <h1>Hand Gesture Detection</h1>
        <div class="video-container">
            <img src="/video_feed" width="640" height="480">
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        # Release resources when the application is closed
        cap.release()