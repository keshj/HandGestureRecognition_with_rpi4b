import cv2
import mediapipe as mp
from flask import Flask, Response
import keyboard
import subprocess

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hand Tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# DroidCam URL
ip_addr = input("Enter the IP address: ")
url = f"http://{ip_addr}:4747/video"
cap = cv2.VideoCapture(url)
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

# Variable to track the last gesture to prevent repeated commands
last_gesture = -1

# Windows machine SSH details
WINDOWS_USER = "windows user"
WINDOWS_IP = "windows ip addr"

def send_command_to_windows(command):
    """Send command to Windows machine via SSH."""
    ssh_command = f"ssh {WINDOWS_USER}@{WINDOWS_IP} python D:\Sem 4\rpi\HandGestureRecognition_with_rpi4b\remote_control.py {command}"
    subprocess.Popen(ssh_command, shell=True)

def generate_frames():
    """Generator function to process frames and stream them."""
    global last_gesture
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB only once
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions directly (no helper function needed)
                landmark_positions = []
                h, w, _ = frame.shape
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmark_positions.append((idx, x, y))
                
                # Count fingers
                if len(landmark_positions) >= 21:
                    fingers_up = []
                    
                    # Check thumb (special case)
                    thumb_tip = landmark_positions[4][1]  # x-coordinate of thumb tip
                    thumb_ip = landmark_positions[3][1]   # x-coordinate of thumb IP joint
                    thumb_cmc = landmark_positions[1][1]  # x-coordinate of thumb CMC joint
                    
                    # Determine left or right hand
                    is_right_hand = landmark_positions[5][1] < landmark_positions[17][1]
                    
                    # Thumb is up if tip is to the left of IP for right hand, or to the right for left hand
                    if (is_right_hand and thumb_tip < thumb_ip) or (not is_right_hand and thumb_tip > thumb_ip):
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)
                    
                    # For other fingers: check if tip is above pip (vertically)
                    for i in range(1, 5):  # Index, middle, ring, pinky
                        tip_y = landmark_positions[tip_ids[i]][2]  
                        pip_y = landmark_positions[pip_ids[i]][2]
                        if tip_y < pip_y:  # Finger is up if tip is above pip
                            fingers_up.append(1)
                        else:
                            fingers_up.append(0)
                    
                    # Count fingers
                    up_count = sum(fingers_up)
                    down_count = 5 - up_count
                    
                    # Display finger counts on frame
                    cv2.putText(frame, f'Up: {up_count}, Down: {down_count}', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    print(f'Fingers Up: {up_count}, Fingers Down: {down_count}')
                    
                    # Only execute command if gesture changed (prevents repeated commands)
                    if up_count != last_gesture:
                        last_gesture = up_count
                        # Execute commands based on finger count
                        if up_count == 4:
                            # Trigger Ctrl+Alt+N on Windows via SSH
                            send_command_to_windows("next")
                            cv2.putText(frame, "Command: Ctrl+Alt+n", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        elif up_count == 3:
                            # Trigger Ctrl+Alt+B on Windows via SSH
                            send_command_to_windows("back")
                            cv2.putText(frame, "Command: Ctrl+Alt+b", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        elif up_count == 2:
                            subprocess.Popen(["amixer", "sset", "Master", "100%"])
                            cv2.putText(frame, "Command: Volume 100%", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        elif up_count == 1:
                            subprocess.Popen(["amixer", "sset", "Master", "0%"])
                            cv2.putText(frame, "Command: Volume 0%", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    """Serve a simple HTML page with the video stream and command info."""
    return """
    <html>
    <head>
        <title>Gesture Control System</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #333; }
            .video-container { margin: 20px auto; }
            .commands { width: 80%; margin: 20px auto; text-align: left; }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Hand Gesture Control System</h1>
        <div class="video-container">
            <img src="/video_feed" width="640" height="480">
        </div>
        <div class="commands">
            <h2>Available Commands:</h2>
            <table>
                <tr>
                    <th>Gesture</th>
                    <th>Command</th>
                </tr>
                <tr>
                    <td>4 Fingers Up</td>
                    <td>Ctrl+Alt+n</td>
                </tr>
                <tr>
                    <td>3 Fingers Up</td>
                    <td>Ctrl+Alt+b</td>
                </tr>
                <tr>
                    <td>2 Fingers Up</td>
                    <td>Volume 100%</td>
                </tr>
                <tr>
                    <td>1 Finger Up</td>
                    <td>Volume 0%</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        # Release resources when application exits
        cap.release()
