import cv2
from flask import Flask, Response

app = Flask(__name__)
ip_addr = input("Enter the IP address: ")
url = f"http://{ip_addr}:4747/video"
cap = cv2.VideoCapture(url)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


# open http://raspberrypi.local:5000/video_feed in a browser to view the video

