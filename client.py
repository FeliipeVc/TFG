import cv2
import requests
import numpy as np

# Open the video file
video_path = 'Sam.mp4'
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()

# Start a session with the Flask app
session = requests.Session()

# Iterate through the frames
while ret:
    # Convert the frame to bytes
    _, img_encoded = cv2.imencode('.jpg', frame)
    frame_bytes = img_encoded.tobytes()

    # Send the frame to the Flask app for processing
    response = session.post('http://localhost:5000/video_feed_client', files={'frame': frame_bytes})

    # Read the next frame
    ret, frame = cap.read()

# Release the video file
cap.release()
