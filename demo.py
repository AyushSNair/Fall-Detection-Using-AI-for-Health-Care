from keras.models import load_model
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os
import platform
import torch  # For YOLOv5
import requests
import datetime
from twilio.rest import Client  # For WhatsApp integration
from flask import Flask, send_from_directory  # For serving the video file
import threading  # For threading and Timer

# Initialize Flask server
app = Flask(__name__)

# Define the folder to store recorded videos
VIDEO_FOLDER = "recorded_videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve the video file from the VIDEO_FOLDER."""
    return send_from_directory(VIDEO_FOLDER, filename)

def run_flask_server():
    """Run the Flask server in a separate thread."""
    app.run(host='0.0.0.0', port=5000, threaded=True)

# Start the Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask_server)
flask_thread.daemon = True
flask_thread.start()

# Load the fall detection model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Load YOLOv5 model for person detection
yolo_model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s', force_reload=False)

# Initialize camera
camera = cv2.VideoCapture(0)

# Global variables
fall_detected = False
running = False
confidence_threshold = 0.8  # Increase confidence threshold to reduce false positives

# Initialize VideoWriter parameters
video_writer = None
recording = False
recording_start_time = None
video_filename = None

# Twilio credentials (replace with your own)
account_sid = ''  # Replace with your Twilio Account SID
auth_token = ''    # Replace with your Twilio Auth Token
twilio_client = Client(account_sid, auth_token)

def start_recording(frame):
    global video_writer, recording, recording_start_time, video_filename
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"fall_detected_{timestamp}.mp4"
    video_path = os.path.join(VIDEO_FOLDER, video_filename)
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    recording = True
    recording_start_time = datetime.datetime.now()

def stop_recording():
    global video_writer, recording
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    recording = False

def send_email_notification():
    try:
        response = requests.post('http://127.0.0.1:5000/send-email')
        if response.status_code == 200:
            print("Email sent successfully!")
        else:
            print("Failed to send email:", response.json())
    except Exception as e:
        print("Error sending email:", str(e))

def send_whatsapp_video(video_filename):
    try:
        # Generate the URL for the video file
        video_url = f"http://localhost:5000/video/{video_filename}"
        # Send the WhatsApp message with the video URL
        message = twilio_client.messages.create(
            from_='whatsapp:+',
            body=f'Fall detected! Here is the recorded video: {video_url}',
            to='whatsapp:+'
        )
        print("WhatsApp message sent successfully! SID:", message.sid)
    except Exception as e:
        print("Error sending WhatsApp message:", str(e))

def play_alert_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('say "Alert! Fall detected"')

def start_detection():
    global running
    running = True
    update_frame()

def stop_detection():
    global running, fall_detected
    running = False
    stop_recording()
    camera.release()
    cv2.destroyAllWindows()

def update_frame():
    global fall_detected, video_writer, recording, recording_start_time, video_filename

    if running:
        ret, frame = camera.read()
        if not ret:
            root.after(10, update_frame)
            return

        # Use YOLOv5 to detect persons in the frame
        results = yolo_model(frame)
        detections = results.xywh[0]

        person_detected = False
        for *xywh, conf, cls in detections:
            if int(cls) == 0:  # Class 0 is 'person' in YOLOv5
                person_detected = True
                x1, y1, w, h = [int(i) for i in xywh]
                y1 = max(y1 - int(h * 0.4), 0)
                x1 = max(x1 - int(w * 0.2), 0)
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2)

        img_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        result_text = f"Fall Detection Result: {class_name}"
        confidence_text = f"Confidence: {confidence_score * 100:.2f}%"
        
        result_label.config(text=result_text)
        confidence_label.config(text=confidence_text)

        # Check for fall detection with higher confidence
        if "fall" in class_name.lower() and confidence_score > confidence_threshold and person_detected:
            if not fall_detected:
                play_alert_sound()
                send_email_notification()
                fall_detected = True
                start_recording(frame)  # Start recording when fall is detected
                # Start a timer to send the video after 10 seconds
                threading.Timer(10, send_whatsapp_video, args=(video_filename,)).start()
        else:
            fall_detected = False

        if recording:
            video_writer.write(frame)  # Continue recording until the camera is turned off

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    if running:
        root.after(10, update_frame)

# Initialize the GUI
root = tk.Tk()
root.title("Fall Detection System")
root.geometry("700x550")  # Increased height to accommodate the new button

video_label = Label(root)
video_label.grid(row=0, column=0, columnspan=3)  # Adjusted columnspan

result_label = Label(root, text="Fall Detection Result: N/A", font=("Arial", 16))
result_label.grid(row=1, column=0)

confidence_label = Label(root, text="Confidence: 0%", font=("Arial", 16))
confidence_label.grid(row=1, column=1)

start_button = tk.Button(root, text="Start", command=start_detection)
start_button.grid(row=2, column=0)

stop_button = tk.Button(root, text="Stop", command=stop_detection)
stop_button.grid(row=2, column=1)

# The "Send Video" button is no longer needed since it will be sent automatically
root.mainloop()