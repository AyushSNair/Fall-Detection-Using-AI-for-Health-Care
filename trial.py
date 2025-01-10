from keras.models import load_model
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os
import platform
import threading
import torch  # For YOLOv5

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s', force_reload=False)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Flag to track the fall detection state and control Start/Stop functionality
fall_detected = False
running = False  # To control Start/Stop functionality

# Function to play an alert sound when a fall is detected
def play_alert_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # frequency (1000 Hz) and duration (500 ms)
    else:
        os.system('say "Alert! Fall detected"')  # MacOS uses 'say', Linux alternatives available

# Function to start the video capture and fall detection
def start_detection():
    global running
    running = True
    update_frame()

# Function to stop the video capture
def stop_detection():
    global running
    running = False
    # Clear the video and labels when stopped
    video_label.config(image='')
    result_label.config(text="Fall Detection Result: N/A")
    confidence_label.config(text="Confidence: N/A")

# Tkinter GUI setup
root = tk.Tk()
root.title("Fall Detection System")
root.geometry("700x500")

# Create a label to display the video feed
video_label = Label(root)
video_label.grid(row=0, column=0, columnspan=2)

# Create a label to display the fall detection result
result_label = Label(root, text="Fall Detection Result: N/A", font=("Arial", 16))
result_label.grid(row=1, column=0)

# Create a label to display the confidence score
confidence_label = Label(root, text="Confidence: 0%", font=("Arial", 16))
confidence_label.grid(row=1, column=1)

# Start button to start fall detection
start_button = tk.Button(root, text="Start", command=start_detection)
start_button.grid(row=2, column=0)

# Stop button to stop the detection
stop_button = tk.Button(root, text="Stop", command=stop_detection)
stop_button.grid(row=2, column=1)

# Function to capture and display frames with fall detection
def update_frame():
    global fall_detected

    if running:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            root.after(10, update_frame)  # Retry after a short delay
            return

        # Run object detection with YOLOv5 to detect persons
        results = yolo_model(frame)  # Run inference
        detections = results.xywh[0]  # Format: [x_center, y_center, width, height, confidence, class_id]

        # Process each detection
        for *xywh, conf, cls in detections:
            if int(cls) == 0:  # Only consider 'person' class (class_id 0)
                x1, y1, w, h = [int(i) for i in xywh]
                y1 = max(y1 - 20, 0)  # Adjust this based on your specific case
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the detected person region for fall detection
                person_crop = frame[y1:y1+h, x1:x1+w]

                # Resize the cropped image to match the input size of the fall detection model
                img_resized = cv2.resize(person_crop, (224, 224), interpolation=cv2.INTER_AREA)
                img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
                data = np.expand_dims(img_normalized, axis=0)  # Shape (1, 224, 224, 3)

                # Predict fall or non-fall
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

                # Update the prediction labels on the GUI
                result_text = f"Fall Detection Result: {class_name}"
                confidence_text = f"Confidence: {confidence_score * 100:.2f}%"
                
                result_label.config(text=result_text)
                confidence_label.config(text=confidence_text)

                # Check for fall detection and play alert sound only on new fall detection
                if "fall" in class_name.lower() and confidence_score > 0.6:
                    if not fall_detected:  # Only beep if fall_detected was previously False
                        play_alert_sound()
                        fall_detected = True
                else:
                    fall_detected = False  # Reset the flag if no fall is detected

        # Convert frame to display in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    # Refresh after a short delay
    if running:
        root.after(10, update_frame)

# Start the Tkinter main loop
root.mainloop()

# Release resources when closing
camera.release()
