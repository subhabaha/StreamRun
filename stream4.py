import streamlit as st
import cv2
import numpy as np
import torch
from datetime import datetime
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to get video stream
def get_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video file: {video_path}")
        print(f"Error: Unable to open video file {video_path}")
    return cap

# Function to detect objects
def detect_objects(frame):
    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects
    results = model(img)
    objects = results.pandas().xyxy[0].to_dict(orient="records")

    # Draw bounding boxes and labels on the frame
    for obj in objects:
        label = obj['name']
        confidence = obj['confidence']
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, objects

# Define paths for the video files
video_paths = [
    "Videos/DSC_0957.MOV",
    "Videos/DSC_1011.MOV",
    "Videos/IMG_9777.MOV",
    "Videos/2ndPart.MOV"
]

streams = [get_video_stream(path) for path in video_paths]

# Streamlit UI
st.title("Multi-Stream Video with YOLOv5 Detection")

# Create placeholders for the video streams and detection results
video_placeholders = [st.empty() for _ in range(4)]
result_placeholder = st.empty()

# Initialize variables for frame capture and detection interval
previous_frames = [None] * 4
start_time = time.time()

while True:
    frames = []
    objects_detected = []

    for i, stream in enumerate(streams):
        if stream.isOpened():
            ret, frame = stream.read()
            if ret:
                frames.append(frame)
                if time.time() - start_time >= 1:
                    frame, objects = detect_objects(frame)
                    objects_detected.append(objects)
                    previous_frames[i] = frame
                else:
                    objects_detected.append([])
            else:
                frames.append(None)
                print(f"Error: Unable to read frame from stream {i+1}")
                stream.release()  # Release and try to reconnect
                streams[i] = get_video_stream(video_paths[i])
        else:
            frames.append(None)
            streams[i] = get_video_stream(video_paths[i])  # Try to reconnect
            print(f"Reconnecting to stream {i+1}")

    # Display video streams in 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        if frames[0] is not None:
            st.markdown(f"### Camera 1")
            video_placeholders[0].image(frames[0], channels="BGR", use_column_width=True)
        else:
            video_placeholders[0].write(f"Stream 1 not available")
        
        if frames[2] is not None:
            st.markdown(f"### Camera 3")
            video_placeholders[2].image(frames[2], channels="BGR", use_column_width=True)
        else:
            video_placeholders[2].write(f"Stream 3 not available")

    with col2:
        if frames[1] is not None:
            st.markdown(f"### Camera 2")
            video_placeholders[1].image(frames[1], channels="BGR", use_column_width=True)
        else:
            video_placeholders[1].write(f"Stream 2 not available")
        
        if frames[3] is not None:
            st.markdown(f"### Camera 4")
            video_placeholders[3].image(frames[3], channels="BGR", use_column_width=True)
        else:
            video_placeholders[3].write(f"Stream 4 not available")

    # Update detection results every 1 second
    if time.time() - start_time >= 1:
        result_text = ""
        for i, objects in enumerate(objects_detected):
            result_text += f"### Camera {i+1} Detection Results\n"
            if objects:
                for obj in objects:
                    result_text += f"Object: {obj['name']}, Confidence: {obj['confidence']:.2f}\n"
            else:
                result_text += "No objects detected\n"
        result_placeholder.markdown(result_text)
        start_time = time.time()

    time.sleep(1)  # Adjust sleep time as needed

# Release video streams
for stream in streams:
    stream.release()
