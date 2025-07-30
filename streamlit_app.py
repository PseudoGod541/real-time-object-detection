import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
from sort import Sort # Make sure sort.py is in the same directory

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Object Tracker",
    page_icon="📹",
    layout="wide"
)

# --- FastAPI Backend URL ---
# IMPORTANT: If using Docker Compose, change '127.0.0.1' to the service name (e.g., 'api')
API_URL = "http://api:8000/detect/"

# --- UI Design ---
st.title("📹 Real-Time Object Detection and Tracking")
st.write(
    "Upload a video file to detect and track objects in real-time. "
    "The application uses a YOLOv8 model served via a FastAPI backend for detection "
    "and the SORT algorithm for tracking."
)

# --- Video Uploader ---
uploaded_file = st.file_uploader(
    "Choose a video file...",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    # A button to start processing
    if st.button("Start Tracking", key="start_button", use_container_width=True):
        
        # Initialize the SORT tracker
        tracker = Sort()
        track_colors = {}
        
        # Create placeholders for the video and stats
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Open the video file using OpenCV
        # We need to write the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        
        frame_count = 0
        total_objects_tracked = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert frame to bytes for the API request
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = io.BytesIO(buffer)
            
            try:
                # Send frame to FastAPI for detection
                files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                
                results = response.json()
                detections = results.get('detections', [])
                
                # Prepare detections for SORT
                detections_for_sort = []
                for det in detections:
                    box = det['box']
                    confidence = det['confidence']
                    detections_for_sort.append([box[0], box[1], box[2], box[3], confidence])
                
                # Update tracker
                if len(detections_for_sort) > 0:
                    tracked_objects = tracker.update(np.array(detections_for_sort))
                else:
                    tracked_objects = tracker.update(np.empty((0, 5)))

                # Draw bounding boxes and track IDs
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = map(int, obj)
                    
                    if track_id not in track_colors:
                        track_colors[track_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    
                    color = track_colors[track_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Update stats
                total_objects_tracked = len(tracker.trackers)
                
                # Display the processed frame
                video_placeholder.image(frame, channels="BGR", caption=f"Processing Frame {frame_count}")
                stats_placeholder.info(f"Total Unique Objects Tracked: {total_objects_tracked}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {e}")
                break
            except Exception as e:
                st.error(f"An error occurred: {e}")
                break

        cap.release()
        st.success("Video processing complete!")
