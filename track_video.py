import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort # Make sure you have the sort.py file

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# --- IMPORTANT: UPDATE THIS PATH ---
# Provide the FULL path to your input video file.
# Example for Windows: r'C:\Users\YourUser\Downloads\my_video.mp4'
# Example for macOS/Linux: '/home/user/Downloads/my_video.mp4'
video_path_in = r'D:\Artificial Intelligence\New folder\object_tracker\store-aisle-detection.mp4' # <-- CHANGE THIS

# Path to save the output video file
video_path_out = r'D:\Artificial Intelligence\New folder\object_tracker\output_video.mp4'

# Initialize the SORT tracker
tracker = Sort()

# Open the video file
cap = cv2.VideoCapture(video_path_in)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path_in}")
    exit()

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Use a more compatible codec like 'avc1' (H.264)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(video_path_out, fourcc, fps, (frame_width, frame_height))

# A dictionary to store the colors for each track ID
track_colors = {}

print("🚀 Starting video processing...")
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection on the frame
    results = model(frame, stream=True)

    # Prepare detections for the SORT tracker
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            detections.append([int(x1), int(y1), int(x2), int(y2), float(confidence)])

    # --- FIX: Ensure detections array has the correct shape ---
    # Convert detections to a numpy array.
    # If there are no detections, create an empty array with shape (0, 5).
    if len(detections) > 0:
        detections_np = np.array(detections)
    else:
        detections_np = np.empty((0, 5))

    # Update the tracker with the new detections
    tracked_objects = tracker.update(detections_np)

    # Draw the bounding boxes and track IDs for tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        
        if track_id not in track_colors:
            track_colors[track_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        color = track_colors[track_id]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optional: Display the frame in a window
    cv2.imshow('Real-Time Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processing complete. Tracked video saved to: {video_path_out}")
