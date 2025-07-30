import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt') # This will download the model on first run

# Path to an image file
image_path =  r'D:\Artificial Intelligence\New folder\object_tracker\image.jpg' # Change this to an image on your computer

# Read the image using OpenCV
frame = cv2.imread(image_path)

# Run the model on the image
results = model(frame)

# Process the results
for result in results:
    for box in result.boxes:
        # Get coordinates of the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get the confidence score and class name
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Draw the bounding box and label on the image
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('YOLOv8 Detection', frame)
cv2.waitKey(0) # Wait for a key press to close the window
cv2.destroyAllWindows()

print("Detection complete. Press any key in the image window to exit.")
