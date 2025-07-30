import io
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel, Field
from typing import List, Tuple

# --- Pydantic Schemas for API Data Structure ---

class Detection(BaseModel):
    """Schema for a single object detection."""
    box: Tuple[int, int, int, int] = Field(..., description="Bounding box as [x1, y1, x2, y2].")
    class_name: str = Field(..., description="Name of the detected object class.")
    confidence: float = Field(..., description="Confidence score of the detection.")

class DetectionResponse(BaseModel):
    """Schema for the API's response."""
    detections: List[Detection]

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="An API to perform object detection on an image using a pre-trained YOLOv8 model.",
    version="1.0.0"
)

# --- Global for the Model ---
model = None

# --- Startup Event to Load the Model ---
@app.on_event("startup")
async def startup_event():
    """
    Load the YOLOv8 model on application startup.
    """
    global model
    try:
        model = YOLO('yolov8n.pt')
        logger.info("✅ YOLOv8 model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading YOLOv8 model: {e}")
        model = None

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the YOLOv8 Object Detection API! Visit /docs for more info."}

@app.post("/detect/", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a list of detected objects.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run the model on the image
        results = model(image)

        # Process results and prepare the response
        detections_list = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detections_list.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=confidence
                    )
                )
        
        return {"detections": detections_list}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
