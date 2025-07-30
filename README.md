# Real-Time Object Detection and Tracking System

This project is a complete, end-to-end computer vision application that performs real-time object detection and tracking on video streams. The system is built with a state-of-the-art YOLOv8 model, served via a high-performance FastAPI backend, and presented through an interactive Streamlit web interface. The entire application is containerized with Docker for seamless setup and deployment.

![Streamlit Frontend GIF](<path_to_your_demo.gif>) <!-- A GIF of the app in action would be very impressive here -->

---

## 📋 Features

-   **State-of-the-Art Detection**: Utilizes a pre-trained **YOLOv8** model for fast and accurate object detection.
-   **Multi-Object Tracking**: Implements the **SORT (Simple Online and Realtime Tracking)** algorithm to assign and maintain a unique ID for each detected object across video frames.
-   **Real-Time Video Processing**: The Streamlit frontend processes uploaded video files frame by frame, sending them to the backend and displaying the annotated results in real-time.
-   **High-Performance Backend**: A robust **FastAPI** server provides a stateless API endpoint for object detection, making the system scalable and efficient.
-   **Fully Containerized**: The entire application, including the API, frontend, and all complex dependencies, is managed by **Docker Compose** for a simple, one-command setup.

---

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Machine Learning / CV**: `ultralytics` (YOLOv8), OpenCV, NumPy, `filterpy`, `scikit-image`
-   **Frontend**: Streamlit, Requests
-   **Deployment**: Docker, Docker Compose

---
```bash
🚀 How to Run

To run this application on your local machine, you need to have Docker and Docker Compose installed.

1. Clone the Repository


git clone https://github.com/PseudoGod541/real-time-object-detection
cd <your-project-directory>

2. Download the SORT tracker script
Ensure the sort.py file from the official repository is present in the root of the project directory.

3. Run with Docker Compose
This single command will build the Docker image (which may take some time on the first run due to the size of the libraries) and start both the FastAPI backend and the Streamlit frontend services.

docker-compose up --build

4. Access the Application
Once the containers are running, you can access the services:

Streamlit Frontend: Open your browser and go to http://localhost:8501

FastAPI Backend Docs: Open your browser and go to http://localhost:8000/docs

📁 Project Structure
.
├── main.py               # FastAPI application for object detection
├── streamlit_app.py      # Streamlit frontend for video processing and tracking
├── sort.py               # The SORT tracking algorithm implementation
├── Dockerfile            # Instructions to build the Docker image
├── docker-compose.yml    # Defines and runs the multi-container setup
├── .dockerignore         # Specifies files to ignore during the Docker build
└── requirements.txt      # Python dependencies for all services
