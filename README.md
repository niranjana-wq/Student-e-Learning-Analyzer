# Student e-Learning Analyzer

An AI-powered computer vision and deep learning system designed to monitor and classify student attentiveness during e-learning sessions in real-time. This project uses webcams to analyze facial landmarks (eye aspect ratio, mouth aspect ratio, gaze direction, head pose) and employs an LSTM neural network to predict continuous engagement.

## Features
*   **Facial Analysis in Real-Time:** Detects faces and extracts 400+ 3D landmarks simultaneously using Google's **MediaPipe**.
*   **Behavioral Metrics:**
    *   **Blink Rate / EAR:** Tracks the Eye Aspect Ratio for drowsiness or reduced blinking.
    *   **Yawn Detection / MAR:** Calculates Mouth Aspect Ratio to detect fatigue.
    *   **Head Pose Estimation:** Leverages OpenCV `solvePnP` to evaluate Pitch, Yaw, and Roll. Detects nodding or head shaking.
    *   **Gaze Tracking:** Approximates pupil alignment to verify visual focus on the screen.
*   **Deep Learning Pipeline:** Includes an LSTM (Long Short-Term Memory) sequential neural network trained with TensorFlow/Keras to look at a 30-frame window and understand temporal engagement context.
*   **Real-time Alerts:** Automatically beeps when the student crosses an inattentive threshold or steps away from the camera.
*   **Web Dashboard:** An elegant, Flask-driven dashboard UI that provides a live video feed alongside a persistent data stream (via SSE) of session statistics and AI confidence scores.

## Project Structure
*   `main.py`: A standalone OpenCV application that calculates all behavioral metrics and provides heuristic diagnostic overlays (does not require the ML model).
*   `train_model.py`: Script to process the collected datasets (from a `student_data.csv`), train the LSTM sequential model, and output the weights/scaler.
*   `run_analyzer.py`: Standalone OpenCV analyzer that uses the trained `.keras` model to actively infer student classification without the web backend.
*   `server.py`: The main Flask backend. Starts the camera and inference loop in a background thread and streams video (MJPEG) and metrics (Server-Sent Events) to the web UI.
*   `static/`: Contains the frontend logic (`index.html`, `app.js`, `style.css`) for the Web Dashboard.
*   `requirements.txt`: Project dependencies list.
*   `attentiveness_model.keras` & `scaler.joblib`: Pre-trained weights and feature scaler for the LSTM.

## Installation

1.  **Clone or Download** the repository to your local machine.
2.  **Install Dependencies:** Ensure you have Python 3.8+ installed. Open a terminal in the project directory and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

This project offers multiple ways to run the analyzer depending on your needs.

### 1. Web Dashboard (Recommended)
This launches a background inference thread and a gorgeous frontend UI accessible in your browser.
```bash
python server.py
```
*Open your browser and navigate to `http://127.0.0.1:5000` to interact with the dashboard.*

### 2. Standalone Inference (OpenCV Window)
Runs the LSTM model inference locally with an OpenCV window showing the diagnostic overlays.
```bash
python run_analyzer.py
```

### 3. Heuristics Only Mode (No Deep Learning)
If you do not have the trained model files, you can still track live facial metrics using pure threshold heuristics constraints.
```bash
python main.py
```

### 4. Training a New Model
If you have collected a new dataset via a tool (`student_data.csv`), you can train a new LSTM model by running:
```bash
python train_model.py
```
This produces a new `attentiveness_model.keras` and `scaler.joblib`.

## Troubleshooting
*   **Camera Not Opening:** The scripts default to Windows `cv2.CAP_DSHOW` on camera index `0`. Unplug and replug your webcam or change the video index in the code if your primary camera isn't registering.
*   **"Model or Scaler not found":** If you receive this running `run_analyzer.py` or `server.py`, ensure `attentiveness_model.keras` and `scaler.joblib` are inside the root folder, or alternatively run `python train_model.py` to compile them if dataset exists.
