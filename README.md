# Integrated Video Evidence Extraction and Digital Forensics

An automated video-forensics framework that processes surveillance footage in real time, detects persons, weapons and suspicious activity with YOLOv8 and OpenCV, and logs every detection with timestamps for evidence review.

## Features

- **Person and weapon detection**: YOLOv8 object detection over live or uploaded surveillance footage, including custom-trained models for dangerous objects.
- **Crowd detection**: counts people in frame and flags crowded areas based on proximity analysis.
- **Violence detection**: identifies violent activity in the video stream.
- **Abandoned baggage detection**: detects and counts items left unattended.
- **Suspicious activity detection**: flags suspicious behaviors and objects.
- **Evidence logging**: every detection is written to a timestamped CSV log (detection_log.csv) for evidence extraction and review.
- **Web interface**: Flask app with live MJPEG video feed, source switching and file upload.

## Tech stack

Python, OpenCV, YOLOv8 (Ultralytics), Flask, NumPy, PyTorch

## Project structure

```
├── app.py                          # Original integrated Flask app
├── pipeline/
│   ├── detection_pipeline.py       # Main pipeline class for parallel processing
│   ├── demo.py                     # Standalone demo script
│   └── integrated_app.py           # Flask app using the pipeline
├── modules/
│   ├── crowd_detection/            # Crowd detection module
│   ├── violence_detection/         # Violence detection module
│   ├── abandoned_detection/        # Abandoned baggage detection module
│   └── suspicious_detection/       # Suspicious activity detection module
└── templates/                      # Flask templates
```

The pipeline orchestrates parallel execution of all detectors using ThreadPoolExecutor for efficient real-time processing.

## Setup

```bash
pip install -r requirements.txt
```

The YOLOv8n base weights (yolov8n.pt) download automatically on first run via Ultralytics.

## Usage

```bash
# Integrated web app (http://localhost:5000)
python pipeline/integrated_app.py

# Standalone demo
python pipeline/demo.py
```
