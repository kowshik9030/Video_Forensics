# Integrated Video Evidence Extraction and Digital Forensics

An automated video-forensics framework that processes surveillance footage in real time, detects persons, weapons and suspicious activity with YOLOv8 and OpenCV, and logs every detection with timestamps for evidence review.

## Features

- **Person and weapon detection**: YOLOv8 object detection over live or uploaded surveillance footage, including custom-trained models for dangerous objects.
- **Crowd detection**: counts people in frame and flags crowded areas based on proximity analysis.
- **Violence detection**: identifies violent activity in the video stream.
- **Abandoned baggage detection**: detects and counts items left unattended.
- **Suspicious activity detection**: flags suspicious behaviors and objects.
- **CLAHE low-quality footage enhancement**: every frame passes through a CLAHE (Contrast Limited Adaptive Histogram Equalization) stage on the LAB lightness channel before detection, recovering local contrast in dark, low-contrast or night-time footage so the detectors work on usable input.
- **Evidence logging**: every detection is written to a timestamped CSV log (detection_log.csv) for evidence extraction and review.
- **Tamper-proof evidence reports**: every log entry is chained with SHA-256 cryptographic hashing (each entry's hash includes the previous entry's hash). Any edit, deletion or insertion in the historical log breaks the chain. The app exposes `/evidence_report` (sealed report with head hash) and `/verify_evidence` (re-computes the chain from the raw CSV and reports the first failure).
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
│   ├── enhancement/                # CLAHE footage enhancement (modules/enhancement/clahe_enhancer.py)
│   ├── evidence_integrity/         # SHA-256 tamper-evidence chain (modules/evidence_integrity/evidence_hasher.py)
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
