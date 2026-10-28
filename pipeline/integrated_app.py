import os
import io
import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from detection_pipeline import DetectionPipeline

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file

# Windows beep
import winsound

# ----------------- Config -----------------
CSV_LOG = "detection_log.csv"
PEOPLE_LIMIT = 10                # how many people constitute crowd
DISTANCE_THRESHOLD = 80          # pixels; tune per camera view
STREAM_FPS_LIMIT = 20            # max yield fps
# ------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Initialize pipeline
pipeline = DetectionPipeline()

# Shared state
state_lock = threading.Lock()
current_status = {"crowd_status": "UNKNOWN", "count": 0, "violence_detected": False, "abandoned_count": 0, "suspicious_detected": False}
prev_status_value = "UNKNOWN"

# Video source control
video_source = {"type": "webcam", "value": 0}
capture = None
capture_lock = threading.Lock()
running = True

# Frame store
latest_frame = None
latest_frame_lock = threading.Lock()

# Ensure CSV exists with header
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "status", "count", "violence_detected", "abandoned_count", "suspicious_detected"])


def log_status(status, count, violence_detected, abandoned_count, suspicious_detected):
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, status, count, violence_detected, abandoned_count, suspicious_detected])


def server_beep_once():
    # Single beep on Windows
    try:
        frequency = 1000  # Hz
        duration = 400    # ms
        winsound.Beep(frequency, duration)
    except Exception as e:
        print("Beep failed:", e)


def video_capture_thread():
    global capture, video_source, latest_frame, current_status, prev_status_value, running
    last_yield = 0
    frame_idx = 0
    while running:
        with capture_lock:
            if capture is None:
                # create capture based on video_source
                try:
                    if video_source["type"] == "webcam":
                        cap_src = int(video_source["value"])
                    else:
                        cap_src = video_source["value"]
                    capture = cv2.VideoCapture(cap_src)
                    # small warm-up
                    time.sleep(0.5)
                except Exception as e:
                    print("Failed to open capture:", e)
                    capture = None

        if capture is None or not capture.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = capture.read()
        if not ret or frame is None:
            # For files, loop by seeking to beginning; for others, restart capture
            if video_source["type"] == "file":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = capture.read()
                if not ret or frame is None:
                    # If still can't read, release
                    with capture_lock:
                        try:
                            capture.release()
                        except Exception:
                            pass
                        capture = None
                    time.sleep(0.5)
                    continue
            else:
                # restart capture (some rtsp streams drop)
                with capture_lock:
                    try:
                        capture.release()
                    except Exception:
                        pass
                    capture = None
                time.sleep(0.5)
                continue

        frame_idx += 1

        # Process frame using pipeline
        annotated, crowded, count, violence_detected, abandoned_count, suspicious_detected = pipeline.process_frame(frame)

        # update shared status (with beep on transition)
        with state_lock:
            new_status_str = "CROWDED" if crowded else "NON-CROWDED"
            # single beep when SAFE -> CROWDED
            global prev_status_value
            prev = current_status.get("crowd_status", "UNKNOWN")
            # update
            current_status["crowd_status"] = new_status_str
            current_status["count"] = int(count)
            current_status["violence_detected"] = violence_detected
            current_status["abandoned_count"] = abandoned_count
            current_status["suspicious_detected"] = suspicious_detected
            # log each time state is evaluated (optional: log only on change)
            log_status(new_status_str, count, violence_detected, abandoned_count, suspicious_detected)

            if prev != "CROWDED" and new_status_str == "CROWDED":
                # server beep once
                try:
                    threading.Thread(target=server_beep_once, daemon=True).start()
                except Exception as e:
                    print("Beep thread error:", e)

        # store latest_frame bytes for MJPEG streaming
        ret2, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        with latest_frame_lock:
            latest_frame = frame_bytes

        # limit server-side processing fps
        now = time.time()
        if STREAM_FPS_LIMIT > 0:
            elapsed = now - last_yield
            min_interval = 1.0 / STREAM_FPS_LIMIT
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_yield = time.time()

    # cleanup on exit
    with capture_lock:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        capture = None


# Start capture thread
t = threading.Thread(target=video_capture_thread, daemon=True)
t.start()


# ------------- Flask routes --------------
@app.route("/")
def index():
    return render_template("index.html")


def generate_mjpeg():
    """Generator yielding multipart JPEG frames from latest_frame"""
    while True:
        with latest_frame_lock:
            frame = latest_frame
        if frame is None:
            # return a blank frame occasionally
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buf = cv2.imencode(".jpg", blank)
            frame = buf.tobytes()
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        yield boundary
        # small sleep to avoid tight loop
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/status")
def status():
    with state_lock:
        return jsonify(current_status)


@app.route("/set_source", methods=["POST"])
def set_source():
    """Change video source: JSON {type:'webcam'|'rtsp'|'file', value:0 or 'rtsp://..'}"""
    data = request.form or request.json or {}
    src_type = data.get("type")
    value = data.get("value")
    if src_type is None or value is None:
        return jsonify({"ok": False, "error": "Provide 'type' and 'value'"}), 400

    with capture_lock:
        global capture
        # close existing capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
            capture = None

        if src_type == "webcam":
            try:
                value_int = int(value)
                video_source["type"] = "webcam"
                video_source["value"] = value_int
            except:
                return jsonify({"ok": False, "error": "webcam value must be int index"}), 400
        elif src_type == "rtsp":
            video_source["type"] = "rtsp"
            video_source["value"] = value
        elif src_type == "file":
            video_source["type"] = "file"
            video_source["value"] = value
        else:
            return jsonify({"ok": False, "error": "invalid type"}), 400

    return jsonify({"ok": True, "type": video_source["type"], "value": video_source["value"]})


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload image or video for quick testing.
    For images: we run detection on the image and return annotated result.
    For videos: set as video source for live analysis and redirect to index.
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # try read as image first
    img = cv2.imread(save_path)
    if img is not None:
        annotated, crowded, count, violence_detected, abandoned_count, suspicious_detected = pipeline.process_frame(img)
        # convert to JPEG and send back
        ret, buf = cv2.imencode(".jpg", annotated)
        return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg', as_attachment=False, download_name="annotated.jpg")

    # else assume video: set as source for live analysis
    # Check if it's a valid video by trying to open
    cap_test = cv2.VideoCapture(save_path)
    if not cap_test.isOpened():
        cap_test.release()
        return "Cannot process uploaded file: not a valid video or image", 400
    cap_test.release()

    # Set video source to the uploaded file
    with capture_lock:
        global capture, video_source
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
            capture = None
        video_source["type"] = "file"
        video_source["value"] = save_path

    return redirect(url_for('index'))


@app.route("/download_log")
def download_log():
    return send_file(CSV_LOG, as_attachment=True)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    # for convenience during development
    global running
    running = False
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "shutting down..."


if __name__ == "__main__":
    # Run Flask on port 5000, accessible locally
    app.run(host="0.0.0.0", port=5000, threaded=True)
