import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.crowd_detection.crowd_detector import CrowdDetector
from modules.violence_detection.violence_detector import ViolenceDetector
from modules.abandoned_detection.abandoned_detector import AbandonedDetector
from modules.suspicious_detection.suspicious_detector import SuspiciousDetector

class DetectionPipeline:
    def __init__(self, conf_thres=0.35):
        self.conf_thres = conf_thres
        self.crowd_detector = CrowdDetector(conf_thres=conf_thres)
        self.violence_detector = ViolenceDetector(conf_thres=conf_thres)
        self.abandoned_detector = AbandonedDetector(conf_thres=conf_thres)
        self.suspicious_detector = SuspiciousDetector(conf_thres=conf_thres)

    def process_frame(self, frame):
        """Process frame with all detectors in parallel"""
        def run_crowd():
            return self.crowd_detector.detect(frame)

        def run_violence():
            return self.violence_detector.detect(frame)

        def run_abandoned():
            return self.abandoned_detector.detect(frame)

        def run_suspicious():
            return self.suspicious_detector.detect(frame)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_crowd): 'crowd',
                executor.submit(run_violence): 'violence',
                executor.submit(run_abandoned): 'abandoned',
                executor.submit(run_suspicious): 'suspicious'
            }
            results = {}
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()

        # Extract results
        crowd_boxes, centroids = results['crowd']
        violence_detected, violence_boxes = results['violence']
        abandoned_count, abandoned_boxes = results['abandoned']
        suspicious_detected, suspicious_boxes = results['suspicious']

        # Compute crowd status
        crowded, count = self.crowd_detector.compute_crowd_status(centroids)

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = self.crowd_detector.annotate_frame(annotated_frame, crowd_boxes)
        annotated_frame = self.violence_detector.annotate_frame(annotated_frame, violence_boxes)
        annotated_frame = self.abandoned_detector.annotate_frame(annotated_frame, abandoned_boxes)
        annotated_frame = self.suspicious_detector.annotate_frame(annotated_frame, suspicious_boxes)

        # Overlay statuses
        y_offset = 20
        # Crowd status
        crowd_text = f"Crowd: {'CROWDED' if crowded else 'NON-CROWDED'} ({count})"
        crowd_color = (0, 0, 255) if crowded else (0, 255, 0)
        cv2.putText(annotated_frame, crowd_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, crowd_color, 2)
        y_offset += 30
        # Violence status
        violence_text = f"Violence: {'DETECTED' if violence_detected else 'NONE'}"
        violence_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        cv2.putText(annotated_frame, violence_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, violence_color, 2)
        y_offset += 30
        # Abandoned status
        abandoned_text = f"Abandoned: {abandoned_count}"
        abandoned_color = (0, 255, 255) if abandoned_count > 0 else (0, 255, 0)
        cv2.putText(annotated_frame, abandoned_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, abandoned_color, 2)
        y_offset += 30
        # Suspicious status
        suspicious_text = f"Suspicious: {'DETECTED' if suspicious_detected else 'NONE'}"
        suspicious_color = (255, 0, 0) if suspicious_detected else (0, 255, 0)
        cv2.putText(annotated_frame, suspicious_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, suspicious_color, 2)

        return annotated_frame, crowded, count, violence_detected, abandoned_count, suspicious_detected
