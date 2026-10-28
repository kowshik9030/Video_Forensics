import cv2
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path="saved_models/best (CD).pt", conf_thres=0.35):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """Detect people in the frame and return centroids and boxes"""
        results = self.model(frame, imgsz=640, conf=self.conf_thres, verbose=False)
        boxes = []
        centroids = []
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones(len(xyxy))
            for i, (box, conf) in enumerate(zip(xyxy, confs)):
                if conf < self.conf_thres:
                    continue
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2, float(conf)))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroids.append((cx, cy))
        return boxes, centroids

    def annotate_frame(self, frame, boxes):
        """Draw bounding boxes on the frame"""
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def compute_crowd_status(self, centroids, people_limit=10, distance_threshold=80):
        """Compute if crowded based on centroids"""
        N = len(centroids)
        if N == 0:
            return False, 0
        pts = np.array(centroids)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        close_counts = (dists < distance_threshold).sum(axis=1)
        crowded = np.any(close_counts >= people_limit)
        return crowded, N
