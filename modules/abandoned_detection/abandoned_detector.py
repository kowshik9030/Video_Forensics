import cv2
from ultralytics import YOLO

class AbandonedDetector:
    def __init__(self, model_path="saved_models/best (UBD).pt", conf_thres=0.35):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """Detect abandoned baggage in the frame"""
        results = self.model(frame, imgsz=640, conf=self.conf_thres, verbose=False)
        abandoned_count = 0
        boxes = []
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else [1.0] * len(xyxy)
            for i, (box, conf) in enumerate(zip(xyxy, confs)):
                if conf < self.conf_thres:
                    continue
                abandoned_count += 1
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2, float(conf)))
        return abandoned_count, boxes

    def annotate_frame(self, frame, boxes):
        """Draw bounding boxes on the frame"""
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Abandoned {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame
