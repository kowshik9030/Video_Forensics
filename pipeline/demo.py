import cv2
from detection_pipeline import DetectionPipeline

def main():
    pipeline = DetectionPipeline()

    # Open webcam or video file
    cap = cv2.VideoCapture(0)  # 0 for webcam, or path to video file

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through pipeline
        annotated_frame, crowded, count, violence_detected, abandoned_count, suspicious_detected = pipeline.process_frame(frame)

        # Display results
        print(f"Crowd: {'CROWDED' if crowded else 'NON-CROWDED'} ({count})")
        print(f"Violence: {'DETECTED' if violence_detected else 'NONE'}")
        print(f"Abandoned: {abandoned_count}")
        print(f"Suspicious: {'DETECTED' if suspicious_detected else 'NONE'}")

        # Show frame
        cv2.imshow('VISTA Pipeline Demo', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
