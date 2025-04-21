from vision.detection_tflite import load_tflite_model, detect_person
from vision.tracking import ObjectTracker
from config.settings import FRAME_WIDTH, FRAME_HEIGHT
import cv2

def main():
    interpreter, input_details, output_details = load_tflite_model("models/ssd_mobilenet_v2.tflite")
    tracker = None

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracker is None or tracker.is_lost:
            box = detect_person(frame, interpreter, input_details, output_details)
            if box:
                tracker = ObjectTracker(frame, box)

        if tracker:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
