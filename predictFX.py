import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = "/Users/barbarosyesilova/Desktop/Yolo/Box_detect_prj/videos/Cat11.mp4"
video_path_out = f"{video_path}_out.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Model dosya yolu
model_path = os.path.join('.', '/Users/barbarosyesilova/Desktop/Yolo/Box_detect_prj/weights/last.pt')

model = YOLO(model_path)  # Modeli yükle

threshold = 0.5  # Daha düşük eşik değeri kullan

while ret:
    results = model(frame)[0]

    if len(results.boxes) == 0:
        print("Warning: No objects detected in this frame!")

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()