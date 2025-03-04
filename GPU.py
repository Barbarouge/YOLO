import torch
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device)  # Modeli GPU'ya taşı
model.train(data="/Users/barbarosyesilova/Desktop/Yolo/Box_detect_prj/config.yaml", epochs=2, device=device)