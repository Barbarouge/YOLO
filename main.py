from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch

# Use the model
results = model.train(data="/Users/barbarosyesilova/Desktop/Yolo/Box_detect_prj/config.yaml", epochs=200) # train the model