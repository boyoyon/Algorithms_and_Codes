# pip install opencv-python numpy ultralytics

from ultralytics import YOLO

# 最軽量の学習済みモデル（yolov8n.pt）を読み込む
model = YOLO("yolov8n.pt")

# OpenCV DNNで読み込めるONNX形式にエクスポート
model.export(format="onnx", opset=12)