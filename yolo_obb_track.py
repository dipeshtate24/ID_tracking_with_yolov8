import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

results = model.track(source=r"C:\Users\Dipesh\Documents\yolo_object_detection_tracking\testing_video\1625968-hd_1920_1080_25fps.mp4", show=True, tracker="bytetrack.yaml")
print(results)