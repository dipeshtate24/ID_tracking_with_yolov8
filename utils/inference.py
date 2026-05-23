import cv2
from ultralytics import YOLO

model = YOLO('model/yolov8n.pt')

def model_inference(frame):

    results = model.track(frame, 
                          persist=True, 
                          classes = [0])
    
    return results


    