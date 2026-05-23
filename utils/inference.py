import cv2
from ultralytics import YOLO

def model_inference(img_tensor, original_frame, video_path):

    model = YOLO('model/yolov8n.pt')

    cap = cv2.VideoCapture(video_path)

    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            break
        img_tensor, original_frame = preprocessing(frame)
        results = model(img_tensor)[0]

        for result in results:
            for box  in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if conf > 0.5:
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(original_frame, f"{cls}: {conf:.2f}", (x1,  y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 0), 5)

            cv2.imshow('inference', original_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()    
        cv2.destroyAllWindows()



    