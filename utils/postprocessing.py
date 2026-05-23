import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def video_postprocessing(video_path):

    cap = cv2.VideoCapture(video_path)

    track_history = defaultdict(lambda:[])

    while cap.isOpened():
        
        ret, frame = cap.read()

        if ret:
        
            result = model.track(frame, persist=True, classes =[0])[0]
            
            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the trace
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id] 
                    track.append(float(x), float(y)) # x, y center point

                    if len(track) > 30: # retain 30 tracks for 30 frames
                    
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color =(230, 230, 230), thickness=5)

            # Display the annoted frame
            cv2.imshow('Person Tracking', frame)

            # Break the loop if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        else:
            # Break the loop if the end of the video is reached
            break

    # Release  the  video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()



        
    