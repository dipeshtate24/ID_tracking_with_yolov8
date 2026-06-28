import cv2
import numpy as np
from collections import defaultdict

def map_tracking(results, frame):

    track_history = defaultdict(lambda:[])

    result = results[0]
        
            
    # Get the boxes and track IDs
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the result on the frame
        annoted_frame = result.plot()

        # Plot the trace
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            track = track_history[track_id] 
            
            track.append((float(x), float(y))) # x, y center point

            if len(track) > 30: # retain 30 tracks for 30 frames
                    
                track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color =(230, 230, 230), thickness=5)

        return annoted_frame
    
    return frame