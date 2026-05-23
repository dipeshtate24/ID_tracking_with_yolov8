import cv2
import numpy as np
import torch

def frame_preprocessing(frame, input_shape=640):

    # Read Frame
    original_frame = frame.copy()
    
    # Resize frame
    img_resize = cv2.resize(original_frame, (input_shape, input_shape))

    # convert BRG to RGB
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # Normalize the image
    img_norm = img_rgb.astype(np.float32) / 255.0

    # HWC to CHW
    img_trasn = np.transpose(img_norm, (2, 0, 1))

    # To tensor
    img_tensor = torch.tensor(img_trasn, dtype=torch.float32)

    # Add batch dimension
    img_BN = img_tensor.unsqueeze(0)

    return img_BN, original_frame