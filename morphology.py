import cv2
import numpy as np

def apply_morphology(img, task):
    
    input_display = img.copy()

    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

   
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    
    kernel = np.ones((5, 5), np.uint8)

    if task == "Image Dilation":
        
        result = cv2.dilate(binary, kernel, iterations=1)

    elif task == "Image Erosion":
        
        result = cv2.erode(binary, kernel, iterations=1)

    elif task == "Image Opening":
        
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    elif task == "Boundary: Internal":
        
        erosion = cv2.erode(binary, kernel, iterations=1)
        result = cv2.subtract(binary, erosion)

    elif task == "Boundary: External":
        
        dilation = cv2.dilate(binary, kernel, iterations=1)
        result = cv2.subtract(dilation, binary)

    elif task == "Boundary: Morphological Gradient":
        
        dilation = cv2.dilate(binary, kernel, iterations=1)
        erosion = cv2.erode(binary, kernel, iterations=1)
        result = cv2.subtract(dilation, erosion)

    else:
        raise ValueError(f"Unknown task: {task}")

    return input_display, result
