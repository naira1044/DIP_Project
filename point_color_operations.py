import cv2
import numpy as np

def apply_point_color(img, task):
    
    input_display = img.copy()

    
    if task == "Addition":
        
        result = cv2.add(img, np.full(img.shape, 50, dtype=np.uint8))

    elif task == "Subtraction":
        
        result = cv2.subtract(img, np.full(img.shape, 50, dtype=np.uint8))

    elif task == "Division":
        
        result = cv2.divide(img, np.full(img.shape, 2, dtype=np.uint8))

    elif task == "Complement":
        
        result = cv2.bitwise_not(img)

   
    elif task == "Change Red Lighting":
        
        result = img.copy()
        result[:, :, 2] = cv2.add(img[:, :, 2], np.full(img[:, :, 2].shape, 80, dtype=np.uint8))

    elif task == "Swap R to G":
        
        b, g, r = cv2.split(img)
        result = cv2.merge((b, r, g))  

    elif task == "Eliminate Red":
        
        result = img.copy()
        result[:, :, 2] = 0

    else:
        
        result = img.copy()

    return input_display, result
