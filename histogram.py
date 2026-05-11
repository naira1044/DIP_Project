import cv2
import numpy as np

def apply_histogram(img, task):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_display = gray.copy()
    result = None

    if task == "Histogram Stretching (Gray)":
        I_low = np.min(gray)
        I_high = np.max(gray)
        
    
        if I_high - I_low == 0:
            result = gray
        else:
            
            gray_float = gray.astype(np.float32)
            result = ((gray_float - I_low) / (I_high - I_low)) * 255.0
            result = np.clip(result, 0, 255).astype(np.uint8)

    elif task == "Histogram Equalization (Gray)":
        
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        
        
        pdf = hist / gray.size
        
        
        cdf = pdf.cumsum()
        
        
        mapping = np.floor(cdf * 255).astype(np.uint8)
        
        
        result = mapping[gray]

    if result is None:
        result = gray.copy()

    return input_display, result