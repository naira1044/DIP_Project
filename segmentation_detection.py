import cv2
import numpy as np


def apply_seg_edge(img, task):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_display = gray.copy()

    if task == "Basic Global Thresholding":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    elif task == "Automatic Thresholding (Otsu)":
        _, result = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif task == "Adaptive Thresholding":
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    elif task == "Sobel Detector":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        result = cv2.convertScaleAbs(magnitude)

    else:
        result = gray.copy()

    return input_display, result
