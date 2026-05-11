import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import mode

# Basic Filters

def average_filter(image):
   
    kernel = np.ones((3, 3)) / 9.0
    filtered_image = ndimage.convolve(image, kernel, mode='reflect')
    return filtered_image

def laplacian_filter(image):
    
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    filtered_image = ndimage.convolve(image, kernel, mode='reflect')
    return filtered_image

def maximum_filter(image):
    
    filtered_image = ndimage.maximum_filter(image, size=3, mode='reflect')
    return filtered_image

def minimum_filter(image):
    
    filtered_image = ndimage.minimum_filter(image, size=3, mode='reflect')
    return filtered_image

def median_filter(image):
    
    filtered_image = ndimage.median_filter(image, size=3, mode='reflect')
    return filtered_image

def mode_filter(image):
    
    def mode_func(neighborhood):
        mode_value, _ = mode(neighborhood, axis=None)
        return mode_value[0]

    filtered_image = ndimage.generic_filter(
        image,
        function=mode_func,
        size=3,
        mode='reflect'
    )
    return filtered_image

# Noise Addition

def add_salt_pepper_noise(image, prob=0.05):
    
    noisy_image = np.copy(image)
    rand = np.random.rand(*image.shape)
    salt_mask = rand < (prob / 2)
    pepper_mask = (rand >= (prob / 2)) & (rand < prob)
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0

    return noisy_image


def add_gaussian_noise(image, mean=0, std=10):
    
    gaussian_noise = np.random.normal(mean, std, size=image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)



# Restoration 


def outlier_removal_filter(image, threshold=50):
    median_img = ndimage.median_filter(image, size=3, mode='reflect')
    diff = np.abs(image.astype(np.int32) - median_img.astype(np.int32))
    result = np.copy(image)
    result[diff > threshold] = median_img[diff > threshold]
    return result


def image_averaging(images):
    sum_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        sum_img += img.astype(np.float64)
    avg = sum_img / len(images)
    return np.clip(avg, 0, 255).astype(np.uint8)



# Restoration Engine

def restore_image(image, noise_type, method=None, auxiliary_images=None):

    # ---------------- Salt & Pepper ----------------
    if noise_type == "salt_pepper":

        if method == "median":
            return median_filter(image)

        elif method == "average":
            return average_filter(image)

        elif method == "outlier":
            return outlier_removal_filter(image)

        else:
            return median_filter(image)

    # ---------------- Gaussian ----------------
    elif noise_type == "gaussian":

        if method == "average":
            return average_filter(image)

        elif method == "image_averaging":
            if auxiliary_images is not None and len(auxiliary_images) > 0:
                return image_averaging(auxiliary_images)
            else:
                return average_filter(image)

        else:
            return average_filter(image)

    
    else:
        return image


# GUI Entry Point


def apply_filters_restoration(img, task):
    
    #entry point called by app.py
    #receives a BGR image and a task name string
    #returns (input_display, result) as NumPy arrays
    
    input_display = img.copy()

    # convert to grayscale for all filter/noise operations
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ---- Linear Filters ----
    if task == "Linear: Average Filter":
        result = average_filter(gray)

    elif task == "Linear: Laplacian Filter":
        result = laplacian_filter(gray)

    # ---- Non-linear Filters ----
    elif task == "Non-linear: Maximum":
        result = maximum_filter(gray)

    elif task == "Non-linear: Minimum":
        result = minimum_filter(gray)

    elif task == "Non-linear: Median":
        result = median_filter(gray)

    elif task == "Non-linear: Mode (Most Frequent)":
        result = mode_filter(gray)

    # ---- Salt & Pepper Noise + Restoration ----
    elif task == "Salt & Pepper: Average":
        noisy = add_salt_pepper_noise(gray)
        input_display = noisy  # show noisy image as input
        result = restore_image(noisy, noise_type="salt_pepper", method="average")

    elif task == "Salt & Pepper: Median":
        noisy = add_salt_pepper_noise(gray)
        input_display = noisy
        result = restore_image(noisy, noise_type="salt_pepper", method="median")

    elif task == "Salt & Pepper: Outlier Method":
        noisy = add_salt_pepper_noise(gray)
        input_display = noisy
        result = restore_image(noisy, noise_type="salt_pepper", method="outlier")

    # ---- Gaussian Noise + Restoration ----
    elif task == "Gaussian: Image Averaging":
        noisy = add_gaussian_noise(gray)
        input_display = noisy
        # Generate a set of noisy copies to simulate image averaging
        noisy_copies = [add_gaussian_noise(gray) for _ in range(8)]
        result = restore_image(noisy, noise_type="gaussian", method="image_averaging",
                               auxiliary_images=noisy_copies)

    elif task == "Gaussian: Average Filter":
        noisy = add_gaussian_noise(gray)
        input_display = noisy
        result = restore_image(noisy, noise_type="gaussian", method="average")

    else:
        raise ValueError(f"Unknown task: {task}")

    return input_display, result
