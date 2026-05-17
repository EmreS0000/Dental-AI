import cv2
import numpy as np

def enhance_dental_image(image):
    """
    Applies medical-grade enhancements to dental X-ray images.
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    2. Unsharp Masking for detail enhancement
    """
    # Convert to grayscale if it's BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. CLAHE
    # clipLimit=3.0 is aggressive enough to show lesions but 2.0 is safer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 2. Sharpening (Unsharp Masking)
    # blurred = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    # sharpened = cv2.addWeighted(enhanced_gray, 1.5, blurred, -0.5, 0)
    
    # Simple sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_gray, -1, kernel)

    # Convert back to BGR for YOLO (which usually expects 3 channels)
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return result
