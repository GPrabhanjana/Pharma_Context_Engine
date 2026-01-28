import cv2
import numpy as np
import os
import tempfile
from pyrxing import read_barcodes

def extract_barcode(image_path):
    """
    Extract barcode from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Barcode text if found, None otherwise
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply gradient detection
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    
    # Blur and threshold
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    # Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    
    # Process contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 50 or h < 50:
            continue
        
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        cropped = img[y1:y2, x1:x2]
        cropped_images.append(cropped)
    
    # If no regions detected, use original image
    if not cropped_images:
        cropped_images.append(img)
    
    for cropped_img in cropped_images:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_barcode.jpg")
        cv2.imwrite(temp_path, cropped_img)
        
        results = read_barcodes(temp_path)
        
        try:
            os.remove(temp_path)
        except:
            pass
        
        if results:
            return results[0].text
    
    return None