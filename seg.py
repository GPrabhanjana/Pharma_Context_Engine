import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

_YOLO_MODEL = YOLO("yolov8s-seg.pt")
_OCR_READER = easyocr.Reader(["en"], gpu=False)

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced


def sharpen_edges(image):
    """
    Apply edge sharpening using unsharp masking
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def apply_gamma_compression(image, gamma=0.6):
    """
    Apply gamma compression to suppress highlights (foil glare)
    gamma < 1 darkens bright regions more than dark ones
    """
    img = image.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def preprocess_image(image):
    """
    Complete preprocessing pipeline
    """
    gamma_corrected = apply_gamma_compression(image, gamma=0.6)
    
    enhanced = apply_clahe(gamma_corrected)
    
    sharpened = sharpen_edges(enhanced)
    
    return sharpened

def crop_from_mask(image, mask):
    """
    Given an image and a binary mask, returns a perspective-correct crop.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    # Order points
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped

def extract_text_from_image(image, conf=0.1):
    """
    Extract text from image using YOLO segmentation + OCR.
    
    Args: 
    image: BGR format from cv2.imread
    conf: Confidence threshold for YOLO detection 
    
    Returns: 
    List of dicts containing: 
    - text
    - center x coordinate 
    - center y coordinate 
    - width
    - height
    - area
    - confidence: OCR confidence score 
    - word_count
    - image_width
    - image_height
    """
    h, w = image.shape[:2]

    # Preprocess image
    preprocessed = preprocess_image(image)

    # Run YOLO segmentation (NO reload)
    result = _YOLO_MODEL.predict(
        preprocessed,
        conf=conf,
        imgsz=640,
        verbose=False
    )[0]

    ocr_image = None
    yolo_seg_successful = False

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))

        idx = int(np.argmax(areas))
        mask = (masks[idx] * 255).astype(np.uint8)

        # Resize mask if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Crop label
        cropped = crop_from_mask(image, mask)
        if cropped is not None:
            ocr_image = cropped
            yolo_seg_successful = True

    # Fallback to original image
    if ocr_image is None:
        ocr_image = image

    ocr_h, ocr_w = ocr_image.shape[:2]

    # Run OCR (NO reload)
    ocr_results = _OCR_READER.readtext(
        ocr_image,
        rotation_info=[0, 90, 180, 270]
    )
    
    # If YOLO segmentation was successful but OCR found less than 3 text boxes, rerun on whole image
    if yolo_seg_successful and len(ocr_results) < 3:
        ocr_image = image
        ocr_h, ocr_w = ocr_image.shape[:2]
        ocr_results = _OCR_READER.readtext(
            ocr_image,
            rotation_info=[0, 90, 180, 270]
        )

    text_info_list = []

    for bbox, text, confidence in ocr_results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]

        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4

        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_area = bbox_width * bbox_height

        text_info_list.append({
            'text': text,
            'x': center_x,
            'y': center_y,
            'width': bbox_width,
            'height': bbox_height,
            'area': bbox_area,
            'confidence': confidence,
            'word_count': len(text.split()),
            'length': len(text),
            'image_width': ocr_w,
            'image_height': ocr_h
        })

    return text_info_list