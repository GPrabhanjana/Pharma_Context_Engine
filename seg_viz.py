import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO


# -----------------------------
# Preprocessing functions
# -----------------------------
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
    
    # Apply CLAHE for contrast enhancement
    enhanced = apply_clahe(gamma_corrected)
    
    # Sharpen edges
    sharpened = sharpen_edges(enhanced)
    
    return sharpened


# -----------------------------
# Utility: perspective crop from rotated rect
# -----------------------------
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
    If no segment is detected, run OCR on the original image.
    
    Args:
        image: Input image (BGR format from cv2.imread)
        conf: Confidence threshold for YOLO detection
    
    Returns:
        List of dicts containing:
        - text: detected text string
        - x: center x coordinate
        - y: center y coordinate
        - width: bounding box width
        - height: bounding box height
        - area: bounding box area (width * height)
        - confidence: OCR confidence score
        - word_count: number of words in the text
        - image_width: width of the image (for edge distance calculation)
        - image_height: height of the image (for edge distance calculation)
    """
    h, w = image.shape[:2]

    # Preprocess image
    preprocessed = preprocess_image(image)

    # Load YOLO segmentation model
    model = YOLO("yolov8s-seg.pt")

    result = model.predict(
        preprocessed,
        conf=conf,
        imgsz=640,
        verbose=False
    )[0]

    ocr_image = None
    
    if result.masks is not None:
        # Pick largest mask (label)
        masks = result.masks.data.cpu().numpy()
        areas = [m.sum() for m in masks]
        idx = int(np.argmax(areas))
        mask = (masks[idx] * 255).astype(np.uint8)

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Crop label using mask
        cropped = crop_from_mask(image, mask)
        if cropped is not None:
            ocr_image = cropped
    
    # If no segment detected or crop failed, use original image
    if ocr_image is None:
        ocr_image = image

    ocr_h, ocr_w = ocr_image.shape[:2]

    # Run OCR
    reader = easyocr.Reader(["en"], gpu=False)
    ocr_results = reader.readtext(ocr_image, rotation_info=[0, 90, 180, 270])

    # Extract text with coordinates and confidence
    text_info_list = []
    for detection in ocr_results:
        bbox, text, confidence = detection
        
        # Calculate center coordinates
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        
        # Calculate bounding box width and height
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_area = bbox_width * bbox_height
        
        # Count words
        word_count = len(text.split())
        
        text_info_list.append({
            'text': text,
            'x': center_x,
            'y': center_y,
            'width': bbox_width,
            'height': bbox_height,
            'area': bbox_area,
            'confidence': confidence,
            'word_count': word_count,
            'image_width': ocr_w,
            'image_height': ocr_h
        })
    
    return text_info_list


# -----------------------------
# Visualization function for testing
# -----------------------------
def visualize_detection(data_folder="archive/data", conf=0.1):
    """
    Test function to visualize the detection and OCR process on a random image.
    """
    data_path = Path(data_folder)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = [p for p in data_path.rglob("*") if p.suffix.lower() in extensions]
    if not images:
        print("No images found.")
        return

    image_path = random.choice(images)
    print(f"\nProcessing: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load image.")
        return

    h, w = image.shape[:2]

    # Get text information
    text_info_list = extract_text_from_image(image, conf=conf)

    print("\nDetected text:")
    print("-" * 60)
    for info in text_info_list:
        print(f"{info['text']} (conf: {info['confidence']:.2f})")

    # Visualization
    preprocessed = preprocess_image(image)
    
    # Get YOLO mask for visualization
    model = YOLO("yolov8s-seg.pt")
    result = model.predict(preprocessed, conf=conf, imgsz=640, verbose=False)[0]
    
    overlay = preprocessed.copy()
    ocr_image = None
    
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        areas = [m.sum() for m in masks]
        idx = int(np.argmax(areas))
        mask = (masks[idx] * 255).astype(np.uint8)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        
        cropped = crop_from_mask(image, mask)
        if cropped is not None:
            ocr_image = cropped

    if ocr_image is None:
        ocr_image = image

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    
    ax[0].imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Preprocessed (CLAHE + Sharpening)")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1].set_title("YOLO-Seg Mask")
    ax[1].axis("off")

    ax[2].imshow(cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Image Used for OCR")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_detection("archive/data")