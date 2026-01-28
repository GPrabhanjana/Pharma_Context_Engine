# Intelligent Pharma-Context Engine

A pipeline for extracting, verifying, and enriching pharmaceutical metadata from medicine packaging images using computer vision, OCR, and fuzzy matching.

## Architecture Overview

The pipeline consists of four modules: barcode extraction, image segmentation, fuzzy matching, and FDA enrichment. The process begins with barcode extraction. If successful, the barcode provides database lookup. When barcode extraction fails or no database match is found, the system uses OCR-based matching that combines image segmentation, text extraction, and fuzzy string matching, followed by FDA API enrichment.

The barcode feature is a proof-of-concept. While functional with barcode-containing images, it lacks a production database mapping barcodes to drug information. The implementation uses morphological operations and contour detection to localize barcode regions before extraction.

---

## Image Segmentation and OCR

### Preprocessing Pipeline

Pharmaceutical packaging presents challenges: curved surfaces causing text warping, foil reflectivity creating glare, inconsistent lighting, and arbitrary orientations. The system applies three preprocessing stages:

**Gamma Compression (γ = 0.6)** - Suppresses highlights and reduces foil glare. Gamma values below 1.0 darken bright regions more than dark regions, compressing the dynamic range where reflections occur.

**CLAHE (clipLimit=3.0, tileGridSize=8x8)** - Enhances text visibility in shadowed or unevenly lit regions. Parameters were empirically tuned to boost local contrast without amplifying noise. A visualization tool enabled side-by-side comparison of preprocessing results for parameter selection.

**Unsharp Masking** - Sharpens edges by computing the weighted difference between the original image and its Gaussian blur, improving OCR accuracy on blurred text.

### Segmentation Strategy

The system uses YOLOv8s-seg for label detection. Selection criteria:
- Instance segmentation masks for precise boundary extraction
- Fast inference suitable for production
- Generalizes to rectangular labels without custom training

The system identifies the largest segmented region, applies morphological closing to clean boundaries, and performs perspective correction for curved surfaces. Perspective correction extracts the minimum area rectangle from the segmentation mask, orders corner points, and applies a perspective transform to unwarp cylindrical bottle labels into flat images.

### OCR Extraction

EasyOCR was selected for:
- Superior handling of complex fonts and non-horizontal text
- Built-in rotation detection for arbitrary orientations
- Reliable confidence scores

The OCR engine processes images with rotation information for 0, 90, 180, and 270 degrees.

### Adaptive Fallback

If YOLO segmentation succeeds but OCR detects fewer than three text segments, the system reruns OCR on the full original image. This handles cases where segmentation crops too tightly and excludes peripheral text like company names or dosage information.

Each text segment includes metadata: text content, bounding box coordinates and dimensions, OCR confidence, word count, character count, and image dimensions.

---

## Fuzzy Matching Engine

### Design Requirements

Pharmaceutical labels lack universal templates. The matching system must:
- Handle OCR errors (character substitutions)
- Distinguish drug names from company names without positional assumptions
- Match against brand names and generic compositions
- Resolve ambiguity when multiple brands share the same active ingredient

### Multi-Level Matching Hierarchy

The system uses a five-level cascade. Each level represents a matching strategy, and the system returns immediately upon finding a match.

**Level 1: Containment Matching** - Checks if any drug name appears as a substring within the three largest text boxes. Example: "PARACETAMOL 500MG" contains "PARACETAMOL". Largest text boxes are prioritized based on the heuristic that prominent text is typically the drug name.

**Level 2: Fuzzy Matching (≥65% similarity)** - Uses Levenshtein distance to calculate character-level similarity between OCR text and database names. The threshold balances tolerance for OCR errors while preventing false positives. Handles cases like "ASPRIN" → "ASPIRIN".

**Level 3: Token Coverage (≥75%)** - Combines the three largest text boxes and calculates the percentage of drug name tokens present. Addresses split text: "AMOXICILLIN" in one box, "CLAVULANATE" in another.

**Level 4: Generic Name Matching** - Invoked only when no confident brand name match exists. The system prioritizes acknowledging uncertainty over guessing. For multi-component drugs, all components must achieve ≥75% similarity. When multiple brands share the same ingredient, the system attempts brand name disambiguation using remaining text boxes. If disambiguation fails, it returns the generic match with a flag indicating lower confidence.

This design ensures that even incorrect brand identification in Level 4 results in correct active ingredient identification. From a clinical perspective, the medicine is functionally equivalent despite different branding. In production, Level 4 matches would include a warning flag prompting user verification.

**Level 5: No Match** - Returns null when all strategies fail, explicitly acknowledging inability to identify the medication.

### Company Name Extraction

Company names lack standardized positions on labels. The system filters for high-confidence single-word segments and scores candidates based on 75% weight for edge proximity and 25% weight for distance from the drug name. This extracts manufacturer information without positional templates.

---

## FDA Enrichment

After drug identification, the system queries the OpenFDA drug label database. For multi-component drugs, each ingredient is queried separately and results are aggregated. Extracted information includes purpose and indications, warnings and contraindications, dosage guidelines, storage requirements, and adverse reactions.

---

## Dataset and Database Enhancement

The original challenge datasets were unsuitable. One contained single pills in controlled angles with minimal text, focused on object detection training. Another showed pills inside bottles without visible label text. The system uses the Mobile-Captured Pharmaceutical Medication Packages dataset from Kaggle, which provides real-world mobile captures with full label text in varied conditions.

The original drug database lacked generic ingredient names, critical for Level 4 matching and FDA queries. This gap was addressed through web scraping of 1mg.com. The scraper searches for each drug, extracts salt composition from embedded JSON, and populates the Active Ingredient column. This enhanced approximately 60-70% of entries.

---

## Performance Analysis

Evaluation on 50 randomly selected images across three independent runs:
- Match rate: 88-92%
- Accuracy: 91-98%
- Character Error Rate: 0.0325-0.1101

CER variance reflects differences in image quality and preprocessing effectiveness across packaging types. Most failures occur when drug names are not detected by OCR, primarily due to extreme angles, orientations, or lighting. The adaptive fallback mechanism mitigates many cases, but some images remain too degraded for reliable extraction.

---

## Edge Case Handling

The system addresses the challenge's reasoning hurdles:
- **Physical distortions**: Perspective correction using homography transforms
- **Foil glare**: Gamma compression and CLAHE preprocessing
- **Layout agnosticism**: Size-based text prioritization and spatial heuristics
- **Fuzzy entity resolution**: Levenshtein-based matching and token-level analysis

Barcode validation is implemented as a framework but requires production databases. With access to GS1 or NDC databases, the system could validate OCR results against barcode identifications.

---

## Project Structure

- **seg.py**: Image segmentation and OCR extraction
- **match.py**: Fuzzy matching and FDA enrichment
- **pipeline.py**: Orchestration (barcode-to-OCR flow)
- **run.py**: Batch processing and evaluation
- **barcode.py**: Barcode and Data Matrix extraction
- **fetch.py**: Web scraping for generic drug names
- **segviz.py**: Preprocessing visualization

Test images are organized in folders named by drug for ground truth validation. Enhanced drug database is stored as Excel. Output includes per-run performance metrics and per-image extraction results in JSON.

---

## Future Improvements

**OCR Robustness**
- Fine-tune EasyOCR on pharmaceutical label fonts
- Ensemble approach combining multiple OCR engines with majority voting
- Confidence-based retry for low-confidence regions

**Mobile Deployment**
- Real-time capture guidance with visual indicators
- Quality gating for blur and lighting checks
- Lightweight models (MobileNet-based segmentation)

**Dataset-Specific Tuning**
- Pre-filter drug database if target medicines are known
- Synthetic data generation through augmentation
- Active learning for failure case prioritization

**Barcode Integration**
- Partnership with pharmaceutical databases (GS1 GTIN, FDA NDC)
- Hybrid validation: barcode primary, OCR fallback

---

## Acknowledgments

This system utilizes the OpenFDA drug label API for public domain pharmaceutical data, the Mobile-Captured Pharmaceutical Medication Packages dataset from Kaggle under CC BY 4.0 license, and open-source libraries including YOLOv8 from Ultralytics, EasyOCR from JaidedAI, OpenCV, and python-Levenshtein.

## Steps to Run

Download the dataset from kaggle and add it to this directory. Replace the excel file within the dataset with the one provided. Downlaod the yolov8s-seg.pt model. Run 'run.py' to reproduce the output json files.

**G Prabhanjana**  
Technical Challenge Submission - January 2026