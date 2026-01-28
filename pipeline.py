import cv2
import barcode as bc
import match as de


# Stand-in database for barcode lookup, empty in this example
BARCODE_DATABASE = {
    
}

def lookup_barcode(barcode_text):
    """
    Look up barcode in database.
    
    Args:
        barcode_text: Barcode string
        
    Returns:
        dict: Drug information if found, None otherwise
    """
    if barcode_text in BARCODE_DATABASE:
        return BARCODE_DATABASE[barcode_text]
    return None


def create_synthetic_ocr_from_drug_data(drug_data, image_width=1000, image_height=1000):
    """
    Create synthetic OCR text info from drug database entry.
    This allows the drug name to be passed to match.py in the same format as seg.py output.
    
    Args:
        drug_data: Dictionary containing drug information
        image_width: Width of the image (for synthetic data)
        image_height: Height of the image (for synthetic data)
        
    Returns:
        List of OCR text info dictionaries matching seg.py format
    """
    text_info_list = [{
        'text': drug_data['Name'],
        'x': 100,
        'y': 100,
        'width': 100,
        'height': 100,
        'area': 10000,
        'confidence': 1.0,
        'word_count': len(drug_data['Name'].split()),
        'length': len(drug_data['Name']),
        'image_width': image_width,
        'image_height': image_height
    }]
    
    return text_info_list


def process_image_pipeline(image_path, drug_df, conf=0.1):
    # Step 1: Try barcode extraction
    barcode_text = bc.extract_barcode(str(image_path))
    
    if barcode_text:
        print(f"Barcode detected: {barcode_text}")
        
        # Step 2: Look up barcode in database
        drug_data = lookup_barcode(barcode_text)
        
        if drug_data:
            print(f"Barcode match found: {drug_data['Name']}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            h, w = image.shape[:2]
            
            ocr_text_info = create_synthetic_ocr_from_drug_data(drug_data, w, h)
            
            # Use match.py to get the full result format
            import pandas as pd
            matched_drug_row = pd.Series(drug_data)
            matched_item = ocr_text_info[0]
            match_field = 'Name'
            
            # Extract company name 
            company_name = matched_drug_row['Company_Name']  
            
            # Get FDA info
            fda_info = None
            generic_name = drug_data.get('Generic Name', '')
            if generic_name and not pd.isna(generic_name):
                print(f"Searching FDA API for: {generic_name}")
                fda_results = de.search_openfda(generic_name)
                fda_info = de.extract_fda_info(fda_results)
            
            # Calculate metrics
            metrics = de.calculate_metrics(ocr_text_info, matched_drug_row)
            
            # Format output
            formatted_output = de.format_output(
                ocr_text_info, 
                matched_drug_row, 
                matched_item, 
                match_field, 
                company_name, 
                fda_info, 
                metrics
            )
            
            return {
                'ocr_text_info': ocr_text_info,
                'matched_drug_row': matched_drug_row,
                'matched_item': matched_item,
                'match_field': match_field,
                'company_name': company_name,
                'fda_info': fda_info,
                'metrics': metrics,
                'formatted_output': formatted_output
            }
    
    # Step 3: Barcode not found or no match, fall back to OCR-based matching
    print("No barcode match, using OCR-based matching...")
    return de.process_image(image_path, drug_df, conf=conf)