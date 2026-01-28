import pandas as pd
import cv2
import re
import requests
from Levenshtein import distance as levenshtein_distance
import seg  
import math


def load_drug_list(excel_path="archive/data/drug list.xlsx"):
    """
    Load drug data from Excel file.
    Returns DataFrame with all columns.
    """
    df = pd.read_excel(excel_path)
    return df


def calculate_similarity(ground_truth, ocr_text):
    """
    Calculate Levenshtein similarity percentage.
    Direction: ground_truth → ocr_text
    """
    if not ground_truth or not ocr_text:
        return 0.0

    gt = ground_truth.lower().strip()
    ocr = ocr_text.lower().strip()

    lev_dist = levenshtein_distance(gt, ocr)
    max_len = len(gt)  

    if max_len == 0:
        return 0.0

    similarity = ((max_len - lev_dist) / max_len) * 100
    return max(0.0, similarity)


def calculate_cer(ocr_text, ground_truth):
    """
    Calculate Character Error Rate (CER).
    CER = edit_distance / length_of_ground_truth
    
    Returns None if ground_truth is empty or invalid.
    """
    if not ground_truth:
        return None

    gt = ground_truth.lower().strip()
    ocr = (ocr_text or "").lower().strip()

    if len(gt) == 0:
        return None

    dist = levenshtein_distance(gt, ocr)
    return dist / len(gt)


def parse_generic_name(generic_name):
    """
    Parse generic name into words, identifying parenthetical groups.
    Returns: (outside_words, inside_parentheses_words)
    """
    if pd.isna(generic_name) or not isinstance(generic_name, str):
        return [], []
    
    parentheses_pattern = r'\(([^)]+)\)'
    parentheses_matches = re.findall(parentheses_pattern, generic_name)    
    text_outside = re.sub(parentheses_pattern, '', generic_name)
    
    outside_words = [w.strip() for w in text_outside.split() if w.strip()]
    inside_words = []
    
    for match in parentheses_matches:
        words = [w.strip() for w in match.split() if w.strip()]
        inside_words.extend(words)
    
    return outside_words, inside_words


def score_word_against_segments(word, ocr_text_info):
    """
    Score a single word against all OCR segments.
    Returns the best similarity score.
    """
    if not ocr_text_info:
        return 0
    
    best_score = 0
    for item in ocr_text_info:
        similarity = calculate_similarity(word, item['text'])
        best_score = max(best_score, similarity)
    
    return best_score


def calculate_generic_bonus(generic_name, ocr_text_info):
    """
    Calculate bonus score from generic name matching.
    Returns total bonus (already weighted by 0.5).
    """
    outside_words, inside_words = parse_generic_name(generic_name)
    
    total_bonus = 0
    
    for word in outside_words:
        word_score = score_word_against_segments(word, ocr_text_info)
        total_bonus += word_score
    
    if inside_words:
        inside_scores = [score_word_against_segments(word, ocr_text_info) for word in inside_words]
        total_bonus += max(inside_scores)
    
    return total_bonus * 0.7


def get_largest_text_boxes(ocr_text_info, n=3, min_letters=3):
    """
    Find the N largest text boxes based on bounding box area,
    considering only boxes with at least `min_letters` alphabetic characters.
    """
    if not ocr_text_info:
        return []

    def letter_count(text):
        return sum(c.isalpha() for c in text)

    filtered_items = [
        item for item in ocr_text_info
        if letter_count(item.get('text', '')) >= min_letters
    ]

    if not filtered_items:
        return []

    sorted_items = sorted(
        filtered_items,
        key=lambda item: item['area'],
        reverse=True
    )

    return sorted_items[:n]

def match_generic_name_components(generic_name, ocr_text_info):
    """
    Match generic name components against OCR text.
    
    For multi-component generics (with '+'):
    - Requires matching ALL components
    - Returns average score of all components
    """
    if pd.isna(generic_name) or not isinstance(generic_name, str):
        return 0, None
    
    if '+' in generic_name:
        components = [comp.strip() for comp in generic_name.split('+')]
        is_multi_component = True
    else:
        outside_words, inside_words = parse_generic_name(generic_name)
        components = outside_words + inside_words
        is_multi_component = False
    
    if not components:
        return 0, None
    
    if is_multi_component:
        component_scores = []
        matched_items = []
        
        for component in components:
            best_component_score = 0
            best_component_item = None
            
            for ocr_item in ocr_text_info:
                similarity = calculate_similarity(component, ocr_item['text'])
                if similarity > best_component_score:
                    best_component_score = similarity
                    best_component_item = ocr_item
            
            component_scores.append(best_component_score)
            matched_items.append(best_component_item)
        
        if all(score >= 75.0 for score in component_scores):
            avg_score = sum(component_scores) / len(component_scores)
            return avg_score, matched_items[0]
        else:
            return 0, None
    
    else:
        best_score = 0
        best_item = None
        
        for component in components:
            for ocr_item in ocr_text_info:
                similarity = calculate_similarity(component, ocr_item['text'])
                if similarity > best_score:
                    best_score = similarity
                    best_item = ocr_item
        
        return best_score, best_item


def find_best_match(ocr_text_info, drug_df):
    """
    Tries matching against the TOP 3 LARGEST text boxes.
    
    1. Exact/containment match on top 3 boxes → return immediately
    2. High fuzzy match (≥75%) on top 3 boxes → return immediately
    3. Combined top 3 boxes fuzzy match (≥75%) with containment
    4. Generic name matching (≥75% per component)
       - Multi-component (+): requires ALL components ≥75%
       - Single component: requires best match ≥75%
    5. Return None if nothing matches
    
    Returns: (matched_row, matched_item, match_field) 
             where match_field is 'Name' or 'Generic Name'
             or (None, None, None)
    """
    largest_items = get_largest_text_boxes(ocr_text_info, n=3)
    
    if not largest_items:
        return None, None, None
    
    print(f"Checking top {len(largest_items)} largest boxes: {[item['text'] for item in largest_items]}")
    
    # LEVEL 1: Exact containment match on top 3 boxes
    print("Level 1: Checking containment matches...")
    for largest_item in largest_items:
        largest_text_lower = largest_item['text'].lower()
        
        containment_matches = []
        for idx, row in drug_df.iterrows():
            drug_name = str(row['Name']).strip().lower()
            if drug_name in largest_text_lower:
                containment_matches.append((drug_name, row))
        
        if containment_matches:
            best_match = max(containment_matches, key=lambda x: len(x[0]))
            print(f"✓ Level 1 match found: {best_match[1]['Name']} (containment in '{largest_item['text']}')")
            return best_match[1], largest_item, 'Name'
    
    # LEVEL 2: High confidence fuzzy match (≥65%) on top 3 boxes
    print("Level 2: Checking high-confidence matches (≥75%)...")
    best_high_conf = None
    best_high_conf_score = 0
    best_high_conf_item = None
    
    for largest_item in largest_items:
        largest_text = largest_item['text']
        
        for idx, row in drug_df.iterrows():
            drug_name = str(row['Name']).strip()
            similarity = calculate_similarity(largest_text, drug_name)
            
            if similarity >= 65.0 and similarity > best_high_conf_score:
                best_high_conf_score = similarity
                best_high_conf = row
                best_high_conf_item = largest_item
    
    if best_high_conf is not None:
        print(f"✓ Level 2 match found: {best_high_conf['Name']} ({best_high_conf_score:.1f}% in '{best_high_conf_item['text']}')")
        return best_high_conf, best_high_conf_item, 'Name'
    
    # LEVEL 3: Combined top 3 boxes with token coverage (≥75%)
    print("Level 3: Checking combined top 3 boxes with token matching...")
    combined_text = ' '.join([item['text'] for item in largest_items])

    print(f"Combined text: '{combined_text}'")

    best_combined = None
    best_combined_score = 0
    best_combined_item = largest_items[0]

    for idx, row in drug_df.iterrows():
        drug_name = str(row['Name']).strip()
        
        # Calculate token coverage
        drug_tokens = set(drug_name.lower().split())
        combined_tokens = set(combined_text.lower().split())
        
        if drug_tokens:
            intersection = drug_tokens.intersection(combined_tokens)
            token_coverage = (len(intersection) / len(drug_tokens)) * 100
            
            if token_coverage >= 75.0 and token_coverage > best_combined_score:
                best_combined_score = token_coverage
                best_combined = row

    if best_combined is not None:
        print(f"✓ Level 3 match found: {best_combined['Name']} ({best_combined_score:.1f}% token coverage)")
        return best_combined, best_combined_item, 'Name'
    
    # LEVEL 4: Generic name matching with brand name disambiguation
    print("Level 4: Checking generic name matches...")
    
    generic_matches = {}  
    
    for idx, row in drug_df.iterrows():
        generic_name = row.get('Generic Name', '')
        if pd.isna(generic_name) or not isinstance(generic_name, str):
            continue
            
        score, matched_item = match_generic_name_components(generic_name, ocr_text_info)
        
        if score >= 75.0:
            generic_key = generic_name.strip().lower()
            
            if generic_key not in generic_matches:
                generic_matches[generic_key] = []
            
            generic_matches[generic_key].append({
                'row': row,
                'generic_score': score,
                'generic_matched_item': matched_item,
                'generic_name': generic_name
            })
    
    if not generic_matches:
        # LEVEL 5: No match found
        print(f"✗ No match found (best scores: level2={best_high_conf_score:.1f}%, level3={best_combined_score:.1f}%, generic=0.0%)")
        return None, None, None
    
    # Get the generic name with the highest score
    best_generic_key = max(generic_matches.keys(), 
                          key=lambda k: max(c['generic_score'] for c in generic_matches[k]))
    candidates = generic_matches[best_generic_key]
    
    if len(candidates) == 1:
        candidate = candidates[0]
        print(f"✓ Level 4 match found: {candidate['row']['Name']} via generic '{candidate['generic_name']}' ({candidate['generic_score']:.1f}%)")
        return candidate['row'], candidate['generic_matched_item'], 'Generic Name'
    
    # Multiple drugs with the SAME generic name - disambiguate by brand name
    print(f"Multiple drugs found with same generic name '{candidates[0]['generic_name']}' ({len(candidates)} candidates). Disambiguating by brand name...")
    
    top_3_items = get_largest_text_boxes(ocr_text_info, n=3)
    
    best_candidate = None
    best_brand_score = 0
    best_brand_item = None
    
    for candidate in candidates:
        brand_name = str(candidate['row']['Name']).strip()
        print(f"  Checking brand name: {brand_name}")
        
        for ocr_item in top_3_items:
            if ocr_item == candidate['generic_matched_item']:
                continue
                
            similarity = calculate_similarity(brand_name, ocr_item['text'])
            
            if similarity > best_brand_score:
                best_brand_score = similarity
                best_candidate = candidate
                best_brand_item = ocr_item
                print(f"    → {similarity:.1f}% match with '{ocr_item['text']}'")
    
    if best_candidate and best_brand_score > 0:
        print(f"✓ Level 4 match found: {best_candidate['row']['Name']} via generic '{best_candidate['generic_name']}' ({best_candidate['generic_score']:.1f}%) + brand name match ({best_brand_score:.1f}%)")
        # Return with 'Name' as match_field since we're using brand name for final decision
        return best_candidate['row'], best_brand_item, 'Name'
    
    best_generic_candidate = max(candidates, key=lambda x: x['generic_score'])
    print(f"✓ Level 4 match found: {best_generic_candidate['row']['Name']} via generic '{best_generic_candidate['generic_name']}' ({best_generic_candidate['generic_score']:.1f}%) [no brand disambiguation possible]")
    return best_generic_candidate['row'], best_generic_candidate['generic_matched_item'], 'Generic Name'


def calculate_edge_distance(item):
    """
    Calculate the minimum distance from the text center to any edge of the image.
    """
    x = item['x']
    y = item['y']
    width = item['image_width']
    height = item['image_height']
    
    dist_left = x
    dist_right = width - x
    dist_top = y
    dist_bottom = height - y
    
    return min(dist_left, dist_right, dist_top, dist_bottom)


def calculate_distance_from_medicine(item, matched_item):
    """
    Calculate Euclidean distance between two text items.
    """
    if matched_item is None:
        return 0
    
    x1, y1 = item['x'], item['y']
    x2, y2 = matched_item['x'], matched_item['y']
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def calculate_company_score(item, matched_item, max_edge_distance, max_medicine_distance):
    """
    Calculate combined score for company name candidate.
    75% weight for edge distance, 25% weight for distance from medicine name.    
    """
    edge_distance = calculate_edge_distance(item)
    medicine_distance = calculate_distance_from_medicine(item, matched_item)
    
    # Normalize distances (0-1 range)
    normalized_edge = edge_distance / max_edge_distance if max_edge_distance > 0 else 0
    normalized_medicine = medicine_distance / max_medicine_distance if max_medicine_distance > 0 else 0
    
    edge_score = (1 - normalized_edge) * 0.75
    medicine_score = normalized_medicine * 0.25
    
    total_score = edge_score + medicine_score
    
    return total_score


def extract_company_name(ocr_text_info, matched_item):
    """
    Extract company name as the text with the best combined score:
    - 75% weight: closest to an edge
    - 25% weight: farthest from medicine name
    Must have confidence > 85% and contain only one word.
    Returns None if no suitable candidate is found.
    """
    if not ocr_text_info:
        return None
    
    candidates = []
    for item in ocr_text_info:
        if item == matched_item:
            continue
        if item['confidence'] > 0.85 and item['word_count'] == 1 and item['length'] >= 3:
            candidates.append(item)
    
    if not candidates:
        return None
    
    # Calculate maximum possible distances for normalization
    if candidates:
        sample_item = candidates[0]
        width = sample_item['image_width']
        height = sample_item['image_height']
        max_edge_distance = min(width, height) / 2
        max_medicine_distance = math.sqrt(width**2 + height**2)
    else:
        max_edge_distance = 1
        max_medicine_distance = 1
    
    best_item = None
    best_score = -1
    
    for item in candidates:
        score = calculate_company_score(item, matched_item, max_edge_distance, max_medicine_distance)
        if score > best_score:
            best_score = score
            best_item = item
    
    return best_item['text'] if best_item else None


def search_openfda(generic_name):
    """
    Search OpenFDA API for drug information using generic name.
    If '+' exists in name, search each component separately and combine results.
    """
    base_url = "https://api.fda.gov/drug/label.json"
    
    # Split by '+' if present
    if '+' in generic_name:
        components = [comp.strip() for comp in generic_name.split('+')]
    else:
        components = [generic_name]
    
    all_results = []
    
    for component in components:
        try:
            params = {
                'search': f'openfda.generic_name:"{component}"',
                'limit': 1
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    all_results.append({
                        'component': component,
                        'data': data['results'][0]
                    })
        except Exception as e:
            print(f"Error searching FDA for '{component}': {e}")
            continue
    
    return all_results


def extract_fda_info(fda_results):
    """
    Extract relevant information from FDA API results.
    """
    if not fda_results:
        return None
    
    enriched_info = {
        'components': []
    }
    
    for result in fda_results:
        component = result['component']
        data = result['data']
        
        component_info = {
            'name': component,
            'purpose': data.get('purpose', ['N/A'])[0] if data.get('purpose') else 'N/A',
            'warnings': data.get('warnings', ['N/A'])[0] if data.get('warnings') else 'N/A',
            'indications_and_usage': data.get('indications_and_usage', ['N/A'])[0] if data.get('indications_and_usage') else 'N/A',
            'dosage_and_administration': data.get('dosage_and_administration', ['N/A'])[0] if data.get('dosage_and_administration') else 'N/A',
            'storage_and_handling': data.get('storage_and_handling', ['N/A'])[0] if data.get('storage_and_handling') else 'N/A',
            'adverse_reactions': data.get('adverse_reactions', ['N/A'])[0] if data.get('adverse_reactions') else 'N/A',
        }
        
        enriched_info['components'].append(component_info)
    
    return enriched_info


def calculate_metrics(ocr_text_info, matched_drug_row):
    """
    Calculate various metrics for the extraction and matching.
    """
    metrics = {
        'total_text_segments': len(ocr_text_info),
        'avg_confidence': sum(item['confidence'] for item in ocr_text_info) / len(ocr_text_info) if ocr_text_info else 0,
        'min_confidence': min(item['confidence'] for item in ocr_text_info) if ocr_text_info else 0,
        'max_confidence': max(item['confidence'] for item in ocr_text_info) if ocr_text_info else 0,
        'total_characters': sum(len(item['text']) for item in ocr_text_info),
    }
    
    return metrics


def format_output(ocr_text_info, matched_drug_row, matched_item, match_field, company_name, fda_info, metrics):
    # Calculate CER if we have a match
    cer = None
    matched_ocr_text = None
    ground_truth_text = None
    
    if matched_drug_row is not None and matched_item is not None and match_field is not None:
        matched_ocr_text = matched_item['text']
        
        # Use the field that was actually used for matching
        if match_field == 'Name':
            ground_truth_text = str(matched_drug_row['Name'])
        elif match_field == 'Generic Name':
            ground_truth_text = str(matched_drug_row['Generic Name'])
        
        if ground_truth_text:
            cer = calculate_cer(matched_ocr_text, ground_truth_text)
    
    output = {
        'Index': int(matched_drug_row['Index']) if matched_drug_row is not None and 'Index' in matched_drug_row and not pd.isna(matched_drug_row['Index']) else None,
        'Name': str(matched_drug_row['Name']) if matched_drug_row is not None and 'Name' in matched_drug_row and not pd.isna(matched_drug_row['Name']) else None,
        'Dose': str(matched_drug_row['Dose']) if matched_drug_row is not None and 'Dose' in matched_drug_row and not pd.isna(matched_drug_row['Dose']) else None,
        'Size': str(matched_drug_row['Size']) if matched_drug_row is not None and 'Size' in matched_drug_row and not pd.isna(matched_drug_row['Size']) else None,
        'Type': str(matched_drug_row['Type']) if matched_drug_row is not None and 'Type' in matched_drug_row and not pd.isna(matched_drug_row['Type']) else None,
        'Generic Name': str(matched_drug_row['Generic Name']) if matched_drug_row is not None and 'Generic Name' in matched_drug_row and not pd.isna(matched_drug_row['Generic Name']) else None,
        'Matched_Field': match_field,
        'Matched_OCR_Text': matched_ocr_text,
        'Ground_Truth_Text': ground_truth_text,
        'CER': cer,
        'Extracted_Company': company_name,
        'FDA_Enrichment': fda_info,
        'OCR_Metrics': metrics
    }
    
    return output


def process_image(image_path, drug_df, conf=0.1):
    """
    Process a single image and extract all drug information.
    
    Args:
        image_path: Path to the image file
        drug_df: DataFrame containing drug database
        conf: Confidence threshold for OCR
    
    Returns:
        Dictionary containing all extracted information:
        - ocr_text_info: List of OCR segments with text, position, confidence
        - matched_drug_row: Matched drug information from database
        - matched_item: The specific OCR item that matched
        - match_field: Which field was used for matching ('Name' or 'Generic Name')
        - company_name: Extracted company name
        - fda_info: FDA enrichment data
        - metrics: OCR performance metrics
        - formatted_output: Structured output with all fields including CER
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Extract text using segmentation script
    ocr_text_info = seg.extract_text_from_image(image, conf=conf)
    
    if not ocr_text_info:
        return {
            'ocr_text_info': [],
            'matched_drug_row': None,
            'matched_item': None,
            'match_field': None,
            'company_name': None,
            'fda_info': None,
            'metrics': None,
            'formatted_output': format_output([], None, None, None, None, None, None)
        }
    
    matched_drug_row, matched_item, match_field = find_best_match(ocr_text_info, drug_df)
    
    company_name = extract_company_name(ocr_text_info, matched_item)
    
    fda_info = None
    if matched_drug_row is not None:
        generic_name = matched_drug_row.get('Generic Name', '')
        if generic_name and not pd.isna(generic_name):
            print(f"Searching FDA API for: {generic_name}")
            fda_results = search_openfda(generic_name)
            fda_info = extract_fda_info(fda_results)
    
    metrics = calculate_metrics(ocr_text_info, matched_drug_row)
    
    formatted_output = format_output(ocr_text_info, matched_drug_row, matched_item, match_field, company_name, fda_info, metrics)
    
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