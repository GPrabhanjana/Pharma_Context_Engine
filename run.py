import random
from pathlib import Path
import json
from datetime import datetime
import match as de


def verify_folder_name_match(folder_name, drug_name):
    """
    True if folder name and drug name overlap
    """
    if not drug_name:
        return False

    f = folder_name.lower().strip()
    d = str(drug_name).lower().strip()

    return d in f or f in d


def get_one_image_per_folder(data_folder="archive/data"):
    """
    Select exactly one random image per subfolder
    """
    data_path = Path(data_folder)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    selected = []

    for folder in data_path.iterdir():
        if not folder.is_dir():
            continue

        images = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in extensions
        ]

        if images:
            selected.append((random.choice(images), folder.name))

    return selected


def process_batch_images(
    data_folder="archive/data",
    excel_path="archive/data/drug list.xlsx",
    num_images=50,
    conf=0.1,
    output_dir="output"
):
    drug_df = de.load_drug_list(excel_path)
    all_images = get_one_image_per_folder(data_folder)

    selected_images = (
        random.sample(all_images, num_images)
        if len(all_images) > num_images
        else all_images
    )

    print(f"\nProcessing {len(selected_images)} images...\n")

    total_images = len(selected_images)
    successful_extractions = 0   
    correct_matches = 0          
    total_cer = 0.0              

    all_results = []

    for idx, (image_path, folder_name) in enumerate(selected_images, 1):
        print("=" * 80)
        print(f"[{idx}/{total_images}] {image_path.name}")
        print(f"Folder: {folder_name}")
        print("=" * 80)

        result = de.process_image(image_path, drug_df, conf=conf)

        if result is None:
            print("❌ Processing failed")
            continue

        formatted_output = result["formatted_output"]
        
        # Check if folder name matches extracted drug name
        folder_match = verify_folder_name_match(
            folder_name,
            formatted_output.get("Name")
        )

        # Get CER and match info from formatted output (already calculated in match.py)
        cer = formatted_output.get("CER")
        matched_ocr_text = formatted_output.get("Matched_OCR_Text")
        ground_truth_text = formatted_output.get("Ground_Truth_Text")
        match_field = formatted_output.get("Matched_Field")
        drug_name = formatted_output.get("Name")

        # Track successful extractions
        if drug_name is not None:
            successful_extractions += 1
            
            # Track correct matches and CER
            if folder_match and cer is not None:
                correct_matches += 1
                total_cer += cer

        all_results.append({
            "image_path": str(image_path),
            "folder_name": folder_name,
            "folder_name_match": folder_match,
            "matched_field": match_field,
            "matched_ocr_text": matched_ocr_text,
            "ground_truth_text": ground_truth_text,
            "cer": cer,
            "extraction": formatted_output,
            "all_ocr_text": [x["text"] for x in result["ocr_text_info"]]
        })

        print(f"Matched Drug: {drug_name}")
        print(f"Matched via: {match_field if match_field else 'N/A'}")
        print(f"Ground Truth: {ground_truth_text if ground_truth_text else 'N/A'}")
        print(f"OCR Text: {matched_ocr_text if matched_ocr_text else 'N/A'}")
        print(f"Correct Match: {'✓' if folder_match else '✗'}")
        print(f"CER: {cer:.4f}" if cer is not None else "CER: N/A")

    # Calculate final metrics
    match_percentage = (
        (successful_extractions / total_images) * 100
        if total_images else 0
    )

    accuracy_percentage = (
        (correct_matches / successful_extractions) * 100
        if successful_extractions else None
    )

    average_cer = (
        total_cer / correct_matches
        if correct_matches > 0 else None
    )

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,

        "successful_extractions": successful_extractions,
        "match_percentage": match_percentage,

        "correct_matches": correct_matches,
        "accuracy_percentage": accuracy_percentage,

        "average_cer": average_cer
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_path / "extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"Total Images: {total_images}")
    print(f"Successful Extractions: {successful_extractions}")
    print(f"Correct Matches: {correct_matches}")
    print(f"Match %: {match_percentage:.2f}%")
    print(
        f"Accuracy %: {accuracy_percentage:.2f}%"
        if accuracy_percentage is not None
        else "Accuracy %: N/A"
    )
    print(
        f"Average CER: {average_cer:.4f}"
        if average_cer is not None
        else "Average CER: N/A"
    )
    print(f"(Total CER sum: {total_cer:.4f})")
    print("=" * 80)
    print(f"Metrics saved to: {output_path / 'metrics.json'}")
    print(f"Detailed results saved to: {output_path / 'extracted_data.json'}")

    return metrics, all_results


if __name__ == "__main__":
    process_batch_images(
        data_folder="archive/data",
        excel_path="archive/data/drug list.xlsx",
        num_images=50,
        conf=0.1,
        output_dir="output"
    )