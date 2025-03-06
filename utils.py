import pandas as pd
import easyocr
import cv2
from ultralytics import YOLO
import os
import re
import string
from difflib import SequenceMatcher

classifier = YOLO(r"Classification_model\best.pt") 
detector = YOLO(r"Detection_model\best.pt")
reader = easyocr.Reader(['en'])

# Common address terms to ignore
ADDRESS_TERMS_TO_IGNORE = [
    "road", "street", "lane", "marg", "nagar", "colony", "township", 
    "apartment", "flat", "sector", "block", "phase", "district", "area",
    "near", "behind", "opposite", "beside", "next to", "across from"
]

def is_aadhar_card(image_path):
    try:
        results = classifier(image_path)
        for result in results:
            probs = result.probs
            aadhar = float(probs.data[0])
            if aadhar >= 0.8:
                return True
        return False
    except Exception as e:
        print(f"Error in is_aadhar_card: {str(e)}")
        return False

def detect_fields(image_path):
    if is_aadhar_card(image_path):
        try:
            results = detector(image_path)
            return results
        except Exception as e:
            print(f"Error in detect_fields: {str(e)}")
            return None
    else:
        return None

def extract_text(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return {"name": "", "uid": "", "address": ""}
            
        results = detect_fields(image_path)

        if results is None or len(results) == 0:
            print(f"No fields detected in image: {image_path}")
            return {"name": "", "uid": "", "address": ""}
            
        extracted_data = {"name": "", "uid": "", "address": ""}
        
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, result[:6])
            field_class = detector.names[class_id]

            cropped_roi = image[y1:y2, x1:x2]
            if cropped_roi.size == 0:
                print(f"Empty ROI for {field_class} in {image_path}")
                continue

            gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
            
            text = reader.readtext(gray_roi, detail=0)
            if text:
                extracted_data[field_class] = ' '.join(text)  

        return extracted_data
    except Exception as e:
        print(f"Error in extract_text: {str(e)}")
        return {"name": "", "uid": "", "address": ""}

def normalize_text(text):
    """Normalize text by removing punctuation, extra spaces, and converting to lowercase"""
    if not text:
        return ""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def name_match(input_name, extracted_name):
    """
    Implement the name matching logic based on the rules:
    1. Exact letter match
    2. Abbreviated names
    3. Ignoring middle names
    4. Matching any part of the name
    5. Circular matching
    6. Single-letter abbreviation
    """
    if not input_name or not extracted_name:
        return False
    
    # Normalize names
    input_name = normalize_text(input_name)
    extracted_name = normalize_text(extracted_name)
    
    # Rule 1: Exact match
    if input_name == extracted_name:
        return True
    
    # Split names into parts
    input_parts = input_name.split()
    extracted_parts = extracted_name.split()
    
    # Rule 2 & 6: Abbreviated names and single-letter abbreviation
    def check_abbreviation(parts1, parts2):
        if len(parts1) != len(parts2):
            return False
        
        for i, (p1, p2) in enumerate(zip(parts1, parts2)):
            # Check if one part is an abbreviation of the other
            if len(p1) == 1 and p2.startswith(p1):
                continue
            elif len(p2) == 1 and p1.startswith(p2):
                continue
            elif p1 != p2:
                return False
        return True
    
    if check_abbreviation(input_parts, extracted_parts) or check_abbreviation(extracted_parts, input_parts):
        return True
    
    # Rule 3: Ignoring middle names
    def check_without_middle(parts1, parts2):
        if len(parts1) < 2 or len(parts2) < 2:
            return False
        
        # Check if first and last names match
        return parts1[0] == parts2[0] and parts1[-1] == parts2[-1]
    
    if (len(input_parts) > 2 and check_without_middle(input_parts, extracted_parts)) or \
       (len(extracted_parts) > 2 and check_without_middle(extracted_parts, input_parts)):
        return True
    
    # Rule 4: Matching any part of the name
    if len(input_parts) == 1 and input_parts[0] in extracted_parts:
        return True
    if len(extracted_parts) == 1 and extracted_parts[0] in input_parts:
        return True
    
    # Rule 5: Circular matching (all parts present but in different order)
    if sorted(input_parts) == sorted(extracted_parts):
        return True
    
    # Check if one name is a subset of the other
    if all(part in extracted_parts for part in input_parts) or all(part in input_parts for part in extracted_parts):
        return True
    
    return False

def extract_pincode(address):
    """Extract 6-digit pincode from an address string"""
    if not address:
        return ""
    
    # Look for 6 consecutive digits
    pincode_match = re.search(r'(\d{6})', address.replace(" ", ""))
    if pincode_match:
        return pincode_match.group(1)
    return ""

def normalize_address(address):
    """Normalize address by removing common terms, punctuation, etc."""
    if not address:
        return ""
    
    # Convert to lowercase
    address = address.lower()
    
    # Remove punctuation
    address = address.translate(str.maketrans('', '', string.punctuation))
    
    # Replace multiple spaces with single space
    address = re.sub(r'\s+', ' ', address).strip()
    
    # Remove common address terms
    words = address.split()
    filtered_words = [word for word in words if word.lower() not in ADDRESS_TERMS_TO_IGNORE]
    
    return ' '.join(filtered_words)

def similarity_ratio(str1, str2):
    """Calculate string similarity ratio using SequenceMatcher"""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio() * 100

def address_match(input_address, extracted_address):
    """
    Implement the address matching logic:
    1. Normalization
    2. Pincode matching
    3. Field-specific matching
    4. Final address match score
    """
    if not input_address or not extracted_address:
        return False
    
    # Extract pincodes
    input_pincode = extract_pincode(input_address)
    extracted_pincode = extract_pincode(extracted_address)
    
    # Pincode matching (100 if match, 0 if not)
    pincode_score = 100 if input_pincode and extracted_pincode and input_pincode == extracted_pincode else 0
    
    # Normalize addresses
    norm_input = normalize_address(input_address)
    norm_extracted = normalize_address(extracted_address)
    
    # Calculate overall string similarity
    similarity_score = similarity_ratio(norm_input, norm_extracted)
    
    # Split addresses into parts for field-specific matching
    input_parts = norm_input.split()
    extracted_parts = norm_extracted.split()
    
    # Check if significant parts of input address are in extracted address
    parts_score = 0
    significant_parts = [part for part in input_parts if len(part) > 3]  # Consider words longer than 3 chars as significant
    
    if significant_parts:
        matches = sum(1 for part in significant_parts if part in extracted_parts)
        parts_score = (matches / len(significant_parts)) * 100
    
    # Calculate final address score
    if pincode_score > 0:
        # If pincode matches, give it high weight
        final_score = (0.4 * pincode_score) + (0.4 * similarity_score) + (0.2 * parts_score)
    else:
        # If no pincode or pincode mismatch, rely more on content matching
        final_score = (0.6 * similarity_score) + (0.4 * parts_score)
    
    # Return True if score is above threshold
    return final_score >= 70

def calculate_match_score(extracted_data, excel_file):
    try:
        if not all(key in extracted_data for key in ["name", "uid", "address"]):
            print("Missing required fields in extracted data")
            return 0
            
        if not os.path.exists(excel_file):
            print(f"Excel file not found: {excel_file}")
            return 0
            
        df = pd.read_excel(excel_file)
        
        required_columns = ["UID", "Name", "Address"]
        if not all(col in df.columns for col in required_columns):
            print(f"Excel file is missing required columns. Available columns: {df.columns.tolist()}")
            return 0
            
        if not extracted_data["uid"]:
            print("Extracted UID is empty")
            return 0
        
        # Normalize UID by removing spaces for comparison
        extracted_uid = extracted_data["uid"].replace(" ", "")
            
        for index, row in df.iterrows():
            db_uid = str(row["UID"]).replace(" ", "")
            
            # Check if UIDs match
            if db_uid == extracted_uid:
                # Apply the improved name and address matching logic
                name_matched = name_match(row["Name"], extracted_data["name"])
                address_matched = address_match(row["Address"], extracted_data["address"])
                
                # Calculate scores based on matches
                name_score = 100 if name_matched else 0
                address_score = 100 if address_matched else 0
                
                # Combined match score
                overall_score = (name_score + address_score) / 2
                
                print(f"Match results for UID {extracted_uid}:")
                print(f"  Name match: {name_matched} ({name_score})")
                print(f"  Address match: {address_matched} ({address_score})")
                print(f"  Overall score: {overall_score}")
                
                return overall_score
                
        print(f"No matching UID found in excel: {extracted_uid}")
        return 0
    except Exception as e:
        print(f"Error in calculate_match_score: {str(e)}")
        return 0