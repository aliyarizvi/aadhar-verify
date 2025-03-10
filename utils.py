import pandas as pd
import easyocr
import cv2
from ultralytics import YOLO
import os
import re
import string
from difflib import SequenceMatcher

classifier = YOLO(r"Classification_model/best.pt") 
detector = YOLO(r"Detection_model/best.pt")
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
        return False, 0
    
    # Normalize names
    input_name = normalize_text(input_name)
    extracted_name = normalize_text(extracted_name)
    
    # Rule 1: Exact match
    if input_name == extracted_name:
        return True, 100
    
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
        return True, 95
    
    # Rule 3: Ignoring middle names
    def check_without_middle(parts1, parts2):
        if len(parts1) < 2 or len(parts2) < 2:
            return False
        
        # Check if first and last names match
        return parts1[0] == parts2[0] and parts1[-1] == parts2[-1]
    
    if (len(input_parts) > 2 and check_without_middle(input_parts, extracted_parts)) or \
       (len(extracted_parts) > 2 and check_without_middle(extracted_parts, input_parts)):
        return True, 90
    
    # Rule 4: Matching any part of the name
    if len(input_parts) == 1 and input_parts[0] in extracted_parts:
        return True, 85
    if len(extracted_parts) == 1 and extracted_parts[0] in input_parts:
        return True, 85
    
    # Rule 5: Circular matching (all parts present but in different order)
    if sorted(input_parts) == sorted(extracted_parts):
        return True, 90
    
    # Check if one name is a subset of the other
    all_input_in_extracted = all(part in extracted_parts for part in input_parts)
    all_extracted_in_input = all(part in input_parts for part in extracted_parts)
    
    if all_input_in_extracted:
        return True, 80
    if all_extracted_in_input:
        return True, 80
    
    # Calculate partial similarity for non-matches
    similarity = SequenceMatcher(None, input_name, extracted_name).ratio() * 100
    if similarity >= 70:
        return True, similarity
        
    return False, similarity

def uid_match(db_uid, extracted_uid):
    """
    Calculate UID match score
    Returns match result (boolean) and score (0-100)
    """
    if not db_uid or not extracted_uid:
        return False, 0
    
    # Normalize UIDs by removing spaces
    db_uid = db_uid.replace(" ", "")
    extracted_uid = extracted_uid.replace(" ", "")
    
    # Exact match
    if db_uid == extracted_uid:
        return True, 100
    
    # Calculate similarity for partial matches
    similarity = SequenceMatcher(None, db_uid, extracted_uid).ratio() * 100
    
    # High similarity threshold for UIDs
    if similarity >= 90:
        return True, similarity
    
    return False, similarity

def construct_address_from_excel(row):
    """Construct address string from the excel row with multiple address fields"""
    address_parts = []
    
    # Add house/flat number if present
    if pd.notna(row.get("House Flat Number")):
        address_parts.append(str(row["House Flat Number"]))
    
    # Add floor number if present
    if pd.notna(row.get(" Floor Number")):
        address_parts.append(str(row[" Floor Number"]))
    
    # Add premise/building name if present
    if pd.notna(row.get("Premise Building Name")):
        address_parts.append(str(row["Premise Building Name"]))
    
    # Add landmark if present
    if pd.notna(row.get("Landmark")):
        address_parts.append(str(row["Landmark"]))
    
    # Add street/road name if present
    if pd.notna(row.get("Street Road Name")):
        address_parts.append(str(row["Street Road Name"]))
    
    # Add town if present
    if pd.notna(row.get("Town")):
        address_parts.append(str(row["Town"]))
    
    # Add city if present
    if pd.notna(row.get("City")):
        address_parts.append(str(row["City"]))
    
    # Add state if present
    if pd.notna(row.get("State")):
        address_parts.append(str(row["State"]))
    
    # Add country if present
    if pd.notna(row.get("Country")):
        address_parts.append(str(row["Country"]))
    
    # Add pincode if present
    if pd.notna(row.get("PINCODE")):
        address_parts.append(str(row["PINCODE"]))
    
    return ", ".join(address_parts)

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

def address_match(input_address, extracted_address):
    """
    Implement the address matching logic:
    1. Normalization
    2. Pincode matching
    3. Field-specific matching
    4. Final address match score
    """
    if not input_address or not extracted_address:
        return False, 0
    
    # Extract pincodes
    input_pincode = extract_pincode(input_address)
    extracted_pincode = extract_pincode(extracted_address)
    
    # Pincode matching (100 if match, 0 if not)
    pincode_score = 100 if input_pincode and extracted_pincode and input_pincode == extracted_pincode else 0
    
    # Normalize addresses
    norm_input = normalize_address(input_address)
    norm_extracted = normalize_address(extracted_address)
    
    # Calculate overall string similarity
    similarity_score = SequenceMatcher(None, norm_input, norm_extracted).ratio() * 100
    
    # Split addresses into parts for field-specific matching
    input_parts = norm_input.split()
    extracted_parts = norm_extracted.split()
    
    # Check if significant parts of input address are in extracted address
    parts_score = 0
    significant_parts = [part for part in input_parts if len(part) > 3]  # Consider words longer than 3 chars as significant
    
    if significant_parts:
        matches = sum(1 for part in significant_parts if part in extracted_parts)
        parts_score = (matches / len(significant_parts)) * 100
    
    # Calculate final address score with weighting
    if pincode_score > 0:
        # If pincode matches, give it high weight
        final_score = (0.4 * pincode_score) + (0.4 * similarity_score) + (0.2 * parts_score)
    else:
        # If no pincode or pincode mismatch, rely more on content matching
        final_score = (0.6 * similarity_score) + (0.4 * parts_score)
    
    # Return match result and score
    return final_score >= 70, final_score

def calculate_match_score(extracted_data, excel_file):
    """
    Calculate match scores for all fields and return a dictionary of scores
    """
    try:
        # Initialize default scores
        match_scores = {
            "name_score": 0,
            "address_score": 0,
            "uid_score": 0,
            "overall_score": 0
        }
        
        if not all(key in extracted_data for key in ["name", "uid", "address"]):
            print("Missing required fields in extracted data")
            return match_scores
            
        if not os.path.exists(excel_file):
            print(f"Excel file not found: {excel_file}")
            return match_scores
            
        df = pd.read_excel(excel_file)
        
        # Check for required columns (based on your actual Excel structure)
        required_columns = ["Name", "UID"]
        if not all(col in df.columns for col in required_columns):
            print(f"Excel file is missing required columns. Available columns: {df.columns.tolist()}")
            return match_scores
            
        if not extracted_data["uid"]:
            print("Extracted UID is empty")
            return match_scores
        
        # Normalize UID by removing spaces for comparison
        extracted_uid = extracted_data["uid"].replace(" ", "")
            
        best_match_scores = match_scores.copy()
        best_match_found = False
            
        for index, row in df.iterrows():
            db_uid = str(row["UID"]).replace(" ", "")
            
            # Calculate UID match score
            uid_matched, uid_score = uid_match(db_uid, extracted_uid)
            
            # If UID doesn't match at all, skip to next record
            if uid_score < 80: 
                continue
                
            # Construct full address from multiple fields in Excel
            db_address = construct_address_from_excel(row)
            
            # Apply name and address matching logic
            name_matched, name_score = name_match(row["Name"], extracted_data["name"])
            addr_matched, addr_score = address_match(db_address, extracted_data["address"])
        
            # Give higher weight to UID and name matches since they're more reliable
            overall_score = (0.4 * uid_score) + (0.4 * name_score) + (0.2 * addr_score)
            
            # Log the match details
            print(f"Match results for UID {extracted_uid}:")
            print(f"  DB UID: '{db_uid}' vs Extracted: '{extracted_uid}' (Score: {uid_score:.1f})")
            print(f"  DB Name: '{row['Name']}' vs Extracted: '{extracted_data['name']}' (Score: {name_score:.1f})")
            print(f"  DB Address: '{db_address}'")
            print(f"  Extracted Address: '{extracted_data['address']}' (Score: {addr_score:.1f})")
            print(f"  Overall score: {overall_score:.1f}")
            
            # Update best match if this is better
            if overall_score > best_match_scores["overall_score"]:
                best_match_scores = {
                    "name_score": round(name_score, 1),
                    "address_score": round(addr_score, 1),
                    "uid_score": round(uid_score, 1),
                    "overall_score": round(overall_score, 1)
                }
                best_match_found = True
        
        if best_match_found:
            return best_match_scores
        else:
            print(f"No matching UID found in excel: {extracted_uid}")
            return match_scores
            
    except Exception as e:
        print(f"Error in calculate_match_score: {str(e)}")
        import traceback
        traceback.print_exc()
        return match_scores