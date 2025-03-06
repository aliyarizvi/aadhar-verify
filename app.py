import zipfile
from flask import Flask, request, jsonify
from bson.json_util import dumps
import os
import shutil
from config import users_collection
from utils import is_aadhar_card
from utils import extract_text
from utils import calculate_match_score
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_uploads_folder():
    try:
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            
            if item == "last_batch.txt":
                continue
                
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")
                
        print("Uploads folder cleaned successfully")
    except Exception as e:
        print(f"Error cleaning uploads folder: {str(e)}")

@app.route("/upload", methods=["POST"])
def upload_files():
    if "zip_file" not in request.files or "excel_file" not in request.files:
        return jsonify({"error": "ZIP file and Excel file are required"}), 400

    clean_uploads_folder()
    
    zip_file = request.files["zip_file"]
    excel_file = request.files["excel_file"]

    batch_id = str(uuid.uuid4())
    
    batch_folder = os.path.join(UPLOAD_FOLDER, batch_id)
    os.makedirs(batch_folder, exist_ok=True)

    zip_path = os.path.join(batch_folder, zip_file.filename)
    excel_path = os.path.join(batch_folder, excel_file.filename)

    zip_file.save(zip_path)
    excel_file.save(excel_path)

    with open(os.path.join(UPLOAD_FOLDER, "last_batch.txt"), "w") as f:
        f.write(batch_id)

    return jsonify({
        "message": "Files uploaded successfully", 
        "zip_path": zip_path, 
        "excel_path": excel_path
    })

@app.route("/process", methods=["POST"])
def process_data():
    try:
        data = request.json
        zip_path = data.get("zip_path")
        excel_path = data.get("excel_path")
        
        if not zip_path or not excel_path:
            return jsonify({"error": "Missing zip_path or excel_path"}), 400
            
        path_parts = zip_path.split(os.sep)
        if len(path_parts) >= 2:
            batch_id = path_parts[-2]
        else:
            return jsonify({"error": "Invalid file path format"}), 400

        extracted_images = extract_zip(zip_path, os.path.join(UPLOAD_FOLDER, batch_id))

        results = []
        bulk_insert = []

        for image in extracted_images:
            if is_aadhar_card(image):
                cropped_data = extract_text(image)
                
                if not all(key in cropped_data for key in ["name", "uid", "address"]):
                    print(f"Warning: Missing fields in cropped data for {image}")
                    continue
                
                score = calculate_match_score(cropped_data, excel_path)

                user_record = {
                    "name": cropped_data.get("name", ""),
                    "uid": cropped_data.get("uid", ""),
                    "address": cropped_data.get("address", ""),
                    "match_score": score,
                    "batch_id": batch_id,
                }

                bulk_insert.append(user_record)
                results.append({**user_record})

        if bulk_insert:
            inserted_ids = users_collection.insert_many(bulk_insert).inserted_ids
            print(f"Inserted {len(inserted_ids)} records into MongoDB for batch {batch_id}.")

        for record in results:
            if "_id" in record:
                record["_id"] = str(record["_id"])

        return jsonify({
            "message": "Processing complete", 
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

def extract_zip(zip_path, output_folder):
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                extracted_files.append(os.path.join(output_folder, file_name))
    return extracted_files

@app.route("/results", methods=["GET"])
def get_results():
    try:
        last_batch_path = os.path.join(UPLOAD_FOLDER, "last_batch.txt")
        if os.path.exists(last_batch_path):
            with open(last_batch_path, "r") as f:
                batch_id = f.read().strip()
                
            users = list(users_collection.find({"batch_id": batch_id}, {"_id": 0}))
            return jsonify({"results": users, "batch_id": batch_id})
        else:
            return jsonify({"error": "No recent batch found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error retrieving results: {str(e)}"}), 500

@app.route("/results/<batch_id>", methods=["GET"])
def get_results_by_batch(batch_id):
    users = list(users_collection.find({"batch_id": batch_id}, {"_id": 0}))
    return jsonify({"results": users, "batch_id": batch_id})

@app.route("/batches", methods=["GET"])
def get_batches():
    batches = users_collection.distinct("batch_id")
    return jsonify(batches)

@app.route("/all-results", methods=["GET"])
def get_all_results():
    users = list(users_collection.find({}, {"_id": 0}))
    return jsonify(users)

@app.route("/")
def home():
    return "Welcome to Aadhaar Fraud Detection API!"

if __name__ == "__main__":
    app.run(debug=True)