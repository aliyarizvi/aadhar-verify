import zipfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from bson.json_util import dumps
import os
import shutil
from config import users_collection
from utils import is_aadhar_card
from utils import extract_text
from utils import calculate_match_score
import uuid
import ntpath

app = Flask(__name__)

# Set up static folder structure
UPLOAD_FOLDER = "/tmp/uploads"
STATIC_FOLDER = "static"
TEMPLATE_FOLDER = "templates"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Copy HTML files to templates folder if they don't exist yet
def setup_template_files():
    html_files = ["index.html", "upload.html", "analytics.html"]
    for file in html_files:
        if os.path.exists(file) and not os.path.exists(os.path.join(TEMPLATE_FOLDER, file)):
            shutil.copy(file, TEMPLATE_FOLDER)
            print(f"Copied {file} to templates folder")

# Call setup on startup
setup_template_files()

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

def get_filename(file_path):
    """Extract filename from path"""
    return ntpath.basename(file_path)

# Routes for serving HTML pages
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload.html")
def upload_page():
    return render_template("upload.html")

@app.route("/analytics.html")
def analytics_page():
    return render_template("analytics.html")

# API Endpoints
@app.route("/api/upload", methods=["POST"])
def upload_files():
    if "zip_file" not in request.files or "excel_file" not in request.files:
        return jsonify({"error": "ZIP file and Excel file are required"}), 400
    
    # Clear the database before processing a new upload
    users_collection.delete_many({})
    print("Database cleared before new upload")    

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
        "excel_path": excel_path,
        "batch_id": batch_id
    })

@app.route("/api/process", methods=["POST"])
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
        non_aadhaar_files = []

        for image in extracted_images:
            filename = get_filename(image)
            
            if is_aadhar_card(image):
                # Process Aadhaar card
                cropped_data = extract_text(image)
                
                # Calculate match scores (name, address, uid, overall)
                match_scores = calculate_match_score(cropped_data, excel_path)
                
                user_record = {
                    "name": cropped_data.get("name", ""),
                    "uid": cropped_data.get("uid", ""),
                    "address": cropped_data.get("address", ""),
                    "filename": filename,
                    "is_aadhaar": True,
                    "name_score": match_scores.get("name_score", 0),
                    "address_score": match_scores.get("address_score", 0),
                    "uid_score": match_scores.get("uid_score", 0),
                    "overall_score": match_scores.get("overall_score", 0),
                    "batch_id": batch_id,
                }

                bulk_insert.append(user_record)
                results.append({**user_record})
            else:
                # Handle non-Aadhaar files
                non_aadhaar_record = {
                    "name": "NA",
                    "uid": "NA",
                    "address": "NA",
                    "filename": filename,
                    "is_aadhaar": False,
                    "name_score": 0,
                    "address_score": 0,
                    "uid_score": 0,
                    "overall_score": 0,
                    "batch_id": batch_id,
                }
                
                bulk_insert.append(non_aadhaar_record)
                results.append({**non_aadhaar_record})
                non_aadhaar_files.append(filename)

        if bulk_insert:
            inserted_ids = users_collection.insert_many(bulk_insert).inserted_ids
            print(f"Inserted {len(inserted_ids)} records into MongoDB for batch {batch_id}.")

        for record in results:
            if "_id" in record:
                record["_id"] = str(record["_id"])

        return jsonify({
            "message": "Processing complete", 
            "results": results,
            "aadhaar_count": len(results) - len(non_aadhaar_files),
            "non_aadhaar_count": len(non_aadhaar_files),
            "non_aadhaar_files": non_aadhaar_files,
            "total_count": len(results)
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

@app.route("/api/results", methods=["GET"])
def get_results():
    try:
        last_batch_path = os.path.join(UPLOAD_FOLDER, "last_batch.txt")
        if os.path.exists(last_batch_path):
            with open(last_batch_path, "r") as f:
                batch_id = f.read().strip()
                
            users = list(users_collection.find({"batch_id": batch_id}))
            
            # Convert ObjectId to string for JSON serialization
            for user in users:
                if "_id" in user:
                    user["_id"] = str(user["_id"])
            
            # Count statistics
            aadhaar_count = sum(1 for user in users if user.get("is_aadhaar", False))
            non_aadhaar_count = sum(1 for user in users if not user.get("is_aadhaar", True))
            non_aadhaar_files = [user.get("filename") for user in users if not user.get("is_aadhaar", True)]
            
            return jsonify({
                "results": users, 
                "batch_id": batch_id,
                "aadhaar_count": aadhaar_count,
                "non_aadhaar_count": non_aadhaar_count,
                "non_aadhaar_files": non_aadhaar_files,
                "total_count": len(users)
            })
        else:
            return jsonify({"error": "No recent batch found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error retrieving results: {str(e)}"}), 500

@app.route("/api/results/<batch_id>", methods=["GET"])
def get_results_by_batch(batch_id):
    users = list(users_collection.find({"batch_id": batch_id}))
    
    # Convert ObjectId to string for JSON serialization
    for user in users:
        if "_id" in user:
            user["_id"] = str(user["_id"])
    
    # Count statistics
    aadhaar_count = sum(1 for user in users if user.get("is_aadhaar", False))
    non_aadhaar_count = sum(1 for user in users if not user.get("is_aadhaar", True))
    non_aadhaar_files = [user.get("filename") for user in users if not user.get("is_aadhaar", True)]
    
    return jsonify({
        "results": users, 
        "batch_id": batch_id,
        "aadhaar_count": aadhaar_count,
        "non_aadhaar_count": non_aadhaar_count,
        "non_aadhaar_files": non_aadhaar_files,
        "total_count": len(users)
    })

@app.route("/api/batches", methods=["GET"])
def get_batches():
    batches = users_collection.distinct("batch_id")
    return jsonify(batches)

@app.route("/api/all-results", methods=["GET"])
def get_all_results():
    users = list(users_collection.find({}))
    
    # Convert ObjectId to string for JSON serialization
    for user in users:
        if "_id" in user:
            user["_id"] = str(user["_id"])
            
    return jsonify(users)

if __name__ == "__main__":
    app.run(debug=True)