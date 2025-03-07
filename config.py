import os
from dotenv import load_dotenv
from pymongo import MongoClient
import urllib.parse

load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "aadhaar_fraud_db")

encoded_username = urllib.parse.quote_plus(MONGO_USERNAME)
encoded_password = urllib.parse.quote_plus(MONGO_PASSWORD)

MONGO_URI = f"mongodb+srv://{encoded_username}:{encoded_password}@{MONGO_CLUSTER}/{MONGO_DB_NAME}?retryWrites=true&w=majority"

client = None
db = None
users_collection = None

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Validate connection
    db = client[MONGO_DB_NAME]
    users_collection = db["users"]
    print("Connected to MongoDB Atlas")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {e}")
    # Create a fallback collection to prevent import errors
    client = MongoClient("mongodb://localhost:27017/")
    db = client["aadhaar_fraud_db_local"]
    users_collection = db["users"]
    print("Using local MongoDB fallback")