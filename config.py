from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/" 

client = MongoClient(MONGO_URI)
db = client["aadhaar_fraud_db"]
users_collection = db["users"] 