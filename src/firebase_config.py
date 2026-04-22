import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Firebase Web Config (for Client-side Auth via REST)
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyBdLEEG8XptYZ-69SbhG84w0Xgh_f3Jwu4",
    "authDomain": "nyayaquest.firebaseapp.com",
    "projectId": "nyayaquest",
    "storageBucket": "nyayaquest.firebasestorage.app",
    "messagingSenderId": "686850588650",
    "appId": "1:686850588650:web:a38c4983b5fa3a0887bc93",
}

def initialize_firebase():
    """Initializes the Firebase Admin SDK for Firestore access."""
    if not firebase_admin._apps:
        cred = None

        # Option 1: JSON content directly in env var (Hugging Face Secrets)
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            try:
                service_account_info = json.loads(service_account_json)
                cred = credentials.Certificate(service_account_info)
                print("[Firebase] Initialized from FIREBASE_SERVICE_ACCOUNT_JSON env var.")
            except Exception as e:
                print(f"[Firebase] Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")

        # Option 2: Path to a JSON file (local development)
        if not cred:
            service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "firebase-service-account.json")
            if os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                print(f"[Firebase] Initialized from file: {service_account_path}")

        if cred:
            firebase_admin.initialize_app(cred)
        else:
            print("[Firebase] WARNING: No service account found. Firestore will be unavailable.")
            return None

    return firestore.client()

db = initialize_firebase()
