import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
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
        # Check for service account JSON
        service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "firebase-service-account.json")
        
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            return firestore.client()
        else:
            # Fallback for local development or if JSON is not yet provided
            # Note: This will only work if the user has logged into GCP locally
            try:
                firebase_admin.initialize_app()
                return firestore.client()
            except Exception:
                return None
    
    return firestore.client()

db = initialize_firebase()
