import requests
import streamlit as st
from .firebase_config import FIREBASE_CONFIG

# Firebase Auth REST API endpoints
AUTH_SIGNUP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + FIREBASE_CONFIG["apiKey"]
AUTH_SIGNIN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=" + FIREBASE_CONFIG["apiKey"]

def sign_up(email, password):
    """Signs up a new user using Firebase REST API."""
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(AUTH_SIGNUP_URL, json=payload)
    data = response.json()
    
    if response.status_code == 200:
        return {"success": True, "user_id": data["localId"], "email": data["email"]}
    else:
        error_msg = data.get("error", {}).get("message", "Signup failed")
        return {"success": False, "error": error_msg}

def sign_in(email, password):
    """Signs in an existing user using Firebase REST API."""
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(AUTH_SIGNIN_URL, json=payload)
    data = response.json()
    
    if response.status_code == 200:
        return {"success": True, "user_id": data["localId"], "email": data["email"]}
    else:
        error_msg = data.get("error", {}).get("message", "Signin failed")
        return {"success": False, "error": error_msg}

def initialize_session_state():
    """Initializes session state variables for Auth."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "is_guest" not in st.session_state:
        st.session_state.is_guest = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
