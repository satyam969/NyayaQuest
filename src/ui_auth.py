import streamlit as st
from .auth_utils import sign_in, sign_up

def render_login_signup():
    """Renders the login/signup toggle and forms."""
    st.markdown("### 🏛️ Welcome to NyayaQuest")
    
    auth_mode = st.radio("Choose Mode", ["Login", "Sign Up", "Continue as Guest"], horizontal=True)
    
    if auth_mode == "Continue as Guest":
        if st.button("Access as Guest"):
            st.session_state.user = None
            st.session_state.is_guest = True
            st.session_state.messages = []
            st.rerun()
        return

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if auth_mode == "Login":
        if st.button("Login"):
            res = sign_in(email, password)
            if res["success"]:
                st.session_state.user = res
                st.session_state.is_guest = False
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(res["error"])
    else:
        if st.button("Create Account"):
            res = sign_up(email, password)
            if res["success"]:
                st.success("Account created! Please login.")
            else:
                st.error(res["error"])
