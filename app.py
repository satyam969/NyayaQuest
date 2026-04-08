import os
import streamlit as st
import base64
import uuid
import datetime
from dotenv import load_dotenv

# Set page configuration - MUST BE FIRST Streamlit command
st.set_page_config(page_title="NyayaQuest", page_icon="logo/logo.png", layout="wide")


from nyayaquest_main import NyayaQuest
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer

from src.auth_utils import initialize_session_state
from src.ui_auth import render_login_signup
from src.db_utils import get_user_conversations, create_conversation, get_conversation_history, db
import extra_streamlit_components as stx

# Initialize Cookie Manager
cookie_manager = stx.CookieManager()


# Custom Embedding class
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

load_dotenv()

# Initialize session state for Auth and Chats
initialize_session_state()

# Custom CSS
def add_custom_css():
    custom_css = """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .st-chat-input { border-radius: 15px; padding: 10px; border: 1px solid #ddd; }
        .stButton > button { border-radius: 20px; transition: 0.3s; }
        .chat-input-container { position: fixed; bottom: 0; width: 100%; background-color: #f0f0f0; padding: 20px; display: flex; gap: 10px; }
        .st-title { font-weight: bold; color: #333; display: flex; align-items: center; gap: 15px; margin-top: 10px; }
        .logo { width: 40px; height: 30px; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()

# Auth Gateway
if not st.session_state.user and not st.session_state.is_guest:
    # Try to recover session from cookie
    auth_data = cookie_manager.get("nyayaquest_auth_session")
    
    if auth_data and isinstance(auth_data, dict):
        st.session_state.user = {
            "success": True, 
            "user_id": auth_data.get("user_id"), 
            "email": auth_data.get("email")
        }
        st.rerun()
    else:
        render_login_signup()
        st.stop()

# Set cookie if just logged in
if st.session_state.user and not cookie_manager.get("nyayaquest_auth_session"):
    auth_payload = {
        "user_id": st.session_state.user["user_id"],
        "email": st.session_state.user["email"]
    }
    cookie_manager.set(
        "nyayaquest_auth_session", 
        auth_payload, 
        expires_at=datetime.datetime.now() + datetime.timedelta(days=7)
    )



# --- Application Main Logic ---

# --- Backend Engine ---
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, groq_api_key=groq_api_key)
embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory="chroma_db_groq_legal", embedding_function=embeddings, collection_name="legal_knowledge")

from hybrid_retriever import HybridRetriever
from langchain.schema import Document
hybrid_retriever = HybridRetriever.from_vector_store(vector_store, k=20, vector_weight=0.5, bm25_weight=0.5)
law = NyayaQuest(llm, embeddings, vector_store, hybrid_retriever=hybrid_retriever)

# Title with Logo
logo_path = "logo/logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(f'<div class="st-title"><img src="data:image/png;base64,{encoded_image}" class="logo"> <span>NyayaQuest - Legal Assistant 🏛️</span></div>', unsafe_allow_html=True)
else:
    st.title("NyayaQuest - Legal Assistant 🏛️")

# Sidebar: History & Actions
st.sidebar.title("💬 Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    if st.session_state.user:
        new_id = create_conversation(st.session_state.user["user_id"])
        st.session_state.thread_id = new_id
    else:
        st.session_state.thread_id = f"guest_{uuid.uuid4()}"
    st.session_state.messages = []
    st.rerun()

# Load User Conversations from Firestore
if st.session_state.user:
    convos = get_user_conversations(st.session_state.user["user_id"])
    for convo in convos:
        if st.sidebar.button(f"📄 {convo['title']}", key=convo['id'], use_container_width=True):
            st.session_state.thread_id = convo['id']
            # Load messages for this thread
            history_docs = get_conversation_history(convo['id'])
            st.session_state.messages = []
            for m in history_docs:
                msg_entry = {"role": m["role"], "content": m["content"]}
                if "metadata" in m and m["metadata"] and "context" in m["metadata"]:
                    from langchain.schema import Document
                    msg_entry["context"] = [
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in m["metadata"]["context"]
                    ]
                st.session_state.messages.append(msg_entry)
            
            # Sync Backend History
            if hasattr(law.cache, 'set_chat_history'):
                law.cache.set_chat_history(convo['id'], st.session_state.messages)
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("Logout"):
    cookie_manager.delete("nyayaquest_auth_session")
    st.session_state.user = None
    st.session_state.is_guest = False
    st.rerun()


# --- Chat UI ---

# Fallback: Initialize thread if empty
if not st.session_state.thread_id:
    if st.session_state.user:
        st.session_state.thread_id = create_conversation(st.session_state.user["user_id"])
    else:
        st.session_state.thread_id = f"guest_{uuid.uuid4()}"

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("📚 Verified Statutory Sources"):
                for idx, doc in enumerate(message["context"]):
                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('law_code', '')} {doc.metadata.get('section_number', '')}")
                    st.info(doc.page_content)

# Input
prompt = st.chat_input("Ask a legal question...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        result, context, updated_history = law.conversational(prompt, st.session_state.thread_id)
        
        # Add to local state
        st.session_state.messages.append({"role": "assistant", "content": result, "context": context})
        
    st.rerun()
