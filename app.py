import os
import uuid
import datetime
import streamlit as st
import extra_streamlit_components as stx
from dotenv import load_dotenv

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NyayaQuest", page_icon="⚖️", layout="wide")

# --- 2. IMPORTS & BACKEND SETUP ---
from nyayaquest_main import NyayaQuest
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from hybrid_retriever import HybridRetriever

from src.auth_utils import initialize_session_state
from src.ui_auth import render_login_signup
from src.db_utils import get_user_conversations, create_conversation, get_conversation_history

# Initialize Environment and Session
load_dotenv()
initialize_session_state()
cookie_manager = stx.CookieManager()

# Custom Embedding class
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# --- 3. CUSTOM CSS ---
def add_custom_css():
    custom_css = """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .st-chat-input { border-radius: 15px; padding: 10px; border: 1px solid #ddd; }
        .stButton > button { border-radius: 20px; transition: 0.3s; }
        .main .block-container { padding-top: 2rem; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()

# --- 4. AUTHENTICATION GATEWAY ---
if not st.session_state.user and not st.session_state.is_guest:
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

if st.session_state.user and not cookie_manager.get("nyayaquest_auth_session"):
    auth_payload = {"user_id": st.session_state.user["user_id"], "email": st.session_state.user["email"]}
    cookie_manager.set(
        "nyayaquest_auth_session", 
        auth_payload, 
        expires_at=datetime.datetime.now() + datetime.timedelta(days=7)
    )

# --- 5. INITIALIZE AI ENGINE ---
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, groq_api_key=os.getenv('GROQ_API_KEY'))
embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory="chroma_db_groq_legal", embedding_function=embeddings, collection_name="legal_knowledge")

hybrid_retriever = HybridRetriever.from_vector_store(vector_store, k=20, vector_weight=0.5, bm25_weight=0.5)
law = NyayaQuest(llm, embeddings, vector_store, hybrid_retriever=hybrid_retriever)

# --- 6. HEADER ---
st.title("NyayaQuest - Legal Assistant ⚖️")

# --- 7. SIDEBAR ---
st.sidebar.title("💬 Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.thread_id = create_conversation(st.session_state.user["user_id"]) if st.session_state.user else f"guest_{uuid.uuid4()}"
    st.session_state.messages = []
    st.rerun()

if st.session_state.user:
    convos = get_user_conversations(st.session_state.user["user_id"])
    for convo in convos:
        if st.sidebar.button(f"📄 {convo['title']}", key=convo['id'], use_container_width=True):
            st.session_state.thread_id = convo['id']
            history_docs = get_conversation_history(convo['id'])
            st.session_state.messages = []
            for m in history_docs:
                msg_entry = {"role": m["role"], "content": m["content"]}
                if "metadata" in m and m.get("metadata") and "context" in m["metadata"]:
                    msg_entry["context"] = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in m["metadata"]["context"]]
                st.session_state.messages.append(msg_entry)
            
            if hasattr(law.cache, 'set_chat_history'):
                law.cache.set_chat_history(convo['id'], st.session_state.messages)
            st.rerun()

st.sidebar.divider()

if st.session_state.user:
    if st.sidebar.button("Logout", use_container_width=True):
        cookie_manager.delete("nyayaquest_auth_session")
        st.session_state.user = None
        st.session_state.is_guest = False
        st.rerun()
elif st.session_state.is_guest:
    if st.sidebar.button("🔐 Exit Guest Mode", use_container_width=True):
        st.session_state.is_guest = False
        st.rerun()

with st.sidebar:
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** NyayaQuest provides informational insights, not formal legal advice.")
    st.caption("**© 2026 NIT Jamshedpur**")

# --- 8. CHAT UI ---
if not st.session_state.thread_id:
    st.session_state.thread_id = create_conversation(st.session_state.user["user_id"]) if st.session_state.user else f"guest_{uuid.uuid4()}"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("📚 Verified Statutory Sources"):
                for idx, doc in enumerate(message["context"]):
                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('law_code', '')} {doc.metadata.get('section_number', '')}")
                    st.info(doc.page_content)

prompt = st.chat_input("Ask a legal question...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Analyzing legal texts..."):
        result, context, _ = law.conversational(prompt, st.session_state.thread_id)
        st.session_state.messages.append({"role": "assistant", "content": result, "context": context})
    st.rerun()