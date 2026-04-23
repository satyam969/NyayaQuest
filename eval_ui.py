import os
import json
import re
import random
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# --- IMPORTS FROM YOUR BACKEND ---
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from hybrid_retriever import HybridRetriever

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NyayaQuest Evaluator", page_icon="📊", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .pass-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 10px; }
    .fail-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 10px; }
    .metric-container { display: flex; justify-content: space-between; background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND INITIALIZATION (Cached for performance) ---
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

@st.cache_resource
def load_retriever():
    load_dotenv()
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal", 
        embedding_function=embeddings, 
        collection_name="legal_knowledge"
    )
    # Using k=10 to check if the correct section is in the top 10 results
    retriever = HybridRetriever.from_vector_store(vector_store, k=10, vector_weight=0.5, bm25_weight=0.5)
    return retriever

with st.spinner("Loading Vector Database and Retriever..."):
    retriever = load_retriever()

# --- HELPER FUNCTIONS ---
def flatten_json(data):
    """Recursively flattens deeply nested lists of JSON objects."""
    flat_list = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                flat_list.extend(flatten_json(item))
            else:
                flat_list.append(item)
    else:
        flat_list.append(data)
    return flat_list

def get_base_section(sec_str):
    """
    Extracts ONLY the first continuous block of digits.
    Stops immediately when any non-digit character appears.
    Example: 'Section 191(1)' -> '191'
    Example: 'Section 86(a), (b)' -> '86'
    Example: '498A' -> '498'
    """
    # \d+ means "one or more digits". It stops at the first non-digit.
    match = re.search(r'(\d+)', str(sec_str))
    if match:
        return match.group(1)
    
    # Fallback if no numbers are found at all
    return str(sec_str).lower().replace("section", "").replace(" ", "")

def evaluate_query(query, expected_sections):
    """Runs a query and checks if the base expected sections are in the retrieved metadata."""
    if hasattr(retriever, 'invoke'):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)
        
    retrieved_sections = [str(doc.metadata.get("section_number", "Unknown")) for doc in docs]
    
    # Extract ONLY the base section numbers for strict digit comparison
    base_expected = [get_base_section(e) for e in expected_sections]
    base_retrieved = [get_base_section(r) for r in retrieved_sections]
    
    passed = False
    for exp in base_expected:
        # Check if our base expected number is anywhere in the retrieved list
        if exp in base_retrieved: 
            passed = True
            break
            
    return passed, retrieved_sections, docs

# --- UI: HEADER ---
st.title("📊 RAG Retrieval Accuracy Evaluator")
st.markdown("Test if the `HybridRetriever` successfully fetches the correct legal chunks based on your test datasets.")

# --- UI: SIDEBAR CONFIGURATION ---
st.sidebar.header("📁 Test Data Configuration")

test_folder = st.sidebar.text_input("Test Data Folder Path", value="data/test_data")

if not os.path.exists(test_folder):
    st.sidebar.warning(f"Folder '{test_folder}' not found. Please create it and add your JSON files.")
    st.stop()

json_files = [f for f in os.listdir(test_folder) if f.endswith('.json')]

if not json_files:
    st.sidebar.warning("No JSON files found in the specified folder.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Test File", json_files)
file_path = os.path.join(test_folder, selected_file)

try:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # Process the data through our flattener to merge all blocks
    test_data = flatten_json(raw_data)
    
except Exception as e:
    st.error(f"Error loading JSON: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(test_data)} test cases.")

# --- UI: EVALUATION EXECUTION ---
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "current_file" not in st.session_state or st.session_state.current_file != selected_file:
    st.session_state.eval_results = None
    st.session_state.current_file = selected_file

# Control Buttons
col_btn1, col_btn2 = st.columns(2)
run_all = col_btn1.button("🚀 Run Full Evaluation", type="primary")
run_random = col_btn2.button("🎲 Run Random 100", type="secondary")

if run_all or run_random:
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Determine the subset of data to run
    if run_random:
        sample_size = min(100, len(test_data))
        eval_data = random.sample(test_data, sample_size)
        st.info(f"Running evaluation on {sample_size} randomly selected queries...")
    else:
        eval_data = test_data
        
    passed_count = 0
    total_count = len(eval_data)
    
    for i, item in enumerate(eval_data):
        query = item.get("query", "")
        expected = item.get("expected_sections", [])
        
        status_text.text(f"Evaluating {i+1}/{total_count}: {query[:50]}...")
        
        passed, retrieved, docs = evaluate_query(query, expected)
        
        if passed:
            passed_count += 1
            
        results.append({
            "Query": query,
            "Expected": ", ".join(expected),
            "Retrieved": ", ".join(list(set(retrieved))), 
            "Passed": passed,
            "Raw Docs": docs
        })
        
        progress_bar.progress((i + 1) / total_count)
        
    status_text.text("Evaluation Complete!")
    
    st.session_state.eval_results = {
        "results": results,
        "passed": passed_count,
        "total": total_count,
        "filename": selected_file
    }

# --- UI: RESULTS DISPLAY ---
if st.session_state.eval_results:
    res_data = st.session_state.eval_results
    passed = res_data["passed"]
    total = res_data["total"]
    accuracy = (passed / total) * 100 if total > 0 else 0
    
    # Generate Download Data & Place in Sidebar
    with st.sidebar:
        st.divider()
        st.header("📥 Export Results")
        
        export_data = [
            {
                "Query": r["Query"], 
                "Expected": r["Expected"], 
                "Retrieved": r["Retrieved"], 
                "Passed": r["Passed"]
            } 
            for r in res_data["results"]
        ]
        json_export = json.dumps(export_data, indent=4)
        
        st.download_button(
            label="Download JSON",
            data=json_export,
            file_name=f"eval_results_{res_data['filename']}",
            mime="application/json",
            use_container_width=True # Makes the button stretch across the sidebar
        )
    
    # Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", total)
    col2.metric("✅ Passed", passed)
    col3.metric("❌ Failed", total - passed)
    col4.metric("🎯 Accuracy", f"{accuracy:.2f}%")
    
    st.divider()
    
    # Detailed Breakdown
    st.subheader("📋 Detailed Breakdown")
    
    tab_all, tab_failed, tab_passed = st.tabs(["All Results", "❌ Failed Only", "✅ Passed Only"])
    
    def render_results(filter_status=None):
        for idx, res in enumerate(res_data["results"]):
            if filter_status is not None and res["Passed"] != filter_status:
                continue
                
            box_class = "pass-box" if res["Passed"] else "fail-box"
            icon = "✅" if res["Passed"] else "❌"
            
            with st.container():
                st.markdown(f"""
                <div class="{box_class}">
                    <strong>{icon} Query {idx+1}:</strong> {res['Query']}<br>
                    <strong>Expected:</strong> <code>{res['Expected']}</code> | <strong>Retrieved:</strong> <code>{res['Retrieved']}</code>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Show Top Retrieved Chunks"):
                    for i, doc in enumerate(res["Raw Docs"][:3]):
                        st.markdown(f"**Chunk {i+1}** (Section: `{doc.metadata.get('section_number', 'N/A')}`)")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()

    with tab_all:
        render_results(filter_status=None)
    with tab_failed:
        if (total - passed) == 0:
            st.success("No failed queries! Perfect score. 🎉")
        else:
            render_results(filter_status=False)
    with tab_passed:
        if passed == 0:
            st.error("No queries passed.")
        else:
            render_results(filter_status=True)