import streamlit as st
import os
import json
import time

# Ensure src in path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from auto_ingest.orchestrator import run_agentic_loop

st.set_page_config(page_title="Auto-Ingestion Admin", layout="wide", page_icon="⚙️")

st.title("⚙️ Universal Bare-Act Ingester")
st.markdown("Upload statutory PDFs. The AI agents will automatically extract samples, write custom Regex configurations, evaluate them natively, and sink them to ChromaDB.")

# Initialize session states
if "quarantine_queue" not in st.session_state:
    st.session_state.quarantine_queue = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Tabs
tab_upload, tab_quarantine = st.tabs(["🚀 Upload & Ingest", "⚠️ Quarantine Docs"])

with tab_upload:
    uploaded_files = st.file_uploader("Upload Indian Bare Acts (PDF)", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files and not st.session_state.processing:
        if st.button("▶️ Launch Auto-Ingest Pipeline", use_container_width=True, type="primary"):
            st.session_state.processing = True
            
            # Temporary directory
            os.makedirs("/tmp/nyayaquest/", exist_ok=True)
            
            for file in uploaded_files:
                st.subheader(f"Processing: {file.name}")
                
                # Save locally
                pdf_path = f"/tmp/nyayaquest/{file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(file.read())
                    
                log_container = st.container(border=True, height=400)
                
                with st.spinner("AI Agents conversing..."):
                    generator = run_agentic_loop(pdf_path, file.name)
                    
                    last_config = None
                    failed = False
                    
                    for update in generator:
                        step_id = update.get("step")
                        status = update.get("status")
                        
                        with log_container:
                            st.markdown(f"**[{step_id.upper()}]** {status}")
                            if "data" in update:
                                if isinstance(update["data"], (dict, list)):
                                    st.json(update["data"], expanded=False)
                                else:
                                    st.text(update["data"])
                            if "error" in update:
                                st.error(update["error"])
                                
                        if step_id == "quarantine":
                            failed = True
                            last_config = update.get("last_config")
                            break
                        if step_id == "complete":
                            last_config = update.get("config")
                            break
                            
                if failed:
                    st.error(f"❌ '{file.name}' defeated the pipeline (Max Retries Reached). Moved to Quarantine!")
                    st.session_state.quarantine_queue.append({
                        "filename": file.name,
                        "path": pdf_path,
                        "best_config": last_config
                    })
                else:
                    st.success(f"✅ Semantic Extraction Config Approved for '{file.name}'!")
                    # In a real app, we would now run the extraction config across the full PDF and push to ChromaDB.
                    st.json(last_config)
                    
            st.session_state.processing = False


with tab_quarantine:
    st.subheader("⚠️ Quarantine Failsafe Dashboard")
    st.markdown("These documents defeated the AI Regex Coder 5 times. You must manually intervene or select Failsafe Fallback (Direct JSON Extraction).")
    
    if not st.session_state.quarantine_queue:
        st.success("The Quarantine queue is empty! All documents were parsed successfully.")
    else:
        for idx, item in enumerate(st.session_state.quarantine_queue):
            with st.expander(f"Fix Required: {item['filename']}", expanded=True):
                st.write(f"File Route: `{item['path']}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**AI's Best Attempt Config**")
                    manual_footer = st.text_input("Footer Regex", value=item['best_config'].get("footer_pattern", ""), key=f"foot_{idx}")
                    manual_split = st.text_input("Section Split Regex", value=item['best_config'].get("section_split_pattern", ""), key=f"split_{idx}")
                    
                    if st.button("Save Manual Fix & Ingest", key=f"save_{idx}", type="secondary"):
                        st.session_state.quarantine_queue.pop(idx)
                        st.success("Config patched manually! Proceeding to Vector Ingestion...")
                        st.rerun()

                with col2:
                    st.markdown("**Failsafe #1: Direct LLM**")
                    st.markdown("Abandon Regex. Let the LLM read the massive document blocks and extract sections contextually via pure compute.")
                    if st.button("🚨 Fallback: Use Direct LLM Context Extraction", type="primary", key=f"fb_{idx}"):
                        st.session_state.quarantine_queue.pop(idx)
                        st.info("Triggering massive LLM context extraction pipeline...")
                        st.rerun()
