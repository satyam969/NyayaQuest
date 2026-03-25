import streamlit as st
import chromadb
import re
import pandas as pd

# Page config
st.set_page_config(page_title="NyayaQuest DB Explorer", page_icon="🔍", layout="wide")

st.title("🔍 NyayaQuest Database Explorer")
st.markdown("Visually inspect the exact structural chunks ingested into the Chroma Vector Database.")

@st.cache_data
def load_db_data():
    """Load all chunks and metadata from the local ChromaDB."""
    try:
        client = chromadb.PersistentClient(path="chroma_db_groq_legal")
        # Get the main collection
        collection = client.get_collection(name="legal_knowledge")
        
        # Get all items in the collection
        results = collection.get(include=['documents', 'metadatas'])
        
        if not results['documents']:
            return [], []
            
        docs = results['documents']
        metas = results['metadatas']
        
        # Parse data into a structured list of dictionaries
        chunk_data = []
        for i in range(len(docs)):
            text = docs[i]
            meta = metas[i] if metas[i] else {}
            
            # Extract section number directly from metadata instead of fragile text regex
            section_num = meta.get("section_number", "Unknown")
            
            chunk_data.append({
                "id": results['ids'][i],
                "section": section_num,
                "chapter": meta.get("chapter", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", "?"), # <-- UPDATED TO FIX THE 1/0 ISSUE
                "law_code": meta.get("law_code", "Unknown"),
                "year": meta.get("year", "Unknown"),
                "source": meta.get("source", "Unknown"),
                "text": text,
                "length": len(text)
            })
            
        # Sort by section number (trying to do it numerically where possible)
        def sort_key(x):
            try:
                # Extract numeric part for sorting
                num_part = re.match(r"(\d+)", x['section'])
                if num_part:
                    return (int(num_part.group(1)), x['section'])
                return (99999, x['section'])
            except:
                return (99999, x['section'])
                
        chunk_data.sort(key=sort_key)
        return chunk_data
        
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return []

# --- LOAD DATA & REFRESH ---
with st.spinner("Connecting to Vector Database..."):
    all_chunks = load_db_data()

# --- SIDEBAR & FILTERING ---
st.sidebar.header("📊 Database Stats")

# Refresh Button to clear cache
if st.sidebar.button("🔄 Refresh Database"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.metric("Total Chunks", len(all_chunks))

# Unnumbered chunks metric (Preamble, Schedules, etc.)
unnumbered_chunks = len([c for c in all_chunks if c['section'] in ["Unknown", "N/A"]])
st.sidebar.metric("ℹ️ Unnumbered Chunks", unnumbered_chunks)
st.sidebar.markdown("---")

if not all_chunks:
    st.warning("No data found in the database. Please run the ingestion script first.")
    st.stop()

# First Filter: Law Act
unique_laws = sorted(list(set([f"{c['law_code']} {c['year']}" for c in all_chunks if c['law_code'] != "Unknown"])))
selected_law = st.sidebar.selectbox("📘 Filter by Law", ["All"] + unique_laws)

# Filter data by Law first to get relevant Chapters/Sections
law_filtered_chunks = all_chunks
if selected_law != "All":
    law_code_val, year_val = selected_law.split(" ", 1)
    law_filtered_chunks = [c for c in all_chunks if c['law_code'] == law_code_val and c['year'] == year_val]

# Second Filter: Chapter (Law-Aware)
unique_chapters = sorted(list(set([c['chapter'] for c in law_filtered_chunks if c['chapter'] != "Unknown"])))
selected_chapter = st.sidebar.selectbox("📂 Filter by Chapter", ["All"] + unique_chapters)

# Filter by Chapter
chapter_filtered_chunks = law_filtered_chunks
if selected_chapter != "All":
    chapter_filtered_chunks = [c for c in law_filtered_chunks if c['chapter'] == selected_chapter]

# Get unique sections from the current level of filtering
unique_sections = sorted(list(set([c['section'] for c in chapter_filtered_chunks if c['section'] != "Unknown"])), 
                         key=lambda x: int(re.match(r"(\d+)", x).group(1)) if re.match(r"(\d+)", x) else 9999)

# Provide a quick visual list of filtered sections
with st.sidebar.expander("📋 List of Sections", expanded=False):
    section_text = " • ".join(unique_sections)
    st.markdown(f"<div style='font-size: 0.8em; line-height: 1.5;'>{section_text}</div>", unsafe_allow_html=True)

with st.sidebar.expander("📂 List of Chapters", expanded=False):
    for ch in unique_chapters:
        st.markdown(f"- {ch}")

# --- SEARCH & NAVIGATION ---
search_query = st.text_input("🔍 Search within filtered results (Text or Section Number)")

# Initialize filtered_chunks for display
filtered_chunks = chapter_filtered_chunks

if search_query:
    filtered_chunks = [c for c in filtered_chunks if search_query.lower() in c['text'].lower() or search_query.lower() == c['section'].lower()]

# --- PAGINATION & RENDER ---
if search_query and any(search_query.lower() == c['section'].lower() for c in all_chunks):
    # EXCLUSIVE SECTION VIEW (Grouped Chunks)
    st.subheader(f"📑 All Chunks for Section {search_query.upper()}")
    
    # Filter only exact matching sections
    exact_chunks = [c for c in all_chunks if search_query.lower() == c['section'].lower()]
    
    st.info(f"Section {search_query.upper()} is split into {len(exact_chunks)} sub-chunks in the database to optimize AI retrieval memory.")
    
    with st.container(border=True):
        st.markdown(f"**Source:** `{exact_chunks[0]['source']}` | **Law:** `{exact_chunks[0]['law_code']} {exact_chunks[0]['year']}` | **Chapter:** `{exact_chunks[0]['chapter']}`")
        
        for i, chunk in enumerate(exact_chunks):
            st.markdown(f"**Chunk {chunk['chunk_index']}/{chunk['total_chunks']}** (Length: {chunk['length']} chars)")
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6; 
                padding: 15px; 
                border-radius: 5px; 
                font-family: monospace;
                font-size: 14px;
                color: #31333F;
                margin-bottom: 20px;
                border-left: 4px solid #FF4B4B;
            ">
                {chunk['text'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
else:
    # STANDARD PAGINATED VIEW
    items_per_page = 10
    total_pages = max(1, (len(filtered_chunks) - 1) // items_per_page + 1)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if total_pages > 1:
            page_number = st.slider("Page", min_value=1, max_value=total_pages, value=1)
        else:
            page_number = 1

    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_chunks))

    chunks_to_display = filtered_chunks[start_idx:end_idx]

    st.markdown(f"***Showing chunks {start_idx + 1} to {end_idx} out of {len(filtered_chunks)}***")

    # --- RENDER CHUNKS ---
    for i, chunk in enumerate(chunks_to_display):
        # Create visually distinct cards for each chunk
        with st.container(border=True):
            col_a, col_b = st.columns([1, 5])
            
            with col_a:
                st.metric(label="Section", value=chunk['section'])
                st.caption(f"📘 `{chunk['law_code']} {chunk['year']}`")
                st.caption(f"📂 {chunk['chapter'][:30]}")
                st.caption(f"🧩 Chunk {chunk['chunk_index']}/{chunk['total_chunks']}")
                st.caption(f"📄 {chunk['length']} chars")
                
            with col_b:
                display_text = chunk['text']
                
                # Put the text inside a stylized box
                st.markdown(f"""
                <div style="
                    background-color: #f0f2f6; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-family: monospace;
                    font-size: 14px;
                    color: #31333F;
                    max-height: 300px;
                    overflow-y: auto;
                    border-left: 4px solid #FF4B4B;
                ">
                    {display_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
        st.write("") # Quick spacer where it is