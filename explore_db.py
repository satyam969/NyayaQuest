import streamlit as st
import chromadb
import re
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NyayaQuest DB Explorer", page_icon="⚖️", layout="wide")

st.title("⚖️ NyayaQuest Database Explorer")
st.markdown("Visually inspect the exact structural chunks ingested into the Chroma Vector Database.")

# ── Shared footer-detection regex ─────────────────────────────────────────────
FOOTER_PATTERN = re.compile(
    r'(?:^|\n)\s*\d{1,2}\.\s+(?=[^\[\n]*?'
    r'(?:Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|'
    r'amended in its application|extended to .{1,40} by Act|'
    r'extended to the .{1,40} by|vide notification|Gazette of India))',
    re.IGNORECASE
)

# ── Load DB ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_db_data():
    """Load all chunks and metadata from the local ChromaDB."""
    try:
        client = chromadb.PersistentClient(path="chroma_db_groq_legal")
        collection = client.get_collection(name="legal_knowledge")
        results = collection.get(include=['documents', 'metadatas'])

        if not results['documents']:
            return []

        docs   = results['documents']
        metas  = results['metadatas']
        ids    = results['ids']

        chunk_data = []
        for i in range(len(docs)):
            text = docs[i]
            meta = metas[i] if metas[i] else {}

            section_num = meta.get("section_number", meta.get("section", "Unknown"))
            has_footer  = bool(FOOTER_PATTERN.search(text))

            chunk_data.append({
                "id":            ids[i],
                "section":       str(section_num),
                "chapter":       meta.get("chapter", "Unknown"),
                "chunk_index":   meta.get("chunk_index", 0),
                "total_chunks":  meta.get("total_chunks", 0),
                "doc_title":     meta.get("doc_title", "Unknown"),
                "doc_type":      meta.get("doc_type", "Unknown"),
                "law_code":      meta.get("law_code", "Unknown"),
                "year":          meta.get("year", "Unknown"),
                "source":        meta.get("source", "Unknown"),
                "section_title": meta.get("section_title", ""),
                "type":          meta.get("type", "Unknown"),
                "order":         meta.get("order", "Unknown"),
                "order_title":   meta.get("order_title", ""),
                "rule":          meta.get("rule", "Unknown"),
                "rule_title":    meta.get("rule_title", ""),
                "text":          text,
                "length":        len(text),
                "has_footer":    has_footer,
            })

        for c in chunk_data:
            if c.get("type") == "rule":
                c["display_id"] = f"Order {c.get('order')} Rule {c.get('rule')}"
            else:
                c["display_id"] = f"Section {c.get('section', 'Unknown')}"

        def sort_key(x):
            is_rule = 1 if x.get("type") == "rule" else 0
            try:
                if is_rule:
                    r_num = re.match(r"(\d+)", str(x.get("rule", "")))
                    r_val = int(r_num.group(1)) if r_num else 999
                    return (is_rule, str(x.get("order")), r_val, x.get("chunk_index", 0))
                else:
                    s_num = re.match(r"(\d+)", str(x.get("section", "")))
                    s_val = int(s_num.group(1)) if s_num else 9999
                    return (is_rule, s_val, str(x.get("section", "")), x.get("chunk_index", 0))
            except:
                return (is_rule, 9999, str(x.get("section", "")), x.get("chunk_index", 0))

        chunk_data.sort(key=sort_key)
        return chunk_data

    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return []

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Connecting to Vector Database..."):
    all_chunks = load_db_data()

if not all_chunks:
    st.warning("No data found in the database. Please run the ingestion script first.")
    st.stop()

# ── Sidebar stats ──────────────────────────────────────────────────────────────
footer_chunks = [c for c in all_chunks if c['has_footer']]

st.sidebar.header("📊 Database Stats")
st.sidebar.metric("Total Chunks",   len(all_chunks))
st.sidebar.metric("Chunks w/ Footer Text ⚠️", len(footer_chunks),
                  delta=f"{'CLEAN ✅' if not footer_chunks else 'Needs review'}")

import os

def build_law_label(c):
    display_law = c.get("doc_title", "").strip() if c.get("doc_title", "Unknown") != "Unknown" else f"{c.get('law_code', '')} {c.get('year', '')}".strip()
    
    source = c.get("source", "Unknown")
    if source != "Unknown":
        if display_law:
            return f"{display_law} [{os.path.basename(source)}]"
        return os.path.basename(source)
    
    return display_law if display_law else "Unknown Document"

for c in all_chunks:
    c['law_dropdown_label'] = build_law_label(c)

unique_laws     = sorted(set(c['law_dropdown_label'] for c in all_chunks))
selected_law    = st.sidebar.selectbox("📘 Filter by Law / PDF", ["All"] + unique_laws)

law_filtered    = all_chunks if selected_law == "All" else [
    c for c in all_chunks
    if c['law_dropdown_label'] == selected_law
]

unique_chapters = sorted(set(c['chapter'] for c in law_filtered if c['chapter'] != "Unknown"))
selected_chapter = st.sidebar.selectbox("📂 Filter by Chapter", ["All"] + unique_chapters)

chapter_filtered = law_filtered if selected_chapter == "All" else [
    c for c in law_filtered if c['chapter'] == selected_chapter
]

unique_units = []
for c in chapter_filtered:
    did = c.get('display_id', '')
    if "Unknown" not in did and did not in unique_units:
        unique_units.append(did)

with st.sidebar.expander("📋 Units in view", expanded=False):
    st.markdown(
        f"<div style='font-size:0.8em;line-height:1.8;'>{'  •  '.join(unique_units)}</div>",
        unsafe_allow_html=True
    )

with st.sidebar.expander("📂 Chapters in view", expanded=False):
    for ch in unique_chapters:
        st.markdown(f"- {ch}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_browser, tab_footer, tab_stats = st.tabs(["🔍 Chunk Browser", "🚨 Footer Audit", "📈 Stats"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHUNK BROWSER
# ══════════════════════════════════════════════════════════════════════════════
with tab_browser:

    search_query = st.text_input("🔍 Search within filtered results (text or section number)",
                                 key="search_main")

    filtered = chapter_filtered
    if search_query:
        filtered = [
            c for c in filtered
            if search_query.lower() in c['text'].lower()
            or search_query.lower() in c.get('display_id', '').lower()
        ]

    # ── Exclusive unit view ─────────────────────────────────────────────
    if search_query and any(search_query.lower() == c.get('display_id', '').lower() for c in all_chunks):
        st.subheader(f"📑 All Chunks for {search_query.upper()}")
        exact = [c for c in all_chunks if search_query.lower() == c.get('display_id', '').lower()]
        st.info(f"**{search_query.upper()}** — {len(exact)} chunk(s) in the database.")

        with st.container(border=True):
            st.markdown(
                f"**Source:** `{exact[0].get('source', 'Unknown')}` | "
                f"**Law:** `{exact[0].get('law_code', 'Unknown')} {exact[0].get('year', '')}` | "
                f"**Chapter:** `{exact[0].get('chapter', 'Unknown')}`"
            )
            for chunk in exact:
                col_idx, col_text = st.columns([1, 5])
                with col_idx:
                    st.metric("Chunk", f"{chunk['chunk_index']}/{chunk['total_chunks']}")
                    st.caption(f"{chunk['length']} chars")
                    if chunk['has_footer']:
                        st.error("⚠️ Footer detected")
                with col_text:
                    _color = "#fff3cd" if chunk['has_footer'] else "#f0f2f6"
                    _border = "#FF4B4B" if chunk['has_footer'] else "#4CAF50"
                    st.markdown(
                        f"""<div style="background:{_color};padding:15px;border-radius:5px;
                        font-family:monospace;font-size:13px;color:#31333F;
                        max-height:300px;overflow-y:auto;border-left:4px solid {_border};">
                        {chunk['text'].replace('<','&lt;').replace('>','&gt;').replace(chr(10),'<br>')}
                        </div>""",
                        unsafe_allow_html=True
                    )

    # ── Paginated view ─────────────────────────────────────────────────────
    else:
        items_per_page = 10
        total_pages    = max(1, (len(filtered) - 1) // items_per_page + 1)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page_number = st.slider("Page", 1, total_pages, 1, key="page_browser") if total_pages > 1 else 1

        start_idx = (page_number - 1) * items_per_page
        end_idx   = min(start_idx + items_per_page, len(filtered))
        st.markdown(f"***Showing chunks {start_idx + 1}–{end_idx} of {len(filtered)}***")

        for chunk in filtered[start_idx:end_idx]:
            with st.container(border=True):
                col_a, col_b = st.columns([1, 5])

                with col_a:
                    if chunk.get('type') == 'rule':
                        st.metric("Order " + str(chunk.get('order')), "Rule " + str(chunk.get('rule')))
                        if chunk.get('rule_title'):
                            st.markdown(f"**{chunk['rule_title']}**")
                    else:
                        st.metric("Section", chunk['section'])
                        if chunk['section_title']:
                            st.markdown(f"**{chunk['section_title']}**")
                            
                    st.caption(f"📘 `{chunk['law_code']} {chunk['year']}`")
                    st.caption(f"📂 {chunk['chapter'][:35]}")
                    st.caption(f"🧩 Chunk {chunk['chunk_index']}/{chunk['total_chunks']}")
                    st.caption(f"📄 {chunk['length']} chars")
                    if chunk['has_footer']:
                        st.error("⚠️ Footer leak")

                with col_b:
                    _color  = "#fff3cd" if chunk['has_footer'] else "#f0f2f6"
                    _border = "#FF4B4B" if chunk['has_footer'] else "#4CAF50"
                    safe_text = chunk['text'].replace('<','&lt;').replace('>','&gt;').replace(chr(10),'<br>')
                    st.markdown(
                        f"""<div style="background:{_color};padding:15px;border-radius:5px;
                        font-family:monospace;font-size:13px;color:#31333F;
                        max-height:300px;overflow-y:auto;border-left:4px solid {_border};">
                        {safe_text}</div>""",
                        unsafe_allow_html=True
                    )
            st.write("")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FOOTER AUDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_footer:
    st.subheader("🚨 Footer Audit — Chunks Containing Footnote Text")
    st.markdown(
        "These chunks contain lines matching the pattern `1. Subs. by...` / `2. Ins. by...` "
        "which are legal amendment footnotes that should NOT be in your ingested content."
    )

    if not footer_chunks:
        st.success("✅ **All clear!** No footer text detected in any chunk.")
    else:
        st.error(f"⚠️ Found **{len(footer_chunks)}** chunk(s) with potential footer leakage.")

        # Filter controls
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            audit_law = st.selectbox("Filter by Law", ["All"] + unique_laws, key="audit_law")
        with col_f2:
            audit_search = st.text_input("Search within flagged chunks", key="audit_search")

        audit_chunks = footer_chunks
        if audit_law != "All":
            audit_chunks = [c for c in audit_chunks if f"{c['law_code']} {c['year']}" == audit_law]
        if audit_search:
            audit_chunks = [c for c in audit_chunks if audit_search.lower() in c['text'].lower()]

        st.markdown(f"**{len(audit_chunks)} chunk(s) shown**")

        for chunk in audit_chunks:
            with st.container(border=True):
                st.markdown(
                    f"🆔 `{chunk['id']}` | "
                    f"📘 `{chunk['law_code']} {chunk['year']}` | "
                    f"📂 `{chunk['chapter']}` | "
                    f"§ `{chunk['section']}` | "
                    f"🧩 Chunk `{chunk['chunk_index']}/{chunk['total_chunks']}`"
                )

                # Highlight the offending footer lines in red
                lines         = chunk['text'].split('\n')
                highlighted   = []
                for line in lines:
                    safe = line.replace('<','&lt;').replace('>','&gt;')
                    if FOOTER_PATTERN.search(line):
                        highlighted.append(
                            f"<span style='background:#ffe0e0;color:#c00;font-weight:bold;'>{safe}</span>"
                        )
                    else:
                        highlighted.append(safe)

                st.markdown(
                    f"""<div style="background:#fff8f8;padding:15px;border-radius:5px;
                    font-family:monospace;font-size:13px;color:#31333F;
                    max-height:400px;overflow-y:auto;border-left:4px solid #FF4B4B;">
                    {'<br>'.join(highlighted)}</div>""",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("📈 Collection Statistics")

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks",   len(all_chunks))
    col2.metric("Unique Sections", len(set(c['section'] for c in all_chunks)))
    col3.metric("Avg Chunk Size", f"{int(sum(c['length'] for c in all_chunks)/len(all_chunks))} chars")
    col4.metric("Footer Leaks", len(footer_chunks), delta_color="inverse")

    st.divider()

    # Chunks per law
    st.markdown("### Chunks per Law")
    law_counts = {}
    for c in all_chunks:
        key = f"{c['law_code']} {c['year']}"
        law_counts[key] = law_counts.get(key, 0) + 1

    df_law = pd.DataFrame(
        [{"Law": k, "Chunks": v} for k, v in sorted(law_counts.items(), key=lambda x: -x[1])]
    )
    st.bar_chart(df_law.set_index("Law"))

    st.divider()

    # Chunk type distribution
    st.markdown("### Chunk Type Distribution")
    type_counts = {}
    for c in all_chunks:
        type_counts[c['type']] = type_counts.get(c['type'], 0) + 1

    df_type = pd.DataFrame(
        [{"Type": k, "Count": v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]
    )
    st.dataframe(df_type, use_container_width=True)

    st.divider()

    # Chunk length distribution
    st.markdown("### Chunk Length Distribution")
    lengths = [c['length'] for c in chapter_filtered]
    if lengths:
        df_len = pd.DataFrame({
            "Chunk Length (chars)": lengths
        })
        st.bar_chart(df_len["Chunk Length (chars)"].value_counts().sort_index())

    st.divider()

    # Full table export
    st.markdown("### Full Chunk Table (filtered view)")
    df_all = pd.DataFrame([{
        "ID":       c['id'],
        "Law":      f"{c['law_code']} {c['year']}",
        "Section":  c['section'],
        "Title":    c['section_title'],
        "Chapter":  c['chapter'],
        "Type":     c['type'],
        "Chunk":    f"{c['chunk_index']}/{c['total_chunks']}",
        "Length":   c['length'],
        "Footer⚠️": "YES" if c['has_footer'] else "",
    } for c in chapter_filtered])

    st.dataframe(df_all, use_container_width=True, height=400)

    csv = df_all.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "chunks_export.csv", "text/csv")
