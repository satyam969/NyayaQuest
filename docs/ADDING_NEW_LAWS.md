# 📚 How to Add New Legal Documents to NyayaQuest

NyayaQuest's high accuracy comes from **Structural Chunking**—the process of breaking down a legal PDF exactly by its formal components (like "Section 103" or "Article 12") rather than blindly slicing it by character count.

This guide explains step-by-step how to add a new legal PDF (such as the Constitution, BNSS, or RTI Act) to the system.

---

## Step 1: Add the PDF
1. Download the official, parseable PDF of the law (preferably from India Code or official gazettes).
2. Place the PDF inside the `data/legal_pdfs/` directory in the repository.

---

## Step 2: Identify the Structural Regex
Before writing the ingestion script, you must open the PDF and look at how the laws are numbered.

*   Are they written as **"Section 1."**, **"Section 2."**?
*   Are they written as **"Article 1."**, **"Article 2."**?
*   Are they written as **"1."**, **"2."**?

You need to create a **Regular Expression (Regex)** pattern that exactly matches this heading. 
For example, the BNS 2023 uses this format: `\n[1-9][0-9]*\.\s` (A newline, followed by a number, a period, and a space).

You can use a tool like [RegExr](https://regexr.com/) to test your pattern against a snippet of text from the PDF.

---

## Step 3: Create the Ingestion Script
Create a new file in the `src/` directory (e.g., `src/ingest_constitution.py`). You can copy the code from `src/ingest_structural.py` as a blueprint.

You will need to modify **three critical variables** in your new script:

### A. The Document Path
Change the `file_path` to point to your new PDF.
```python
file_path = "data/legal_pdfs/Constitution_of_India.pdf"
```

### B. The Regex Splitter
Update the `regex_pattern` to match the structure you found in Step 2.
```python
from langchain_text_splitters import CharacterTextSplitter

# Example for splitting by "Article 1.", "Article 2.", etc.
regex_pattern = r"(?=\nArticle \d+\.)" 

text_splitter = CharacterTextSplitter(
    separator=regex_pattern,
    is_separator_regex=True,
    # Keep chunk size small (e.g., 250) and overlap around 20 for BGE models to prevent Memory Errors
)
```

### C. The Metadata Injector
When the script splits the chunk, it needs to attach the "Name" of the law to the chunk so the AI knows what it is reading. Update the logic that prepends the header to the chunk:
```python
# Old logic for BNS 2023:
# injected_content = f"Section {section_num}: {cleaned_chunk}"

# New logic for Constitution:
injected_content = f"Constitution of India, Article {article_num}: {cleaned_chunk}"
```
*Note: Make sure to also update the metadata dictionary to point to the correct source file name if you add metadata attributes.*

---

## Step 4: Run the Ingestion
Once your script is ready, activate the virtual environment and run your script from the root directory:
```powershell
.venv\Scripts\activate
uv run python src/ingest_constitution.py
```

> **⚠️ CRITICAL HARDWARE NOTE:** 
> Do not increase the ChromaDB embedding `batch_size` (keep it at 4 to 10) or the `chunk_size` over 250 characters if you are running this on a local machine with limited RAM. The `BAAI/bge-small-en-v1.5` embedding model requires significant system memory.

---

## Step 5: Test the Retrieval (Verification)
You must verify that the database successfully ingested the document constraints! 

You can do this safely using the Streamlit app:
1. Run the app:
   ```powershell
   uv run streamlit run app.py
   ```
2. Ask a highly specific question about the new law (e.g., "What does Article 14 of the Constitution say about equality?").
3. Check the console output where the chunks are printed (if you retain logging) to see if it correctly pulled chunks titled "Constitution of India, Article X".
4. The LLM should perfectly recite the text from the newly ingested PDF!

### Fixing Bad Ingestions
If the LLM hallucinates or cannot find it, it means your **Regex Pattern in Step 2** was incorrect and the document was severely mangled during ingestion. If this happens:
1. Delete the specific collection `chroma_db_groq_legal` folder, or create a script to remove only those documents.
2. Fix your regex and test it locally on a small text file.
3. Re-run your ingestion scripts.

---

## Step 6: Update the Prompts and UI (Optional but Recommended)
Once multiple laws are in the system, you should update the Application so the AI knows its new knowledge domains.

1. Open `prompts.py`
2. Update the `Current Legal Knowledge Domains:` section in the `SYSTEM_PROMPT` to include the new law.
3. Open `app.py`
4. Update the `st.sidebar.markdown` to list the new law so users know they can ask about it!
