import os
import sys
import json
import re
import fitz # PyMuPDF
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .prompts import CODER_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT
except ImportError:
    from prompts import CODER_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT

load_dotenv()

# ==============================================================================
# SAMPLER
# ==============================================================================
def sample_pdf(pdf_path: str, num_chunks: int = 5, words_per_chunk: int = 500) -> str:
    """Extracts strategic sample blocks from Start, 25%, 50%, 75%, and End of PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"Error opening PDF: {e}"
        
    num_pages = len(doc)
    if num_pages == 0:
        return ""
        
    # Pick strategic pages
    indices = [
        0,                                   # Beginning
        max(1, int(num_pages * 0.25)),       # 25%
        max(1, int(num_pages * 0.50)),       # Middle
        max(1, int(num_pages * 0.75)),       # 75%
        num_pages - 1                        # End
    ]
    # Remove duplicates if it's a short PDF
    indices = sorted(list(set(indices)))
    
    samples = []
    for idx in indices:
        page = doc[idx]
        text = page.get_text()
        # Take first 1000 chars of that page to keep context limit safe
        samples.append(f"--- SAMPLE FROM PAGE {idx+1} ---\n{text[:1000]}\n")
        
    return "\n".join(samples)

# ==============================================================================
# SANDBOX ENVIRONMENT
# ==============================================================================
def sandbox_run(config: dict, sample_text: str) -> dict:
    """Runs the regex config native python code safely."""
    results = {"chunks": [], "errors": None}
    try:
        footer_pattern = config.get("footer_pattern", "")
        split_pattern = config.get("section_split_pattern", "")
        
        # 1. Strip footers line by line
        clean_lines = []
        for line in sample_text.split('\n'):
            if footer_pattern and re.search(footer_pattern, line, re.IGNORECASE):
                continue
            clean_lines.append(line)
        clean_text = "\n".join(clean_lines)
        
        # 2. Split sections
        if split_pattern:
            chunks = re.split(split_pattern, clean_text)
            chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
            results["chunks"] = chunks
        else:
            results["chunks"] = [clean_text]
            
    except Exception as e:
        results["errors"] = str(e)
        
    return results

# ==============================================================================
# ORCHESTRATOR LOOP
# ==============================================================================
def run_agentic_loop(pdf_path: str, law_name: str, yield_logs=False):
    """
    Generator that orchestrates the Code-Gen and Critic pipeline.
    Yields log dictionaries to the UI gracefully.
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
    
    yield {"step": "sampler", "status": "Extracting strategic samples..."}
    samples = sample_pdf(pdf_path)
    if not samples or "Error" in samples:
        yield {"step": "error", "status": f"Failed to sample PDF: {samples}"}
        return
        
    yield {"step": "sampler", "status": "Samples extracted successfully.", "data": samples[:500] + "..."}
    
    max_retries = 5
    attempt = 1
    feedback = "No feedback yet. This is the first attempt."
    config = {}
    
    while attempt <= max_retries:
        yield {"step": f"coder_{attempt}", "status": f"Attempt {attempt}/{max_retries} - Coder AI generating JSON Regex Config..."}
        
        # --- CODER AI ---
        coder_prompt = f"{CODER_SYSTEM_PROMPT}\n\nFEEDBACK FROM CRITIC:\n{feedback}\n\nSAMPLE TEXT:\n{samples}"
        raw_text = ""
        try:
            res = llm.invoke([SystemMessage(content=coder_prompt)])
            raw_text = res.content.strip()
            # Extract JSON block even if conversational padding exists
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                raw_text = match.group(0)
            config = json.loads(raw_text)
        except Exception as e:
            feedback = f"JSON Parsing failed. Do not return markdown. Error: {e}\nRaw output was: {raw_text}"
            yield {"step": f"coder_{attempt}", "status": f"Coder failed to return valid JSON. Retrying.", "error": str(e)}
            attempt += 1
            continue

        yield {"step": f"sandbox_{attempt}", "status": "Running Config in Sandbox...", "data": dict(config)}
        
        # --- SANDBOX ---
        sandbox_out = sandbox_run(config, samples)
        if sandbox_out["errors"]:
            feedback = f"Sandbox Exception executing Regex: {sandbox_out['errors']}"
            yield {"step": f"sandbox_{attempt}", "status": "Sandbox crashed with regex errors.", "error": sandbox_out["errors"]}
            attempt += 1
            continue
            
        chunks = sandbox_out["chunks"]
        chunk_preview = "\n\n=== CHUNK BOUNDARY ===\n\n".join([c[:200] for c in chunks[:5]]) # show first 5
        
        # --- CRITIC AI ---
        yield {"step": f"critic_{attempt}", "status": "Critic AI evaluating chunks..."}
        critic_prompt = f"{CRITIC_SYSTEM_PROMPT}\n\nTOTAL CHUNKS GENERATED: {len(chunks)}\n\nCHUNK PREVIEWS (First 200 chars of first 5 chunks):\n{chunk_preview}"
        
        c_raw = ""
        try:
            c_res = llm.invoke([SystemMessage(content=critic_prompt)])
            c_raw = c_res.content.strip()
            match = re.search(r'\{.*\}', c_raw, re.DOTALL)
            if match:
                c_raw = match.group(0)
            evaluation = json.loads(c_raw)
        except Exception as e:
            feedback = f"Critic failed to evaluate. Ensure it outputs valid JSON. Output was: {c_raw}"
            yield {"step": f"critic_{attempt}", "status": "Critic JSON extraction failed.", "error": str(e)}
            attempt += 1
            continue
            
        yield {"step": f"evaluation_{attempt}", "status": f"Critic Evaluation Result: {'PASSED' if evaluation.get('success') else 'FAILED'}", "data": evaluation}
        
        if evaluation.get("success") == True or evaluation.get("success") == "true":
            # SUCCESS!
            yield {"step": "complete", "status": "Pipeline Succeeded!", "config": config}
            return config
            
        feedback = evaluation.get("feedback", "General Failure.")
        attempt += 1
        
    # --- FAILSAFE #1 & #3 TRIGGERS ---
    yield {"step": "quarantine", "status": "Max retries (5) reached. PDF routed to Quarantine Failsafe.", "last_config": config}
    return None
