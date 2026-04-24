import os
import re
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from chains import get_rag_chain
from prompts import SYSTEM_PROMPT, QA_PROMPT

load_dotenv()

class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# ---------------------------------------------------------
# SYSTEM-WIDE TEST CASES ACROSS 4 DOMAINS
# ---------------------------------------------------------
SYSTEM_TEST_CASES = [
    # --- BNS 2023 ---
    {
        "question": "What is the punishment for murder under Section 103 of BNS 2023?",
        "expected_sections": ["103"],
        "expected_keywords": ["death", "imprisonment for life", "fine"],
        "domain": "BNS 2023"
    },
    {
        "question": "What constitutes organised crime under Section 111 of the new BNS?",
        "expected_sections": ["111"],
        "expected_keywords": ["organised crime", "syndicate", "kidnapping", "robbery", "extortion"],
        "domain": "BNS 2023"
    },
    
    # --- RTI 2005 ---
    {
        "question": "What is the definition of 'information' under the RTI Act 2005?",
        "expected_sections": ["2"],
        "expected_keywords": ["material", "records", "documents", "memos", "emails"],
        "domain": "RTI 2005"
    },
    {
        "question": "How many days does a Public Information Officer have to respond to a request involving life or liberty?",
        "expected_sections": ["7"],
        "expected_keywords": ["forty-eight hours", "48 hours"],
        "domain": "RTI 2005"
    },
    
    # --- Consumer Protection Act 2019 ---
    {
        "question": "What is the definition of a 'consumer' under the Consumer Protection Act 2019?",
        "expected_sections": ["2"],
        "expected_keywords": ["buys any goods", "consideration", "hires or avails", "commercial purpose"],
        "domain": "CPA 2019"
    },
    {
        "question": "What is the monetary jurisdiction of the District Commission for consumer complaints?",
        "expected_sections": ["34"],
        "expected_keywords": ["one crore rupees", "does not exceed one crore", "jurisdiction"],
        "domain": "CPA 2019"
    },
    
    # --- Four Labour Codes ---
    {
        "question": "Under the Code on Wages 2019, how is the minimum wage fixed?",
        "expected_sections": ["6", "7", "8", "9"],
        "expected_keywords": ["appropriate Government", "minimum rate of wages", "time work", "piece work", "skill"],
        "domain": "Labour Code (Wages)"
    },
    {
        "question": "What constitutes a 'strike' under the Industrial Relations Code 2020?",
        "expected_sections": ["2"],
        "expected_keywords": ["cessation of work", "body of persons", "concerted refusal", "mass casual leave"],
        "domain": "Labour Code (IR)"
    }
]

def log_output(file, text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))
    if file:
        file.write(text + "\n")

def run_system_evaluation():
    with open("system_eval_report.txt", "w", encoding="utf-8") as f:
        log_output(f, "=" * 80)
        log_output(f, "  NyayaQuest GLOBAL RAG Evaluation (Multi-Law System Test)")
        log_output(f, f"  Total Cases: {len(SYSTEM_TEST_CASES)}")
        log_output(f, "=" * 80)
    
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            log_output(f, "❌ Error: GROQ_API_KEY not found in environment!")
            return
    
        log_output(f, "Loading embedding model...")
        embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        log_output(f, "Connecting to ChromaDB...")
        vector_store = Chroma(
            persist_directory="chroma_db_groq_legal",
            embedding_function=embeddings,
            collection_name="legal_knowledge"
        )
        
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1, groq_api_key=groq_api_key)
        rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)
    
        total_passed = 0
    
        for i, test in enumerate(SYSTEM_TEST_CASES):
            log_output(f, f"\n[{i+1}/{len(SYSTEM_TEST_CASES)}] [{test['domain']}] Q: {test['question']}")
            
            # 1. Retrieval Analysis (Using robust DB metadata)
            start_time = time.time()
            docs = vector_store.similarity_search(test["question"], k=10)
            found_sections = set()
            for d in docs:
                sec = d.metadata.get("section_number")
                if sec:
                    found_sections.add(str(sec).strip())
                    
                act = d.metadata.get("act_name")
                log_output(f, f"     - Retrieved: [{act}] Section {sec}")
            
            retrieval_hit = any(sec in found_sections for sec in test["expected_sections"])
            
            # 2. RAG Generation
            try:
                response = rag_chain.invoke({"input": test["question"], "chat_history": []})
                answer = response.get("answer", "")
                gen_time = time.time() - start_time
                
                section_cited = any(sec in answer for sec in test["expected_sections"])
                kw_hits = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer.lower())
                kw_score = kw_hits / len(test["expected_keywords"]) if test["expected_keywords"] else 0
                
                # Weighted pass
                passed = section_cited and kw_score >= 0.4
                if passed: total_passed += 1
                
                log_output(f, f"   Response Time: {gen_time:.1f}s")
                log_output(f, f"   Retrieval Match: {'Yes' if retrieval_hit else 'No'}")
                log_output(f, f"   Section Cited in Answer: {'Yes' if section_cited else 'No'}")
                log_output(f, f"   Keyword Coverage: {kw_score:.0%}")
                log_output(f, f"   Result: {'✅ PASS' if passed else '❌ FAIL'}")
                
                if not passed:
                    log_output(f, f"   [Debug Answer]: {answer[:250]}...")
                
                time.sleep(2) # rate limit prevention
                
            except Exception as e:
                log_output(f, f"   ❌ Error generating response: {e}")
                time.sleep(5)
    
        log_output(f, "\n" + "=" * 80)
        log_output(f, f"  FINAL SYSTEM SCORE: {total_passed}/{len(SYSTEM_TEST_CASES)} ({total_passed/len(SYSTEM_TEST_CASES):.0%})")
        log_output(f, "=" * 80)

if __name__ == "__main__":
    run_system_evaluation()
