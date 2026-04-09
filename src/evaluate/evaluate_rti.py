import os
import re
import sys
import json
import time
from datetime import datetime
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

# RTI-Specific Test Cases
RTI_TEST_CASES = [
    {
        "question": "What is the definition of 'information' under the RTI Act 2005?",
        "expected_sections": ["2"],
        "expected_keywords": ["information", "definition", "records", "documents", "memos"],
        "category": "Definitions",
        "query_type": "definition"
    },
    {
        "question": "What are the obligations of public authorities under Section 4?",
        "expected_sections": ["4"],
        "expected_keywords": ["obligations", "public authority", "maintain", "records", "publish"],
        "category": "Public Authorities",
        "query_type": "provision"
    },
    {
        "question": "How many days does a Public Information Officer have to respond to a request?",
        "expected_sections": ["7"],
        "expected_keywords": ["thirty days", "30 days", "rejection", "request"],
        "category": "Procedure",
        "query_type": "provision"
    },
    {
        "question": "What are the exemptions from disclosure of information under Section 8?",
        "expected_sections": ["8"],
        "expected_keywords": ["exemptions", "disclosure", "sovereignty", "integrity", "security"],
        "category": "Exemptions",
        "query_type": "provision"
    },
    {
        "question": "What is the penalty for a Public Information Officer who fails to provide information?",
        "expected_sections": ["20"],
        "expected_keywords": ["penalty", "two hundred and fifty", "250", "per day", "twenty-five thousand"],
        "category": "Penalties",
        "query_type": "punishment"
    },
    {
        "question": "Can information be provided if it involves the life or liberty of a person?",
        "expected_sections": ["7"],
        "expected_keywords": ["life or liberty", "forty-eight hours", "48 hours"],
        "category": "Procedure",
        "query_type": "provision"
    },
    {
        "question": "What is the constitution of the Central Information Commission?",
        "expected_sections": ["12"],
        "expected_keywords": ["Central Information Commission", "Chief Information Commissioner"],
        "category": "Commission",
        "query_type": "provision"
    },
    {
        "question": "What are the powers and functions of the Information Commissions?",
        "expected_sections": ["18"],
        "expected_keywords": ["powers", "functions", "inquiry", "complaint"],
        "category": "Commission",
        "query_type": "provision"
    }
]

def run_evaluation():
    print("=" * 70)
    print("  NyayaQuest RAG Evaluation: RTI Act 2005")
    print(f"  Cases: {len(RTI_TEST_CASES)}")
    print("=" * 70)

    groq_api_key = os.getenv('GROQ_API_KEY')
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )
    
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2, groq_api_key=groq_api_key)
    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)

    results = []
    total_passed = 0

    for i, test in enumerate(RTI_TEST_CASES):
        print(f"\n[{i+1}/{len(RTI_TEST_CASES)}] Q: {test['question']}")
        
        # 1. Retrieval
        docs = vector_store.similarity_search(test["question"], k=10)
        found_sections = set()
        for d in docs:
            sec = d.metadata.get("section_number")
            if sec: found_sections.add(str(sec))
        
        retrieval_hit = any(sec in found_sections for sec in test["expected_sections"])
        
        # 2. Generation
        try:
            response = rag_chain.invoke({"input": test["question"], "chat_history": []})
            answer = response.get("answer", "")
            
            section_cited = any(sec in answer for sec in test["expected_sections"])
            kw_hits = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer.lower())
            kw_score = kw_hits / len(test["expected_keywords"]) if test["expected_keywords"] else 0
            
            # Weighted pass
            passed = section_cited and kw_score >= 0.5
            if passed: total_passed += 1
            
            print(f"   Retrieval Hit: {retrieval_hit} | Section Cited: {section_cited} | KW Score: {kw_score:.0%}")
            print(f"   Status: {'✅' if passed else '❌'}")
            
            results.append({
                "question": test["question"],
                "passed": passed,
                "retrieval_hit": retrieval_hit,
                "section_cited": section_cited,
                "answer": answer[:200] + "..."
            })
            
            time.sleep(2) # rate limit
        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(5)

    print("\n" + "=" * 70)
    print(f"  FINAL SCORE: {total_passed}/{len(RTI_TEST_CASES)} ({total_passed/len(RTI_TEST_CASES):.0%})")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
