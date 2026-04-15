import os
import sys
import json
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

# ----------------------------------------------------------------------
# CONSUMER PROTECTION ACT, 2019 - COMPREHENSIVE TEST SUITE
# ----------------------------------------------------------------------
CPA_TEST_CASES = [
    # --- EASY: Definitions & Basic Authority ---
    {
        "question": "Does a person who buys goods for a commercial purpose qualify as a 'consumer'?",
        "expected_sections": ["2"],
        "expected_keywords": ["does not include", "commercial purpose", "resale", "livelihood", "self-employment"],
        "category": "Definitions",
        "query_type": "provision",
        "difficulty": "Easy"
    },
    {
        "question": "Which specific authority is established by the Central Government to regulate false or misleading advertisements?",
        "expected_sections": ["10"],
        "expected_keywords": ["Central Consumer Protection Authority", "Central Authority", "CCPA", "protect", "enforce"],
        "category": "Authorities",
        "query_type": "factual",
        "difficulty": "Easy"
    },
    {
        "question": "Does the Consumer Protection Act apply to goods bought through teleshopping or online internet transactions?",
        "expected_sections": ["2"],
        "expected_keywords": ["electronic means", "teleshopping", "direct selling", "multi-level marketing", "offline or online"],
        "category": "Applicability",
        "query_type": "provision",
        "difficulty": "Easy"
    },

    # --- MEDIUM: Jurisdictions & Procedures ---
    {
        "question": "What is the maximum value of goods or services paid as consideration for a complaint to be filed in the District Commission?",
        "expected_sections": ["34"],
        "expected_keywords": ["one crore rupees", "does not exceed", "District Commission", "consideration"],
        "category": "Jurisdiction",
        "query_type": "factual",
        "difficulty": "Medium"
    },
    {
        "question": "Within what time frame must an appeal be filed before the State Commission against an order passed by the District Commission?",
        "expected_sections": ["41"],
        "expected_keywords": ["forty-five days", "45 days", "date of the order", "appeal", "fifty per cent"],
        "category": "Procedure",
        "query_type": "factual",
        "difficulty": "Medium"
    },
    {
        "question": "If a consumer wants to settle their dispute outside of the formal commission hearings, what alternative mechanism is established by the Act?",
        "expected_sections": ["74", "37"],
        "expected_keywords": ["Mediation", "Consumer Mediation Cell", "settlement", "alternative", "written consent"],
        "category": "Mediation",
        "query_type": "procedure",
        "difficulty": "Medium"
    },

    # --- HARD: Penalties, Liability & Exceptions ---
    {
        "question": "Under what specific conditions can a 'product manufacturer' be held liable in a product liability action?",
        "expected_sections": ["84"],
        "expected_keywords": ["manufacturing defect", "defective in design", "deviation from manufacturing specifications", "express warranty", "adequate instructions"],
        "category": "Product Liability",
        "query_type": "provision",
        "difficulty": "Hard"
    },
    {
        "question": "What is the maximum penalty and imprisonment the Central Authority can impose on a manufacturer for a subsequent offence of false or misleading advertisement?",
        "expected_sections": ["21", "89"],
        "expected_keywords": ["fifty lakh rupees", "50 lakh", "imprisonment", "five years", "subsequent"],
        "category": "Penalties",
        "query_type": "calculation",
        "difficulty": "Hard"
    },
    {
        "question": "Is a product seller (who is not the manufacturer) liable if the product causes harm due to the seller failing to exercise reasonable care in assembling the product?",
        "expected_sections": ["86"],
        "expected_keywords": ["product seller", "substantial control", "assembling", "reasonable care", "liable"],
        "category": "Product Liability",
        "query_type": "provision",
        "difficulty": "Hard"
    },

    # --- TWISTED: Edge Cases & Multi-Step Logic ---
    {
        "question": "Can a celebrity or athlete who endorses a misleading advertisement be penalized, and if so, how can they avoid this penalty?",
        "expected_sections": ["21"],
        "expected_keywords": ["endorser", "ten lakh rupees", "due diligence", "verify", "prohibit"],
        "category": "Penalties",
        "query_type": "twisted",
        "difficulty": "Twisted"
    },
    {
        "question": "A consumer signs an 'unfair contract' with a builder for a flat worth 3 Crore Rupees. Where is the original jurisdiction to file a complaint specifically regarding the unfair contract?",
        "expected_sections": ["47"],
        "expected_keywords": ["State Commission", "unfair contract", "does not exceed ten crore", "National Commission"],
        "category": "Jurisdiction",
        "query_type": "twisted", # Twisted because District Commissions cannot hear unfair contract cases regardless of amount.
        "difficulty": "Twisted"
    },
    {
        "question": "If a doctor provides a medical service free of charge at a government hospital, can a patient sue for deficiency of service under the Consumer Protection Act?",
        "expected_sections": ["2"],
        "expected_keywords": ["free of charge", "does not include", "consideration", "contract of personal service"],
        "category": "Applicability",
        "query_type": "twisted",
        "difficulty": "Twisted"
    }
]

def run_cpa_evaluation():
    print("=" * 80)
    print("  NyayaQuest RAG Evaluation: Consumer Protection Act, 2019")
    print(f"  Total Cases: {len(CPA_TEST_CASES)}")
    print("=" * 80)

    groq_api_key = os.getenv('GROQ_API_KEY')
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Point to the master legal database
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )
    
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1, groq_api_key=groq_api_key)
    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)

    results = []
    total_passed = 0

    for i, test in enumerate(CPA_TEST_CASES):
        print(f"\n[{i+1}/{len(CPA_TEST_CASES)}] Q: {test['question']}")
        
        # 1. Retrieval Check
        docs = vector_store.similarity_search(test["question"], k=10)
        found_sections = set()
        
        for d in docs:
            sec = d.metadata.get("section_number")
            if sec: found_sections.add(str(sec))
            
        retrieval_hit = any(sec in found_sections for sec in test["expected_sections"])
        
        # 2. Generation Check
        try:
            response = rag_chain.invoke({"input": test["question"], "chat_history": []})
            answer = response.get("answer", "")
            
            section_cited = any(sec in answer for sec in test["expected_sections"])
            kw_hits = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer.lower())
            kw_score = kw_hits / len(test["expected_keywords"]) if test["expected_keywords"] else 0
            
            # Weighted pass: Must cite the section AND get at least 50% keyword match
            passed = section_cited and kw_score >= 0.5
            if passed: total_passed += 1
            
            print(f"   Retrieval Hit: {retrieval_hit} | Section Cited: {section_cited} | KW Score: {kw_score:.0%}")
            print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
            
            results.append({
                "question": test["question"],
                "difficulty": test["difficulty"],
                "passed": passed,
                "retrieval_hit": retrieval_hit,
                "section_cited": section_cited,
                "answer_preview": answer[:200] + "..." 
            })
            
            time.sleep(2) # Rate limit prevention
        except Exception as e:
            print(f"   Error during generation: {e}")
            time.sleep(5)

    print("\n" + "=" * 80)
    print(f"  FINAL SCORE: {total_passed}/{len(CPA_TEST_CASES)} ({total_passed/len(CPA_TEST_CASES):.0%})")
    print("=" * 80)
    
    with open("cpa_2019_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_cpa_evaluation()