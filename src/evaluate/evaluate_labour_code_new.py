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
# THE FOUR LABOUR CODES - COMPREHENSIVE TEST SUITE
# ----------------------------------------------------------------------
LABOUR_CODES_TEST_CASES = [
    # --- EASY: Definitions & Applicability ---
    {
        "question": "Does the Code on Wages allow an employer to pay wages in kind instead of money?",
        "expected_sections": ["15"],
        "expected_keywords": ["coin", "currency notes", "cheque", "bank account", "electronic mode"],
        "category": "Wages",
        "difficulty": "Easy"
    },
    {
        "question": "Under the Industrial Relations Code, what is the required notice period before employees can go on a strike?",
        "expected_sections": ["62"],
        "expected_keywords": ["sixty days", "fourteen days", "notice of strike", "conciliation"],
        "category": "Industrial Relations",
        "difficulty": "Easy"
    },
    {
        "question": "Does the Code on Social Security provide any benefits specifically for 'gig workers' and 'platform workers'?",
        "expected_sections": ["114"],
        "expected_keywords": ["gig workers", "platform workers", "Central Government", "schemes", "life and disability"],
        "category": "Social Security",
        "difficulty": "Easy"
    },

    # --- MEDIUM: Procedures & Limits ---
    {
        "question": "What is the minimum number of continuous years of service required for an employee to be eligible for gratuity under the Social Security Code?",
        "expected_sections": ["53"],
        "expected_keywords": ["five years", "continuous service", "termination", "superannuation", "retirement"],
        "category": "Social Security",
        "difficulty": "Medium"
    },
    {
        "question": "According to the OSH Code, what is the maximum number of hours a worker can be required to work in a week?",
        "expected_sections": ["25"],
        "expected_keywords": ["forty-eight hours", "week", "maximum", "establishment"],
        "category": "OSH",
        "difficulty": "Medium"
    },
    {
        "question": "Under the Code on Wages, what is the concept of a 'floor wage' and who has the authority to fix it?",
        "expected_sections": ["9"],
        "expected_keywords": ["Central Government", "floor wage", "minimum rate", "geographical areas"],
        "category": "Wages",
        "difficulty": "Medium"
    },

    # --- HARD: Penalties, Liability & Thresholds ---
    {
        "question": "If an employer contravenes the provisions for payment of minimum wages, what is the maximum penalty and fine they can face for a first offence?",
        "expected_sections": ["54"],
        "expected_keywords": ["fine", "fifty thousand rupees", "punishable"],
        "category": "Wages",
        "difficulty": "Hard"
    },
    {
        "question": "Under the Industrial Relations Code, how many workers must an industrial establishment have to be legally required to prepare standing orders?",
        "expected_sections": ["28"],
        "expected_keywords": ["three hundred", "workers", "standing orders", "certifying officer"],
        "category": "Industrial Relations",
        "difficulty": "Hard"
    },
    {
        "question": "What are the employer's obligations regarding free annual health examinations for workers under the OSH Code?",
        "expected_sections": ["6"],
        "expected_keywords": ["free of costs", "annual health examination", "prescribed age", "class of employees"],
        "category": "OSH",
        "difficulty": "Hard"
    },

    # --- TWISTED: Edge Cases & Multi-Step Logic ---
    {
        "question": "If a female employee takes maternity leave, is that period counted as a break in service when calculating her 5-year continuous service for gratuity?",
        "expected_sections": ["2"], 
        "expected_keywords": ["maternity leave", "continuous service", "interruption", "twenty-six weeks"],
        "category": "Social Security",
        "difficulty": "Twisted"
    },
    {
        "question": "Can an employer deduct wages from an employee to recover a loan/advance, and is there a maximum limit on total deductions in a single wage period?",
        "expected_sections": ["18"],
        "expected_keywords": ["deductions", "advances", "fifty per cent", "written authorisation"],
        "category": "Wages",
        "difficulty": "Twisted"
    }
]

def run_labour_codes_evaluation():
    print("=" * 80)
    print(" ⚖️ NyayaQuest RAG Evaluation: The Four Labour Codes")
    print(f" 📊 Total Cases: {len(LABOUR_CODES_TEST_CASES)}")
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

    for i, test in enumerate(LABOUR_CODES_TEST_CASES):
        print(f"\n[{i+1}/{len(LABOUR_CODES_TEST_CASES)}] Q: {test['question']}")
        
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
            
            print(f" 🔍 Retrieval Hit: {retrieval_hit} | Section Cited: {section_cited} | KW Score: {kw_score:.0%}")
            print(f" 🏁 Status: {'✅ PASS' if passed else '❌ FAIL'}")
            
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
            print(f" ⚠️ Error during generation: {e}")
            time.sleep(5)

    print("\n" + "=" * 80)
    print(f" 🏆 FINAL SCORE: {total_passed}/{len(LABOUR_CODES_TEST_CASES)} ({total_passed/len(LABOUR_CODES_TEST_CASES):.0%})")
    print("=" * 80)
    
    with open("labour_codes_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_labour_codes_evaluation()