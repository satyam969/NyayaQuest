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

# Labour Codes Specific Test Cases
# These cover The Code on Wages, Industrial Relations, Social Security, and OSH Codes
# Expanded Comprehensive Test Cases for the Four Labour Codes (Involves 4 level each chapter wise and section wise covering teh importatn provisions and definitions and related queries includes (easy, medium, hard and twisted - hard) questions or queries with expected sections and keywords to be cited in the answer for a pass. These cases are designed to evaluate both retrieval accuracy and generation quality of the RAG system on the labour codes compendium.)
LABOUR_TEST_CASES = [
    # ----------------------------------------------------------------------
    # THE CODE ON WAGES, 2019
    # ----------------------------------------------------------------------
    {
        "question": "Does the Code on Wages prohibit discrimination on the ground of gender?",
        "expected_sections": ["3"],
        "expected_keywords": ["discrimination", "gender", "equal work", "remuneration", "employer"],
        "category": "Wages",
        "query_type": "provision",
        "difficulty": "Easy"
    },
    {
        "question": "By what specific day should wages be paid to an employee employed on a monthly basis?",
        "expected_sections": ["17"],
        "expected_keywords": ["seventh day", "succeeding month", "monthly basis"],
        "category": "Wages",
        "query_type": "procedure",
        "difficulty": "Easy"
    },
    {
        "question": "What is the maximum limit for total deductions that can be made from an employee's wages in a single wage period?",
        "expected_sections": ["18"],
        "expected_keywords": ["fifty per cent", "50%", "deductions", "wage period"],
        "category": "Wages",
        "query_type": "provision",
        "difficulty": "Medium"
    },
    {
        "question": "What is the minimum statutory bonus payable to an employee under the Code on Wages, regardless of whether the employer has an allocable surplus?",
        "expected_sections": ["26"],
        "expected_keywords": ["8.33 per cent", "eight point three three", "one hundred rupees", "whichever is higher"],
        "category": "Wages",
        "query_type": "calculation",
        "difficulty": "Medium"
    },
    {
        "question": "If an employee whose minimum rate of wages is fixed by the hour works on a designated rest day, is he entitled to an overtime rate, and if so, at what rate?",
        "expected_sections": ["13", "14"],
        "expected_keywords": ["overtime rate", "twice", "ordinary rate", "rest day"],
        "category": "Wages",
        "query_type": "calculation",
        "difficulty": "Hard"
    },
    {
        "question": "Under what specific conditions is an employee disqualified from receiving a bonus under the Code on Wages?",
        "expected_sections": ["29"],
        "expected_keywords": ["fraud", "riotous", "violent behaviour", "theft", "misappropriation", "sabotage", "sexual harassment"],
        "category": "Wages",
        "query_type": "provision",
        "difficulty": "Twisted"
    },

    # ----------------------------------------------------------------------
    # THE INDUSTRIAL RELATIONS CODE, 2020
    # ----------------------------------------------------------------------
    {
        "question": "How is a 'Trade Union' defined under the Industrial Relations Code?",
        "expected_sections": ["2"],
        "expected_keywords": ["trade union", "combination", "temporary or permanent", "workmen and employers"],
        "category": "Industrial Relations",
        "query_type": "definition",
        "difficulty": "Easy"
    },
    {
        "question": "How many days of notice must an employer give before changing the conditions of service specified in the Third Schedule?",
        "expected_sections": ["40"],
        "expected_keywords": ["twenty-one days", "21 days", "notice", "conditions of service"],
        "category": "Industrial Relations",
        "query_type": "procedure",
        "difficulty": "Easy"
    },
    {
        "question": "What is the maximum number of members allowed in a Grievance Redressal Committee in an industrial establishment?",
        "expected_sections": ["4"],
        "expected_keywords": ["ten", "10", "women workers", "proportionate"],
        "category": "Industrial Relations",
        "query_type": "provision",
        "difficulty": "Medium"
    },
    {
        "question": "Under the Industrial Relations Code, what are the conditions for a legal strike in a public utility service?",
        "expected_sections": ["62"],
        "expected_keywords": ["strike", "notice", "sixty days", "fourteen days", "conciliation"],
        "category": "Industrial Relations",
        "query_type": "provision",
        "difficulty": "Hard"
    },
    {
        "question": "What is the exact compensation payable to a workman who is retrenched after continuous service of five years?",
        "expected_sections": ["70"],
        "expected_keywords": ["fifteen days", "15 days", "average pay", "completed year", "continuous service"],
        "category": "Industrial Relations",
        "query_type": "calculation",
        "difficulty": "Hard"
    },
    {
        "question": "Can an employer declare a lock-out during the pendency of proceedings before a National Industrial Tribunal?",
        "expected_sections": ["62"],
        "expected_keywords": ["prohibit", "sixty days", "pendency", "Tribunal", "lock-out"],
        "category": "Industrial Relations",
        "query_type": "twisted",
        "difficulty": "Twisted"
    },

    # ----------------------------------------------------------------------
    # THE CODE ON SOCIAL SECURITY, 2020
    # ----------------------------------------------------------------------
    {
        "question": "What is the threshold of the number of employees for the Employees' Provident Fund (EPF) scheme to be applicable to an establishment?",
        "expected_sections": ["16", "First Schedule"],
        "expected_keywords": ["twenty", "20", "employees", "establishment"],
        "category": "Social Security",
        "query_type": "provision",
        "difficulty": "Easy"
    },
    {
        "question": "Who acts as the Chairperson of the National Social Security Board for Unorganised Workers?",
        "expected_sections": ["6"],
        "expected_keywords": ["Union Minister", "Labour and Employment", "Chairperson"],
        "category": "Social Security",
        "query_type": "provision",
        "difficulty": "Easy"
    },
    {
        "question": "What is the continuous service requirement for the payment of gratuity to a working journalist?",
        "expected_sections": ["53"],
        "expected_keywords": ["three years", "five years", "working journalist", "termination", "superannuation"],
        "category": "Social Security",
        "query_type": "eligibility",
        "difficulty": "Medium"
    },
    {
        "question": "What are the provisions for the framing of schemes for gig workers and platform workers?",
        "expected_sections": ["114"],
        "expected_keywords": ["gig workers", "platform workers", "aggregators", "welfare", "scheme"],
        "category": "Social Security",
        "query_type": "provision",
        "difficulty": "Medium"
    },
    {
        "question": "Under the Employees' Compensation provisions, what is the minimum amount of compensation payable in case of death resulting from an injury?",
        "expected_sections": ["76"],
        "expected_keywords": ["one lakh twenty thousand", "1,20,000", "fifty per cent", "monthly wages", "multiplier"],
        "category": "Social Security",
        "query_type": "calculation",
        "difficulty": "Hard"
    },
    {
        "question": "If a woman works for 75 days in the preceding 12 months, is she eligible to claim maternity benefit under the Social Security Code?",
        "expected_sections": ["59", "60"],
        "expected_keywords": ["maternity benefit", "eighty days", "80 days", "twelve months", "not eligible"],
        "category": "Social Security",
        "query_type": "twisted",
        "difficulty": "Twisted"
    },

    # ----------------------------------------------------------------------
    # THE OCCUPATIONAL SAFETY, HEALTH AND WORKING CONDITIONS CODE, 2020
    # ----------------------------------------------------------------------
    {
        "question": "What is the minimum number of women workers required in an establishment for the employer to be legally obligated to provide a crèche facility?",
        "expected_sections": ["24"],
        "expected_keywords": ["fifty", "50", "women workers", "creche", "under the age of six"],
        "category": "Occupational Safety",
        "query_type": "provision",
        "difficulty": "Easy"
    },
    {
        "question": "Under the OSH Code, how many workers are required for a premises using electrical power to be classified as a 'factory'?",
        "expected_sections": ["2"],
        "expected_keywords": ["twenty", "20", "power", "manufacturing process"],
        "category": "Occupational Safety",
        "query_type": "definition",
        "difficulty": "Medium"
    },
    {
        "question": "An establishment employs 250 workers. Is the employer required to provide a canteen under the OSH Code?",
        "expected_sections": ["24"],
        "expected_keywords": ["one hundred", "100", "canteen", "appropriate Government"],
        "category": "Occupational Safety",
        "query_type": "procedure",
        "difficulty": "Medium"
    },
    {
        "question": "What is the maximum limit of daily and weekly working hours specified under the Occupational Safety Code?",
        "expected_sections": ["25"],
        "expected_keywords": ["eight hours", "forty-eight hours", "week", "daily", "intervals"],
        "category": "Occupational Safety",
        "query_type": "provision",
        "difficulty": "Hard"
    },
    {
        "question": "If a contractor fails to pay wages to contract labour, who is ultimately responsible for ensuring the payment under the OSH Code?",
        "expected_sections": ["55"],
        "expected_keywords": ["principal employer", "contractor", "recover", "amount", "wages"],
        "category": "Occupational Safety",
        "query_type": "twisted",
        "difficulty": "Hard"
    },
    {
        "question": "While calculating annual leave with wages under the OSH Code, how is a fraction of leave of half a day or more mathematically treated?",
        "expected_sections": ["32"],
        "expected_keywords": ["one full day", "fraction", "half a day", "omitted"],
        "category": "Occupational Safety",
        "query_type": "twisted",
        "difficulty": "Twisted"
    }
]

def run_evaluation():
    print("=" * 80)
    print("  NyayaQuest RAG Evaluation: Four Labour Codes Compendium")
    print(f"  Cases: {len(LABOUR_TEST_CASES)}")
    print("=" * 80)

    groq_api_key = os.getenv('GROQ_API_KEY')
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Ensuring it points to the same DB where ingest_A_Compendium_on_new_Four_Labour_Codes-labor_law.py wrote the data
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )
    
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2, groq_api_key=groq_api_key)
    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)

    results = []
    total_passed = 0

    for i, test in enumerate(LABOUR_TEST_CASES):
        print(f"\n[{i+1}/{len(LABOUR_TEST_CASES)}] Q: {test['question']}")
        
        # 1. Retrieval
        docs = vector_store.similarity_search(test["question"], k=10)
        found_sections = set()
        
        for d in docs:
            # Matches the "section_number" key created in the compendium chunking script
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
            
            # Weighted pass: Must cite the section AND get at least 50% keyword match
            passed = section_cited and kw_score >= 0.5
            if passed: total_passed += 1
            
            print(f"   Retrieval Hit: {retrieval_hit} | Section Cited: {section_cited} | KW Score: {kw_score:.0%}")
            print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
            
            results.append({
                "question": test["question"],
                "passed": passed,
                "retrieval_hit": retrieval_hit,
                "section_cited": section_cited,
                "answer": answer[:250] + "..." # Storing first 250 chars of response for review
            })
            
            time.sleep(2) # rate limit prevention for Groq API
        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(5)

    print("\n" + "=" * 80)
    print(f"  FINAL SCORE: {total_passed}/{len(LABOUR_TEST_CASES)} ({total_passed/len(LABOUR_TEST_CASES):.0%})")
    print("=" * 80)
    
    # Optionally save results to a JSON for review
    with open("labour_codes_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_evaluation()