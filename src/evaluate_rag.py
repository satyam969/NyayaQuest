"""
NyayaQuest RAG Evaluation Benchmark
====================================
Tests both RETRIEVAL accuracy and CITATION accuracy of the RAG pipeline.

Usage:
    uv run python src/evaluate_rag.py

Output:
    - A detailed report showing pass/fail for each test case
    - Overall retrieval hit rate and citation accuracy percentages
"""

import os
import re
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to sys.path so we can import chains, prompts, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from chains import get_rag_chain
from prompts import SYSTEM_PROMPT, QA_PROMPT

load_dotenv()

# ── Custom Embedding (same as app.py) ──────────────────────────────────
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# ── Test Cases ─────────────────────────────────────────────────────────
# Each test has:
#   - question: what the user asks
#   - expected_sections: list of Section numbers that MUST appear in retrieved chunks
#   - expected_keywords: keywords the LLM answer MUST contain for citation accuracy
TEST_CASES = [
    {
        "question": "What is the punishment for murder?",
        "expected_sections": ["103"],
        "expected_keywords": ["103", "murder", "death", "imprisonment"],
        "category": "Criminal - Murder"
    },
    {
        "question": "Define culpable homicide under BNS",
        "expected_sections": ["101"],
        "expected_keywords": ["101", "culpable homicide"],
        "category": "Criminal - Homicide"
    },
    {
        "question": "What is criminal trespass?",
        "expected_sections": ["329"],
        "expected_keywords": ["329", "trespass"],
        "category": "Property Offences"
    },
    {
        "question": "What are the provisions related to theft?",
        "expected_sections": ["303"],
        "expected_keywords": ["303", "theft"],
        "category": "Property Offences"
    },
    {
        "question": "What is the punishment for robbery?",
        "expected_sections": ["309"],
        "expected_keywords": ["309", "robbery"],
        "category": "Property Offences"
    },
    {
        "question": "What constitutes defamation under BNS?",
        "expected_sections": ["356"],
        "expected_keywords": ["356", "defamation"],
        "category": "Defamation"
    },
    {
        "question": "What is the age of criminal responsibility in India?",
        "expected_sections": ["15"],
        "expected_keywords": ["15"],
        "category": "General Exceptions"
    },
    {
        "question": "Define forgery under BNS 2023",
        "expected_sections": ["336"],
        "expected_keywords": ["336", "forgery"],
        "category": "Document Offences"
    },
    {
        "question": "What is the punishment for kidnapping?",
        "expected_sections": ["137"],
        "expected_keywords": ["137", "kidnapping"],
        "category": "Criminal - Kidnapping"
    },
    {
        "question": "What are the provisions about hurt and grievous hurt?",
        "expected_sections": ["114", "115"],
        "expected_keywords": ["hurt"],
        "category": "Criminal - Hurt"
    },
    {
        "question": "Define cheating under Bharatiya Nyaya Sanhita",
        "expected_sections": ["318"],
        "expected_keywords": ["318", "cheating"],
        "category": "Fraud"
    },
    {
        "question": "What is sedition or acts against sovereignty of India?",
        "expected_sections": ["152"],
        "expected_keywords": ["152"],
        "category": "State Offences"
    },
    {
        "question": "What does BNS say about attempt to commit an offence?",
        "expected_sections": ["62"],
        "expected_keywords": ["62", "attempt"],
        "category": "General Provisions"
    },
    {
        "question": "What is the right of private defence?",
        "expected_sections": ["34", "35", "36", "37"],
        "expected_keywords": ["private defence"],
        "category": "General Exceptions"
    },
    {
        "question": "What are the provisions about criminal conspiracy?",
        "expected_sections": ["61"],
        "expected_keywords": ["61", "conspiracy"],
        "category": "Criminal Conspiracy"
    },
]


def run_evaluation():
    print("=" * 70)
    print("  NyayaQuest RAG Evaluation Benchmark")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Cases: {len(TEST_CASES)}")
    print("=" * 70)

    # ── Initialize Models ──────────────────────────────────────────────
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in .env file!")
        return

    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2, groq_api_key=groq_api_key)
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )

    # ── Test 1: RETRIEVAL ACCURACY ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Phase 1: RETRIEVAL ACCURACY (Does vector search find the right chunks?)")
    print("─" * 70)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    retrieval_results = []

    for i, test in enumerate(TEST_CASES):
        docs = retriever.invoke(test["question"])

        # Extract all section numbers from retrieved chunk text
        found_sections = set()
        for doc in docs:
            matches = re.findall(r'Section (\d+[A-Z]?):', doc.page_content)
            found_sections.update(matches)

        # Check if expected sections are in the retrieved set
        expected = set(test["expected_sections"])
        hits = expected.intersection(found_sections)
        passed = len(hits) == len(expected)

        retrieval_results.append({
            "question": test["question"],
            "expected": list(expected),
            "found": sorted(list(found_sections), key=lambda x: int(re.match(r'\d+', x).group()) if re.match(r'\d+', x) else 9999),
            "hits": list(hits),
            "passed": passed,
            "category": test["category"]
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  [{i+1}/{len(TEST_CASES)}] {status} | {test['category']}")
        print(f"    Q: {test['question']}")
        print(f"    Expected Sections: {list(expected)}")
        print(f"    Found in Top-20: {sorted(list(found_sections))[:10]}...")
        if not passed:
            missing = expected - hits
            print(f"    ⚠ Missing: {list(missing)}")

    retrieval_pass_count = sum(1 for r in retrieval_results if r["passed"])
    retrieval_accuracy = (retrieval_pass_count / len(TEST_CASES)) * 100

    print(f"\n  ── Retrieval Summary ──")
    print(f"  Passed: {retrieval_pass_count}/{len(TEST_CASES)}")
    print(f"  Retrieval Accuracy: {retrieval_accuracy:.1f}%")

    # ── Test 2: FULL RAG CITATION ACCURACY ─────────────────────────────
    print("\n" + "─" * 70)
    print("  Phase 2: CITATION ACCURACY (Does the LLM cite correctly?)")
    print("─" * 70)

    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)
    citation_results = []

    for i, test in enumerate(TEST_CASES):
        try:
            response = rag_chain.invoke(
                {"input": test["question"], "chat_history": []},
                config={"configurable": {"session_id": "eval_session"}},
            )
            answer = response.get("answer", "")

            # Check if expected keywords appear in the answer
            keywords_found = []
            keywords_missing = []
            for kw in test["expected_keywords"]:
                if kw.lower() in answer.lower():
                    keywords_found.append(kw)
                else:
                    keywords_missing.append(kw)

            keyword_score = len(keywords_found) / len(test["expected_keywords"]) if test["expected_keywords"] else 0
            passed = keyword_score >= 0.5  # At least half the keywords must be present

            citation_results.append({
                "question": test["question"],
                "answer_snippet": answer[:200] + "..." if len(answer) > 200 else answer,
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "score": keyword_score,
                "passed": passed,
                "category": test["category"]
            })

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n  [{i+1}/{len(TEST_CASES)}] {status} | {test['category']} (Score: {keyword_score:.0%})")
            print(f"    Q: {test['question']}")
            print(f"    A: {answer[:150]}...")
            if keywords_missing:
                print(f"    ⚠ Missing keywords: {keywords_missing}")

            # Rate limit for Groq API
            time.sleep(2)

        except Exception as e:
            print(f"\n  [{i+1}/{len(TEST_CASES)}] ⚠ ERROR | {test['category']}")
            print(f"    {str(e)}")
            citation_results.append({
                "question": test["question"],
                "answer_snippet": f"ERROR: {str(e)}",
                "keywords_found": [],
                "keywords_missing": test["expected_keywords"],
                "score": 0,
                "passed": False,
                "category": test["category"]
            })
            time.sleep(3)

    citation_pass_count = sum(1 for r in citation_results if r["passed"])
    citation_accuracy = (citation_pass_count / len(TEST_CASES)) * 100
    avg_keyword_score = sum(r["score"] for r in citation_results) / len(citation_results) * 100

    print(f"\n  ── Citation Summary ──")
    print(f"  Passed: {citation_pass_count}/{len(TEST_CASES)}")
    print(f"  Citation Accuracy: {citation_accuracy:.1f}%")
    print(f"  Average Keyword Score: {avg_keyword_score:.1f}%")

    # ── Final Report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"  📊 Retrieval Accuracy:  {retrieval_accuracy:.1f}%  ({retrieval_pass_count}/{len(TEST_CASES)} tests passed)")
    print(f"  📝 Citation Accuracy:   {citation_accuracy:.1f}%  ({citation_pass_count}/{len(TEST_CASES)} tests passed)")
    print(f"  🎯 Avg Keyword Score:   {avg_keyword_score:.1f}%")
    print(f"  📦 Total Chunks in DB:  {vector_store._collection.count()}")
    print("=" * 70)

    # ── Save Report to JSON ────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(TEST_CASES),
        "retrieval_accuracy": retrieval_accuracy,
        "citation_accuracy": citation_accuracy,
        "avg_keyword_score": avg_keyword_score,
        "retrieval_results": retrieval_results,
        "citation_results": citation_results
    }

    report_path = "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
