"""
NyayaQuest RAG Evaluation Benchmark V2
=======================================
Addresses all critical issues from V1:
  1. Rank-aware retrieval (Top-1, Top-3, Top-10 accuracy)
  2. Section presence + semantic match for citation scoring
  3. Retrieval precision (relevant chunks / total retrieved)
  4. Section coverage (did we get definition + punishment chunks?)
  5. Failure analysis (what was retrieved instead? top docs preview)
  6. Query type tags (definition / punishment / multi-hop / provision)

Usage:
    uv run python src/evaluate_rag_v2.py
"""

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


# ── Custom Embedding ───────────────────────────────────────────────────
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ── Test Cases (with query types) ──────────────────────────────────────
TEST_CASES = [
    {
        "question": "What is the punishment for murder?",
        "expected_sections": ["103"],
        "expected_keywords": ["103", "murder", "death", "imprisonment"],
        "category": "Criminal - Murder",
        "query_type": "punishment"
    },
    {
        "question": "Define culpable homicide under BNS",
        "expected_sections": ["101"],
        "expected_keywords": ["101", "culpable homicide"],
        "category": "Criminal - Homicide",
        "query_type": "definition"
    },
    {
        "question": "What is criminal trespass?",
        "expected_sections": ["329"],
        "expected_keywords": ["329", "trespass"],
        "category": "Property Offences",
        "query_type": "definition"
    },
    {
        "question": "What are the provisions related to theft?",
        "expected_sections": ["303"],
        "expected_keywords": ["303", "theft"],
        "category": "Property Offences",
        "query_type": "provision"
    },
    {
        "question": "What is the punishment for robbery?",
        "expected_sections": ["309"],
        "expected_keywords": ["309", "robbery"],
        "category": "Property Offences",
        "query_type": "punishment"
    },
    {
        "question": "What constitutes defamation under BNS?",
        "expected_sections": ["356"],
        "expected_keywords": ["356", "defamation"],
        "category": "Defamation",
        "query_type": "definition"
    },
    {
        "question": "What is the age of criminal responsibility in India?",
        "expected_sections": ["20"],
        "expected_keywords": ["20", "child", "seven"],
        "category": "General Exceptions",
        "query_type": "provision"
    },
    {
        "question": "Define forgery under BNS 2023",
        "expected_sections": ["336"],
        "expected_keywords": ["336", "forgery"],
        "category": "Document Offences",
        "query_type": "definition"
    },
    {
        "question": "What is the punishment for kidnapping?",
        "expected_sections": ["137"],
        "expected_keywords": ["137", "kidnapping"],
        "category": "Criminal - Kidnapping",
        "query_type": "punishment"
    },
    {
        "question": "What are the provisions about hurt and grievous hurt?",
        "expected_sections": ["114", "115"],
        "expected_keywords": ["114", "115", "hurt"],
        "category": "Criminal - Hurt",
        "query_type": "multi-hop"
    },
    {
        "question": "Define cheating under Bharatiya Nyaya Sanhita",
        "expected_sections": ["318"],
        "expected_keywords": ["318", "cheating"],
        "category": "Fraud",
        "query_type": "definition"
    },
    {
        "question": "What is sedition or acts against sovereignty of India?",
        "expected_sections": ["152"],
        "expected_keywords": ["152"],
        "category": "State Offences",
        "query_type": "definition"
    },
    {
        "question": "What does BNS say about attempt to commit an offence?",
        "expected_sections": ["62"],
        "expected_keywords": ["62", "attempt"],
        "category": "General Provisions",
        "query_type": "provision"
    },
    {
        "question": "What is the right of private defence?",
        "expected_sections": ["34", "35", "36", "37"],
        "expected_keywords": ["private defence"],
        "category": "General Exceptions",
        "query_type": "multi-hop"
    },
    {
        "question": "What are the provisions about criminal conspiracy?",
        "expected_sections": ["61"],
        "expected_keywords": ["61", "conspiracy"],
        "category": "Criminal Conspiracy",
        "query_type": "provision"
    },
]


# ── Helper Functions ───────────────────────────────────────────────────

def extract_sections_from_doc(doc):
    """Extract all Section numbers mentioned in a document's page_content."""
    return set(re.findall(r'Section (\d+[A-Z]?):', doc.page_content))


def get_first_hit_rank(docs, expected_sections):
    """Return the 1-indexed rank of the first document containing any expected section."""
    expected = set(expected_sections)
    for i, doc in enumerate(docs):
        found = extract_sections_from_doc(doc)
        if expected.intersection(found):
            return i + 1
    return None


def compute_retrieval_precision(docs, expected_sections):
    """What fraction of the retrieved chunks are actually relevant?"""
    expected = set(expected_sections)
    relevant = 0
    for doc in docs:
        found = extract_sections_from_doc(doc)
        if expected.intersection(found):
            relevant += 1
    return relevant / len(docs) if docs else 0


def check_section_coverage(docs, expected_sections):
    """Combine all retrieved text and check if key legal content types are present."""
    expected = set(expected_sections)
    relevant_docs = [d for d in docs if extract_sections_from_doc(d).intersection(expected)]
    
    if not relevant_docs:
        return {"chunks_found": 0, "has_definition": False, "has_punishment": False, "combined_length": 0}
    
    combined = " ".join([d.page_content for d in relevant_docs]).lower()
    
    return {
        "chunks_found": len(relevant_docs),
        "has_definition": any(w in combined for w in ["means", "defined", "includes", "denotes", "said to"]),
        "has_punishment": any(w in combined for w in ["punished", "imprisonment", "fine", "death", "rigorous"]),
        "combined_length": len(combined)
    }


def get_top_docs_preview(docs, n=3):
    """Return the first n characters of the top-n retrieved docs for failure analysis."""
    return [doc.page_content[:120] + "..." for doc in docs[:n]]


# ── Main Evaluation ────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("  NyayaQuest RAG Evaluation Benchmark V2")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Cases: {len(TEST_CASES)}")
    print("=" * 70)

    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in .env!")
        return

    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2, groq_api_key=groq_api_key)
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    # ── Accumulators ───────────────────────────────────────────────────
    all_results = []
    top1_hits = 0
    top3_hits = 0
    top10_hits = 0
    top20_hits = 0
    total_precision = 0
    query_type_stats = {}  # {type: {"pass": n, "fail": n}}

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: DEEP RETRIEVAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 1: DEEP RETRIEVAL ANALYSIS")
    print("─" * 70)

    for i, test in enumerate(TEST_CASES):
        docs = retriever.invoke(test["question"])
        expected = set(test["expected_sections"])

        # 1. Rank awareness
        rank = get_first_hit_rank(docs, test["expected_sections"])
        if rank:
            top20_hits += 1
            if rank <= 10: top10_hits += 1
            if rank <= 3:  top3_hits += 1
            if rank == 1:  top1_hits += 1

        # 2. Precision
        precision = compute_retrieval_precision(docs, test["expected_sections"])
        total_precision += precision

        # 3. Section coverage
        coverage = check_section_coverage(docs, test["expected_sections"])

        # 4. All sections found in top-20
        all_found_sections = set()
        for doc in docs:
            all_found_sections.update(extract_sections_from_doc(doc))
        hits = expected.intersection(all_found_sections)

        # 5. Failure analysis
        top_preview = get_top_docs_preview(docs)

        result = {
            "question": test["question"],
            "category": test["category"],
            "query_type": test["query_type"],
            "expected_sections": list(expected),
            "first_hit_rank": rank,
            "precision": round(precision, 3),
            "coverage": coverage,
            "all_found_sections": sorted(list(all_found_sections))[:15],
            "missing_sections": list(expected - hits),
            "top_3_docs_preview": top_preview,
            "retrieval_passed": len(hits) == len(expected)
        }
        all_results.append(result)

        # Print
        status = "✅" if result["retrieval_passed"] else "❌"
        rank_str = f"Rank {rank}" if rank else "NOT FOUND"
        print(f"\n  [{i+1}/{len(TEST_CASES)}] {status} | {test['category']} [{test['query_type']}]")
        print(f"    Q: {test['question']}")
        print(f"    First Hit: {rank_str} | Precision: {precision:.1%} | Relevant Chunks: {coverage['chunks_found']}")
        print(f"    Coverage: Def={coverage['has_definition']} | Punishment={coverage['has_punishment']}")
        if not result["retrieval_passed"]:
            print(f"    ⚠ Missing: {result['missing_sections']}")
            print(f"    Top-1 retrieved: {top_preview[0][:80]}...")

    # Retrieval summary
    n = len(TEST_CASES)
    avg_precision = total_precision / n
    print(f"\n  ══ Retrieval Summary ══")
    print(f"  Top-1  Accuracy: {top1_hits}/{n} ({top1_hits/n:.0%})")
    print(f"  Top-3  Accuracy: {top3_hits}/{n} ({top3_hits/n:.0%})")
    print(f"  Top-10 Accuracy: {top10_hits}/{n} ({top10_hits/n:.0%})")
    print(f"  Top-20 Accuracy: {top20_hits}/{n} ({top20_hits/n:.0%})")
    print(f"  Avg Precision:   {avg_precision:.1%}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: CITATION ACCURACY (Section presence + keyword match)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Phase 2: CITATION ACCURACY (Section + Semantic)")
    print("─" * 70)

    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)
    citation_pass = 0

    for i, test in enumerate(TEST_CASES):
        try:
            response = rag_chain.invoke(
                {"input": test["question"], "chat_history": []},
                config={"configurable": {"session_id": "eval_v2"}},
            )
            answer = response.get("answer", "")

            # Section presence check
            section_present = any(sec in answer for sec in test["expected_sections"])

            # Semantic keyword match
            kw_hits = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer.lower())
            kw_score = kw_hits / len(test["expected_keywords"]) if test["expected_keywords"] else 0

            # Combined pass: section cited AND at least half keywords present
            passed = section_present and kw_score >= 0.5

            if passed:
                citation_pass += 1

            # Track by query type
            qt = test["query_type"]
            if qt not in query_type_stats:
                query_type_stats[qt] = {"pass": 0, "fail": 0, "total": 0}
            query_type_stats[qt]["total"] += 1
            query_type_stats[qt]["pass" if passed else "fail"] += 1

            all_results[i]["citation"] = {
                "answer_snippet": answer[:250],
                "section_cited": section_present,
                "keyword_score": round(kw_score, 2),
                "passed": passed
            }

            status = "✅" if passed else "❌"
            print(f"\n  [{i+1}/{n}] {status} | {test['category']} [{test['query_type']}]")
            print(f"    Q: {test['question']}")
            print(f"    Section Cited: {section_present} | Keyword Score: {kw_score:.0%}")
            print(f"    A: {answer[:120]}...")

            time.sleep(2)  # Groq rate limit

        except Exception as e:
            print(f"\n  [{i+1}/{n}] ⚠ ERROR | {test['category']}: {e}")
            all_results[i]["citation"] = {
                "answer_snippet": f"ERROR: {e}",
                "section_cited": False,
                "keyword_score": 0,
                "passed": False
            }
            time.sleep(3)

    citation_accuracy = citation_pass / n * 100

    # ══════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL REPORT — NyayaQuest RAG Benchmark V2")
    print("=" * 70)
    print(f"  📊 Top-1  Retrieval: {top1_hits/n:.0%}  ({top1_hits}/{n})")
    print(f"  📊 Top-3  Retrieval: {top3_hits/n:.0%}  ({top3_hits}/{n})")
    print(f"  📊 Top-10 Retrieval: {top10_hits/n:.0%}  ({top10_hits}/{n})")
    print(f"  📊 Top-20 Retrieval: {top20_hits/n:.0%}  ({top20_hits}/{n})")
    print(f"  🎯 Avg Precision:    {avg_precision:.1%}")
    print(f"  📝 Citation Accuracy: {citation_accuracy:.1f}%  ({citation_pass}/{n})")
    print(f"  📦 Total Chunks:     {vector_store._collection.count()}")

    print(f"\n  ── Breakdown by Query Type ──")
    for qt, stats in sorted(query_type_stats.items()):
        rate = stats["pass"] / stats["total"] * 100 if stats["total"] else 0
        print(f"    {qt:12s}: {stats['pass']}/{stats['total']} passed ({rate:.0f}%)")

    # Save full report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "top1_accuracy": round(top1_hits / n * 100, 1),
            "top3_accuracy": round(top3_hits / n * 100, 1),
            "top10_accuracy": round(top10_hits / n * 100, 1),
            "top20_accuracy": round(top20_hits / n * 100, 1),
            "avg_precision": round(avg_precision * 100, 1),
            "citation_accuracy": round(citation_accuracy, 1),
            "total_chunks_in_db": vector_store._collection.count(),
        },
        "query_type_breakdown": query_type_stats,
        "detailed_results": all_results
    }

    report_path = "eval_report_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
