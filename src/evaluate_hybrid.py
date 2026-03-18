"""
NyayaQuest: Hybrid vs Vector-Only Retrieval Comparison
=======================================================
Compares retrieval quality between:
  1. Vector-Only (bge-small-en-v1.5)
  2. Hybrid (Vector + BM25 via Reciprocal Rank Fusion)

NO LLM calls required — this tests ONLY retrieval, so no Groq rate limits.

Usage:
    uv run python src/evaluate_hybrid.py
"""

import os
import re
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from hybrid_retriever import HybridRetriever


# ── Custom Embedding ───────────────────────────────────────────────────
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ── Test Cases ─────────────────────────────────────────────────────────
TEST_CASES = [
    {"question": "What is the punishment for murder?", "expected_sections": ["103"], "category": "Murder", "query_type": "punishment"},
    {"question": "Define culpable homicide under BNS", "expected_sections": ["101"], "category": "Homicide", "query_type": "definition"},
    {"question": "What is criminal trespass?", "expected_sections": ["329"], "category": "Trespass", "query_type": "definition"},
    {"question": "What are the provisions related to theft?", "expected_sections": ["303"], "category": "Theft", "query_type": "provision"},
    {"question": "What is the punishment for robbery?", "expected_sections": ["309"], "category": "Robbery", "query_type": "punishment"},
    {"question": "What constitutes defamation under BNS?", "expected_sections": ["356"], "category": "Defamation", "query_type": "definition"},
    {"question": "What is the age of criminal responsibility?", "expected_sections": ["20"], "category": "Age", "query_type": "provision"},
    {"question": "Define forgery under BNS 2023", "expected_sections": ["336"], "category": "Forgery", "query_type": "definition"},
    {"question": "What is the punishment for kidnapping?", "expected_sections": ["137"], "category": "Kidnapping", "query_type": "punishment"},
    {"question": "What are the provisions about hurt and grievous hurt?", "expected_sections": ["114", "115"], "category": "Hurt", "query_type": "multi-hop"},
    {"question": "Define cheating under Bharatiya Nyaya Sanhita", "expected_sections": ["318"], "category": "Cheating", "query_type": "definition"},
    {"question": "What is sedition or acts against sovereignty?", "expected_sections": ["152"], "category": "Sedition", "query_type": "definition"},
    {"question": "What does BNS say about attempt to commit offence?", "expected_sections": ["62"], "category": "Attempt", "query_type": "provision"},
    {"question": "What is the right of private defence?", "expected_sections": ["34", "35", "36", "37"], "category": "Private Defence", "query_type": "multi-hop"},
    {"question": "What are the provisions about criminal conspiracy?", "expected_sections": ["61"], "category": "Conspiracy", "query_type": "provision"},
]


# ── Helper Functions ───────────────────────────────────────────────────

def extract_sections(doc):
    return set(re.findall(r'Section (\d+[A-Z]?):', doc.page_content))

def get_first_hit_rank(docs, expected):
    expected = set(expected)
    for i, doc in enumerate(docs):
        if expected.intersection(extract_sections(doc)):
            return i + 1
    return None

def compute_precision(docs, expected):
    expected = set(expected)
    relevant = sum(1 for d in docs if expected.intersection(extract_sections(d)))
    return relevant / len(docs) if docs else 0

def check_coverage(docs, expected):
    expected = set(expected)
    relevant = [d for d in docs if extract_sections(d).intersection(expected)]
    if not relevant:
        return {"chunks": 0, "def": False, "pun": False}
    combined = " ".join([d.page_content for d in relevant]).lower()
    return {
        "chunks": len(relevant),
        "def": any(w in combined for w in ["means", "defined", "includes", "denotes", "said to"]),
        "pun": any(w in combined for w in ["punished", "imprisonment", "fine", "death", "rigorous"]),
    }


def evaluate_retriever(name, retriever, test_cases):
    """Run all test cases through a single retriever and return metrics."""
    results = []
    top1 = top3 = top10 = top20 = 0
    total_precision = 0

    for test in test_cases:
        docs = retriever.invoke(test["question"])
        expected = set(test["expected_sections"])

        rank = get_first_hit_rank(docs, test["expected_sections"])
        precision = compute_precision(docs, test["expected_sections"])
        coverage = check_coverage(docs, test["expected_sections"])

        all_found = set()
        for d in docs:
            all_found.update(extract_sections(d))
        hits = expected.intersection(all_found)

        if rank:
            top20 += 1
            if rank <= 10: top10 += 1
            if rank <= 3:  top3 += 1
            if rank == 1:  top1 += 1
        total_precision += precision

        results.append({
            "question": test["question"],
            "category": test["category"],
            "query_type": test["query_type"],
            "expected": list(expected),
            "rank": rank,
            "precision": round(precision, 3),
            "coverage": coverage,
            "missing": list(expected - hits),
            "passed": len(hits) == len(expected),
            "top3_preview": [d.page_content[:100] for d in docs[:3]]
        })

    n = len(test_cases)
    return {
        "name": name,
        "top1": top1, "top3": top3, "top10": top10, "top20": top20,
        "top1_pct": round(top1/n*100, 1),
        "top3_pct": round(top3/n*100, 1),
        "top10_pct": round(top10/n*100, 1),
        "top20_pct": round(top20/n*100, 1),
        "avg_precision": round(total_precision/n*100, 1),
        "detailed": results
    }


def print_comparison(vec_metrics, hyb_metrics, n):
    """Print a beautiful side-by-side comparison table."""

    def delta(hyb, vec):
        d = hyb - vec
        if d > 0: return f"  ⬆ +{d:.1f}%"
        if d < 0: return f"  ⬇ {d:.1f}%"
        return "  ─ same"

    print(f"\n{'─'*75}")
    print(f"  {'Metric':<22} {'Vector-Only':>14} {'Hybrid (V+BM25)':>16} {'Delta':>12}")
    print(f"{'─'*75}")
    print(f"  {'Top-1 Accuracy':<22} {vec_metrics['top1_pct']:>13}% {hyb_metrics['top1_pct']:>15}% {delta(hyb_metrics['top1_pct'], vec_metrics['top1_pct'])}")
    print(f"  {'Top-3 Accuracy':<22} {vec_metrics['top3_pct']:>13}% {hyb_metrics['top3_pct']:>15}% {delta(hyb_metrics['top3_pct'], vec_metrics['top3_pct'])}")
    print(f"  {'Top-10 Accuracy':<22} {vec_metrics['top10_pct']:>13}% {hyb_metrics['top10_pct']:>15}% {delta(hyb_metrics['top10_pct'], vec_metrics['top10_pct'])}")
    print(f"  {'Top-20 Accuracy':<22} {vec_metrics['top20_pct']:>13}% {hyb_metrics['top20_pct']:>15}% {delta(hyb_metrics['top20_pct'], vec_metrics['top20_pct'])}")
    print(f"  {'Avg Precision':<22} {vec_metrics['avg_precision']:>13}% {hyb_metrics['avg_precision']:>15}% {delta(hyb_metrics['avg_precision'], vec_metrics['avg_precision'])}")
    print(f"{'─'*75}")

    # Per-query comparison
    print(f"\n{'─'*75}")
    print(f"  {'Query':<35} {'Vec Rank':>10} {'Hyb Rank':>10} {'Winner':>10}")
    print(f"{'─'*75}")
    for v, h in zip(vec_metrics["detailed"], hyb_metrics["detailed"]):
        vr = v["rank"] if v["rank"] else "Miss"
        hr = h["rank"] if h["rank"] else "Miss"
        
        if v["rank"] and h["rank"]:
            winner = "Hybrid ✅" if h["rank"] < v["rank"] else ("Vector ✅" if v["rank"] < h["rank"] else "Tie")
        elif h["rank"] and not v["rank"]:
            winner = "Hybrid ✅"
        elif v["rank"] and not h["rank"]:
            winner = "Vector ✅"
        else:
            winner = "Both Miss"
        
        q = v["question"][:33]
        print(f"  {q:<35} {str(vr):>10} {str(hr):>10} {winner:>10}")
    print(f"{'─'*75}")

    # Query type breakdown
    print(f"\n  ── Breakdown by Query Type ──")
    for qt in ["definition", "punishment", "provision", "multi-hop"]:
        v_pass = sum(1 for r in vec_metrics["detailed"] if r["query_type"] == qt and r["passed"])
        h_pass = sum(1 for r in hyb_metrics["detailed"] if r["query_type"] == qt and r["passed"])
        total = sum(1 for r in vec_metrics["detailed"] if r["query_type"] == qt)
        if total > 0:
            print(f"    {qt:12s}: Vector {v_pass}/{total} | Hybrid {h_pass}/{total}")


def run_comparison():
    print("=" * 75)
    print("  NyayaQuest: Hybrid vs Vector-Only Retrieval Comparison")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Cases: {len(TEST_CASES)}")
    print(f"  Note: NO LLM calls — retrieval-only benchmark")
    print("=" * 75)

    # ── Initialize ─────────────────────────────────────────────────────
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )

    # Vector-only retriever
    vec_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    # Hybrid retriever (Vector + BM25)
    print("\n  Building BM25 index...")
    hyb_retriever = HybridRetriever.from_vector_store(vector_store, k=20, vector_weight=0.5, bm25_weight=0.5)

    n = len(TEST_CASES)

    # ── Run Both ───────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("  Running Vector-Only retrieval...")
    print(f"{'─'*75}")
    vec_metrics = evaluate_retriever("Vector-Only", vec_retriever, TEST_CASES)

    for r in vec_metrics["detailed"]:
        status = "✅" if r["passed"] else "❌"
        rank_str = f"Rank {r['rank']}" if r['rank'] else "NOT FOUND"
        print(f"    {status} {r['category']:<16} | {rank_str:<12} | Prec: {r['precision']:.0%} | Cov: {r['coverage']['chunks']} chunks")

    print(f"\n{'─'*75}")
    print("  Running Hybrid (Vector + BM25) retrieval...")
    print(f"{'─'*75}")
    hyb_metrics = evaluate_retriever("Hybrid (V+BM25)", hyb_retriever, TEST_CASES)

    for r in hyb_metrics["detailed"]:
        status = "✅" if r["passed"] else "❌"
        rank_str = f"Rank {r['rank']}" if r['rank'] else "NOT FOUND"
        print(f"    {status} {r['category']:<16} | {rank_str:<12} | Prec: {r['precision']:.0%} | Cov: {r['coverage']['chunks']} chunks")

    # ── Side-by-Side Comparison ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 75)
    print_comparison(vec_metrics, hyb_metrics, n)

    # ── Final Verdict ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  FINAL VERDICT")
    print(f"{'='*75}")
    hyb_wins = sum(1 for v, h in zip(vec_metrics["detailed"], hyb_metrics["detailed"])
                   if h["rank"] and (not v["rank"] or h["rank"] < v["rank"]))
    vec_wins = sum(1 for v, h in zip(vec_metrics["detailed"], hyb_metrics["detailed"])
                   if v["rank"] and (not h["rank"] or v["rank"] < h["rank"]))
    ties = n - hyb_wins - vec_wins

    print(f"  Hybrid wins:  {hyb_wins}/{n}")
    print(f"  Vector wins:  {vec_wins}/{n}")
    print(f"  Ties/Both miss: {ties}/{n}")

    if hyb_wins > vec_wins:
        print(f"\n  🏆 HYBRID RETRIEVAL IS BETTER! (+{hyb_metrics['top1_pct'] - vec_metrics['top1_pct']:.1f}% Top-1)")
    elif vec_wins > hyb_wins:
        print(f"\n  🏆 VECTOR-ONLY IS BETTER!")
    else:
        print(f"\n  🤝 TIE!")
    print(f"{'='*75}")

    # ── Save Report ────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": n,
        "vector_only": {k: v for k, v in vec_metrics.items() if k != "detailed"},
        "hybrid": {k: v for k, v in hyb_metrics.items() if k != "detailed"},
        "per_query_comparison": [
            {
                "question": v["question"],
                "category": v["category"],
                "query_type": v["query_type"],
                "vector_rank": v["rank"],
                "hybrid_rank": h["rank"],
                "vector_precision": v["precision"],
                "hybrid_precision": h["precision"],
                "vector_coverage": v["coverage"],
                "hybrid_coverage": h["coverage"],
                "vector_passed": v["passed"],
                "hybrid_passed": h["passed"],
                "winner": "hybrid" if (h["rank"] and (not v["rank"] or h["rank"] < v["rank"])) else 
                          ("vector" if (v["rank"] and (not h["rank"] or v["rank"] < h["rank"])) else "tie")
            }
            for v, h in zip(vec_metrics["detailed"], hyb_metrics["detailed"])
        ]
    }

    with open("eval_hybrid_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report saved to: eval_hybrid_comparison.json")


if __name__ == "__main__":
    run_comparison()
