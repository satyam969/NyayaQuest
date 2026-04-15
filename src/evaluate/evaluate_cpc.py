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


# ============================================================
# CPC 1908 Test Cases
# 6 categories × 4 difficulty levels (Easy / Medium / Hard / Twisted)
# Each test carries:
#   - expected_sections : section numbers OR order/rule refs that MUST appear in the answer
#   - expected_keywords : key phrases the answer must include (≥50% → PASS)
#   - meta_key          : the ChromaDB metadata key to look up during retrieval check
#                         ("section" for sections/definitions, "order" for Order/Rule chunks)
# ============================================================

CPC_TEST_CASES = [

    # ------------------------------------------------------------------
    # CATEGORY 1: PRELIMINARY & DEFINITIONS (Sections 1–8)
    # ------------------------------------------------------------------
    {
        "question": "What does the term 'decree' mean under the Code of Civil Procedure, 1908?",
        "expected_sections": ["2"],
        "expected_keywords": ["decree", "formal expression", "adjudication", "conclusively determine", "rights"],
        "category": "Definitions",
        "query_type": "definition",
        "difficulty": "Easy",
        "meta_key": "section"
    },
    {
        "question": "How is the term 'mesne profits' defined under Section 2 of the CPC?",
        "expected_sections": ["2"],
        "expected_keywords": ["mesne profits", "profits", "wrongful possession", "person in possession", "improvements"],
        "category": "Definitions",
        "query_type": "definition",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "What is the meaning of 'judgment-debtor' and 'decree-holder' under the CPC?",
        "expected_sections": ["2"],
        "expected_keywords": ["judgment-debtor", "decree-holder", "decree", "legal representative"],
        "category": "Definitions",
        "query_type": "definition",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "Does Section 4 of the CPC override the provisions of any special or local law? Under what circumstances can the Code apply to Revenue Courts?",
        "expected_sections": ["4", "5"],
        "expected_keywords": ["savings", "special law", "local law", "Revenue Courts", "apply"],
        "category": "Definitions",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "section"
    },

    # ------------------------------------------------------------------
    # CATEGORY 2: JURISDICTION OF COURTS & RES JUDICATA (Sections 9–25)
    # ------------------------------------------------------------------
    {
        "question": "What is the principle of res judicata under the CPC?",
        "expected_sections": ["11"],
        "expected_keywords": ["res judicata", "former suit", "directly and substantially", "litigated", "bar"],
        "category": "Jurisdiction",
        "query_type": "provision",
        "difficulty": "Easy",
        "meta_key": "section"
    },
    {
        "question": "Under which section of the CPC can a court stay the proceedings of a suit, and what conditions must be met?",
        "expected_sections": ["10"],
        "expected_keywords": ["stay of suit", "directly and substantially", "same parties", "previously instituted", "matter in issue"],
        "category": "Jurisdiction",
        "query_type": "provision",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "In which court should a suit for compensation for wrongs to a person or movables be instituted where the wrong was committed in one place and the defendant resides in another?",
        "expected_sections": ["19"],
        "expected_keywords": ["wrong committed", "defendant resides", "plaintiff", "compensation", "movables"],
        "category": "Jurisdiction",
        "query_type": "procedure",
        "difficulty": "Hard",
        "meta_key": "section"
    },
    {
        "question": "A plaintiff seeks to set aside a decree on the ground that the court that passed it had no jurisdiction over the place of suing. Can the plaintiff file a fresh suit for this purpose after the decree has been passed?",
        "expected_sections": ["21A"],
        "expected_keywords": ["bar", "set aside", "decree", "objection", "place of suing", "suit"],
        "category": "Jurisdiction",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "section"
    },

    # ------------------------------------------------------------------
    # CATEGORY 3: SUITS IN GENERAL — SUMMONS, JUDGMENT & COSTS (Sections 26–35B)
    # ------------------------------------------------------------------
    {
        "question": "How is a suit instituted under the CPC?",
        "expected_sections": ["26"],
        "expected_keywords": ["plaint", "institution of suits", "presentation", "court"],
        "category": "Suits",
        "query_type": "procedure",
        "difficulty": "Easy",
        "meta_key": "section"
    },
    {
        "question": "Under Section 34 of the CPC, at what rate can a court award interest on the principal sum adjudged, and from what date does it run?",
        "expected_sections": ["34"],
        "expected_keywords": ["interest", "principal sum", "date of suit", "date of decree", "court may"],
        "category": "Suits",
        "query_type": "calculation",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "Under what circumstances can a court award compensatory costs under Section 35A of the CPC, and what is the maximum amount?",
        "expected_sections": ["35A"],
        "expected_keywords": ["compensatory costs", "false", "vexatious", "three thousand rupees", "claim or defence"],
        "category": "Suits",
        "query_type": "provision",
        "difficulty": "Hard",
        "meta_key": "section"
    },
    {
        "question": "A defendant, in order to delay the trial, takes multiple frivolous adjournments. Can the court impose costs under Section 35B for the delay caused, and does the defendant lose the right to contest later steps if those costs are not paid?",
        "expected_sections": ["35B"],
        "expected_keywords": ["costs for causing delay", "adjournment", "not paid", "shall not be allowed", "further steps"],
        "category": "Suits",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "section"
    },

    # ------------------------------------------------------------------
    # CATEGORY 4: EXECUTION OF DECREES (Sections 36–74)
    # ------------------------------------------------------------------
    {
        "question": "Which court has the power to execute a decree under the CPC?",
        "expected_sections": ["38"],
        "expected_keywords": ["court by which decree may be executed", "passed", "transferred", "executing court"],
        "category": "Execution",
        "query_type": "provision",
        "difficulty": "Easy",
        "meta_key": "section"
    },
    {
        "question": "What questions are to be determined by the executing court under Section 47 of the CPC, and can a fresh suit be brought on those questions?",
        "expected_sections": ["47"],
        "expected_keywords": ["questions", "executing decree", "no suit", "bar", "determine"],
        "category": "Execution",
        "query_type": "provision",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "What are the modes of execution available to a decree-holder under Section 51 of the CPC?",
        "expected_sections": ["51"],
        "expected_keywords": ["delivery of property", "attachment", "sale", "arrest", "detention", "appointing receiver"],
        "category": "Execution",
        "query_type": "provision",
        "difficulty": "Hard",
        "meta_key": "section"
    },
    {
        "question": "A decree was passed against a judgment-debtor who subsequently died. Can the decree be executed against his legal representative personally, and under what limitation does his liability fall?",
        "expected_sections": ["52"],
        "expected_keywords": ["legal representative", "personal liability", "assets", "deceased", "execution against"],
        "category": "Execution",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "section"
    },

    # ------------------------------------------------------------------
    # CATEGORY 5: APPEALS (Sections 96–115)
    # ------------------------------------------------------------------
    {
        "question": "What is the right of appeal from an original decree under the CPC?",
        "expected_sections": ["96"],
        "expected_keywords": ["appeal", "original decree", "appellate court", "every decree"],
        "category": "Appeals",
        "query_type": "provision",
        "difficulty": "Easy",
        "meta_key": "section"
    },
    {
        "question": "Under Section 100 of the CPC, what are the grounds on which a second appeal can lie to the High Court?",
        "expected_sections": ["100"],
        "expected_keywords": ["second appeal", "High Court", "substantial question of law", "arising out of"],
        "category": "Appeals",
        "query_type": "provision",
        "difficulty": "Medium",
        "meta_key": "section"
    },
    {
        "question": "What are the powers of an appellate court with respect to the decree or order it can pass under Section 107 of the CPC?",
        "expected_sections": ["107"],
        "expected_keywords": ["powers of appellate court", "determine", "remand", "frame issues", "take evidence", "additional evidence"],
        "category": "Appeals",
        "query_type": "provision",
        "difficulty": "Hard",
        "meta_key": "section"
    },
    {
        "question": "A High Court exercises revisional jurisdiction under Section 115 to interfere with an order of the subordinate court. Can the High Court alter the finding of fact while exercising this jurisdiction?",
        "expected_sections": ["115"],
        "expected_keywords": ["revision", "jurisdiction", "subordinate court", "illegally", "material irregularity", "finding of fact"],
        "category": "Appeals",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "section"
    },

    # ------------------------------------------------------------------
    # CATEGORY 6: FIRST SCHEDULE — ORDERS & RULES
    # ------------------------------------------------------------------
    {
        "question": "Who can be joined as plaintiffs in a suit under Order I of the CPC?",
        "expected_sections": ["I"],      # Order number
        "expected_keywords": ["joined as plaintiffs", "same act", "same transaction", "common question", "right to relief"],
        "category": "Orders & Rules",
        "query_type": "provision",
        "difficulty": "Easy",
        "meta_key": "order"
    },
    {
        "question": "What are the consequences when a plaint is not properly signed or verified under Order VI Rule 15 of the CPC?",
        "expected_sections": ["VI"],
        "expected_keywords": ["verification", "pleading", "signed", "struck off", "amendment"],
        "category": "Orders & Rules",
        "query_type": "procedure",
        "difficulty": "Medium",
        "meta_key": "order"
    },
    {
        "question": "Under Order XXXVII of the CPC, what is the procedure for a summary suit, and when can a defendant be granted leave to defend?",
        "expected_sections": ["XXXVII"],
        "expected_keywords": ["summary suit", "leave to defend", "negotiable instrument", "acknowledgment of debt", "unconditional leave"],
        "category": "Orders & Rules",
        "query_type": "procedure",
        "difficulty": "Hard",
        "meta_key": "order"
    },
    {
        "question": "Under Order XXXIX of the CPC, a plaintiff applies for a temporary injunction but fails to serve notice on the defendant before the order is passed ex parte. What conditions must the court satisfy before granting such an injunction, and what happens if the plaintiff provides false averments?",
        "expected_sections": ["XXXIX"],
        "expected_keywords": ["temporary injunction", "ex parte", "notice", "irreparable injury", "balance of convenience", "false averment", "compensate"],
        "category": "Orders & Rules",
        "query_type": "twisted",
        "difficulty": "Twisted",
        "meta_key": "order"
    },
]


def run_evaluation():
    print("=" * 80)
    print("  NyayaQuest RAG Evaluation: Code of Civil Procedure, 1908")
    print(f"  Total Cases: {len(CPC_TEST_CASES)}")
    print("=" * 80)

    groq_api_key = os.getenv("GROQ_API_KEY")
    embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_store = Chroma(
        persist_directory="chroma_db_groq_legal",
        embedding_function=embeddings,
        collection_name="legal_knowledge"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=groq_api_key)
    rag_chain = get_rag_chain(llm, vector_store, SYSTEM_PROMPT, QA_PROMPT)

    results = []
    total_passed = 0

    # Track per-category scores for the summary
    category_scores: dict[str, dict] = {}

    for i, test in enumerate(CPC_TEST_CASES):
        cat = test["category"]
        diff = test["difficulty"]
        print(f"\n[{i+1}/{len(CPC_TEST_CASES)}] [{cat}] [{diff}]")
        print(f"  Q: {test['question']}")

        # ----------------------------------------------------------------
        # 1. Retrieval check
        # ----------------------------------------------------------------
        docs = vector_store.similarity_search(test["question"], k=10)
        found_vals = set()

        for d in docs:
            # CPC chunks use "section" for sections/definitions and "order" for Order/Rule chunks
            val = d.metadata.get(test["meta_key"])
            if val:
                found_vals.add(str(val))

        retrieval_hit = any(sec in found_vals for sec in test["expected_sections"])

        # ----------------------------------------------------------------
        # 2. Generation & scoring
        # ----------------------------------------------------------------
        try:
            response = rag_chain.invoke({"input": test["question"], "chat_history": []})
            answer = response.get("answer", "")

            # Section/Order citation check in generated answer
            section_cited = any(sec in answer for sec in test["expected_sections"])

            # Keyword matching (case-insensitive)
            kw_hits = sum(
                1 for kw in test["expected_keywords"]
                if kw.lower() in answer.lower()
            )
            kw_score = kw_hits / len(test["expected_keywords"]) if test["expected_keywords"] else 0

            # Pass criterion: section cited AND ≥50% keyword match
            passed = section_cited and kw_score >= 0.5
            if passed:
                total_passed += 1

            print(f"  Retrieval Hit  : {'✅' if retrieval_hit else '❌'}  ({found_vals & set(test['expected_sections'])})")
            print(f"  Section Cited  : {'✅' if section_cited else '❌'}  (expected: {test['expected_sections']})")
            print(f"  Keyword Score  : {kw_score:.0%}  ({kw_hits}/{len(test['expected_keywords'])} keywords matched)")
            print(f"  Status         : {'✅ PASS' if passed else '❌ FAIL'}")

            # Per-category tracking
            if cat not in category_scores:
                category_scores[cat] = {"passed": 0, "total": 0}
            category_scores[cat]["total"] += 1
            if passed:
                category_scores[cat]["passed"] += 1

            results.append({
                "question": test["question"],
                "category": cat,
                "difficulty": diff,
                "passed": passed,
                "retrieval_hit": retrieval_hit,
                "section_cited": section_cited,
                "kw_score": round(kw_score, 2),
                "expected_sections": test["expected_sections"],
                "expected_keywords": test["expected_keywords"],
                "answer_snippet": answer[:300] + "..."
            })

            time.sleep(2)  # Rate-limit guard for Groq API

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            results.append({
                "question": test["question"],
                "category": cat,
                "difficulty": diff,
                "passed": False,
                "error": str(e)
            })
            time.sleep(5)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  CATEGORY BREAKDOWN")
    print("=" * 80)
    for cat, score in category_scores.items():
        pct = score["passed"] / score["total"] if score["total"] else 0
        print(f"  {cat:<25}  {score['passed']}/{score['total']}  ({pct:.0%})")

    print("\n" + "=" * 80)
    overall_pct = total_passed / len(CPC_TEST_CASES) if CPC_TEST_CASES else 0
    print(f"  FINAL SCORE: {total_passed}/{len(CPC_TEST_CASES)}  ({overall_pct:.0%})")
    print("=" * 80)

    # Save results
    out_path = "cpc_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    run_evaluation()
