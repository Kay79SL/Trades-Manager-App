"""
eval/run_baselines.py
=====================
Compares three retrieval systems across the 20-question eval set:
  1. MongoDB $text  — simple keyword search (weakest baseline)
  2. BM25           — smarter term-frequency keyword ranking
  3. DACARag full   — your polyglot RAG pipeline (should win)

Outputs eval/results/baselines.csv with per-question and per-category scores.

Usage:
    cd C:\\Users\\Kasia\\DACARag
    venv\\Scripts\\activate
    python eval\\run_baselines.py
"""

import os, sys, json, csv, time, re
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from rank_bm25 import BM25Okapi

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
RESULTS_DIR   = PROJECT_ROOT / "eval" / "results"
OUTPUT_CSV    = RESULTS_DIR / "baselines.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = os.getenv("MONGO_DB", "trades_quotes")

if not MONGO_URI:
    sys.exit("ERROR: MONGO_URI not set in .env")

client = MongoClient(MONGO_URI)
db     = client[MONGO_DB]

# ---------------------------------------------------------------------------
# Import your RAG orchestrator
# ---------------------------------------------------------------------------
# Make sure the project root is on the path so 'retrieve' package is found
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from retrieve.orchestrator import answer_query
    RAG_AVAILABLE = True
    print("✓ RAG orchestrator imported successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠ Could not import RAG orchestrator: {e}")
    print("  Continuing with MongoDB $text and BM25 baselines only.")

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list, gold_ids: list, k: int = 5) -> float:
    """
    What fraction of the gold documents did we find in the top-k results?
    A score of 1.0 means we found everything relevant.
    A score of 0.0 means we found nothing relevant.
    Returns 0.0 if gold_ids is empty (unannotated question — skip in averages).
    """
    if not gold_ids:
        return None   # unannotated — excluded from averages
    hits = set(str(x) for x in retrieved_ids[:k]) & set(str(x) for x in gold_ids)
    return len(hits) / len(gold_ids)


def precision_at_k(retrieved_ids: list, gold_ids: list, k: int = 5) -> float:
    """
    Of the top-k documents we returned, what fraction were actually correct?
    A score of 1.0 means every returned document was relevant.
    Returns None if gold_ids is empty.
    """
    if not gold_ids:
        return None
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = set(str(x) for x in top_k) & set(str(x) for x in gold_ids)
    return len(hits) / len(top_k)


def mrr(retrieved_ids: list, gold_ids: list) -> float:
    """
    Mean Reciprocal Rank — rewards finding the right answer higher in the list.
    Score of 1.0 = correct doc was first. Score of 0.33 = correct doc was third.
    Returns None if gold_ids is empty.
    """
    if not gold_ids:
        return None
    gold_set = set(str(x) for x in gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if str(doc_id) in gold_set:
            return 1.0 / rank
    return 0.0


def safe_avg(values: list) -> float:
    """Average a list, ignoring None values (unannotated questions)."""
    filtered = [v for v in values if v is not None]
    return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

# ---------------------------------------------------------------------------
# System 1 — MongoDB $text search
# ---------------------------------------------------------------------------

def ensure_text_index():
    """Create a text index on the chunks collection if it doesn't exist."""
    existing = db.chunks.index_information()
    has_text = any("text" in str(v.get("key", "")) for v in existing.values())
    if not has_text:
        db.chunks.create_index([("text", "text"), ("source_text", "text")])
        print("  Created $text index on chunks collection")


def mongo_text_search(query: str, k: int = 5) -> list:
    """
    MongoDB full-text search using the $text operator.
    This is the weakest baseline — it only matches exact keywords.
    'Water dripping' will NOT match an email saying 'leak under sink'.
    """
    try:
        results = db.chunks.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "source_id": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(k)
        return [str(r.get("source_id", r.get("_id"))) for r in results]
    except Exception as e:
        print(f"    $text search error: {e}")
        return []

# ---------------------------------------------------------------------------
# System 2 — BM25
# ---------------------------------------------------------------------------

def build_bm25_corpus():
    """
    Load all chunks from MongoDB and build a BM25 index over their text.
    BM25 is smarter than $text: it down-weights common words and rewards
    rare terms. Still keyword-based — no semantic understanding.
    Takes ~5 seconds the first time.
    """
    print("  Building BM25 corpus from chunks collection...")
    docs = list(db.chunks.find({}, {"text": 1, "source_text": 1, "source_id": 1}))
    corpus_texts  = []
    corpus_ids    = []
    for doc in docs:
        text = doc.get("text") or doc.get("source_text") or ""
        corpus_texts.append(text.lower().split())
        corpus_ids.append(str(doc.get("source_id", doc.get("_id"))))
    if not corpus_texts:
        print("  WARNING: No chunks found. Run ingest/embed_chunks.py first.")
        return None, []
    bm25 = BM25Okapi(corpus_texts)
    print(f"  BM25 corpus: {len(corpus_texts)} chunks indexed")
    return bm25, corpus_ids


def bm25_search(bm25, corpus_ids: list, query: str, k: int = 5) -> list:
    """Return top-k chunk source_ids ranked by BM25 score."""
    if bm25 is None:
        return []
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(scores, corpus_ids), reverse=True)
    return [doc_id for _, doc_id in ranked[:k]]

# ---------------------------------------------------------------------------
# System 3 — DACARag full pipeline
# ---------------------------------------------------------------------------

def parse_source_id(raw: str):
    # Handles two formats from answer_query():
    # Format A - vector chunks: 'email_msg_0041 (emails, similarity 0.68)' -> 'msg_0041'
    # Format B - exact/pricing: 'Customer cust_0001' -> 'cust_0001'
    #                            'PO PO-2026-P0042'  -> 'PO-2026-P0042'
    #                            'Neo4j USES_ITEM recipe (5 items)' -> None (skip)
    raw = raw.strip()

    # Skip aggregate description strings that are not real document IDs
    skip_phrases = ["Neo4j", "MongoDB", "recipe", "records", "graph", "vector"]
    if any(p in raw for p in skip_phrases):
        return None

    # Format B: strip human-readable label prefix
    label_prefixes = ["Customer ", "PO ", "Invoice ", "Email ", "Item ", "JobType "]
    for prefix in label_prefixes:
        if raw.startswith(prefix):
            return raw[len(prefix):].strip()

    # Format A: strip trailing similarity score
    if " (" in raw:
        raw = raw[:raw.index(" (")]

    # Strip leading collection prefix from chunk IDs
    chunk_prefixes = ["email_", "jobtype_", "job_type_", "item_", "po_", "inv_", "cust_cust_"]
    for prefix in chunk_prefixes:
        if raw.startswith(prefix):
            return raw[len(prefix):].strip()

    return raw.strip()


def rag_search(query: str, k: int = 5) -> list:
    if not RAG_AVAILABLE:
        return []
    try:
        result = answer_query(query)
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            return []
        parsed = []
        for s in sources[:k]:
            raw = s.get("id") or s.get("source_id") or str(s) if isinstance(s, dict) else str(s)
            pid = parse_source_id(raw)
            if pid is not None:
                parsed.append(pid)
        return parsed
    except Exception as e:
        print(f"    RAG pipeline error: {e}")
        return []

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run():
    print("\n" + "="*60)
    print("DACARag — Baseline Evaluation")
    print("="*60)

    # Load the eval set
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)
    print(f"\nLoaded {len(eval_set)} questions from eval_set.json")

    # Count annotated vs placeholder questions
    annotated = [q for q in eval_set if q["gold_doc_ids"]]
    print(f"  Fully annotated (gold_doc_ids filled): {len(annotated)}")
    print(f"  Placeholders (need your gold labels):  {len(eval_set) - len(annotated)}")
    print()

    # Set up MongoDB $text index
    ensure_text_index()

    # Build BM25 corpus (done once, reused for all 20 questions)
    bm25_index, corpus_ids = build_bm25_corpus()

    # Collect per-row results
    rows = []

    for q in eval_set:
        qid    = q["id"]
        qtype  = q["type"]
        query  = q["question"]
        gold   = q["gold_doc_ids"]

        print(f"\n[{qid}] ({qtype}) {query[:60]}...")

        # --- MongoDB $text ---
        t0 = time.perf_counter()
        text_hits = mongo_text_search(query)
        text_time = round(time.perf_counter() - t0, 3)

        # --- BM25 ---
        t0 = time.perf_counter()
        bm25_hits = bm25_search(bm25_index, corpus_ids, query)
        bm25_time = round(time.perf_counter() - t0, 3)

        # --- DACARag ---
        t0 = time.perf_counter()
        rag_hits = rag_search(query)
        rag_time = round(time.perf_counter() - t0, 3)

        # --- Metrics ---
        row = {
            "id":             qid,
            "type":           qtype,
            "question":       query,
            "gold_count":     len(gold),

            "text_r5":        recall_at_k(text_hits, gold, 5),
            "text_p5":        precision_at_k(text_hits, gold, 5),
            "text_mrr":       mrr(text_hits, gold),
            "text_time_s":    text_time,

            "bm25_r5":        recall_at_k(bm25_hits, gold, 5),
            "bm25_p5":        precision_at_k(bm25_hits, gold, 5),
            "bm25_mrr":       mrr(bm25_hits, gold),
            "bm25_time_s":    bm25_time,

            "rag_r5":         recall_at_k(rag_hits, gold, 5),
            "rag_p5":         precision_at_k(rag_hits, gold, 5),
            "rag_mrr":        mrr(rag_hits, gold),
            "rag_time_s":     rag_time,
        }
        rows.append(row)

        # Print quick summary for this question
        def fmt(v):
            return f"{v:.2f}" if v is not None else " -- "
        print(f"  $text  recall@5={fmt(row['text_r5'])}  prec@5={fmt(row['text_p5'])}  mrr={fmt(row['text_mrr'])}")
        print(f"  BM25   recall@5={fmt(row['bm25_r5'])}  prec@5={fmt(row['bm25_p5'])}  mrr={fmt(row['bm25_mrr'])}")
        print(f"  RAG    recall@5={fmt(row['rag_r5'])}  prec@5={fmt(row['rag_p5'])}  mrr={fmt(row['rag_mrr'])}")

    # --- Save to CSV ---
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Per-question results saved to {OUTPUT_CSV}")

    # --- Summary table ---
    print("\n" + "="*60)
    print("SUMMARY — Averages across annotated questions")
    print("="*60)

    for qtype in ["exact_lookup", "semantic_paraphrase", "multi_hop_graph", "aggregation", "ALL"]:
        subset = rows if qtype == "ALL" else [r for r in rows if r["type"] == qtype]
        label  = "ALL TYPES" if qtype == "ALL" else qtype
        print(f"\n  {label} (n={len(subset)}):")
        for system, prefix in [("$text", "text"), ("BM25", "bm25"), ("RAG", "rag")]:
            r5  = safe_avg([r[f"{prefix}_r5"]  for r in subset])
            p5  = safe_avg([r[f"{prefix}_p5"]  for r in subset])
            mrr_ = safe_avg([r[f"{prefix}_mrr"] for r in subset])
            print(f"    {system:6s}  recall@5={r5:.2f}  prec@5={p5:.2f}  mrr={mrr_:.2f}")

    print("\n" + "="*60)
    print("NEXT STEP: Check eval/results/baselines.csv")
    print("Copy the summary table numbers into Section 4 of your report.")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()
