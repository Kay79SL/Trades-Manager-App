"""
vector_search.py
=================
Module 1 of Phase 5 retrieval orchestration.

Reusable wrapper around Atlas Vector Search. Encodes a query string into a
384-dim vector using sentence-transformers, then runs a $vectorSearch
aggregation against the chunks collection. Returns top-k semantically
similar chunks ranked by cosine similarity.

The first call loads the model into memory (~80MB, takes 2-3 sec). All
subsequent calls reuse the loaded model — encoding is then milliseconds.

Used by:
  - orchestrator.py (when router picks "vector" path)
  - context_assembler.py (for "find similar past content" queries)
  - predict_quote.py (to find similar past invoices/POs by job description)
"""

from __future__ import annotations

import os
from functools import lru_cache  # used to cache the model and DB connection across calls
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim model, must match what embed_chunks.py used at ingest
DEFAULT_INDEX = "vector_index"       # Atlas Search index name configured in MongoDB Atlas UI
DEFAULT_COLLECTION = "chunks"        # collection where embedded text chunks are stored


# ---------------------------------------------------------------------------
# Connection cache (load model + DB connection once, reuse forever)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)  # load model once per process — ~80MB, 2-3 sec first call
def _get_model() -> SentenceTransformer:
    """Load embedding model once, cache for all subsequent calls."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)  # reuse the same MongoClient — avoids repeated handshake overhead
def _get_db():
    """Get MongoDB connection once, cache."""
    mongo_uri = os.getenv("MONGO_URI")           # Atlas connection string from .env
    db_name = os.getenv("MONGO_DB", "trades_quotes")  # default DB name if not set in .env
    return MongoClient(mongo_uri)[db_name]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    top_k: int = 5,
    num_candidates: int = 50,
    source_filter: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    Find chunks semantically similar to the query.

    Args:
        query:          The natural-language query string.
        top_k:          How many results to return (default 5).
        num_candidates: How many candidates Atlas considers before ranking
                        (more = more accurate but slower; 50 is a good default).
        source_filter:  Optional filter to one source collection
                        (e.g. "emails", "job_types", "items", "pos").
        min_score:      Optional minimum cosine similarity (0-1) to include.

    Returns:
        List of dicts: [{"chunk_id", "source_collection", "source_id",
                         "text", "metadata", "score"}, ...]
    """
    if not query or not query.strip():  # guard against empty queries — Atlas would error
        return []

    db = _get_db()
    model = _get_model()

    # Encode query into 384-dim vector (must match what embed_chunks.py used)
    query_vec = model.encode(query, normalize_embeddings=True).tolist()  # normalise for cosine similarity

    # Build the aggregation pipeline
    pipeline: list[dict] = [
        {
            "$vectorSearch": {
                "index":         DEFAULT_INDEX,
                "path":          "embedding",      # field name in chunks collection holding the vector
                "queryVector":   query_vec,
                "numCandidates": num_candidates,   # ANN search pool — higher = more accurate, slower
                "limit":         top_k,
            }
        },
        {
            "$project": {
                "_id":               0,   # suppress MongoDB internal ID
                "chunk_id":          1,
                "source_collection": 1,   # e.g. "emails", "pos", "job_types"
                "source_id":         1,   # original document ID in its source collection
                "text":              1,   # the raw text that was embedded
                "metadata":          1,   # any extra fields stored at chunk creation time
                "score":             {"$meta": "vectorSearchScore"},  # cosine similarity from Atlas
            }
        }
    ]

    # Optional source filter (post-vector-search filter)
    if source_filter:  # e.g. restrict to "emails" only — applied after vector ranking
        pipeline.append({"$match": {"source_collection": source_filter}})

    # Optional minimum-score filter
    if min_score is not None:  # e.g. 0.75 — discard weak matches before returning to orchestrator
        pipeline.append({"$match": {"score": {"$gte": min_score}}})

    results = list(db[DEFAULT_COLLECTION].aggregate(pipeline))
    return results


def vector_search_multi_source(
    query: str,
    top_k_per_source: int = 3,
    sources: tuple[str, ...] = ("emails", "job_types", "items", "pos"),
) -> dict[str, list[dict]]:
    """
    Run a vector search across multiple sources separately and return
    grouped results. Useful when you want diverse evidence from each
    source collection rather than a single ranked list.
    """
    grouped: dict[str, list[dict]] = {}
    for source in sources:  # one Atlas query per source — ensures each collection gets representation
        grouped[source] = vector_search(
            query, top_k=top_k_per_source, source_filter=source
        )
    return grouped


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "boiler installation quote"  # default test query
    print(f"Query: {q}\n")
    for r in vector_search(q, top_k=5):
        print(f"  [{r['score']:.3f}] {r['chunk_id']:25} ({r['source_collection']})")
        print(f"          {r['text'][:90]}...")  # preview first 90 chars of the matching chunk
