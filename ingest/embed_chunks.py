"""
embed_chunks.py
================
vector embeddings for hybrid retrieval.

Generates a 384-dimensional vector embedding for every retrievable text
chunk in the dataset. Saves all chunks to a new MongoDB `chunks` collection
that will be indexed by Atlas Vector Search.

Why:
  Hybrid retrieval needs three search modes:
    1. Exact lookup (MongoDB)       
    2. Graph traversal (Neo4j)        
    3. Semantic search (vectors)      
  Without embeddings, the chatbot can only match by exact words.
  With embeddings, "my heating's broken" finds boiler-related emails
  even when no words match.

What gets embedded:
  - Email bodies (100)        text = subject + body + key_phrases
  - Job type names (60)       text = job_name
  - Items (88)                text = item_name + category
  - PO summaries (15)         text = job_type + job_desc + items + customer

Storage shape (chunks collection):
  {
    "chunk_id":          "email_msg_0042",
    "source_collection": "emails",
    "source_id":         "msg_0042",
    "text":              "Subject: Quote for boiler... Body: Hi, my boiler...",
    "metadata": {
        "trade":            "plumber",
        "job_type_id":      "pl_01",
        "urgency":          "low",
        "is_returning":     true
    },
    "embedding":         [0.0234, -0.1827, 0.4421, ... (384 floats)]
  }

Cost: €0 — sentence-transformers runs entirely on your CPU.
Time: ~3-5 minutes for first run (model download), ~30 sec on re-runs.
Disk: ~80MB for the model (cached locally), ~2MB MongoDB storage.

Usage:
    python ingest\\embed_chunks.py                    # full embedding run
    python ingest\\embed_chunks.py --emails           # emails only
    python ingest\\embed_chunks.py --pos              # POs only
    python ingest\\embed_chunks.py --job-types        # job types only
    python ingest\\embed_chunks.py --items            # items only
    python ingest\\embed_chunks.py --force            # re-embed everything
    python ingest\\embed_chunks.py --dry-run          # show what would be embedded

Dependencies:
    python -m pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

# Conditional import — explain how to install if missing
# sentence-transformers is a popular library that provides easy access to many pre-trained embedding models. We use the "all-MiniLM-L6-v2" model, 
# which is a good balance of performance and speed for our use case.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Install with: python -m pip install sentence-transformers")
    print("(First-run downloads ~80MB model. One-time cost.)")
    sys.exit(1)

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TARGET_COLLECTION = "chunks"
BATCH_SIZE = 32   # texts processed per batch — keeps memory usage low


# ---------------------------------------------------------------------------
# Text builders — turn each source document into the text that gets embedded
# ---------------------------------------------------------------------------

def build_email_text(email: dict) -> str:
    """
    Combine subject, body, and key phrases into a single text for embedding.
    Key phrases (extracted earlier in Phase 3A) help the embedding capture
    the email's intent more precisely.
    """
    # Include subject and body if they exist — captures the main content of the email
    parts = []
    if email.get("subject"): # some emails may not have a subject, so we check before including it
        parts.append(f"Subject: {email['subject']}") # we prefix with "Subject:" to give the model a hint about the structure of the text (this can help it learn that the subject is a distinct part of the email)
    if email.get("body"): 
        parts.append(f"Body: {email['body']}") #  it helps the model understand the structure and focus on the main content of the email.

    # Add LLM-extracted key phrases if available
    extracted = email.get("extracted") or {} # the "extracted" field may not exist if the email hasn't gone through the LLM extraction phase, so we default to an empty dict to avoid errors
    key_phrases = extracted.get("key_phrases") or [] # key_phrases may not exist or may be None, so we default to an empty list to avoid errors
    if key_phrases: # if there are any key phrases, we include them in the text. This helps the embedding capture the main topics and intent of the email more effectively, 
        # especially for short emails where the subject and body may not have enough context on their own.
        parts.append("Key topics: " + "; ".join(key_phrases)) # we join the key phrases with semicolons to create a clear list of topics. This can help the model understand that these are distinct key points extracted from the email.

    return " ".join(parts).strip()

#   The metadata we save with each chunk can be used later for filtered retrieval (e.g. "find similar emails with urgency = high") or for evaluation 
# (e.g. "was the retrieved email actually about a boiler installation?"). We include any fields that might be useful for those purposes, even if they're not directly used in the embedding text.
def build_email_metadata(email: dict) -> dict:
    """Metadata travels with the chunk — used later for filtered retrieval."""
    extracted = email.get("extracted") or {}
    return {
        "trade":              extracted.get("trade_needed"),
        "job_type_id":        extracted.get("job_type_id"),
        "urgency":            extracted.get("urgency"),
        "is_returning":       extracted.get("is_returning_customer"),
        "matched_customer_id": extracted.get("matched_customer_id"),
        "from_email":         email.get("from_email"),
    }

# The following text builders are simpler than the email one, since they don't have multiple fields to combine or LLM-extracted key phrases. 
# They just format the most relevant fields into a concise text for embedding.
def build_job_type_text(jt: dict) -> str:
    """Just the job name — short and distinctive."""
    return f"{jt.get('job_name', '')} ({jt.get('trade', '')})"

# For items, we combine the item name with the category (if it exists) to give more context about what the item is and what it's used for. 
# This can help the embedding capture similarities between items that might have different names but belong to the same category.
def build_item_text(item: dict) -> str:
    """Item name plus category — captures what the item is for."""
    parts = [item.get("item_name", "")]
    if item.get("category"):
        parts.append(f"category: {item['category']}")
    return " — ".join(parts).strip()

# For POs, we want to capture the overall context of the order, including the job type, job description, customer name, and key items. 
# This helps the embedding find similar past POs based on the overall scenario, not just exact words. For example, a PO for a boiler installation might be similar to another PO for a boiler repair, even if the exact words differ.
def build_po_text(po: dict) -> str:
    """
    PO embedding text — captures what the PO is for, who it's for,
    and what materials are involved. Useful for "find similar past POs".
    """
    parts = []
    if po.get("job_type"):
        parts.append(f"Job: {po['job_type']}")
    if po.get("job_desc"):
        parts.append(po["job_desc"])
    if po.get("cust_name"):
        parts.append(f"Customer: {po['cust_name']}")

    # Include item descriptions for material-based similarity
    #  If a PO has line items, we take the first few item descriptions and include them in the text. This helps the embedding capture the materials involved in the order, 
    # which can be important for finding similar POs based on what was installed or repaired.
    items = po.get("line_items", [])
    if items:
        item_strs = [li.get("description", "") for li in items[:5] if li.get("description")]
        if item_strs:
            parts.append("Items: " + ", ".join(item_strs))

    return ". ".join(parts).strip()

# The metadata for POs includes the status, matched customer ID, matched job type ID, trade name, and total cost. This information can be useful for filtered retrieval 
# (e.g. "find similar POs that were for plumbers and had a high total cost") or for evaluation (e.g. "was the retrieved PO actually for a boiler installation?").
def build_po_metadata(po: dict) -> dict:
    return {
        "po_status":           po.get("po_status"),
        "matched_customer_id": po.get("matched_customer_id"),
        "matched_job_type_id": po.get("matched_job_type_id"),
        "trade_name":          po.get("trade_name"),
        "total_inc_vat":       po.get("total_inc_vat"),
    }


# ---------------------------------------------------------------------------
# Chunk collection — fetch each source, build chunks, return list
# ---------------------------------------------------------------------------
# Each function below fetches documents from a specific MongoDB collection, builds the text and metadata for each chunk using the builders above, and returns a list of chunk dicts ready for embedding.
def collect_email_chunks(db) -> list[dict]:
    chunks = []
    for email in db.emails.find({}):
        text = build_email_text(email)
        if not text:
            continue
        chunks.append({
            "chunk_id":          f"email_{email['email_id']}",
            "source_collection": "emails",
            "source_id":         email["email_id"],
            "text":              text,
            "metadata":          build_email_metadata(email),
        })
    return chunks

# The following collection functions follow the same pattern as collect_email_chunks: they fetch documents from their respective collections, 
# build the text and metadata for each chunk, and return a list of chunk dicts. The main differences are in how the text is built (using the specific builders for each source) and what metadata fields are included.
def collect_job_type_chunks(db) -> list[dict]:
    chunks = []
    for jt in db.job_types.find({}):
        text = build_job_type_text(jt)
        if not text:
            continue
        chunks.append({
            "chunk_id":          f"jobtype_{jt['job_type_id']}",
            "source_collection": "job_types",
            "source_id":         jt["job_type_id"],
            "text":              text,
            "metadata": {
                "trade":       jt.get("trade"),
                "job_name":    jt.get("job_name"),
            },
        })
    return chunks

# For items, we include the category and unit price in the metadata, which can be useful for filtered retrieval 
# (e.g. "find similar items in the same category" or "find similar items with a high unit price") or for evaluation (e.g. "was the retrieved item actually a boiler part?").
def collect_item_chunks(db) -> list[dict]: # def to collect item chunks — we build the text using the build_item_text function, which combines the item name and category into a single text for embedding.
    chunks = []
    for item in db.items.find({}):
        text = build_item_text(item)
        if not text:
            continue
        chunks.append({
            "chunk_id":          f"item_{item['item_id']}",
            "source_collection": "items",
            "source_id":         item["item_id"],
            "text":              text,
            "metadata": {
                "category":         item.get("category"),
                "unit_price_ex_vat": item.get("unit_price_ex_vat"),
                "unit":             item.get("unit"),
            },
        })
    return chunks

# For POs, we include the job type, job description, customer name, and key items in the text to capture the overall context of the order. 
# The metadata includes the status, matched customer ID, matched job type ID, trade name, and total cost for filtered retrieval and evaluation purposes.
def collect_po_chunks(db) -> list[dict]:
    chunks = []
    for po in db.pos.find({}):
        text = build_po_text(po)
        if not text:
            continue
        chunks.append({
            "chunk_id":          f"po_{po['po_number']}",
            "source_collection": "pos",
            "source_id":         po["po_number"],
            "text":              text,
            "metadata":          build_po_metadata(po),
        })
    return chunks


# ---------------------------------------------------------------------------
# Embedding & saving
# ---------------------------------------------------------------------------
# This function takes a list of chunk dicts, embeds their text using the provided model, and saves the chunks with their embeddings to MongoDB. It processes the chunks in batches for efficiency.
def embed_and_save(chunks: list[dict], model: SentenceTransformer, db, batch_size: int = 32):
    """Embed each chunk's text and upsert to MongoDB."""
    if not chunks:
        return 0

    # Batch encode for speed and memory efficiency — we use tqdm to show a progress bar for embedding, which can take some time for large numbers of chunks. 
    # The batch_size parameter controls how many texts are processed at once, which can help manage memory usage.
    texts = [c["text"] for c in chunks]
    print(f"  Encoding {len(texts)} text(s) (batch size = {batch_size})...")
    # The model.encode function returns a numpy array of shape (num_texts, embedding_dim). We set convert_to_numpy=True to get a numpy array directly, 
    # and normalize_embeddings=True to ensure the embeddings are unit vectors, which can improve cosine similarity performance in retrieval.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity to work cleanly
    )

    # Save each chunk and embedding to MongoDB
    saved = 0
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # numpy array -> Python list (BSON-safe)
        chunk["_embedded_at"] = time.time()
        chunk["_model"] = MODEL_NAME

        db[TARGET_COLLECTION].update_one( # upsert by chunk_id to avoid duplicates, and to allow re-running with --force to update embeddings if needed
            {"chunk_id": chunk["chunk_id"]},
            {"$set": chunk},
            upsert=True,
        )
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# def to run the embedding process. It parses command-line arguments to determine which sources to embed, connects to MongoDB, 
# collects chunks, embeds them, and saves them back to the database. It also provides a dry-run option to preview what would be embedded without actually doing it, 
# and a force option to re-embed everything even if it already exists in the database.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emails",    action="store_true", help="Embed emails only")
    parser.add_argument("--job-types", action="store_true", help="Embed job_types only")
    parser.add_argument("--items",     action="store_true", help="Embed items only")
    parser.add_argument("--pos",       action="store_true", help="Embed POs only")
    parser.add_argument("--force",     action="store_true",
                        help="Re-embed even if chunk already exists")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would be embedded, no model loading or DB writes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Encoding batch size (default {BATCH_SIZE})")
    args = parser.parse_args()

    # Load environment variables from .env file (MONGO_URI, MONGO_DB) and connect to MongoDB. We use the MONGO_URI to connect to the database, and MONGO_DB to specify which database to use (defaulting to "trades_quotes" if not set).
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

    db = MongoClient(mongo_uri)[db_name]

    # Decide which sources to embed (default = all if no flag specified)
    do_all = not (args.emails or args.job_types or args.items or args.pos)
    do_emails    = args.emails    or do_all
    do_job_types = args.job_types or do_all
    do_items     = args.items     or do_all
    do_pos       = args.pos       or do_all

    #  Collect chunks
    print(f"\nCollecting chunks from MongoDB ({db_name})...")
    all_chunks: list[dict] = []
    if do_emails:
        chunks = collect_email_chunks(db)
        print(f"  Emails:    {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if do_job_types:
        chunks = collect_job_type_chunks(db)
        print(f"  JobTypes:  {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if do_items:
        chunks = collect_item_chunks(db)
        print(f"  Items:     {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if do_pos:
        chunks = collect_po_chunks(db)
        print(f"  POs:       {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to consider: {len(all_chunks)}")

    if not all_chunks:
        print("Nothing to embed. (Empty source collections?)")
        return

    # Skip already-embedded unless --force is used — this allows you to run the script multiple times without re-embedding everything, 
    # which can save time if you add new data or want to embed just one source. The script checks the database for existing chunk_ids and filters out any chunks that have already been embedded, 
    # unless the --force flag is set, in which case it will re-embed all chunks regardless of whether they already exist in the database.
    if not args.force and not args.dry_run:
        existing = {
            d["chunk_id"]
            for d in db[TARGET_COLLECTION].find({}, {"chunk_id": 1, "_id": 0})
        }
        new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing]
        skipped = len(all_chunks) - len(new_chunks)
        if skipped:
            print(f"  Skipping {skipped} already-embedded chunks (use --force to redo)")
        all_chunks = new_chunks

    if not all_chunks:
        print("Nothing new to embed.")
        return

    #  Dry-run preview to show what would be embedded without actually doing it. This is useful for sanity-checking that the text and metadata look correct before running the full embedding process, 
    # which can take some time. If the --dry-run flag is set, the script will print out the first few chunks that would be embedded, including their chunk_id, a preview of the text, and the metadata fields.
    # This allows you to verify that the chunk construction logic is working as intended before committing to embedding and saving to the database.
    if args.dry_run:
        print("\nDRY RUN — first 5 chunks that would be embedded:\n")
        for chunk in all_chunks[:5]:
            print(f"  {chunk['chunk_id']}")
            print(f"    text: {chunk['text'][:120]}...")
            print(f"    metadata: {list(chunk['metadata'].keys())}")
            print()
        return

    #  Load model (downloads ~80MB on first run)
    print(f"\nLoading embedding model: {MODEL_NAME}")
    print("(First run downloads ~80MB. Subsequent runs use cached model.)")
    model = SentenceTransformer(MODEL_NAME)

    #  Embed and save 
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    saved = embed_and_save(all_chunks, model, db, batch_size=args.batch_size)

    # Indexes 
    print("\nCreating helper indexes...")
    db[TARGET_COLLECTION].create_index("chunk_id", unique=True)
    db[TARGET_COLLECTION].create_index("source_collection")
    db[TARGET_COLLECTION].create_index("source_id")

    #  Summary stats and next steps — after embedding, we print out a summary of how many chunks were embedded, the total number of chunks in the database, 
    # the embedding dimension, and the model used. We also show a distribution of chunks by source collection, which can help verify that the expected data was embedded. 
    # Finally, we provide instructions for creating the Atlas Vector Search index in the MongoDB Atlas dashboard, including the JSON definition for the vector index and a suggestion for the index name.
    total = db[TARGET_COLLECTION].count_documents({})
    print("\n" + "=" * 60)
    print(f"  Embedding complete")
    print("=" * 60)
    print(f"  Saved this run:           {saved}")
    print(f"  Total chunks in DB:       {total}")
    print(f"  Embedding dimension:      {EMBEDDING_DIM}")
    print(f"  Model:                    {MODEL_NAME}")

    # Distribution by source — this shows how many chunks were embedded from each source collection (emails, job_types, items, pos).
    # This can help verify that the expected data was embedded and identify any issues with missing or extra data.
    print(f"\n  Distribution by source:")
    for r in db[TARGET_COLLECTION].aggregate([
        {"$group": {"_id": "$source_collection", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]):
        print(f"    {str(r['_id']):20} {r['count']:>5}")

    print(f"\nNEXT STEP: create the Atlas Vector Search index in cloud.mongodb.com")
    print(f"  → Atlas Search → Create Index → JSON Editor")
    print(f"  → Database: {db_name}, Collection: {TARGET_COLLECTION}")
    print(f"  → Paste this JSON definition:")
    print()
    print('  {')
    print('    "fields": [{')
    print('      "type":          "vector",')
    print('      "path":          "embedding",')
    print(f'      "numDimensions": {EMBEDDING_DIM},')
    print('      "similarity":    "cosine"')
    print('    }]')
    print('  }')
    print()
    print("  Index name suggestion: vector_index")
    print("  Atlas takes ~5 minutes to build the index.")


if __name__ == "__main__":
    main()
