"""
load_po_pdfs.py
================
Phase 3B step: enrich PO PDF metadata in GridFS bucket `po_files`
and cross-link to the pos collection.

PDFs are already in GridFS (uploaded via Streamlit or original ingest).
This script:
  1. Iterates po_files.files documents
  2. Adds missing metadata: po_number, original_filename, size_bytes
  3. Cross-links each PDF to its matching pos document

Usage:
    python ingest\\load_po_pdfs.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gridfs
from pymongo import MongoClient
from tqdm import tqdm

BUCKET_NAME  = "po_files"
PO_NUMBER_RE = re.compile(r"PO-\d{4}-[A-Z]\d{4}", re.IGNORECASE)


def extract_po_number(filename: str) -> str:
    """Extract the PO number from a filename."""
    match = PO_NUMBER_RE.search(filename)
    if match:
        return match.group(0).upper()
    return Path(filename).stem


def main() -> None:
    # ── Connect ───────────────────────────────────────────────
    # Works on Streamlit Cloud (env injected by ingest_runner)
    # and locally (env set by .env via load_dotenv in calling script)
    mongo_uri = os.environ.get("MONGO_URI")
    db_name   = os.environ.get("MONGO_DB", "trades_quotes")

    if not mongo_uri:
        # Last-resort local fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            mongo_uri = os.environ.get("MONGO_URI")
        except ImportError:
            pass

    if not mongo_uri:
        print("ERROR: MONGO_URI not set."); sys.exit(1)

    print(f"Connecting to MongoDB ({db_name})...")
    db = MongoClient(mongo_uri)[db_name]
    fs = gridfs.GridFS(db, collection=BUCKET_NAME)

    # ── Enrich metadata on existing po_files.files docs ──────
    files_coll = db[f"{BUCKET_NAME}.files"]
    all_files  = list(files_coll.find({}))

    if not all_files:
        print("ERROR: No files found in GridFS po_files bucket.")
        print("Upload PDFs via the Streamlit Data Upload tab first.")
        sys.exit(1)

    print(f"\nFound {len(all_files)} file(s) in GridFS po_files bucket.")
    print("Enriching metadata...")

    enriched = 0
    for doc in tqdm(all_files, desc="  enriching"):
        filename  = doc.get("filename", "")
        po_number = extract_po_number(filename)

        update = {}

        # Add po_number if missing
        if not doc.get("po_number"):
            update["po_number"] = po_number

        # Add original_filename if missing
        if not doc.get("original_filename"):
            update["original_filename"] = filename

        # Add size_bytes from GridFS length field if missing
        if not doc.get("size_bytes") and doc.get("length"):
            update["size_bytes"] = doc["length"]

        if update:
            files_coll.update_one({"_id": doc["_id"]}, {"$set": update})
            enriched += 1

    print(f"  Enriched: {enriched} document(s)")
    print(f"  Already complete: {len(all_files) - enriched} document(s)")

    # ── Cross-link to pos collection ──────────────────────────
    print(f"\nCross-linking to pos collection...")
    matched = 0
    pos_count = db.pos.count_documents({})

    for doc in files_coll.find({}, {"po_number": 1, "filename": 1}):
        po_number = doc.get("po_number")
        if po_number:
            result = db.pos.update_one(
                {"po_number": po_number},
                {"$set": {"pdf_gridfs_filename": doc.get("filename")}},
            )
            if result.matched_count:
                matched += 1

    print(f"  Cross-linked: {matched}/{pos_count} PO records")

    # ── Summary ───────────────────────────────────────────────
    files_count  = files_coll.count_documents({})
    chunks_count = db[f"{BUCKET_NAME}.chunks"].count_documents({})
    print(f"\nFinal state:")
    print(f"  {BUCKET_NAME}.files:  {files_count} files")
    print(f"  {BUCKET_NAME}.chunks: {chunks_count} chunks")
    print(f"  pos documents:        {pos_count}")


if __name__ == "__main__":
    main()
