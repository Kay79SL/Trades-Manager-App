"""
extract_pos_from_pdf.py
========================
Purchase Order ingestion via PDF + AI.

This is the PRIMARY (and only) PO ingestion path for DACARag. It mirrors the
realistic production scenario: customers send PDF purchase orders, and the
chatbot must extract structured data from them automatically.

Pipeline:
  1. Read each PDF from GridFS po_files bucket
  2. Extract text content with pypdf (fast, no OCR needed for digital PDFs)
  3. Send the text to Claude with a structured-extraction prompt
  4. Parse the JSON response
  5. Match cust_email back to a customer_id in the customers collection.
     If no match found, AUTO-ADD a new customer record from the PO data
     (with provenance markers) so referential integrity is maintained.
  6. Save to MongoDB `pos` collection — primary PO storage

Customer matching strategy (in order):
  a) Email match (case-insensitive, whitespace-stripped) on existing customers
  b) First-name + last-name exact match on existing customers
  c) If neither matches: create new customer record with PO data,
     tagged with _source='added_from_po' and _added_via_po=<po_number>

Within-batch deduplication: customers added during this run are visible to
subsequent POs (database is re-queried each time), so multiple POs from
the same new customer all link to a single customer_id.

After this script runs, downstream pipeline steps work unchanged:
  - load_pos_to_neo4j.py — adds PO nodes to Neo4j with FOR_CUSTOMER, FOR_JOB, CONTAINS_ITEM edges

"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Conditional imports - tell user what to install if missing
try:
    import pypdf
except ImportError:
    print("ERROR: missing dependency 'pypdf'.")
    print("Install with: python -m pip install pypdf")
    sys.exit(1)

import gridfs
from anthropic import Anthropic
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_COLLECTION = "pos"   # main PO collection (PDF + AI is the only ingestion path)
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 2000   # POs are bigger than emails — more line items, more text

PO_NUMBER_RE = re.compile(r"PO-\d{4}-[A-Z]\d{4}", re.IGNORECASE)


SYSTEM_PROMPT_PDF_PO = """You are a purchase order extraction assistant for a chatbot serving Irish trades.
You read text extracted from PDF purchase orders and produce a structured JSON representation
suitable for downstream storage in MongoDB.

You always respond with a single valid JSON object — no preamble, no commentary, no markdown
code fences. The JSON must conform exactly to the schema provided in the user message.

Be precise:
  - Currency values must be extracted as floats (no euro symbol, no thousands separators).
    Example: "€1,712.60" must become 1712.60.
  - For line_items, capture every line shown in the PDF. Each line MUST include
    item_id, description, quantity, unit, unit_price, and line_total.
  - Dates must be in YYYY-MM-DD format if discernible, otherwise the literal string shown.
  - If a field is not present in the PDF, set it to null. Don't fabricate.
  - For job_type, use the literal text from the PDF's "Job type" field (do NOT try to
    map it to a canonical job_type_id — that's done in a later step by load_pos_to_neo4j.py).
"""


# ---------------------------------------------------------------------------
# PDF reading — from bytes (GridFS) instead of file path
# ---------------------------------------------------------------------------

def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str: # def to extract text from PDF bytes using pypdf
    """
    Extract all text from PDF bytes. Works for digital PDFs (which yours are
    since they were generated from Excel). Would need OCR for scanned PDFs.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes)) # read from bytes, not file path
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def detect_po_number(filename: str, pdf_text: str) -> str: # def to detect PO number from filename or PDF text
    """
    Try filename first (most reliable), fall back to scanning PDF text.
    """
    match = PO_NUMBER_RE.search(filename)
    if match:
        return match.group(0).upper()
    match = PO_NUMBER_RE.search(pdf_text)
    if match:
        return match.group(0).upper()
    return Path(filename).stem


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def build_prompt(pdf_text: str, po_number_hint: str) -> str: # def to build the user prompt for Claude, including the PDF text and the required output schema
    """Build the user prompt for one PO."""
    return f"""Extract a structured purchase order from this PDF text.

PO_NUMBER (from filename, treat as authoritative if present): {po_number_hint}

PDF TEXT (extracted by pypdf — formatting may be slightly off):
\"\"\"
{pdf_text}
\"\"\"

REQUIRED OUTPUT SCHEMA (respond with this JSON only): 
{{
  "po_number":           "<PO number e.g. PO-2026-P0042>",
  "po_date":             "<YYYY-MM-DD or null>",
  "trade_name":          "<trade business name>",
  "trade_addr":          "<trade address line 1>",
  "trade_city":          "<trade town/city>",
  "trade_eircode":       "<eircode or null>",
  "trade_email":         "<trade email>",
  "trade_phone":         "<trade phone>",
  "trade_vat":           "<trade VAT number or null>",
  "cust_name":           "<customer full name>",
  "cust_addr":           "<customer address>",
  "cust_city":           "<customer town/city>",
  "cust_eircode":        "<eircode or null>",
  "cust_email":          "<customer email>",
  "cust_phone":          "<customer phone>",
  "po_status":           "<status text or null>",
  "job_type":            "<job type as written, e.g. 'Bathroom suite installation'>",
  "job_desc":            "<job description>",
  "delivery_date":       "<YYYY-MM-DD or null>",
  "payment_terms":       "<payment terms>",
  "line_items": [
    {{
      "item_id":            "<e.g. it_pl_009>",
      "description":        "<item description>",
      "quantity":           <number>,
      "unit":               "<e.g. unit, metre, pair>",
      "unit_price_ex_vat":  <number>,
      "line_total_ex_vat":  <number>
    }}
  ],
  "materials_subtotal":  <number>,
  "labour_cost":         <number>,
  "subtotal_ex_vat":     <number>,
  "vat_23pct":           <number>,
  "total_inc_vat":       <number>,
  "extraction_confidence": "high" | "medium" | "low",
  "extraction_notes":    "<any concerns or ambiguities the parser noticed>"
}}
"""


def parse_json_response(text: str) -> dict[str, Any]: # def to parse the JSON response from Claude, tolerating optional markdown fences
    """Parse Claude's response, tolerating optional markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n") # find the end of the opening ``` line
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def extract_one(client: Anthropic, system_prompt: str, user_prompt: str) -> dict: # def to extract information from a single PDF using Claude
    """Call Claude and parse the response as JSON."""
    response = client.messages.create( # Call the Claude API with the system prompt and user prompt, and parse the response as JSON
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return parse_json_response(response.content[0].text)


# ---------------------------------------------------------------------------
# Customer matching with auto-add
# ---------------------------------------------------------------------------

def normalize_email(email: str | None) -> str: # def to normalize email addresses for matching (lowercase, strip whitespace)
    """Lowercase and strip whitespace from an email address."""
    if not email:
        return ""
    return email.strip().lower()


def next_customer_id(db) -> str: # def to generate the next sequential customer_id (cust_NNNN) for auto-adding new customers
    """Generate the next sequential customer_id (cust_NNNN)."""
    last = db.customers.find_one(
        {"customer_id": {"$regex": r"^cust_\d+$"}},
        sort=[("customer_id", -1)],
    )
    if not last:
        return "cust_0001"
    last_num = int(last["customer_id"].split("_")[1])
    return f"cust_{last_num + 1:04d}"


def infer_trade_from_po(po: dict) -> str: # def to infer the preferred trade from the trade_name field in the PO, using simple keyword matching
    """Guess preferred_trade from the trade_name in the PO."""
    name = (po.get("trade_name") or "").lower()
    if "plumb" in name or "heating" in name:
        return "plumber"
    if "carpent" in name or "joiner" in name:
        return "carpenter"
    if "electric" in name:
        return "electrician"
    return "plumber"  # safe default


def match_or_add_customer(db, po: dict, auto_add: bool = True) -> tuple[str | None, str]: # def to match a PO's customer to an existing customer record, or auto-add if no match found (with provenance)
    """
    Try to match a PO's customer to an existing customers record. If no match
    found and auto_add=True, create a new customer record from the PO data.

    Returns (customer_id_or_None, action_taken).
    Possible actions: "matched_email", "matched_name", "added_new", "no_match"
    """
    cust_email = normalize_email(po.get("cust_email")) # normalize the email for matching means stripping whitespace and lowercasing; if no email, becomes empty string
    cust_name = (po.get("cust_name") or "").strip() # get the customer name, stripping whitespace (but not altering case, since matching is exact)

    # Strategy 1: email match
    if cust_email: # if there's an email, try to match it to an existing customer (case-insensitive)
        existing = db.customers.find_one( # find a customer with a matching email (case-insensitive)
            {"email": cust_email},
            {"customer_id": 1, "_id": 0}
        )
        if existing:
            return existing["customer_id"], "matched_email"

    # Strategy 2: name match (first + last)
    if cust_name: # if there's a customer name, try to match it to an existing customer by splitting into first and last name (exact match)
        parts = cust_name.split(maxsplit=1)
        if len(parts) == 2:
            first, last = parts
            existing = db.customers.find_one( # find a customer with a matching first and last name (exact match)
                {"first_name": first, "last_name": last},
                {"customer_id": 1, "_id": 0}
            )
            if existing:
                return existing["customer_id"], "matched_name"

    # Strategy 3: auto-add
    if not auto_add: # if auto-add is disabled, return no match instead of adding a new customer
        return None, "no_match"

    if not cust_email and not cust_name: # if we have neither email nor name, we have no way to match or add, so return no match
        return None, "no_match"

    name_parts = cust_name.split(maxsplit=1)
    first = name_parts[0] if len(name_parts) >= 1 else ""
    last = name_parts[1] if len(name_parts) >= 2 else ""

    new_id = next_customer_id(db) # generate the next customer_id (cust_NNNN) for the new customer
    new_customer = {
        "customer_id":          new_id,
        "first_name":           first,
        "last_name":            last,
        "email":                cust_email,
        "phone":                po.get("cust_phone", ""),
        "address_line_1":       po.get("cust_addr", ""),
        "address_line_2":       "",
        "county":               po.get("cust_city", ""),
        "eircode":              po.get("cust_eircode", ""),
        "preferred_trade":      infer_trade_from_po(po),
        "first_contact_date":   time.strftime("%Y-%m-%d"),
        "_source":              "added_from_po",
        "_added_via_po":        po.get("po_number", ""),
        "_added_at":            time.time(),
    }
    db.customers.insert_one(new_customer)

    return new_id, "added_new"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None: # main function to orchestrate the entire extraction process, including reading PDFs from GridFS, extracting text, calling Claude, matching/adding customers, and saving results to MongoDB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max PDFs to process (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract PDFs even if already in pos collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the prompt for the first PDF, no API calls")
    parser.add_argument("--no-auto-add", action="store_true",
                        help="Don't auto-add unknown customers")
    args = parser.parse_args()

    load_dotenv() # Load environment variables from .env file (MongoDB URI, Anthropic API key)
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

    db = MongoClient(mongo_uri)[db_name]

    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

    initial_customer_count = db.customers.count_documents({}) # Log the initial number of customers in the database for later comparison
    print(f"Current customers in DB: {initial_customer_count}")
    if args.no_auto_add:
        print("Auto-add of unknown customers: DISABLED")
    else:
        print("Auto-add of unknown customers: ENABLED")
    print()

    # ── Read PDFs from GridFS po_files bucket ────────────────────────────────
    fs = gridfs.GridFS(db, collection="po_files") # connect to the GridFS bucket named "po_files" in MongoDB, where the PDFs are stored
    grid_docs = list(fs.find())

    if not grid_docs:
        print("ERROR: no PDFs found in GridFS po_files bucket.")
        print("Upload PDFs via the Streamlit Data Upload tab first.")
        sys.exit(1)

    print(f"Found {len(grid_docs)} PDF(s) in GridFS po_files bucket\n")

    # Skip already-extracted unless --force
    if not args.force: # if --force is not set, filter out PDFs that have already been extracted (based on po_number in the target collection) to avoid duplicates
        existing = {d["po_number"] for d in db[TARGET_COLLECTION].find(
            {}, {"po_number": 1, "_id": 0}
        )}
        new_docs = [
            g for g in grid_docs
            if detect_po_number(g.filename, "") not in existing
        ]
        if len(new_docs) < len(grid_docs): # if some PDFs are being skipped because they've already been extracted, log that information
            print(f"  {len(grid_docs) - len(new_docs)} already extracted (skipping). "
                  f"Use --force to redo all.")
        grid_docs = new_docs

    if args.limit and len(grid_docs) > args.limit: # if a limit is set and we have more PDFs than the limit, truncate the list to the limit and log that we're limiting the number of PDFs being processed
        grid_docs = grid_docs[:args.limit]
        print(f"  Limiting to {args.limit} PDF(s)")

    if not grid_docs: # if there are no PDFs left to process after filtering, log that and exit
        print("Nothing to process. Use --force to re-extract.")
        return

    # Dry run
    if args.dry_run: # if --dry-run is set, do a dry run by showing the prompt for the first PDF without making any API calls, to allow the user to verify the prompt before running the full extraction
        sample = grid_docs[0]
        pdf_bytes = sample.read()
        text = extract_pdf_text_from_bytes(pdf_bytes)
        po_num = detect_po_number(sample.filename, text)
        prompt = build_prompt(text, po_num)
        print("=" * 60)
        print(f"DRY RUN — would extract from: {sample.filename}")
        print(f"PDF text length: {len(text)} chars")
        print(f"PO number detected: {po_num}")
        print("=" * 60)
        print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT_PDF_PO)
        print("\n--- USER (truncated to 1500 chars) ---")
        print(prompt[:1500])
        print("\n... (full prompt is " + str(len(prompt)) + " chars)")
        return

    # Real run
    client = Anthropic(api_key=api_key)
    success = 0
    failures: list[tuple[str, str]] = []

    action_stats = {
        "matched_email":  0,
        "matched_name":   0,
        "added_new":      0,
        "no_match":       0,
    }
    new_customers_log: list[tuple[str, str, str]] = []

    for grid_out in tqdm(grid_docs, desc="Extracting"):
        try:
            # Read bytes from GridFS and extract text
            pdf_bytes = grid_out.read() # read the PDF bytes from GridFS
            text = extract_pdf_text_from_bytes(pdf_bytes)
            if not text.strip():
                raise ValueError("PDF text extraction returned empty")

            po_num = detect_po_number(grid_out.filename, text)
            prompt = build_prompt(text, po_num)

            extracted = extract_one(client, SYSTEM_PROMPT_PDF_PO, prompt)

            # Match or auto-add customer
            customer_id, action = match_or_add_customer( # match the extracted PO's customer to an existing customer in the database, or auto-add if no match found (depending on args.no_auto_add)
                db, extracted, auto_add=not args.no_auto_add
            )
            extracted["matched_customer_id"] = customer_id
            extracted["_customer_match_action"] = action
            action_stats[action] += 1

            if action == "added_new":
                new_customers_log.append((
                    extracted.get("po_number", po_num),
                    customer_id,
                    extracted.get("cust_name", ""),
                ))

            extracted["_source_pdf"] = grid_out.filename
            extracted["_extracted_at"] = time.time()
            extracted["_model"] = MODEL
            extracted["_extraction_method"] = "pdf_via_llm"

            db[TARGET_COLLECTION].update_one(
                {"po_number": extracted.get("po_number", po_num)},
                {"$set": extracted},
                upsert=True,
            )
            success += 1
        except Exception as e:
            failures.append((grid_out.filename, str(e)[:200]))

    db[TARGET_COLLECTION].create_index("po_number", unique=True)
    db[TARGET_COLLECTION].create_index("matched_customer_id")
    db[TARGET_COLLECTION].create_index("po_status")

    # Summary
    print("\n" + "=" * 50)
    print(f"Successful extractions: {success}")
    print(f"Failures:               {len(failures)}")
    if failures:
        print("\nFailures:")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")

    print(f"\nCustomer matching outcomes:")
    print(f"  Matched by email:       {action_stats['matched_email']}")
    print(f"  Matched by name:        {action_stats['matched_name']}")
    print(f"  Added as new customer:  {action_stats['added_new']}")
    print(f"  No match (unresolved):  {action_stats['no_match']}")

    if new_customers_log: # if we added new customers, log their PO number, assigned customer_id, and name for reference
        print(f"\nNew customers added to DB this run:")
        for po_num, cust_id, name in new_customers_log:
            print(f"  {po_num} → {cust_id} ({name})")

    final_customer_count = db.customers.count_documents({}) # count the final number of customers in the database after processing, to see how many new customers were added
    if final_customer_count != initial_customer_count: # if the customer count has changed, log the before and after counts and how many were added
        print(f"\nCustomer count: {initial_customer_count} → {final_customer_count} "
              f"(+{final_customer_count - initial_customer_count})")

    total = db[TARGET_COLLECTION].count_documents({}) # count the total number of POs in the target collection after processing, and how many of them are linked to a customer (matched_customer_id not null)
    matched_cust = db[TARGET_COLLECTION].count_documents({"matched_customer_id": {"$ne": None}}) # count how many POs have a matched_customer_id that is not null, indicating they are linked to a customer record
    print(f"\nTotal POs in {TARGET_COLLECTION}: {total}")
    print(f"POs linked to customer:   {matched_cust}/{total}")

    print(f"\nConfidence distribution:")
    for r in db[TARGET_COLLECTION].aggregate([
        {"$match": {"extraction_confidence": {"$exists": True}}},
        {"$group": {"_id": "$extraction_confidence", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]):
        print(f"  {str(r['_id']):15} {r['count']}")

    print(f"\nNext steps:")
    print(f"  python ingest\\load_pos_to_neo4j.py   # POs into graph")


if __name__ == "__main__":
    main()
