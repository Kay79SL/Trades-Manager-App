"""
extract_entities.py
====================
Phase 3 of DACARag: LLM entity extraction from email bodies.

For each email in MongoDB `emails` collection:
  1. Build a structured-output prompt with reference data (job types,
     customers, items)
  2. Call Claude Haiku to extract structured fields from the email body
  3. Save the extraction back to MongoDB as `email.extracted` sub-document

Extracted fields per email:
  - trade_needed          plumber | carpenter | electrician
  - job_type_id           one of 60 valid job_type_ids (or null)
  - job_type_name         human-readable job name
  - mentioned_items       list of item_ids referenced/implied in body
  - urgency               low | medium | high
  - is_returning_customer bool
  - matched_customer_id   customer_id if returning, else null
  - confidence            high | medium | low
  - key_phrases           list of distinctive phrases
  - reasoning             one-sentence explanation

Cost: ~€0.10 total for 100 emails using claude-haiku-4-5.

Usage:
    python ingest\\extract_entities.py
    python ingest\\extract_entities.py --limit 5     # test on 5 emails
    python ingest\\extract_entities.py --force        # re-extract all (overwrite)
    python ingest\\extract_entities.py --dry-run      # print first prompt, no API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"  # Haiku is fast and cheap — ~€0.10 for 100 emails
MAX_TOKENS = 800  # emails are short so 800 tokens is enough for the JSON response

SYSTEM_PROMPT = """You are an entity extraction assistant for a chatbot serving Irish trades \
(plumbers, carpenters, electricians). You read incoming customer quote-request emails and \
extract structured data needed for downstream retrieval.

You always respond with a single valid JSON object — no preamble, no commentary, no markdown \
code fences. The JSON must conform exactly to the schema provided in the user message.

Be conservative and precise:
  - If you cannot determine a field with reasonable confidence, set it to null.
  - For mentioned_items, only include items explicitly named or unambiguously implied.
  - For matched_customer_id, only fill it in if the from_email clearly matches a known
    customer's email address.
  - urgency is "high" only when the email contains explicit urgency markers
    (asap, today, emergency, leak now, water damage, no power, gas smell, etc).
  - urgency is "medium" for normal weekly-timeline requests.
  - urgency is "low" for general enquiries with no time pressure mentioned.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prompt(
    email: dict,
    job_types: list[dict],
    customers_by_email: dict[str, dict],
    items_summary: str,
) -> str:
    """Build the per-email user prompt."""
    job_options = "\n".join( # NB: job_type_id is what we want to extract, not job_name
        f"  - {j['job_type_id']}: {j['job_name']} ({j['trade']})" # include trade in job options to help disambiguate for the model
        for j in job_types
    )

    sender_email = (email.get("from_email") or "").strip().lower()
    matched = customers_by_email.get(sender_email)
    customer_hint = ""
    if matched:
        # If the sender's email already exists in our customers table, tell the model
        # explicitly — this avoids ambiguity and prevents a known customer being missed
        customer_hint = (
            f"\nIMPORTANT: The from_email matches a known customer in our database: "
            f"{matched['customer_id']} ({matched['first_name']} {matched['last_name']}, "
            f"prefers {matched['preferred_trade']}). Set is_returning_customer=true and "
            f"matched_customer_id='{matched['customer_id']}'."
        )

    return f"""Extract entities from this email.

EMAIL METADATA
  email_id:        {email['email_id']}
  from_name:       {email.get('from_name', '')}
  from_email:      {email.get('from_email', '')}
  subject:         {email.get('subject', '')}
  trade_contacted: {email.get('trade_contacted', '')}
{customer_hint}

EMAIL BODY
\"\"\"
{email.get('body', '')}
\"\"\"

VALID job_type_id VALUES (pick the BEST single match):
{job_options}

ITEMS CATALOGUE (88 total — match items mentioned in the email by name or description):
{items_summary}

REQUIRED OUTPUT SCHEMA (respond with this JSON only — no fences, no commentary):
{{
  "trade_needed":          "plumber" | "carpenter" | "electrician",
  "job_type_id":           "<one of the IDs above, or null>",
  "job_type_name":         "<human-readable job name>",
  "mentioned_items":       ["<item_id>", ...],
  "urgency":               "low" | "medium" | "high",
  "is_returning_customer": true | false,
  "matched_customer_id":   "<customer_id or null>",
  "confidence":            "high" | "medium" | "low",
  "key_phrases":           ["<distinctive phrase 1>", ...],
  "reasoning":             "<one-sentence explanation of your classification>"
}}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse Claude's response, tolerating optional markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip the ```json or ``` opening line before parsing
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def extract_one(client: Anthropic, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Call Claude and parse the response as JSON."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text  # Claude always returns content[0] for non-streaming calls
    return parse_json_response(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max emails to process (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract emails that already have extracted data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the first prompt and exit (no API calls)")
    args = parser.parse_args()

    load_dotenv()  # load MONGO_URI, MONGO_DB, ANTHROPIC_API_KEY from .env
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

    print(f"Connecting to MongoDB ({db_name})...")
    db = MongoClient(mongo_uri)[db_name]

    print("Loading reference data...")
    job_types = list(db.job_types.find(
        {}, {"_id": 0, "job_type_id": 1, "job_name": 1, "trade": 1}
    ))
    customers = list(db.customers.find(
        {}, {"_id": 0, "customer_id": 1, "first_name": 1, "last_name": 1,
             "email": 1, "preferred_trade": 1}
    ))
    items = list(db.items.find(
        {}, {"_id": 0, "item_id": 1, "item_name": 1, "category": 1}
    ))

    customers_by_email = {
        c["email"].lower(): c for c in customers if c.get("email")
    }
    items_summary = "\n".join(
        f"  - {i['item_id']}: {i['item_name']}" for i in items
    )

    print(f"  {len(job_types)} job types · {len(customers)} customers · {len(items)} items")

    # Find emails to process
    # Skip emails that already have an extracted field unless --force is set
    query = {} if args.force else {"extracted": {"$exists": False}}
    cursor = db.emails.find(query).sort("email_id", 1)
    if args.limit:
        cursor = cursor.limit(args.limit)
    emails = list(cursor)
    print(f"\nEmails to process: {len(emails)}")

    if not emails:
        print("Nothing to do. Use --force to re-extract.")
        return

    if args.dry_run:
        sample_prompt = build_user_prompt(
            emails[0], job_types, customers_by_email, items_summary
        )
        print("\n" + "=" * 60)
        print("DRY RUN — first email prompt")
        print("=" * 60)
        print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT)
        print("\n--- USER ---\n" + sample_prompt)
        return

    client = Anthropic(api_key=api_key)
    success = 0
    failures: list[tuple[str, str]] = []

    for email in tqdm(emails, desc="Extracting"):
        prompt = build_user_prompt(
            email, job_types, customers_by_email, items_summary
        )
        try:
            extracted = extract_one(client, SYSTEM_PROMPT, prompt)
            extracted["_extracted_at"] = time.time()  # unix timestamp for audit trail
            extracted["_model"] = MODEL  # record which model was used in case we upgrade later
            db.emails.update_one(
                {"email_id": email["email_id"]},
                {"$set": {"extracted": extracted}},
            )
            success += 1
        except Exception as e:
            failures.append((email["email_id"], str(e)[:200]))

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Successful extractions: {success}")
    print(f"Failures:               {len(failures)}")
    if failures:
        print("\nFirst 5 failures:")
        for eid, err in failures[:5]:
            print(f"  {eid}: {err}")

    # Sanity check
    print(f"\nVerifying in MongoDB...")
    total = db.emails.count_documents({})
    extracted_count = db.emails.count_documents({"extracted": {"$exists": True}})
    print(f"  Emails with extracted field: {extracted_count}/{total}")

    # Distribution check
    print(f"\n  Trade distribution:")
    pipeline = [
        {"$match": {"extracted": {"$exists": True}}},
        {"$group": {"_id": "$extracted.trade_needed", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    for r in db.emails.aggregate(pipeline):
        print(f"    {str(r['_id']):15} {r['count']:>5}")

    print(f"\n  Urgency distribution:")
    pipeline = [
        {"$match": {"extracted": {"$exists": True}}},
        {"$group": {"_id": "$extracted.urgency", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    for r in db.emails.aggregate(pipeline):
        print(f"    {str(r['_id']):15} {r['count']:>5}")

    print(f"\n  Returning customers identified: " +
          str(db.emails.count_documents({"extracted.is_returning_customer": True})))


if __name__ == "__main__":
    main()
