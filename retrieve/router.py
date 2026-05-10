"""
router.py
==========
Module 6 of Phase 5 retrieval orchestration.

LLM-FIRST ROUTER — uses Claude Haiku for almost all classification.

Only structured ID patterns (PO/INV/cust_/email) bypass the LLM because
they're deterministic and unambiguous. Everything else is classified by
the LLM, which handles natural language variations gracefully.

Trade-off accepted: ~1-2 sec extra latency per query, ~€0.001 per query
in exchange for zero keyword maintenance and natural language understanding.
"""

from __future__ import annotations

import os
import re
import json
from functools import lru_cache

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

ROUTER_MODEL = "claude-haiku-4-5-20251001"


@lru_cache(maxsize=1)
def _get_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Cheap deterministic patterns (only kept because they're unambiguous)
# ---------------------------------------------------------------------------

PO_ID_RE = re.compile(r"\b(PO-\d{4}-[A-Z]\d{4})\b", re.IGNORECASE) # e.g. PO-2026-P0042
INVOICE_ID_RE = re.compile(r"\b(INV-\d{4}-\d{4})\b", re.IGNORECASE) # e.g. INV-2026-0001
EMAIL_MSG_RE = re.compile(r"\b(msg_\d{4})\b", re.IGNORECASE) # e.g. msg_2026, added for email-specific queries like "find emails about X" where we can extract an email ID if mentioned
CUSTOMER_ID_RE = re.compile(r"\b(cust_\d{4})\b", re.IGNORECASE) # e.g. cust_0001, added for queries that mention a customer ID directly like "find cust_0001" or "show me cust_0001's info"
EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b") # generic email pattern, added for queries that mention an email address directly like "find info about


def _try_id_pattern_match(query: str) -> dict | None: # def to return dict with paths, params, intent, method if matched, else None
    """Match deterministic ID patterns — fast and zero false positives."""
    q = query.strip()

    if m := PO_ID_RE.search(q): # e.g. "Show me PO-2026-P0042"
        return {
            "paths":  ["mongo", "graph"],
            "params": {"po_number": m.group(1).upper()},
            "intent": "po_lookup",
            "method": "pattern_id",
        }

    if m := INVOICE_ID_RE.search(q): # e.g. "Show me INV-2026-0001"
        return {
            "paths":  ["mongo"],
            "params": {"invoice_id": m.group(1).upper()},
            "intent": "invoice_lookup",
            "method": "pattern_id",
        }

    if m := CUSTOMER_ID_RE.search(q): # e.g. "find cust_0001" or "show me cust_0001's info"
        return {
            "paths":  ["mongo", "graph"],
            "params": {"customer_id": m.group(1).lower()},
            "intent": "customer_lookup",
            "method": "pattern_id",
        }
    if m := EMAIL_MSG_RE.search(q): # e.g. "find msg_2026"
        return {
            "paths":  ["mongo"],
            "params": {"email_id": m.group(1).lower()},
            "intent": "email_lookup",
            "method": "pattern_id",
        }

    if m := EMAIL_RE.search(q): # e.g. "find info about
        return {
            "paths":  ["mongo"],
            "params": {"customer_email": m.group(1).lower()},
            "intent": "customer_lookup",
            "method": "pattern_id",
        }

    return None


# ---------------------------------------------------------------------------
# LLM classification — handles everything else
# ---------------------------------------------------------------------------

LLM_CLASSIFIER_SYSTEM = """You classify user queries about an Irish trades business
chatbot (plumbing, carpentry, electrical). The chatbot has access to:
- A MongoDB database of customers, invoices, POs, items, job types, emails
- A Neo4j graph of relationships between them
- A vector search over text content
- A pricing predictor that uses all three for job estimates

Return ONLY a single valid JSON object. No preamble, no markdown fences.

Available paths (pick 1-3):
- "vector"        : Atlas Vector Search for fuzzy/semantic content search
- "mongo"         : Direct MongoDB lookups by ID, name, email
- "graph"         : Neo4j relationship queries (customer history, etc.)
- "predict_quote" : Pricing prediction for a specific job type
- "list"          : Browse-style queries that list many records

Available intents:
- "pricing_query"           : User asks how much something costs
- "customer_lookup_by_name" : User asks about a specific named customer
- "po_lookup"               : User asks about a specific PO
- "invoice_lookup"          : User asks about a specific invoice
- "list_pos"                : "show me all POs", "list purchase orders"
- "list_customers"          : "show me all customers"
- "list_jobs"               : "what jobs are there"
- "list_invoices"           : "show me recent invoices"
- "list_emails"             : "recent emails"
- "list_summary"            : "give me an overview"
- "general_question"        : Open-ended question requiring search
- "open_search"             : Free-form information retrieval

Required JSON shape:
{
  "intent":  "<one of the intents above>",
  "paths":   ["<path>", ...],
  "params":  {
    "job_name":      "<canonical job name if pricing query>",
    "customer_name": "<full name if customer lookup>",
    "list_target":   "<pos|customers|jobs|invoices|emails|summary if list>"
  }
}

Rules:
- Pricing questions ("how much", "cost", "price", "quote", "estimate") → "pricing_query" + ["predict_quote"]
  - Extract the job_name. Use the EXACT canonical phrase from these known job types:
    Plumbing: "Boiler installation", "Boiler service", "Annual boiler service",
              "Bathroom suite installation", "Radiator replacement",
              "Underfloor heating installation", "Outside tap installation",
              "Tap replacement", "Toilet installation", "Shower installation",
              "Hot water cylinder replacement", "Mains stop valve replacement",
              "Leak repair", "Drain unblocking", "Pipe insulation", "Power flush"
    Electrical: "Full house rewire", "Light fitting installation", "Extra socket installation",
                "EV charger installation", "Smoke alarm installation", "CCTV installation",
                "Cooker wiring", "Electric shower fit", "Consumer unit upgrade",
                "Doorbell installation", "Fan installation", "Garden lighting",
                "Immersion timer fit", "Outdoor socket fit", "Emergency lighting"
    Carpentry: "Floor sanding", "Wooden floor laying", "Kitchen installation",
               "Fitted wardrobe building", "Stairs balustrade replacement",
               "Attic flooring", "Architrave fitting", "Door hanging", "Decking",
               "Skirting fitting", "Shelving installation", "Custom shelving"
  - If user phrasing doesn't match exactly, pick the CLOSEST canonical name
  - "electric shower installation" → "Electric shower fit"
  - "EV charging point" → "EV charger installation"

- Questions about specific named customer ("Gerard Walsh", "Niamh Byrne") → 
  "customer_lookup_by_name" + ["mongo", "graph"], extract customer_name

- "show/list me all X" or "give me an overview" → "list_<X>" + ["list"]

- Questions about emails, "find emails about", "recent enquiries" → ["vector"] + maybe ["mongo"]

- Free-form questions you can't classify → "open_search" + ["vector", "graph"]
"""


def _llm_classify(query: str) -> dict: # def to call Claude Haiku and return dict with paths, params, intent, method
    """Use Claude Haiku to classify the query. Returns routing dict."""
    client = _get_client() # cached client for efficiency
    response = client.messages.create( # response from Claude Haiku with the system prompt and user query
        model=ROUTER_MODEL,
        max_tokens=400,
        system=LLM_CLASSIFIER_SYSTEM, # system prompt with instructions for classification
        messages=[{"role": "user", "content": f"Classify this query: {query}"}],
    )
    text = response.content[0].text.strip()

    # Strip markdown fences if Claude added them despite instructions
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?|\n?```$", "", text).strip()

    try:
        result = json.loads(text) # parse the JSON response from Claude
        # Sanity check — must have intent and paths
        if not result.get("intent") or not result.get("paths"):
            raise ValueError("Missing required fields")
        # Default empty params if missing
        if "params" not in result:
            result["params"] = {}
        # Strip any null params
        result["params"] = {k: v for k, v in result["params"].items() if v}
        result["method"] = "llm"
        return result
    except (json.JSONDecodeError, ValueError): # if parsing fails or required fields are missing, return a fallback
        # Last-resort fallback
        return { # if Claude fails to respond properly, we return a safe fallback that tries both vector and graph search for open-ended queries
            "intent": "open_search",
            "paths":  ["vector", "graph"],
            "params": {},
            "method": "llm_fallback",
        }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> dict: # main function to classify a user query, first trying deterministic patterns, then falling back to LLM classification
    """
    Classify a user query.

    Strategy:
      1. Try ID patterns (PO/INV/cust_/email) — instant if matched
      2. Otherwise, ask the LLM to classify

    Returns dict: {paths, params, intent, method}
    """
    if not query or not query.strip(): #    handle empty queries gracefully
        return {
            "paths":  [],
            "params": {},
            "intent": "empty",
            "method": "skip",
        }

    # Step 1 — fast ID patterns
    id_match = _try_id_pattern_match(query) # try to match deterministic ID patterns first for speed and accuracy
    if id_match: #  if a deterministic pattern is matched, return it immediately without calling the LLM
        return id_match # if a deterministic pattern is matched, return the corresponding routing dict immediately

    # Step 2 — LLM classification for everything else
    return _llm_classify(query)


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__": # simple command-line interface for testing the router with example queries
    import sys
    test_queries = sys.argv[1:] if len(sys.argv) > 1 else [ # default test queries if none provided via command line
        "Show me PO-2026-P0042",
        "find gerard walsh info",
        "What's Gerard Walsh's phone number?",
        "How much for a boiler installation?",
        "give me price for electric shower installation",
        "cost of EV charging point",
        "rate for cooker wiring please",
        "show me all POs",
        "list all customers",
        "what jobs are there?",
        "give me an overview",
        "find emails about leaks",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = classify_query(q)
        print(f"  Intent: {result['intent']}  ({result['method']})")
        print(f"  Paths:  {result['paths']}")
        print(f"  Params: {result['params']}")
