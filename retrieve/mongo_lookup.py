"""
mongo_lookup.py
================
Module 2 of Phase 5 retrieval orchestration.

Direct, exact fetches from MongoDB. No AI involved — just CRUD lookups by
ID, email, customer, etc. Used when the user's intent is "give me this
specific record" rather than "find me something similar".

Includes list/summary functions for open-ended queries like
"show me all POs" or "give me an overview".

Used by:
  - orchestrator.py (when router picks "mongo" or "list" path)
  - graph_search.py (to enrich graph results with full Mongo documents)
  - predict_quote.py (to fetch invoice statistics)
"""

from __future__ import annotations

import os
from functools import lru_cache
from shutil import which
from typing import Any
import re

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


@lru_cache(maxsize=1)
def _get_db():# Cached MongoDB client for reuse across lookups
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    return MongoClient(mongo_uri)[db_name]


# ---------------------------------------------------------------------------
# Customer lookups
# ---------------------------------------------------------------------------

def get_customer(customer_id: str) -> dict | None: #  def to fetch a customer by their canonical customer_id (e.g. "cust_1234")
    """Fetch a single customer by canonical customer_id."""
    return _get_db().customers.find_one({"customer_id": customer_id}, {"_id": 0})


def get_customer_by_email(email: str) -> dict | None: # def to fetch a customer by email address, case-insensitive (e.g. "
    """Find a customer by email (case-insensitive)."""
    if not email:
        return None
    return _get_db().customers.find_one(
        {"email": email.strip().lower()}, {"_id": 0}
    )


def get_customers_by_name(name: str) -> list[dict]: # def to fetch customers by name, which may be ambiguous. If the name has two parts, treat them as first and last name; otherwise, search for matches in either field.
    """
    Find customers matching a name. May return multiple
    (e.g. 3 Gerards in the dataset). Caller must disambiguate.
    """
    if not name:
        return []
    parts = name.strip().split(maxsplit=1) # If there are two parts, treat as first and last name; otherwise search both fields
    if len(parts) == 2:
        first, last = parts
        return list(_get_db().customers.find(
            {"first_name": first, "last_name": last}, {"_id": 0}
        ))
    return list(_get_db().customers.find(
        {"$or": [{"first_name": name}, {"last_name": name}]}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Invoice lookups
# ---------------------------------------------------------------------------

def get_invoice(invoice_id: str) -> dict | None: # def to fetch an invoice by its canonical invoice_id (e.g. "INV-2025-0001")
    """Fetch a single invoice by invoice_id."""
    return _get_db().invoices.find_one({"invoice_id": invoice_id}, {"_id": 0})


def get_invoices_for_customer(customer_id: str, limit: int = 20) -> list[dict]: # def to fetch recent invoices for a customer, sorted by invoice_date descending. Limit to 20 by default to avoid overwhelming the user if they have many invoices.
    """All invoices for a customer, most recent first."""
    return list(_get_db().invoices.find(
        {"customer_id": customer_id}, {"_id": 0}
    ).sort("invoice_date", -1).limit(limit))


def get_invoices_for_job_type(job_type_id: str, limit: int = 50) -> list[dict]: # def to fetch invoices by job_type_id, which is useful for pricing statistics. Sorted by invoice_date descending, limited to 50 to avoid overwhelming the user.
    """All invoices for a particular job type — used for pricing statistics."""
    return list(_get_db().invoices.find(
        {"job_type_id": job_type_id}, {"_id": 0}
    ).limit(limit))


def get_invoice_line_items(invoice_id: str) -> list[dict]: # def to fetch line items for a given invoice_id, which is used to show the breakdown of an invoice or to calculate pricing statistics. Returns an empty list if there are no line items for the invoice.
    """All line items belonging to a particular invoice."""
    return list(_get_db().invoice_items.find(
        {"invoice_id": invoice_id}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# PO lookups
# ---------------------------------------------------------------------------

def get_po(po_number: str) -> dict | None: #     def to fetch a PO by its po_number (e.g. "PO-2026-P0058")
    """Fetch a single PO by po_number."""
    return _get_db().pos.find_one({"po_number": po_number}, {"_id": 0})


def get_pos_for_customer(customer_id: str) -> list[dict]: # def to fetch all POs linked to a particular customer_id. This is based on the "matched_customer_id" field that was extracted by the LLM, 
    # so it may not be perfect, but it's the best way to find relevant POs for a customer when we don't have a direct ID match. Returns an empty list if there are no POs for the customer.
    """All POs linked to a particular customer."""
    return list(_get_db().pos.find(
        {"matched_customer_id": customer_id}, {"_id": 0}
    ))


def get_pos_for_job_type(job_type_text: str) -> list[dict]: #  def to fetch POs by their job_type field, which is a free-form string extracted from the PDF. 
    # This allows us to find POs that are relevant to a particular job type even if we don't have a perfect match on the canonical job_type_id. Uses a case-insensitive substring match, 
    # so searching for "boiler" would match POs with job_type like "Boiler installation and repair". Returns an empty list if no matches are found or if the input text is empty.
    """
    Fuzzy match POs by their job_type field (free-form string from PDF).
    Case-insensitive substring match.
    """
    if not job_type_text:
        return []
    pattern = re.escape(job_type_text)
    return list(_get_db().pos.find(
        {"job_type": {"$regex": pattern, "$options": "i"}}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Email lookups
# ---------------------------------------------------------------------------

def get_email(email_id: str) -> dict | None: # def to fetch an email by its canonical email_id (e.g. "msg_0001")
    """Fetch a single email by email_id."""
    return _get_db().emails.find_one({"email_id": email_id}, {"_id": 0})


def get_emails_for_customer(customer_id: str) -> list[dict]: # def to fetch all emails linked to a particular customer_id based on the "extracted.matched_customer_id" field. This allows us to find emails that are relevant to a customer 
    # even if we don't have a perfect ID match, since the LLM extraction may not be 100% accurate. Returns an empty list if there are no emails for the customer.
    """All emails for a customer (matched via the LLM extraction)."""
    return list(_get_db().emails.find(
        {"extracted.matched_customer_id": customer_id}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Job type and item lookups
# ---------------------------------------------------------------------------

def get_job_type(job_type_id: str) -> dict | None: #   def to fetch a job type by its canonical job_type_id (e.g. "pl_01"). This is used to resolve the job_type_id from an invoice or PO to get the display name and trade, 
    # which can be helpful for showing more user-friendly information or for matching to POs that only have free-form job type text.
    """Fetch a job type by canonical id (e.g. 'pl_01')."""
    return _get_db().job_types.find_one({"job_type_id": job_type_id}, {"_id": 0})


def get_job_type_by_name(job_name: str) -> dict | None: # def to fetch a job type by its display name (e.g. "Boiler installation"). This is used for fuzzy matching when we only have a job name from a PO or email and want to find the corresponding canonical job type. 
    # Uses a case-insensitive exact match (i.e. regex with ^ and $) to allow for
    """Fetch a job type by display name (e.g. 'Boiler installation')."""
    if not job_name:
        return None
    return _get_db().job_types.find_one(
        {"job_name": {"$regex": f"^{re.escape(job_name)}$", "$options": "i"}},
        {"_id": 0}
    )


def get_item(item_id: str) -> dict | None: # def to fetch an item by its canonical item_id (e.g. "it_pl_009"). This is used to resolve the item_id from an invoice line item to get the display name and details of the item, 
    # which can be helpful for showing more user-friendly information or for calculating pricing statistics.
    """Fetch an item by canonical id (e.g. 'it_pl_009')."""
    return _get_db().items.find_one({"item_id": item_id}, {"_id": 0})


def get_items_by_ids(item_ids: list[str]) -> list[dict]: # def to fetch multiple items by a list of item_ids, which is useful for batch fetching the details of all items on an invoice or PO. Returns an empty list if the input list is empty or if no matches are found.
    """Bulk fetch items by a list of ids."""
    if not item_ids:
        return []
    return list(_get_db().items.find(
        {"item_id": {"$in": item_ids}}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# High-level "by intent" dispatcher
# ---------------------------------------------------------------------------

def lookup_by_intent(params: dict) -> dict[str, Any]: # def to take a dict of parameters that indicate what the user is looking for (e.g. "email_id", "po_number", "customer_name") and return a structured dict with the relevant information fetched from MongoDB. 
    # This function serves as a single entry point for the orchestrator to fetch data based on the user's intent, without needing to know the details of which specific lookup functions to call. It checks for specific keys in the input params to determine what to look up, 
    # and can return a combination of related data (e.g. fetching a PO and also including the linked customer information).
    """
    Takes a dict of params indicating what to
    look up and returns a structured result.
    """
    out: dict[str, Any] = {}

    # Email lookup (look up an email by ID, used for "find email msg_0001" type queries)
    if "email_id" in params:
        email = get_email(params["email_id"])
        out["email"] = email
        return out

    # PO lookup (look up a PO by po_number, used for "show me PO-2026-P0058" type queries)
    if "po_number" in params:
        po = get_po(params["po_number"])
        out["po"] = po
        if po and po.get("matched_customer_id"):
            out["customer"] = get_customer(po["matched_customer_id"])
        return out

    # Invoice lookup    (look up an invoice by invoice_id, used for "get me invoice INV-2025-0001" type queries)
    if "invoice_id" in params:
        inv = get_invoice(params["invoice_id"])
        out["invoice"] = inv
        if inv:
            out["line_items"] = get_invoice_line_items(params["invoice_id"])
            if inv.get("customer_id"):
                out["customer"] = get_customer(inv["customer_id"])
        return out

    # Customer lookup (resolve email or name to customer_id first)
    customer_id = params.get("customer_id")
    if not customer_id and params.get("customer_email"):
        cust = get_customer_by_email(params["customer_email"])
        if cust:
            customer_id = cust["customer_id"]
    if not customer_id and params.get("customer_name"):
        matches = get_customers_by_name(params["customer_name"])
        if len(matches) == 1:
            customer_id = matches[0]["customer_id"]
        elif len(matches) > 1:
            out["ambiguous_customers"] = matches
            return out

    if customer_id:
        out["customer"] = get_customer(customer_id)
        out["invoices"] = get_invoices_for_customer(customer_id)
        out["pos"] = get_pos_for_customer(customer_id)
        out["emails"] = get_emails_for_customer(customer_id)

    return out


# ---------------------------------------------------------------------------
# List/summary functions for open-ended queries
# ---------------------------------------------------------------------------

def list_all_pos(limit: int = 50) -> list[dict]: # def to list all POs with summary fields, which is useful for "show me all POs" type queries. Returns a list of dicts with key information about each PO, limited to 50 by default to avoid overwhelming the user.
    """All POs with summary fields. Used for 'show me POs' queries."""
    return list(_get_db().pos.find(
        {},
        {
            "_id": 0,
            "po_number": 1,
            "cust_name": 1,
            "matched_customer_id": 1,
            "trade_name": 1,
            "job_type": 1,
            "po_status": 1,
            "total_inc_vat": 1,
            "po_date": 1,
        }
    ).limit(limit))


def list_all_customers(limit: int = 20) -> list[dict]: # def to list all customers with summary fields, which is useful for "show me all customers" type queries. Returns a list of dicts with key information about each customer, limited to 20 by default to avoid overwhelming the user.
    """Sample customers with summary fields."""
    return list(_get_db().customers.find(
        {},
        {
            "_id": 0,
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "email": 1,
            "preferred_trade": 1,
        }
    ).limit(limit))


def list_all_job_types(trade: str | None = None) -> list[dict]: # def to list all job types, optionally filtered by trade. This is useful for "what job types do you have" type queries, and the optional trade filter allows the user to narrow down to job types relevant to a particular trade (e.g. "show me all plumbing job types"). Returns a list of dicts with key information about each job type.
    """All job types, optionally filtered by trade."""
    query = {"trade": trade} if trade else {}
    return list(_get_db().job_types.find(
        query,
        {
            "_id": 0,
            "job_type_id": 1,
            "job_name": 1,
            "trade": 1,
        }
    ).sort("trade"))


def list_all_invoices(limit: int = 20) -> list[dict]: # def to list recent invoices with summary fields, which is useful for "show me recent invoices" type queries. Returns a list of dicts with key information about each invoice, 
    # sorted by invoice_date descending and limited to 20 by default to avoid overwhelming the user.
    """Recent invoices with summary fields."""
    return list(_get_db().invoices.find(
        {},
        {
            "_id": 0,
            "invoice_id": 1,
            "customer_id": 1,
            "job_type_id": 1,
            "invoice_date": 1,
            "total_inc_vat": 1,
        }
    ).sort("invoice_date", -1).limit(limit))


def list_recent_emails(limit: int = 10) -> list[dict]: # def to list recent emails with summary fields, which is useful for "show me recent emails" type queries. Returns a list of dicts with key information about each email, sorted by received_at descending and limited to 10 by default to avoid overwhelming the user.
    """Recent emails with summary."""
    return list(_get_db().emails.find(
        {},
        {
            "_id": 0,
            "email_id": 1,
            "from_email": 1,
            "subject": 1,
            "received_at": 1,
            "extracted.trade_needed": 1,
            "extracted.urgency": 1,
        }
    ).limit(limit))


def get_collection_summary() -> dict: # def to get counts of each collection, which is useful for "what data do you have" type queries. Returns a dict with the count of documents in each collection, which can help the user understand what data is available in the system.
    """Counts of each collection - useful for 'what data do you have' questions."""
    db = _get_db()
    return {
        "customers":     db.customers.count_documents({}),
        "emails":        db.emails.count_documents({}),
        "invoices":      db.invoices.count_documents({}),
        "invoice_items": db.invoice_items.count_documents({}),
        "items":         db.items.count_documents({}),
        "job_types":     db.job_types.count_documents({}),
        "pos":           db.pos.count_documents({}),
        "chunks":        db.chunks.count_documents({}),
    }


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python mongo_lookup.py <customer_id|po_number|invoice_id|name>")
        sys.exit(0)

    arg = sys.argv[1]
    if arg.startswith("PO-"):
        result = lookup_by_intent({"po_number": arg})
    elif arg.startswith("INV-"):
        result = lookup_by_intent({"invoice_id": arg})
    elif arg.startswith("cust_"):
        result = lookup_by_intent({"customer_id": arg})
    else:
        result = lookup_by_intent({"customer_name": arg})
    print(json.dumps(result, indent=2, default=str))
