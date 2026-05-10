"""
Add these functions to your retrieve/mongo_lookup.py file
(at the bottom, before the if __name__ == "__main__": block).

These provide "list/summary" views for open-ended questions like
"show me all POs" that the previous version couldn't handle.
"""

# ---------------------------------------------------------------------------
# List/summary functions for open-ended queries
# ---------------------------------------------------------------------------

def list_all_pos(limit: int = 50) -> list[dict]:
    """All POs with summary fields. Used for 'show me POs' queries."""
    return list(_get_db().pos.find(
        {},                  # no filter — return all POs
        {
            "_id": 0,        # exclude internal MongoDB _id field
            "po_number": 1,
            "cust_name": 1,
            "matched_customer_id": 1,
            "trade_name": 1,
            "job_type": 1,
            "po_status": 1,
            "total_inc_vat": 1,
            "po_date": 1,
        }
    ).limit(limit))  # default cap of 50 to avoid sending too much to the LLM


def list_all_customers(limit: int = 20) -> list[dict]:
    """Sample customers with summary fields."""
    return list(_get_db().customers.find(
        {},              # no filter — return all customers up to the limit
        {
            "_id": 0,    # exclude internal MongoDB _id field
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "email": 1,
            "preferred_trade": 1,
        }
    ).limit(limit))  # default cap of 20 for concise context


def list_all_job_types(trade: str | None = None) -> list[dict]:
    """All job types, optionally filtered by trade."""
    query = {"trade": trade} if trade else {}  # e.g. {"trade": "plumber"} or {} for all trades
    return list(_get_db().job_types.find(
        query,
        {
            "_id": 0,           # exclude internal MongoDB _id field
            "job_type_id": 1,   # e.g. "pl_01"
            "job_name": 1,      # e.g. "Boiler installation"
            "trade": 1,         # e.g. "plumber"
        }
    ).sort("trade"))  # sort alphabetically by trade so results group naturally


def list_all_invoices(limit: int = 20) -> list[dict]:
    """Recent invoices with summary fields."""
    return list(_get_db().invoices.find(
        {},              # no filter — return most recent invoices
        {
            "_id": 0,    # exclude internal MongoDB _id field
            "invoice_id": 1,
            "customer_id": 1,
            "job_type_id": 1,
            "invoice_date": 1,
            "total_inc_vat": 1,
        }
    ).sort("invoice_date", -1).limit(limit))  # -1 = descending, newest first


def list_recent_emails(limit: int = 10) -> list[dict]:
    """Recent emails with summary."""
    return list(_get_db().emails.find(
        {},              # no filter — return most recent emails
        {
            "_id": 0,    # exclude internal MongoDB _id field
            "email_id": 1,
            "from_email": 1,
            "subject": 1,
            "received_at": 1,
            "extracted.trade_needed": 1,   # LLM-extracted trade from email body at ingest time
            "extracted.urgency": 1,        # LLM-extracted urgency flag e.g. "high", "normal"
        }
    ).limit(limit))  # default cap of 10 — emails are verbose, keep context lean


def get_collection_summary() -> dict:
    """Counts of each collection - useful for 'what data do you have' questions."""
    db = _get_db()  # reuse cached connection
    return {
        "customers":     db.customers.count_documents({}),
        "emails":        db.emails.count_documents({}),
        "invoices":      db.invoices.count_documents({}),
        "invoice_items": db.invoice_items.count_documents({}),  # line-item breakdown table
        "items":         db.items.count_documents({}),          # catalogue of materials/parts
        "job_types":     db.job_types.count_documents({}),
        "pos":           db.pos.count_documents({}),
        "chunks":        db.chunks.count_documents({}),         # vector-embedded text chunks
    }
