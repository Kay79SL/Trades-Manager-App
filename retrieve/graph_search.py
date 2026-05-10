"""
graph_search.py
================
Module 3 of Phase 5 retrieval orchestration.

Cypher query templates for Neo4j. Used when the user's question is
fundamentally about *relationships* — "who's done similar work", "what
materials are typically used for X", "show me past customers like this".

Vector search can't elegantly answer multi-hop relationship queries.
Cypher can. This module wraps a handful of common templates.

Used by:
  - orchestrator.py (when router picks "graph" path)
  - predict_quote.py (uses items_for_job to find the materials recipe)
  - context_assembler.py (to enrich vector results with relationship context)
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


# ---------------------------------------------------------------------------
# Driver cache with auto-recovery (AuraDB Free pauses on idle)
# ---------------------------------------------------------------------------

_DRIVER = {"instance": None}  # module-level cache so the driver is only created once per process


def _create_driver():
    """Create a fresh Neo4j driver from .env credentials."""
    uri = os.getenv("NEO4J_URI")         # e.g. neo4j+s://xxxx.databases.neo4j.io
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_lifetime=300,      # recycle connections after 5 min to avoid AuraDB idle drops
        connection_acquisition_timeout=30, # fail fast if AuraDB is still waking up
    )


def _get_driver():
    """
    Get a working Neo4j driver. If the cached driver is dead (AuraDB
    paused), transparently reconnect. End user never sees the error.
    """
    if _DRIVER["instance"] is None:  # first call — no driver yet
        _DRIVER["instance"] = _create_driver()
        return _DRIVER["instance"]

    try:
        with _DRIVER["instance"].session() as s:  # ping the DB to check it's alive
            s.run("RETURN 1").single()
        return _DRIVER["instance"]
    except Exception:  # driver is stale (AuraDB paused) — recreate silently
        try:
            _DRIVER["instance"].close()
        except Exception:
            pass
        _DRIVER["instance"] = _create_driver()
        return _DRIVER["instance"]


def close_driver():
    """Close the cached driver (call at app shutdown)."""
    if _DRIVER["instance"] is not None:
        try:
            _DRIVER["instance"].close()
        except Exception:
            pass
        _DRIVER["instance"] = None


# ---------------------------------------------------------------------------
# Material recipe queries (used by predict_quote.py)
# ---------------------------------------------------------------------------

def items_for_job(job_type_id: str) -> list[dict]:
    """
    Find typical materials for a job type via the USES_ITEM graph edge.
    This is the 'recipe' that drives the materials cost prediction.
    """
    cypher = """
    MATCH (j:JobType {job_type_id: $job_type_id})-[r:USES_ITEM]->(i:Item)
    RETURN
      i.item_id            AS item_id,
      i.item_name          AS item_name,
      i.unit_price_ex_vat  AS unit_price,
      i.unit               AS unit,
      r.typical_quantity   AS typical_quantity
    ORDER BY i.item_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_type_id=job_type_id)]  # list of item dicts from Neo4j


def items_for_job_by_name(job_name: str) -> list[dict]:
    """Same as items_for_job but lookup by display name."""
    cypher = """
    MATCH (j:JobType)-[r:USES_ITEM]->(i:Item)
    WHERE toLower(j.job_name) = toLower($job_name)
    RETURN
      i.item_id            AS item_id,
      i.item_name          AS item_name,
      i.unit_price_ex_vat  AS unit_price,
      i.unit               AS unit,
      r.typical_quantity   AS typical_quantity
    ORDER BY i.item_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_name=job_name)]  # case-insensitive name match


# ---------------------------------------------------------------------------
# Customer history queries
# ---------------------------------------------------------------------------

def customer_full_history(customer_id: str) -> dict[str, list]:
    """
    Pull everything connected to a customer in the graph: past invoices,
    POs, emails, preferred trade. Returns grouped by relationship type.
    """
    cypher = """
    MATCH (c:Customer {customer_id: $customer_id})
    OPTIONAL MATCH (c)<-[:FOR_CUSTOMER]-(i:Invoice)
    OPTIONAL MATCH (c)<-[:FOR_CUSTOMER]-(po:PO)
    OPTIONAL MATCH (c)<-[:FROM_CUSTOMER]-(e:Email)
    OPTIONAL MATCH (c)-[:PREFERS]->(t:Trade)
    RETURN
      c.customer_id                                  AS customer_id,
      c.first_name                                   AS first_name,
      c.last_name                                    AS last_name,
      collect(DISTINCT i.invoice_id)                 AS invoice_ids,
      collect(DISTINCT po.po_number)                 AS po_numbers,
      collect(DISTINCT e.email_id)                   AS email_ids,
      collect(DISTINCT t.name)                       AS preferred_trades
    """
    with _get_driver().session() as s:
        rec = s.run(cypher, customer_id=customer_id).single()  # single() returns None if no match
        if not rec:
            return {}
        return dict(rec)


def similar_customers_by_job(job_type_id: str, limit: int = 5) -> list[dict]:
    """
    Find customers who have had work done in the same job_type.
    """
    cypher = """
    MATCH (j:JobType {job_type_id: $job_type_id})<-[:FOR_JOB|ABOUT_JOB]-(rec)
    MATCH (rec)-[:FOR_CUSTOMER|FROM_CUSTOMER]->(c:Customer)
    RETURN
      c.customer_id     AS customer_id,
      c.first_name      AS first_name,
      c.last_name       AS last_name,
      count(DISTINCT rec) AS interaction_count
    ORDER BY interaction_count DESC
    LIMIT $limit
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_type_id=job_type_id, limit=limit)]  # ranked by most interactions


# ---------------------------------------------------------------------------
# PO and invoice graph queries
# ---------------------------------------------------------------------------

def po_full_context(po_number: str) -> dict:
    """Pull a PO with all its connected entities."""
    cypher = """
    MATCH (po:PO {po_number: $po_number})
    OPTIONAL MATCH (po)-[:FOR_CUSTOMER]->(c:Customer)
    OPTIONAL MATCH (po)-[:FOR_JOB]->(j:JobType)
    OPTIONAL MATCH (po)-[:CONTAINS_ITEM]->(i:Item)
    RETURN
      po.po_number       AS po_number,
      c.customer_id      AS customer_id,
      c.first_name + ' ' + c.last_name AS customer_name,
      j.job_name         AS job_name,
      collect(DISTINCT i.item_name) AS items
    """
    with _get_driver().session() as s:
        rec = s.run(cypher, po_number=po_number).single()
        return dict(rec) if rec else {}  # empty dict if PO not found in graph


def similar_pos_by_job_type(job_name: str, limit: int = 5) -> list[dict]:
    """Find past POs for the same job_type — used for price benchmarking."""
    cypher = """
    MATCH (po:PO)-[:FOR_JOB]->(j:JobType)
    WHERE toLower(j.job_name) CONTAINS toLower($job_name)
    RETURN
      po.po_number          AS po_number,
      po.subtotal_ex_vat    AS subtotal,
      po.total_inc_vat      AS total,
      j.job_name            AS job_name
    ORDER BY po.po_number DESC
    LIMIT $limit
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_name=job_name, limit=limit)]  # most recent POs first


# ---------------------------------------------------------------------------
# Trade-based queries
# ---------------------------------------------------------------------------

def jobs_for_trade(trade: str) -> list[dict]:
    """All job types served by a trade."""
    cypher = """
    MATCH (j:JobType)-[:OF_TRADE]->(t:Trade {name: $trade})
    RETURN
      j.job_type_id   AS job_type_id,
      j.job_name      AS job_name
    ORDER BY j.job_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, trade=trade.lower())]  # trade name must be lowercase e.g. "plumber"


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def graph_traversal(params: dict) -> dict[str, Any]:
    """Dispatcher for the router. Maps params to the right Cypher template."""
    out: dict[str, Any] = {}  # accumulate results from whichever sub-queries apply

    if "job_type_id" in params:  # e.g. "pl_01" — look up materials and similar customers
        out["items"] = items_for_job(params["job_type_id"])
        out["similar_customers"] = similar_customers_by_job(params["job_type_id"])

    if "job_name" in params:  # e.g. "Boiler installation" — name-based fallback
        out["items_by_name"] = items_for_job_by_name(params["job_name"])
        out["similar_pos"] = similar_pos_by_job_type(params["job_name"])

    if "customer_id" in params:  # e.g. "cust_0001" — full customer graph history
        out["customer_history"] = customer_full_history(params["customer_id"])

    if "po_number" in params:  # e.g. "PO-2026-P0042" — PO with connected customer, job, items
        out["po_context"] = po_full_context(params["po_number"])

    if "trade" in params:  # e.g. "plumber" — list all job types for that trade
        out["jobs"] = jobs_for_trade(params["trade"])

    return out


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python graph_search.py <customer_id|po_number|job_type_id|trade>")
        sys.exit(0)

    arg = sys.argv[1]
    if arg.startswith("cust_"):       # e.g. cust_0001
        out = graph_traversal({"customer_id": arg})
    elif arg.startswith("PO-"):       # e.g. PO-2026-P0042
        out = graph_traversal({"po_number": arg})
    elif arg in ("plumber", "carpenter", "electrician"):  # trade name
        out = graph_traversal({"trade": arg})
    else:                             # assume job_type_id e.g. pl_01
        out = graph_traversal({"job_type_id": arg})
    print(json.dumps(out, indent=2, default=str))
