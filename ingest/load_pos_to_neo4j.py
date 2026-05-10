"""
load_pos_to_neo4j.py
=====================
Phase 3C: lift POs into the Neo4j graph.

Two responsibilities:

1. JOB-TYPE MATCHING
   The pos collection currently has a free-form `job_type` string (e.g.
   "Annual boiler service", "Full house rewire"). This script matches each
   to a canonical `job_type_id` from the job_types collection using:
     - exact case-insensitive match on job_name
     - substring containment in either direction
     - token overlap (Jaccard) with stopword filtering
   Updates each pos document with `matched_job_type_id`.

2. NEO4J GRAPH LOAD
   For each PO, creates:
     - (PO {po_number, po_date, status, totals...})
     - (PO)-[:FOR_CUSTOMER]->(Customer)        when matched_customer_id present
     - (PO)-[:FOR_JOB]->(JobType)              when matched_job_type_id present
     - (PO)-[:CONTAINS_ITEM {qty, price}]->(Item)  for each line item

After this completes, the graph supports queries like:
    "Show all past invoices AND open POs for Gerard Walsh"
    "Average labour cost across invoices + POs for boiler installation"
    "Items most commonly bundled with EV charger jobs (across both)"

Usage:
    python ingest\\load_pos_to_neo4j.py
    python ingest\\load_pos_to_neo4j.py --clear     # delete all PO nodes first
    python ingest\\load_pos_to_neo4j.py --skip-match # skip job_type matching
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Job-type matching
# ---------------------------------------------------------------------------

# Generic words that don't help disambiguate jobs
# Filtering these out before token overlap scoring reduces false positives
STOPWORDS = {
    "service", "installation", "repair", "replacement", "upgrade",
    "construction", "building", "installation", "of", "the", "a",
    "and", "with", "new", "full", "annual",
}


def match_job_type(po_job_type: str, job_types: list[dict]) -> tuple[str | None, str]:
    """
    Match a PO's free-form job_type string to a canonical job_type_id.
    Returns (job_type_id_or_None, strategy_used).
    """
    if not po_job_type:
        return None, "no_input"

    needle = po_job_type.strip().lower()

    # Strategy 1: exact match
    for jt in job_types:
        if jt["job_name"].strip().lower() == needle:
            return jt["job_type_id"], "exact"

    # Strategy 2: substring containment (either direction)
    for jt in job_types:
        canonical = jt["job_name"].strip().lower()
        if needle in canonical or canonical in needle:
            return jt["job_type_id"], "substring"

    # Strategy 3: token overlap with stopword filtering
    # Jaccard-style score: shared tokens / max(needle tokens, canonical tokens)
    needle_tokens = set(needle.split()) - STOPWORDS
    if not needle_tokens:
        return None, "all_stopwords"

    best: tuple[str, float] | None = None
    for jt in job_types:
        canonical_tokens = set(jt["job_name"].strip().lower().split()) - STOPWORDS
        if not canonical_tokens:
            continue
        overlap = len(needle_tokens & canonical_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(needle_tokens), len(canonical_tokens))
        if best is None or score > best[1]:
            best = (jt["job_type_id"], score)

    if best and best[1] >= 0.4:  # 0.4 = at least 40% token overlap required — avoids weak matches
        return best[0], f"token_overlap_{best[1]:.2f}"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Neo4j operations
# ---------------------------------------------------------------------------

CONSTRAINT_CYPHER = (
    "CREATE CONSTRAINT po_number IF NOT EXISTS "
    "FOR (p:PO) REQUIRE p.po_number IS UNIQUE"
)

CLEAR_PO_CYPHER = "MATCH (p:PO) DETACH DELETE p"  # DETACH DELETE removes the node AND all its relationships

CREATE_PO_CYPHER = """
MERGE (po:PO {po_number: $po_number})
SET po.po_date           = $po_date,
    po.status            = $status,
    po.delivery_date     = $delivery_date,
    po.payment_terms     = $payment_terms,
    po.materials_subtotal= $materials_subtotal,
    po.labour_cost       = $labour_cost,
    po.subtotal_ex_vat   = $subtotal_ex_vat,
    po.vat_23pct         = $vat_23pct,
    po.total_inc_vat     = $total_inc_vat,
    po.cust_name         = $cust_name,
    po.trade_name        = $trade_name,
    po.job_type_text     = $job_type_text
"""

LINK_TO_CUSTOMER_CYPHER = """
MATCH (po:PO       {po_number:    $po_number})
MATCH (c:Customer  {customer_id:  $customer_id})
MERGE (po)-[:FOR_CUSTOMER]->(c)
"""

LINK_TO_JOB_CYPHER = """
MATCH (po:PO     {po_number:    $po_number})
MATCH (j:JobType {job_type_id:  $job_type_id})
MERGE (po)-[:FOR_JOB]->(j)
"""

LINK_TO_ITEM_CYPHER = """
MATCH (po:PO   {po_number: $po_number})
MATCH (i:Item  {item_id:   $item_id})
MERGE (po)-[r:CONTAINS_ITEM]->(i)
SET r.quantity            = $quantity,
    r.unit_price_ex_vat   = $unit_price_ex_vat,
    r.line_total_ex_vat   = $line_total_ex_vat
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true",
                        help="Delete all PO nodes from Neo4j before loading")
    parser.add_argument("--skip-match", action="store_true",
                        help="Skip job_type matching (use existing matched_job_type_id only)")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    for var, val in [
        ("MONGO_URI", mongo_uri),
        ("NEO4J_URI", neo4j_uri),
        ("NEO4J_PASSWORD", neo4j_password),
    ]:
        if not val:
            print(f"ERROR: {var} not set in .env"); sys.exit(1)

    print(f"Connecting to MongoDB ({db_name})...")
    db = MongoClient(mongo_uri)[db_name]
    pos = list(db.pos.find({}))
    print(f"  Found {len(pos)} POs in pos collection")

    if not pos:
        print("Nothing to load. Run extract_pos.py first."); sys.exit(0)

    job_types = list(db.job_types.find({}, {"_id": 0, "job_type_id": 1, "job_name": 1, "trade": 1}))
    print(f"  Loaded {len(job_types)} canonical job_types for matching")

    # ===== Step 1: match job_type strings to job_type_ids =====
    if not args.skip_match:
        print("\nMatching PO job_type strings to canonical job_type_ids...")
        match_summary: dict[str, int] = {}
        unmatched_examples: list[str] = []
        for po in pos:
            jt_id, strategy = match_job_type(po.get("job_type", ""), job_types)
            match_summary[strategy] = match_summary.get(strategy, 0) + 1

            if jt_id is None and len(unmatched_examples) < 5:
                unmatched_examples.append(
                    f"  {po['po_number']}: \"{po.get('job_type', '')}\""
                )

            db.pos.update_one(
                {"po_number": po["po_number"]},
                {"$set": {"matched_job_type_id": jt_id, "match_strategy": strategy}},
            )

        for strategy, count in sorted(match_summary.items(), key=lambda x: -x[1]):
            print(f"    {strategy:20} {count:>3}")
        if unmatched_examples:
            print(f"\n  Unmatched POs (first 5):")
            for ex in unmatched_examples:
                print(ex)

        # Reload pos with updated matches
        pos = list(db.pos.find({}))

    # ===== Step 2: Neo4j graph load =====
    print(f"\nConnecting to Neo4j ({neo4j_uri})...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()
    print("  Connected")

    with driver.session() as session:
        if args.clear:
            print("\nClearing existing PO nodes...")
            session.run(CLEAR_PO_CYPHER)
            print("  PO nodes cleared")

        print("\nCreating PO uniqueness constraint...")
        session.run(CONSTRAINT_CYPHER)

        # Create PO nodes
        print("\nCreating PO nodes...")
        po_nodes_created = 0
        for po in tqdm(pos, desc="  PO nodes"):
            session.run(CREATE_PO_CYPHER, {
                "po_number":          po["po_number"],
                "po_date":            po.get("po_date", ""),
                "status":             po.get("po_status", ""),
                "delivery_date":      po.get("delivery_date", ""),
                "payment_terms":      po.get("payment_terms", ""),
                "materials_subtotal": _num(po.get("materials_subtotal")),
                "labour_cost":        _num(po.get("labour_cost")),
                "subtotal_ex_vat":    _num(po.get("subtotal_ex_vat")),
                "vat_23pct":          _num(po.get("vat_23pct")),
                "total_inc_vat":      _num(po.get("total_inc_vat")),
                "cust_name":          po.get("cust_name", ""),
                "trade_name":         po.get("trade_name", ""),
                "job_type_text":      po.get("job_type", ""),
            })
            po_nodes_created += 1
        print(f"  {po_nodes_created} PO nodes")

        # Link to customers
        print("\nLinking POs to Customers...")
        customer_links = 0
        for po in tqdm(pos, desc="  customer edges"):
            cid = po.get("matched_customer_id")
            if not cid:
                continue
            session.run(LINK_TO_CUSTOMER_CYPHER, {
                "po_number":   po["po_number"],
                "customer_id": cid,
            })
            customer_links += 1
        print(f"  {customer_links}/{len(pos)} POs linked to customers")

        # Link to job_types
        print("\nLinking POs to JobTypes...")
        job_links = 0
        for po in tqdm(pos, desc="  job edges"):
            jid = po.get("matched_job_type_id")
            if not jid:
                continue
            session.run(LINK_TO_JOB_CYPHER, {
                "po_number":   po["po_number"],
                "job_type_id": jid,
            })
            job_links += 1
        print(f"  {job_links}/{len(pos)} POs linked to job_types")

        # Link to items
        print("\nLinking POs to Items via line items...")
        item_links = 0
        for po in tqdm(pos, desc="  item edges"):
            for li in po.get("line_items", []):
                item_id = li.get("item_id")
                if not item_id or str(item_id).startswith("["):  # skip malformed item_ids from LLM extraction errors
                    continue
                session.run(LINK_TO_ITEM_CYPHER, {
                    "po_number":         po["po_number"],
                    "item_id":           item_id,
                    "quantity":          _num(li.get("quantity")),
                    "unit_price_ex_vat": _num(li.get("unit_price_ex_vat")),
                    "line_total_ex_vat": _num(li.get("line_total_ex_vat")),
                })
                item_links += 1
        print(f"  {item_links} CONTAINS_ITEM edges created")

        # Final summary
        print("\n" + "=" * 50)
        print("Final graph state for POs:")
        print("=" * 50)

        result = session.run("MATCH (p:PO) RETURN count(p) AS n")
        po_count = result.single()["n"]
        print(f"\n  PO nodes:                 {po_count}")

        result = session.run(
            "MATCH (p:PO)-[r:FOR_CUSTOMER]->() RETURN count(r) AS n"
        )
        print(f"  PO -[:FOR_CUSTOMER]:      {result.single()['n']}")

        result = session.run(
            "MATCH (p:PO)-[r:FOR_JOB]->() RETURN count(r) AS n"
        )
        print(f"  PO -[:FOR_JOB]:           {result.single()['n']}")

        result = session.run(
            "MATCH (p:PO)-[r:CONTAINS_ITEM]->() RETURN count(r) AS n"
        )
        print(f"  PO -[:CONTAINS_ITEM]:     {result.single()['n']}")

    driver.close()
    print("\nDone.")


def _num(value: Any) -> float:
    """Coerce a value to float (returns 0.0 if unparseable)."""
    # LLM sometimes returns strings like "€120.00" or None — this handles both safely
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    main()
