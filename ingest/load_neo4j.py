"""
load_neo4j.py
=============
Loads the synthetic trades dataset CSVs into Neo4j as a property graph.
Idempotent — safe to re-run; nodes and relationships are MERGEd.

Graph model
-----------
Nodes:
    (:Customer {customer_id, first_name, last_name, email, phone,
                address_line_1, address_line_2, county, eircode,
                preferred_trade, first_contact_date})
    (:Trade    {name})
    (:JobType  {job_type_id, job_name, typical_hours,
                base_labour_cost_eur, skill_level, description})
    (:Item     {item_id, item_name, unit, unit_price_ex_vat, category})
    (:Invoice  {invoice_id, invoice_date, labour_cost_ex_vat,
                materials_cost_ex_vat, subtotal_ex_vat, vat_23pct,
                total_inc_vat, status})
    (:Email    {email_id, from_name, from_email, to_email, subject,
                sent_at, template_kind, is_returning_customer, body})

Relationships:
    (Customer)-[:PREFERS]->(Trade)
    (JobType)-[:OF_TRADE]->(Trade)
    (JobType)-[:USES_ITEM {typical_quantity, position}]->(Item)
    (Invoice)-[:FOR_CUSTOMER]->(Customer)
    (Invoice)-[:FOR_JOB]->(JobType)
    (Invoice)-[:OF_TRADE]->(Trade)
    (Invoice)-[:CONTAINS {line_no, quantity, unit, unit_price_ex_vat,
                          line_total_ex_vat}]->(Item)
    (Email)-[:CONTACTED]->(Trade)
    (Email)-[:FROM_CUSTOMER]->(Customer)        (when customer_id_if_known)
    (Email)-[:ABOUT_JOB]->(JobType)             (when target_job_type_id)

Usage (from project root):
    python ingest/load_neo4j.py
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 500  # rows per Cypher UNWIND call — balances memory and round-trip overhead


def _parse_date(s: str) -> str:
    """Return YYYY-MM-DD for Cypher date()."""
    return datetime.strptime(s, "%Y-%m-%d").date().isoformat()  # Neo4j date() expects ISO string


def _parse_datetime(s: str) -> str:
    """Return ISO-8601 datetime for Cypher datetime()."""
    # emails have timestamps; customers/invoices have date-only strings
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_batched(
    driver: Driver, query: str, rows: list[dict], label: str
) -> None:
    if not rows:
        print(f"  {label:<22} 0 rows")
        return
    with driver.session() as session:
        for i in tqdm(range(0, len(rows), BATCH_SIZE),
                      desc=f"  {label}", unit="batch", leave=False):
            session.run(query, rows=rows[i:i + BATCH_SIZE])
    print(f"  {label:<22} {len(rows):>5} rows")


def create_constraints(driver: Driver) -> None:
    # IF NOT EXISTS makes this idempotent — safe to re-run without errors
    stmts = [
        "CREATE CONSTRAINT customer_id IF NOT EXISTS "
        "FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE",
        "CREATE CONSTRAINT trade_name IF NOT EXISTS "
        "FOR (n:Trade) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT jobtype_id IF NOT EXISTS "
        "FOR (n:JobType) REQUIRE n.job_type_id IS UNIQUE",
        "CREATE CONSTRAINT item_id IF NOT EXISTS "
        "FOR (n:Item) REQUIRE n.item_id IS UNIQUE",
        "CREATE CONSTRAINT invoice_id IF NOT EXISTS "
        "FOR (n:Invoice) REQUIRE n.invoice_id IS UNIQUE",
        "CREATE CONSTRAINT email_id IF NOT EXISTS "
        "FOR (n:Email) REQUIRE n.email_id IS UNIQUE",
    ]
    with driver.session() as session:
        for stmt in stmts:
            session.run(stmt)
    print("  constraints created")


def load_customers(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "customer_id":        r["customer_id"],
            "first_name":         r["first_name"],
            "last_name":          r["last_name"],
            "email":              r["email"],
            "phone":              r["phone"],
            "address_line_1":     r["address_line_1"],
            "address_line_2":     r["address_line_2"],
            "county":             r["county"],
            "eircode":            r["eircode"],
            "preferred_trade":    r["preferred_trade"],
            "first_contact_date": _parse_date(r["first_contact_date"]),
        }
        for r in _read_csv(data_dir / "customers.csv")
    ]
    query = """
    UNWIND $rows AS row
    MERGE (c:Customer {customer_id: row.customer_id})
    SET c.first_name         = row.first_name,
        c.last_name          = row.last_name,
        c.email              = row.email,
        c.phone              = row.phone,
        c.address_line_1     = row.address_line_1,
        c.address_line_2     = row.address_line_2,
        c.county             = row.county,
        c.eircode            = row.eircode,
        c.preferred_trade    = row.preferred_trade,
        c.first_contact_date = date(row.first_contact_date)
    MERGE (t:Trade {name: row.preferred_trade})
    MERGE (c)-[:PREFERS]->(t)
    """
    _run_batched(driver, query, rows, "customers")


def load_job_types(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "job_type_id":          r["job_type_id"],
            "trade":                r["trade"],
            "job_name":             r["job_name"],
            "typical_hours":        float(r["typical_hours"]),
            "base_labour_cost_eur": float(r["base_labour_cost_eur"]),
            "skill_level":          int(r["skill_level"]),
            "description":          r["description"],
        }
        for r in _read_csv(data_dir / "job_types.csv")
    ]
    query = """
    UNWIND $rows AS row
    MERGE (j:JobType {job_type_id: row.job_type_id})
    SET j.job_name             = row.job_name,
        j.typical_hours        = row.typical_hours,
        j.base_labour_cost_eur = row.base_labour_cost_eur,
        j.skill_level          = row.skill_level,
        j.description          = row.description
    MERGE (t:Trade {name: row.trade})
    MERGE (j)-[:OF_TRADE]->(t)
    """
    _run_batched(driver, query, rows, "job_types")


def load_items(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "item_id":           r["item_id"],
            "item_name":         r["item_name"],
            "unit":              r["unit"],
            "unit_price_ex_vat": float(r["unit_price_ex_vat"]),
            "category":          r["category"],
        }
        for r in _read_csv(data_dir / "items.csv")
    ]
    query = """
    UNWIND $rows AS row
    MERGE (i:Item {item_id: row.item_id})
    SET i.item_name         = row.item_name,
        i.unit              = row.unit,
        i.unit_price_ex_vat = row.unit_price_ex_vat,
        i.category          = row.category
    """
    _run_batched(driver, query, rows, "items")


def load_job_items(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "job_type_id":      r["job_type_id"],
            "item_id":          r["item_id"],
            "typical_quantity": int(r["typical_quantity"]),
            "position":         int(r["position"]),
        }
        for r in _read_csv(data_dir / "job_items.csv")
    ]
    query = """
    UNWIND $rows AS row
    MATCH (j:JobType {job_type_id: row.job_type_id})
    MATCH (i:Item    {item_id:     row.item_id})
    MERGE (j)-[r:USES_ITEM {position: row.position}]->(i)
    SET r.typical_quantity = row.typical_quantity
    """
    _run_batched(driver, query, rows, "job_items")


def load_invoices(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "invoice_id":            r["invoice_id"],
            "customer_id":           r["customer_id"],
            "trade":                 r["trade"],
            "job_type_id":           r["job_type_id"],
            "invoice_date":          _parse_date(r["invoice_date"]),
            "labour_cost_ex_vat":    float(r["labour_cost_ex_vat"]),
            "materials_cost_ex_vat": float(r["materials_cost_ex_vat"]),
            "subtotal_ex_vat":       float(r["subtotal_ex_vat"]),
            "vat_23pct":             float(r["vat_23pct"]),
            "total_inc_vat":         float(r["total_inc_vat"]),
            "status":                r["status"],
        }
        for r in _read_csv(data_dir / "invoices.csv")
    ]
    query = """
    UNWIND $rows AS row
    MERGE (inv:Invoice {invoice_id: row.invoice_id})
    SET inv.invoice_date          = date(row.invoice_date),
        inv.labour_cost_ex_vat    = row.labour_cost_ex_vat,
        inv.materials_cost_ex_vat = row.materials_cost_ex_vat,
        inv.subtotal_ex_vat       = row.subtotal_ex_vat,
        inv.vat_23pct             = row.vat_23pct,
        inv.total_inc_vat         = row.total_inc_vat,
        inv.status                = row.status
    MERGE (c:Customer {customer_id: row.customer_id})
    MERGE (j:JobType  {job_type_id: row.job_type_id})
    MERGE (t:Trade    {name:        row.trade})
    MERGE (inv)-[:FOR_CUSTOMER]->(c)
    MERGE (inv)-[:FOR_JOB]->(j)
    MERGE (inv)-[:OF_TRADE]->(t)
    """
    _run_batched(driver, query, rows, "invoices")


def load_invoice_items(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "invoice_id":        r["invoice_id"],
            "line_no":           int(r["line_no"]),
            "item_id":           r["item_id"],
            "quantity":          int(r["quantity"]),
            "unit":              r["unit"],
            "unit_price_ex_vat": float(r["unit_price_ex_vat"]),
            "line_total_ex_vat": float(r["line_total_ex_vat"]),
        }
        for r in _read_csv(data_dir / "invoice_items.csv")
    ]
    query = """
    UNWIND $rows AS row
    MATCH (inv:Invoice {invoice_id: row.invoice_id})
    MATCH (i:Item      {item_id:    row.item_id})
    MERGE (inv)-[c:CONTAINS {line_no: row.line_no}]->(i)
    SET c.quantity          = row.quantity,
        c.unit              = row.unit,
        c.unit_price_ex_vat = row.unit_price_ex_vat,
        c.line_total_ex_vat = row.line_total_ex_vat
    """
    _run_batched(driver, query, rows, "invoice_items")


def load_emails(driver: Driver, data_dir: Path) -> None:
    rows = [
        {
            "email_id":              r["email_id"],
            "from_name":             r["from_name"],
            "from_email":            r["from_email"],
            "to_email":              r["to_email"],
            "trade_contacted":       r["trade_contacted"],
            "subject":               r["subject"],
            "sent_at":               _parse_datetime(r["sent_at"]),
            "template_kind":         r["template_kind"],
            "is_returning_customer":
                r["is_returning_customer"].lower() == "true",
            "customer_id_if_known":  r["customer_id_if_known"] or None,
            "target_job_type_id":    r["target_job_type_id"] or None,
            "body":                  r["body"],
        }
        for r in _read_csv(data_dir / "emails.csv")
    ]
    query = """
    UNWIND $rows AS row
    MERGE (e:Email {email_id: row.email_id})
    SET e.from_name             = row.from_name,
        e.from_email            = row.from_email,
        e.to_email              = row.to_email,
        e.subject               = row.subject,
        e.sent_at               = datetime(row.sent_at),
        e.template_kind         = row.template_kind,
        e.is_returning_customer = row.is_returning_customer,
        e.body                  = row.body
    MERGE (t:Trade {name: row.trade_contacted})
    MERGE (e)-[:CONTACTED]->(t)
    FOREACH (_ IN
        CASE WHEN row.customer_id_if_known IS NULL THEN [] ELSE [1] END |
        MERGE (c:Customer {customer_id: row.customer_id_if_known})
        MERGE (e)-[:FROM_CUSTOMER]->(c)
    )
    FOREACH (_ IN
        -- FOREACH [1] / [] is a Cypher trick for conditional MERGE (no IF/ELSE in Cypher)
        CASE WHEN row.target_job_type_id IS NULL THEN [] ELSE [1] END |
        MERGE (j:JobType {job_type_id: row.target_job_type_id})
        MERGE (e)-[:ABOUT_JOB]->(j)
    )
    """
    _run_batched(driver, query, rows, "emails")


def main() -> None:
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")  # Neo4j AuraDB default username is "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")  # AuraDB free tier only has the default "neo4j" database
    if not uri or not password:
        sys.exit("NEO4J_URI / NEO4J_PASSWORD not set — check your .env file")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    print(f"Connected  uri={uri}  db={database!r}")
    print(f"Data dir:  {DATA_DIR}\n")

    print("Creating constraints ...")
    create_constraints(driver)
    print()

    load_customers(driver, DATA_DIR)
    load_job_types(driver, DATA_DIR)
    load_items(driver, DATA_DIR)
    load_job_items(driver, DATA_DIR)
    load_invoices(driver, DATA_DIR)
    load_invoice_items(driver, DATA_DIR)
    load_emails(driver, DATA_DIR)

    print("\nDone.")
    driver.close()


if __name__ == "__main__":
    main()
