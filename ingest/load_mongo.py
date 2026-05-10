"""
load_mongo.py
=============
Loads all synthetic trades dataset CSVs and email .eml files into MongoDB.
Idempotent — safe to re-run; existing documents are updated in place.

Usage (from project root):
    python ingest/load_mongo.py
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import gridfs
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.operations import UpdateOne
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
EML_DIR = DATA_DIR / "emails"


def _parse_date(s: str) -> datetime:
    # Try both date-only and date-time formats so CSVs with either style work
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bulk_upsert(collection, docs: list[dict], filter_keys: list[str], label: str) -> None:
    if not docs:
        print(f"  {label}: 0 docs")
        return
    ops = [
        # upsert=True creates the document if it doesn't exist, updates it if it does
        UpdateOne({k: doc[k] for k in filter_keys}, {"$set": doc}, upsert=True)
        for doc in tqdm(docs, desc=f"  {label}", unit="doc", leave=False)
    ]
    result = collection.bulk_write(ops, ordered=False)
    print(f"  {label:<22} {len(docs):>5} docs  "
          f"(inserted={result.upserted_count}, updated={result.modified_count})")


def load_customers(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "customers.csv")
    docs = [
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
        for r in rows
    ]
    _bulk_upsert(db.customers, docs, ["customer_id"], "customers")


def load_job_types(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "job_types.csv")
    docs = [
        {
            "job_type_id":          r["job_type_id"],
            "trade":                r["trade"],
            "job_name":             r["job_name"],
            "typical_hours":        float(r["typical_hours"]),
            "base_labour_cost_eur": float(r["base_labour_cost_eur"]),
            "skill_level":          int(r["skill_level"]),
            "description":          r["description"],
        }
        for r in rows
    ]
    _bulk_upsert(db.job_types, docs, ["job_type_id"], "job_types")


def load_items(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "items.csv")
    docs = [
        {
            "item_id":           r["item_id"],
            "item_name":         r["item_name"],
            "unit":              r["unit"],
            "unit_price_ex_vat": float(r["unit_price_ex_vat"]),
            "category":          r["category"],
        }
        for r in rows
    ]
    _bulk_upsert(db.items, docs, ["item_id"], "items")


def load_job_items(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "job_items.csv")
    docs = [
        {
            "job_type_id":      r["job_type_id"],
            "item_id":          r["item_id"],
            "typical_quantity": int(r["typical_quantity"]),
            "position":         int(r["position"]),
        }
        for r in rows
    ]
    _bulk_upsert(db.job_items, docs, ["job_type_id", "position"], "job_items")


def load_invoices(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "invoices.csv")
    docs = [
        {
            "invoice_id":            r["invoice_id"],
            "customer_id":           r["customer_id"],
            "trade":                 r["trade"],
            "job_type_id":           r["job_type_id"],
            "job_name":              r["job_name"],
            "invoice_date":          _parse_date(r["invoice_date"]),
            "labour_cost_ex_vat":    float(r["labour_cost_ex_vat"]),
            "materials_cost_ex_vat": float(r["materials_cost_ex_vat"]),
            "subtotal_ex_vat":       float(r["subtotal_ex_vat"]),
            "vat_23pct":             float(r["vat_23pct"]),
            "total_inc_vat":         float(r["total_inc_vat"]),
            "status":                r["status"],
        }
        for r in rows
    ]
    _bulk_upsert(db.invoices, docs, ["invoice_id"], "invoices")


def load_invoice_items(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "invoice_items.csv")
    docs = [
        {
            "invoice_id":        r["invoice_id"],
            "line_no":           int(r["line_no"]),
            "item_id":           r["item_id"],
            "item_name":         r["item_name"],
            "quantity":          int(r["quantity"]),
            "unit":              r["unit"],
            "unit_price_ex_vat": float(r["unit_price_ex_vat"]),
            "line_total_ex_vat": float(r["line_total_ex_vat"]),
        }
        for r in rows
    ]
    _bulk_upsert(db.invoice_items, docs, ["invoice_id", "line_no"], "invoice_items")


def load_emails(db, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "emails.csv")
    docs = [
        {
            "email_id":              r["email_id"],
            "from_name":             r["from_name"],
            "from_email":            r["from_email"],
            "to_email":              r["to_email"],
            "trade_contacted":       r["trade_contacted"],
            "subject":               r["subject"],
            "sent_at":               _parse_date(r["sent_at"]),
            "template_kind":         r["template_kind"],
            "is_returning_customer": r["is_returning_customer"].lower() == "true",
            "customer_id_if_known":  r["customer_id_if_known"] or None,
            "target_job_type_id":    r["target_job_type_id"],
            "target_job_name":       r["target_job_name"],
            "body":                  r["body"],
            "extracted":             None,  # placeholder — filled in by extract_entities.py (Phase 3A)
        }
        for r in rows
    ]
    _bulk_upsert(db.emails, docs, ["email_id"], "emails")


def load_eml_files(db, eml_dir: Path) -> None:
    bucket = gridfs.GridFS(db, collection="email_files")  # store raw .eml bytes in GridFS — keeps MongoDB document size under the 16MB BSON limit
    eml_files = sorted(eml_dir.glob("*.eml"))
    skipped = uploaded = 0
    for eml_path in tqdm(eml_files, desc="  email_files (GridFS)", unit="file"):
        email_id = eml_path.stem  # filename without extension = email_id (e.g. msg_0042)
        if bucket.exists({"metadata.email_id": email_id}):  # skip if already in GridFS to keep the script idempotent
            skipped += 1
            continue
        bucket.put(
            eml_path.read_bytes(),
            filename=eml_path.name,
            contentType="message/rfc822",
            metadata={"email_id": email_id},
        )
        uploaded += 1
    print(f"  {'email_files (GridFS)':<22} {len(eml_files):>5} files "
          f"(uploaded={uploaded}, skipped={skipped})")


def create_indexes(db) -> None:
    # Unique constraints prevent duplicate documents on re-runs
    db.customers.create_index([("customer_id", ASCENDING)], unique=True)
    db.job_types.create_index([("job_type_id", ASCENDING)], unique=True)
    db.items.create_index([("item_id", ASCENDING)], unique=True)

    db.invoices.create_index([("invoice_id", ASCENDING)], unique=True)
    db.invoices.create_index([("customer_id", ASCENDING)])
    db.invoices.create_index([("trade", ASCENDING)])
    db.invoices.create_index([("invoice_date", DESCENDING)])

    db.invoice_items.create_index([("invoice_id", ASCENDING)])

    db.emails.create_index([("email_id", ASCENDING)], unique=True)
    db.emails.create_index([("trade_contacted", ASCENDING)])
    db.emails.create_index([("from_email", ASCENDING)])

    print("  indexes created")


def main() -> None:
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    if not uri:
        sys.exit("MONGO_URI not set — check your .env file")

    client = MongoClient(uri)
    db = client[db_name]

    print(f"Connected  db={db_name!r}")
    print(f"Data dir:  {DATA_DIR}\n")

    load_customers(db, DATA_DIR)
    load_job_types(db, DATA_DIR)
    load_items(db, DATA_DIR)
    load_job_items(db, DATA_DIR)
    load_invoices(db, DATA_DIR)
    load_invoice_items(db, DATA_DIR)
    load_emails(db, DATA_DIR)
    load_eml_files(db, EML_DIR)

    print("\nCreating indexes ...")
    create_indexes(db)

    print("\nDone.")
    client.close()


if __name__ == "__main__":
    main()
