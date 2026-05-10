"""
ingest/ingest_runner.py
=======================

Wrappers called by the Streamlit Data Upload tab pipeline buttons.
Secrets come exclusively from st.secrets (Streamlit Cloud dashboard).
No .env file needed or used.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from pymongo import MongoClient

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
INGEST_DIR   = PROJECT_ROOT / "ingest"
PYTHON       = sys.executable

# Keys that every ingest script needs
SECRET_KEYS = [
    "MONGO_URI",
    "MONGO_DB",
    "ANTHROPIC_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]


# ── Secret resolution ─────────────────────────────────────────
def _get_env() -> dict:
    """
    Build a subprocess environment dict populated from st.secrets.
    This is the only source of secrets — no .env file involved.
    """
    env = os.environ.copy()

    import streamlit as st
    for key in SECRET_KEYS:
        try:
            val = st.secrets[key]
            if val:
                env[key] = str(val)
        except (KeyError, Exception):
            pass

    return env


def _build_db():
    """Return MongoDB db object using st.secrets."""
    import streamlit as st
    uri     = st.secrets["MONGO_URI"]
    db_name = st.secrets.get("MONGO_DB", "trades_quotes")
    return MongoClient(uri)[db_name]


def _count(db, collection: str) -> int:
    try:
        return db[collection].count_documents({})
    except Exception:
        return 0


def _now() -> str:
    return time.strftime("%H:%M:%S")


# ── Subprocess runner ─────────────────────────────────────────
def _run(script_name: str) -> tuple[bool, str]:
    """
    Run an ingest script as a subprocess.
    Injects st.secrets into the child process environment so every
    os.getenv("MONGO_URI") call inside the script resolves correctly.
    """
    script_path = INGEST_DIR / script_name
    if not script_path.exists():
        return False, f"{script_name} not found in repo."

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=_get_env(),
    )

    output = (result.stdout or "").strip()
    errors = (result.stderr or "").strip()
    full   = "\n".join(filter(None, [output, errors]))

    if result.returncode != 0:
        err_lines = [l for l in errors.splitlines() if l.strip()]
        friendly  = err_lines[-1] if err_lines else f"Exit code {result.returncode}"
        return False, friendly

    return True, full


# ─────────────────────────────────────────────────────────────
# Step 1 — Load CSVs → MongoDB collections
# ─────────────────────────────────────────────────────────────
def run_load_mongo() -> dict:
    success, output = _run("load_mongo.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    db = _build_db()
    collections = [
        {"name": "customers",     "store": "MongoDB"},
        {"name": "invoices",      "store": "MongoDB"},
        {"name": "job_types",     "store": "MongoDB"},
        {"name": "items",         "store": "MongoDB"},
        {"name": "job_items",     "store": "MongoDB"},
        {"name": "invoice_items", "store": "MongoDB"},
    ]
    for c in collections:
        c["count"] = _count(db, c["name"])

    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     f"{len(collections)} collections updated",
    }


# ─────────────────────────────────────────────────────────────
# Step 2 — Extract PO PDFs → pos collection
# ─────────────────────────────────────────────────────────────
def run_extract_pos() -> dict:
    success, output = _run("extract_pos_from_pdf.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    success2, output2 = _run("load_po_pdfs.py")
    if not success2:
        return {"status": "error", "timestamp": ts, "message": output2}

    db = _build_db()
    collections = [
        {"name": "pos", "store": "MongoDB", "count": _count(db, "pos")},
    ]
    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     "PO PDFs extracted and loaded into pos collection",
    }


# ─────────────────────────────────────────────────────────────
# Step 3 — Extract entities from emails
# ─────────────────────────────────────────────────────────────
def run_extract_entities() -> dict:
    success, output = _run("extract_entities.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    db = _build_db()
    collections = [
        {"name": "emails", "store": "MongoDB", "count": _count(db, "emails")},
    ]
    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     "Email entities extracted",
    }


# ─────────────────────────────────────────────────────────────
# Step 4 — Load MongoDB → Neo4j graph
# ─────────────────────────────────────────────────────────────
def run_load_neo4j() -> dict:
    success, output = _run("load_neo4j.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    nodes = next(
        (l for l in output.splitlines() if "node" in l.lower()), None
    )
    collections = [
        {"name": "Customer · Invoice · JobType · Item nodes", "store": "Neo4j", "count": None},
    ]
    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     nodes or "Graph nodes and relationships created",
    }


# ─────────────────────────────────────────────────────────────
# Step 5 — Load POs → Neo4j graph
# ─────────────────────────────────────────────────────────────
def run_load_pos_neo4j() -> dict:
    success, output = _run("load_pos_to_neo4j.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    db = _build_db()
    po_count = _count(db, "pos")
    collections = [
        {"name": "PO nodes + relationships", "store": "Neo4j", "count": po_count},
    ]
    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     f"{po_count} PO nodes linked in graph",
    }


# ─────────────────────────────────────────────────────────────
# Step 6 — Generate embeddings → Vector index
# ─────────────────────────────────────────────────────────────
def run_embed_documents() -> dict:
    success, output = _run("embed_chunks.py")
    ts = _now()
    if not success:
        return {"status": "error", "timestamp": ts, "message": output}

    db = _build_db()
    emb_count = _count(db, "embeddings")
    collections = [
        {"name": "embeddings", "store": "MongoDB · Atlas Vector Search", "count": emb_count},
    ]
    summary = next(
        (l for l in reversed(output.splitlines()) if "chunk" in l.lower() or "embed" in l.lower()),
        f"{emb_count} embedding chunks in index",
    )
    return {
        "status":      "done",
        "timestamp":   ts,
        "collections": collections,
        "message":     summary,
    }
