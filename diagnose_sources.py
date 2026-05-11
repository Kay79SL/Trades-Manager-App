import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from retrieve.orchestrator import answer_query
    print("OK orchestrator imported\n")
except ImportError as e:
    sys.exit("Cannot import orchestrator: " + str(e))

test_queries = [
    ("exact_lookup", "Show me PO-2026-P0042"),
    ("aggregation",  "How much does boiler installation usually cost?"),
    ("semantic",     "Any customers complaining about water dripping?"),
]

for qtype, query in test_queries:
    print("=" * 60)
    print("TYPE:  " + qtype)
    print("QUERY: " + query)
    print("=" * 60)
    try:
        result = answer_query(query)
    except Exception as e:
        print("  ERROR: " + str(e))
        continue

    print("Result type: " + type(result).__name__)
    if isinstance(result, dict):
        print("Keys: " + str(list(result.keys())))
        for k, v in result.items():
            if k == "answer":
                print("\n  answer: " + str(v)[:200])
            elif k == "sources":
                count = len(v) if isinstance(v, list) else "?"
                print("\n  sources (" + str(count) + " items):")
                for i, s in enumerate((v or [])[:5]):
                    print("    [" + str(i) + "] " + repr(s)[:120])
            else:
                print("\n  " + k + ": " + repr(v)[:120])
    elif isinstance(result, str):
        print("\n  Plain string result (no sources dict):")
        print("  " + result[:300])
    print()

try:
    from pymongo import MongoClient
    db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]
    print("CHUNKS collection - sample source_ids:")
    for s in db.chunks.find({}, {"source_id": 1, "chunk_type": 1, "_id": 0}).limit(10):
        print("  source_id=" + repr(s.get("source_id")) + "  chunk_type=" + str(s.get("chunk_type")))
except Exception as e:
    print("MongoDB error: " + str(e))
