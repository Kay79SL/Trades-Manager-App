"""
predict_quote.py
=================
Module 4 of Phase 5 retrieval orchestration — THE KILLER FEATURE.

Generates a pricing prediction for a job type by combining data from all
three storage stores. This is the architectural climax of the project:
no single store can answer this question alone.

The prediction has three components:
  1. Materials (recipe)  — Neo4j USES_ITEM relationships give "what items"
                          MongoDB items collection gives "current price each"
  2. Labour              — MongoDB invoices give past labour cost statistics
                          (median, range, n) for the same job_type
  3. Cross-check         — Neo4j similar PO query benchmarks against recent quotes

Output structure is designed to feed cleanly into context_assembler so the
chatbot can present a confidence-weighted price to the user.

Used by:
  - orchestrator.py (when router picks "predict_quote" path)
"""

from __future__ import annotations

from statistics import median  # use built-in median — no extra dependency needed
from typing import Any

from . import graph_search   # Neo4j queries: USES_ITEM recipe and similar POs
from . import mongo_lookup   # MongoDB queries: invoices, job_type lookup


VAT_RATE = 0.23  # Irish standard VAT rate applied to all trades quotes


# ---------------------------------------------------------------------------
# Materials estimation
# ---------------------------------------------------------------------------

def estimate_materials(job_name: str) -> dict[str, Any]:
    """
    Use Neo4j USES_ITEM recipe + current item prices from MongoDB to
    produce a materials cost estimate.
    """
    # 1. Get the recipe from Neo4j
    recipe = graph_search.items_for_job_by_name(job_name)  # list of items with typical_quantity and unit_price

    if not recipe:  # no USES_ITEM edges found for this job in the graph
        return {
            "items":         [],
            "subtotal":      0.0,
            "n_recipe_items": 0,
            "source":        "no_recipe_in_graph",  # signals low confidence to compute_confidence()
        }

    # 2. Resolve full item details from MongoDB and compute line totals
    items_out = []
    subtotal = 0.0
    for entry in recipe:
        qty = entry.get("typical_quantity") or 1         # default to 1 if not set on the edge
        unit_price = entry.get("unit_price") or 0.0      # ex-VAT price from Neo4j (mirrors MongoDB items)
        line_total = qty * unit_price
        items_out.append({
            "item_id":     entry.get("item_id"),
            "item_name":   entry.get("item_name"),
            "quantity":    qty,
            "unit":        entry.get("unit"),             # e.g. "each", "m", "litre"
            "unit_price":  unit_price,
            "line_total":  line_total,
        })
        subtotal += line_total

    return {
        "items":          items_out,
        "subtotal":       round(subtotal, 2),
        "n_recipe_items": len(items_out),
        "source":         "neo4j_recipe + mongo_prices",  # provenance for the academic report
    }


# ---------------------------------------------------------------------------
# Labour estimation
# ---------------------------------------------------------------------------

def estimate_labour(job_type_id: str | None, job_name: str | None) -> dict[str, Any]:
    """
    Compute labour cost statistics from past invoices for this job type.
    Returns median (the headline number), min, max, and n.
    """
    invoices: list[dict] = []

    if job_type_id:  # preferred path — direct lookup by canonical ID
        invoices = mongo_lookup.get_invoices_for_job_type(job_type_id)

    # Fallback: try matching by job name in case job_type_id isn't known
    if not invoices and job_name:  # e.g. router returned job_name but not job_type_id
        # job_types collection lookup to resolve job_name → job_type_id
        jt = mongo_lookup.get_job_type_by_name(job_name)
        if jt and jt.get("job_type_id"):
            invoices = mongo_lookup.get_invoices_for_job_type(jt["job_type_id"])

    # Extract labour costs (handle different field names that may exist)
    labour_costs: list[float] = []
    for inv in invoices:
        cost = inv.get("labour_cost_ex_vat") or inv.get("labour_cost") or 0  # field name varies by invoice version
        if cost > 0:
            labour_costs.append(float(cost))

    if not labour_costs:  # no historical labour data — return zero estimate with low-confidence signal
        return {
            "median_eur": 0.0,
            "min_eur":    0.0,
            "max_eur":    0.0,
            "n_invoices": 0,
            "source":     "no_past_invoices",  # signals low confidence to compute_confidence()
        }

    return {
        "median_eur": round(median(labour_costs), 2),  # headline figure — robust to outliers
        "min_eur":    round(min(labour_costs), 2),
        "max_eur":    round(max(labour_costs), 2),
        "n_invoices": len(labour_costs),
        "source":     "mongo_invoices",
    }


# ---------------------------------------------------------------------------
# Cross-check against recent POs
# ---------------------------------------------------------------------------

def benchmark_against_pos(job_name: str) -> dict[str, Any]:
    """
    Pull recent POs for the same job type to benchmark our prediction.
    Returns the average and count for cross-validation.
    """
    pos = graph_search.similar_pos_by_job_type(job_name)  # Neo4j CONTAINS query, newest POs first
    if not pos:  # no matching POs in graph — skip benchmark
        return {
            "n_pos":          0,
            "avg_subtotal":   0.0,
            "avg_total":      0.0,
            "po_numbers":     [],
        }

    subtotals = [p["subtotal"] for p in pos if p.get("subtotal")]  # ex-VAT totals for comparison
    totals = [p["total"] for p in pos if p.get("total")]           # inc-VAT totals for headline benchmark

    return {
        "n_pos":        len(pos),
        "avg_subtotal": round(sum(subtotals) / len(subtotals), 2) if subtotals else 0.0,
        "avg_total":    round(sum(totals) / len(totals), 2) if totals else 0.0,
        "po_numbers":   [p["po_number"] for p in pos],  # returned so context_assembler can cite them as sources
    }


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(materials: dict, labour: dict, benchmark: dict) -> str:
    """
    Aggregate confidence based on data availability across all three sources.
    high   = all three have data, predicted vs benchmark within 20%
    medium = most sources have data
    low    = sparse data (no recipe, or no past invoices)
    """
    has_recipe = materials.get("n_recipe_items", 0) > 0     # Neo4j has a USES_ITEM recipe for this job
    has_labour = labour.get("n_invoices", 0) >= 3           # at least 3 past invoices for meaningful stats
    has_benchmark = benchmark.get("n_pos", 0) > 0           # at least one similar PO to cross-check against

    score = sum([has_recipe, has_labour, has_benchmark])  # 0-3 based on how many sources contributed

    # Cross-check: if we have both prediction and benchmark, see if they agree
    if has_recipe and has_benchmark and benchmark.get("avg_subtotal", 0) > 0:
        predicted_subtotal = materials["subtotal"] + labour["median_eur"]
        benchmark_subtotal = benchmark["avg_subtotal"]
        diff_pct = abs(predicted_subtotal - benchmark_subtotal) / benchmark_subtotal  # % divergence
        if diff_pct < 0.20 and score == 3:    # within 20% and all sources present — high confidence
            return "high"
        elif diff_pct < 0.35 and score >= 2:  # within 35% with most sources — medium confidence
            return "medium"
        else:                                  # large divergence or sparse data — low confidence
            return "low"

    if score == 3:    # all three sources have data but no benchmark to compare
        return "high"
    elif score == 2:  # two of three sources have data
        return "medium"
    else:             # only one or zero sources — not enough to trust the estimate
        return "low"


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def predict_quote(
    job_name: str,
    job_type_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a complete quote prediction for a job type.

    Args:
        job_name:    Display name (e.g. "Boiler installation")
        job_type_id: Optional canonical id (e.g. "pl_01") — speeds up lookup

    Returns:
        Structured prediction with materials, labour, totals, confidence,
        and provenance for every figure.
    """
    materials = estimate_materials(job_name)                    # step 1: Neo4j recipe + MongoDB prices
    labour = estimate_labour(job_type_id, job_name)             # step 2: MongoDB past invoice stats
    benchmark = benchmark_against_pos(job_name)                 # step 3: Neo4j similar PO cross-check

    subtotal_ex_vat = materials["subtotal"] + labour["median_eur"]  # combine materials + labour before VAT
    vat = round(subtotal_ex_vat * VAT_RATE, 2)                      # 23% Irish VAT
    total_inc_vat = round(subtotal_ex_vat + vat, 2)

    confidence = compute_confidence(materials, labour, benchmark)  # "high" / "medium" / "low"

    return {
        "job_type":     job_name,
        "job_type_id":  job_type_id,
        "materials":    materials,   # detailed item breakdown for context_assembler
        "labour":       labour,      # median/min/max and source invoice count
        "totals": {
            "subtotal_ex_vat": round(subtotal_ex_vat, 2),
            "vat_23pct":       vat,
            "total_inc_vat":   total_inc_vat,
        },
        "benchmark":    benchmark,   # PO cross-check for confidence calibration
        "confidence":   confidence,
        "evidence": {
            "recipe_items_count":  materials.get("n_recipe_items", 0),
            "past_invoices_count": labour.get("n_invoices", 0),
            "benchmark_pos_count": benchmark.get("n_pos", 0),
            "stores_used":         ["mongo_invoices", "mongo_items", "neo4j_recipe", "neo4j_pos"],  # all three stores
        },
    }


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    job = sys.argv[1] if len(sys.argv) > 1 else "Boiler installation"  # default test job if none supplied
    result = predict_quote(job)
    print(json.dumps(result, indent=2))
