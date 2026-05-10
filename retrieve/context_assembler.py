"""
context_assembler.py
=====================
Module 5 of Phase 5 retrieval orchestration.

Updated to support list/browse query results.
"""

from __future__ import annotations

from typing import Any


MAX_CONTEXT_CHARS = 12_000  # hard cap on context passed to the LLM to avoid token overruns


def assemble_context(
    results: dict[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """Take retrieval results and produce context_text + sources."""
    sections: list[str] = []  # ordered text blocks that will be joined into final context
    sources: list[str] = []   # provenance labels shown to the user alongside the answer

    # Pricing prediction (highest priority)
    if "predict" in results and results["predict"]:  # price estimate from predict_quote.py
        s, src = _format_predict(results["predict"])
        if s:
            sections.append(s)
            sources.extend(src)

    # NEW: List/browse data
    if "list" in results and results["list"]:  # open-ended browse results e.g. "show all POs"
        s, src = _format_list(results["list"])
        if s:
            sections.append(s)
            sources.extend(src)

    # Direct lookups
    if "mongo" in results and results["mongo"]:  # structured MongoDB lookup results
        s, src = _format_mongo(results["mongo"])
        if s:
            sections.append(s)
            sources.extend(src)

    # Graph relationships
    if "graph" in results and results["graph"]:  # Neo4j relationship traversal results
        s, src = _format_graph(results["graph"])
        if s:
            sections.append(s)
            sources.extend(src)

    # Vector search
    if "vector" in results and results["vector"]:  # Atlas Vector Search semantic matches
        seen = set(sources)  # deduplicate: skip chunks already surfaced by mongo/graph
        s, src = _format_vector(results["vector"], seen)
        if s:
            sections.append(s)
            sources.extend(src)

    if not sections:  # nothing found in any store — return a safe fallback message
        return {
            "context_text": "(No relevant context found in the data stores.)",
            "sources":      [],
        }

    context_text = "\n\n".join(sections)  # merge all sections with blank lines between them
    if len(context_text) > MAX_CONTEXT_CHARS:  # trim if too long to avoid LLM context overflow
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[... truncated ...]"

    return {
        "context_text": context_text,
        "sources":      sources,
    }


def _format_predict(predict: dict) -> tuple[str, list[str]]:
    if not predict.get("job_type"):  # guard: skip if prediction is empty or incomplete
        return "", []

    sources = []
    lines = ["## PRICING PREDICTION"]
    lines.append(f"**Job type:** {predict['job_type']}")
    lines.append(f"**Confidence:** {predict.get('confidence', 'unknown')}")  # high/medium/low from compute_confidence()
    lines.append("")

    mat = predict.get("materials", {})  # materials sub-dict from predict_quote.py
    if mat.get("items"):
        lines.append(f"**Materials estimate:** €{mat.get('subtotal', 0):.2f} ex VAT")
        for item in mat["items"][:8]:  # cap at 8 items to keep context readable
            lines.append(
                f"  - {item['item_name']:30}  "
                f"qty {item['quantity']} × €{item['unit_price']:.2f} = €{item['line_total']:.2f}"
            )
        if len(mat["items"]) > 8:  # tell user if more items were omitted
            lines.append(f"  - ...and {len(mat['items']) - 8} more items")
        sources.append(f"Neo4j USES_ITEM recipe ({mat['n_recipe_items']} items)")

    lab = predict.get("labour", {})  # labour sub-dict with median/min/max from past invoices
    if lab.get("n_invoices", 0) > 0:  # only show labour if we have real historical data
        lines.append("")
        lines.append(
            f"**Labour estimate:** €{lab['median_eur']:.2f} (median), "
            f"range €{lab['min_eur']:.2f} - €{lab['max_eur']:.2f} "
            f"(from {lab['n_invoices']} past invoices)"
        )
        sources.append(f"MongoDB invoices ({lab['n_invoices']} records)")

    totals = predict.get("totals", {})  # combined subtotal + VAT breakdown
    lines.append("")
    lines.append(f"**Total estimate:** €{totals.get('total_inc_vat', 0):.2f} inc VAT "
                 f"(subtotal €{totals.get('subtotal_ex_vat', 0):.2f} + VAT €{totals.get('vat_23pct', 0):.2f})")

    bench = predict.get("benchmark", {})  # cross-check against recent POs for same job type
    if bench.get("n_pos", 0) > 0:  # only show benchmark if similar POs exist
        lines.append("")
        lines.append(
            f"**Benchmark:** average of {bench['n_pos']} similar past PO(s) was "
            f"€{bench.get('avg_total', 0):.2f} inc VAT"
        )
        for po in bench.get("po_numbers", [])[:3]:  # show up to 3 reference PO numbers
            sources.append(f"PO {po}")

    return "\n".join(lines), sources


def _format_list(list_data: dict) -> tuple[str, list[str]]:
    """Format a list/summary result as Markdown."""
    if not list_data or not list_data.get("items"):  # nothing to render — return empty
        return "", []

    sources = []
    list_type = list_data.get("type", "")   # e.g. "pos", "customers", "jobs", "invoices", "emails", "summary"
    items = list_data.get("items", [])
    title = list_data.get("title", "Records")  # section heading shown to the user

    lines = [f"## {title}"]

    if list_type == "pos":  # purchase order list
        lines.append(f"Found {len(items)} purchase order(s):\n")
        for po in items:
            lines.append(
                f"- **{po.get('po_number')}**: {po.get('cust_name', '?')} "
                f"({po.get('matched_customer_id', '?')}), "
                f"{po.get('job_type', '?')} — "
                f"€{po.get('total_inc_vat', 0):.2f} — "
                f"{po.get('po_status', '?')}"
            )
            sources.append(f"PO {po.get('po_number')}")

    elif list_type == "customers":  # customer directory listing
        lines.append(f"Customers (showing {len(items)}):\n")
        for c in items:
            lines.append(
                f"- **{c.get('customer_id')}**: {c.get('first_name')} {c.get('last_name')} "
                f"<{c.get('email')}> — {c.get('preferred_trade', '?')}"
            )
        sources.append(f"MongoDB customers ({len(items)} records)")

    elif list_type == "jobs":  # job type catalogue, grouped by trade
        by_trade = {}
        for j in items:
            trade = j.get("trade", "other")
            by_trade.setdefault(trade, []).append(j)
        lines.append(f"Available job types ({len(items)} total):\n")
        for trade in sorted(by_trade.keys()):  # alphabetical trade order
            lines.append(f"\n**{trade.title()}** ({len(by_trade[trade])} jobs):")
            for j in by_trade[trade]:
                lines.append(f"  - {j.get('job_name')} ({j.get('job_type_id')})")
        sources.append("MongoDB job_types")

    elif list_type == "invoices":  # recent invoices sorted newest-first (from mongo_lookup)
        lines.append(f"Recent invoices ({len(items)}):\n")
        for inv in items:
            lines.append(
                f"- **{inv.get('invoice_id')}** ({inv.get('invoice_date', '?')}): "
                f"customer {inv.get('customer_id')}, "
                f"job {inv.get('job_type_id', '?')}, "
                f"€{inv.get('total_inc_vat', 0):.2f}"
            )
        sources.append(f"MongoDB invoices ({len(items)} records)")

    elif list_type == "emails":  # recent inbound emails with extracted trade/urgency fields
        lines.append(f"Recent emails ({len(items)}):\n")
        for e in items:
            extracted = e.get("extracted") or {}  # LLM-extracted metadata stored at ingest time
            lines.append(
                f"- **{e.get('email_id')}** from {e.get('from_email', '?')}: "
                f"{e.get('subject', '?')} "
                f"[{extracted.get('trade_needed', '?')}, {extracted.get('urgency', '?')}]"
            )
        sources.append(f"MongoDB emails ({len(items)} records)")

    elif list_type == "summary":  # database health check — record counts per collection
        lines.append("Database statistics:\n")
        for collection, count in items.items():
            lines.append(f"- **{collection}**: {count:,} records")
        sources.append("MongoDB collection counts")

    return "\n".join(lines), sources


def _format_mongo(mongo: dict) -> tuple[str, list[str]]:
    sources = []
    lines = ["## DIRECT LOOKUPS"]
    has_content = False  # flag: only return a section if at least one sub-key has data

    if "ambiguous_customers" in mongo:  # more than one customer matched the name query
        lines.append(f"**Multiple customers match this name:**")
        for c in mongo["ambiguous_customers"]:
            lines.append(f"  - {c['customer_id']}: {c.get('first_name')} {c.get('last_name')} <{c.get('email')}>")
        has_content = True

    if "customer" in mongo and mongo["customer"]:  # single resolved customer record
        c = mongo["customer"]
        lines.append(
            f"**Customer:** {c['customer_id']} - "
            f"{c.get('first_name', '')} {c.get('last_name', '')} "
            f"<{c.get('email', '')}> ({c.get('preferred_trade', '?')})"
        )
        sources.append(f"Customer {c['customer_id']}")
        has_content = True

    if "invoices" in mongo and mongo["invoices"]:  # past invoices for a customer or job type
        lines.append(f"**Past invoices ({len(mongo['invoices'])}):**")
        for inv in mongo["invoices"][:5]:  # cap at 5 to keep context concise
            lines.append(
                f"  - {inv.get('invoice_id')} on {inv.get('invoice_date', '?')}: "
                f"{inv.get('job_type_id', '?')}, total €{inv.get('total_inc_vat', 0):.2f}"
            )
            sources.append(f"Invoice {inv.get('invoice_id')}")
        has_content = True

    if "pos" in mongo and mongo["pos"]:  # active POs for this customer (plural)
        lines.append(f"**Active POs ({len(mongo['pos'])}):**")
        for po in mongo["pos"]:
            lines.append(
                f"  - {po.get('po_number')}: {po.get('job_type', '?')}, "
                f"total €{po.get('total_inc_vat', 0):.2f} - {po.get('po_status', '?')}"
            )
            sources.append(f"PO {po.get('po_number')}")
        has_content = True

    if "po" in mongo and mongo["po"]:  # single PO detail lookup (by PO number pattern)
        po = mongo["po"]
        lines.append(f"**PO {po.get('po_number')}:**")
        lines.append(f"  - Customer: {po.get('cust_name')} ({po.get('matched_customer_id')})")
        lines.append(f"  - Job: {po.get('job_type')}")
        lines.append(f"  - Status: {po.get('po_status')}")
        lines.append(f"  - Total: €{po.get('total_inc_vat', 0):.2f} inc VAT")
        sources.append(f"PO {po.get('po_number')}")
        has_content = True

    if "invoice" in mongo and mongo["invoice"]:  # single invoice detail lookup (by INV-ID pattern)
        inv = mongo["invoice"]
        lines.append(f"**Invoice {inv.get('invoice_id')}:**")
        lines.append(f"  - Date: {inv.get('invoice_date')}")
        lines.append(f"  - Job: {inv.get('job_type_id')}")
        lines.append(f"  - Total: €{inv.get('total_inc_vat', 0):.2f} inc VAT")
        sources.append(f"Invoice {inv.get('invoice_id')}")
        has_content = True
        
    if "email" in mongo and mongo["email"]:  # single email record fetched by email_id (msg_XXXX pattern)
        e = mongo["email"]
        lines.append(f"**Email {e.get('email_id')}:**")
        lines.append(f"  - From: {e.get('from_name')} <{e.get('from_email')}>")
        lines.append(f"  - Subject: {e.get('subject')}")
        lines.append(f"  - Sent: {e.get('sent_at')}")
        lines.append(f"  - Body: {e.get('body', '')[:500]}")  # truncate body to 500 chars for context brevity
        sources.append(f"Email {e.get('email_id')}")
        has_content = True

    return ("\n".join(lines), sources) if has_content else ("", [])  # suppress section entirely if nothing resolved


def _format_graph(graph: dict) -> tuple[str, list[str]]:
    sources = []
    lines = ["## GRAPH RELATIONSHIPS"]
    has_content = False  # only emit the section if at least one graph query returned data

    if "items" in graph and graph["items"]:  # materials from Neo4j USES_ITEM — looked up by job_type_id
        lines.append(f"**Materials typically used (from graph):**")
        for item in graph["items"][:8]:  # cap at 8 items for readability
            lines.append(
                f"  - {item.get('item_name')} (qty {item.get('typical_quantity', '?')}, "
                f"€{item.get('unit_price', 0):.2f}/{item.get('unit', 'unit')})"
            )
        sources.append("Neo4j USES_ITEM graph")
        has_content = True

    if "items_by_name" in graph and graph["items_by_name"]:  # same as above but looked up by job display name
        lines.append(f"**Materials typically used:**")
        for item in graph["items_by_name"][:8]:
            lines.append(f"  - {item.get('item_name')}")
        sources.append("Neo4j USES_ITEM graph")
        has_content = True

    if "similar_customers" in graph and graph["similar_customers"]:  # customers who've had this job type before
        lines.append(f"**Similar customers (had this job type before):**")
        for c in graph["similar_customers"][:5]:  # top 5 by interaction count
            lines.append(
                f"  - {c.get('customer_id')}: {c.get('first_name')} {c.get('last_name')} "
                f"({c.get('interaction_count')} interactions)"
            )
        sources.append("Neo4j similar_customers traversal")
        has_content = True

    if "customer_history" in graph and graph["customer_history"]:  # full graph context for a specific customer
        h = graph["customer_history"]
        if h.get("invoice_ids") or h.get("po_numbers") or h.get("email_ids"):  # skip if all empty lists
            lines.append(f"**Customer graph context for {h.get('customer_id')}:**")
            if h.get("invoice_ids"):
                lines.append(f"  - Past invoices: {', '.join(h['invoice_ids'][:5])}")  # limit to 5 IDs
            if h.get("po_numbers"):
                lines.append(f"  - Active POs: {', '.join(h['po_numbers'])}")
            if h.get("preferred_trades"):
                lines.append(f"  - Preferred trades: {', '.join(h['preferred_trades'])}")
            sources.append(f"Neo4j customer history for {h.get('customer_id')}")
            has_content = True

    if "po_context" in graph and graph["po_context"]:  # PO enriched with graph neighbours (customer, job, items)
        po = graph["po_context"]
        lines.append(f"**PO graph context:**")
        lines.append(f"  - Customer: {po.get('customer_name')} ({po.get('customer_id')})")
        lines.append(f"  - Job: {po.get('job_name')}")
        if po.get("items"):
            lines.append(f"  - Items: {', '.join(po['items'][:5])}")  # first 5 item names
        sources.append(f"Neo4j PO graph for {po.get('po_number')}")
        has_content = True

    return ("\n".join(lines), sources) if has_content else ("", [])  # suppress section if no graph data


def _format_vector(vector: list[dict], seen_sources: set) -> tuple[str, list[str]]:
    if not vector:  # no semantic matches returned from Atlas Vector Search
        return "", []

    sources = []
    lines = ["## SEMANTICALLY SIMILAR CONTENT"]

    for r in vector[:6]:  # cap at 6 vector results to avoid context bloat
        chunk_id = r.get("chunk_id", "?")
        score = r.get("score", 0)  # cosine similarity score 0-1 from $vectorSearch
        text = (r.get("text", "") or "")[:200]  # truncate chunk text preview to 200 chars
        source_coll = r.get("source_collection", "?")  # e.g. "emails", "pos", "job_types"

        source_label = f"{chunk_id} ({source_coll}, similarity {score:.2f})"
        if source_label in seen_sources:  # skip chunks already cited by mongo or graph sections
            continue

        lines.append(f"- **[{score:.2f}]** {source_coll}/{chunk_id}")
        lines.append(f"  {text}{'...' if len(r.get('text', '')) > 200 else ''}")
        sources.append(source_label)

    return ("\n".join(lines), sources) if len(lines) > 1 else ("", [])  # only emit section if at least one chunk was added
