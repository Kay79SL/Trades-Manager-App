"""
orchestrator.py
================
Module 7 of Phase 5 retrieval orchestration — THE GLUE.

Single entry point function `answer_query(user_message)` that ties all
six Phase 5 modules together. Updated to support list/browse queries.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

from . import router
from . import vector_search
from . import mongo_lookup
from . import graph_search
from . import predict_quote
from . import context_assembler


load_dotenv()

ANSWER_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = """You are a helpful assistant for an Irish trades business
(plumbing, carpentry, electrical). You answer questions about customers,
quotes, jobs, and pricing.

CRITICAL RULES:
- Use ONLY the context provided below. Do not invent customer names, prices,
  invoice numbers, or any other facts.
- If the context contains a list of records, present them clearly and
  comprehensively. Don't say "I can only see one" if the list has more.
- If the context does not contain the answer, say so honestly.
- Quote specific numbers (€ amounts, customer ids, PO numbers, dates) when
  they appear in the context.
- For pricing questions, present the prediction clearly and mention the
  confidence level.
- Keep responses professional and well-formatted (use bullet points for lists).
- No unnecessary preamble."""


@lru_cache(maxsize=1)
def _get_client(): # Cache the Anthropic client since it doesn't need to be re-created for each query
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _handle_list_query(list_target: str) -> dict: # New helper function to handle list/browse queries
    """Fetch list/summary data based on what the user wants to see."""
    if list_target == "pos": # List all purchase orders
        return {
            "type": "pos",
            "title": "All Purchase Orders",
            "items": mongo_lookup.list_all_pos(),
        }
    elif list_target == "customers": # List all customers
        return {
            "type": "customers",
            "title": "Customers",
            "items": mongo_lookup.list_all_customers(),
        }
    elif list_target == "jobs": # List all job types    
        return {
            "type": "jobs",
            "title": "Job Types",
            "items": mongo_lookup.list_all_job_types(),
        }
    elif list_target == "invoices": # List all invoices
        return {
            "type": "invoices",
            "title": "Recent Invoices",
            "items": mongo_lookup.list_all_invoices(),
        }
    elif list_target == "emails": # List recent emails
        return {
            "type": "emails",
            "title": "Recent Emails",
            "items": mongo_lookup.list_recent_emails(),
        }
    elif list_target == "summary": # Provide a high-level summary of the database contents
        return {
            "type": "summary",
            "title": "Database Overview",
            "items": mongo_lookup.get_collection_summary(),
        }
    return {}


def answer_query(user_message: str, verbose: bool = False) -> dict[str, Any]:
    """The single entry point for the chatbot."""
    t_start = time.time()

    # Step 1: Router decides which retrievers to invoke
    routing = router.classify_query(user_message) # Returns dict with keys: "intent", "method", "paths" (list), and "params" (dict)

    # Step 2: Run the selected retrievers
    results: dict[str, Any] = {} # Will hold results from vector search, mongo lookup, graph search, predict_quote, and list/browse (if applicable)
    paths = routing.get("paths", []) # List of retriever paths to execute, e.g. ["vector", "mongo"] or ["list"]
    params = routing.get("params", {}) # Parameters extracted by the router, e.g. {"customer_name": "John Doe"} or {"list_target": "pos"}

    if "vector" in paths: # Always run vector search if it's in the paths, even for list/browse queries, since it can provide relevant context. The router can decide to include vector search for list queries if it thinks it will help answer the user's question more effectively.
        results["vector"] = vector_search.vector_search(user_message, top_k=5)

    if "mongo" in paths: # Always run mongo lookup if it's in the paths
        results["mongo"] = mongo_lookup.lookup_by_intent(params)

    if "graph" in paths:
        # Enrich params with customer_id resolved by mongo lookup
        # so graph_traversal can call customer_full_history()
        graph_params = dict(params)
        mongo_result = results.get("mongo", {}) # Get the mongo lookup result to extract customer_id if available
        if mongo_result.get("customer", {}).get("customer_id"):
            graph_params["customer_id"] = mongo_result["customer"]["customer_id"]
        results["graph"] = graph_search.graph_traversal(graph_params)

    if "predict_quote" in paths: # Only run predict_quote if it's in the paths, and pass job_name and job_type_id as parameters
        job_name = params.get("job_name", "")
        if job_name: # Only call predict_quote if we have a job_name, otherwise skip it to avoid errors
            results["predict"] = predict_quote.predict_quote(
                job_name,
                job_type_id=params.get("job_type_id"),
            )

    # NEW: list/browse path
    if "list" in paths: # If the router has determined that this is a list/browse query, call the helper function to fetch the appropriate list data based on the "list_target" parameter extracted by the router (e.g. "pos", "customers", "jobs", "invoices", "emails", or "summary")
        results["list"] = _handle_list_query(params.get("list_target", "summary"))

    # Step 3: Assemble context
    context = context_assembler.assemble_context(results, user_query=user_message) #    Takes all the retriever results and compiles them into a single context string, also formats list/browse results in a clear way if present

    # Step 4: Call Claude
    client = _get_client()
    response = client.messages.create( # Pass the assembled context and the user question to Claude, and get the answer. The system prompt instructs Claude to use only the provided context to answer, and to present list responses clearly if the context contains a list of records.
        model=ANSWER_MODEL,
        max_tokens=2000,  # Increased to handle list responses
        system=SYSTEM_PROMPT,
        messages=[ # The user message includes the full context and the original user question. This way, Claude has all the information it needs to answer accurately, even for list/browse queries where the context may contain multiple records. The system prompt also instructs Claude to present list responses clearly and comprehensively, so it should format the answer in a way that is easy to read and understand.
            {
                "role": "user",
                "content": (
                    f"CONTEXT (use this to answer):\n\n"
                    f"{context['context_text']}\n\n"
                    f"---\n\n"
                    f"USER QUESTION: {user_message}"
                ),
            }
        ],
    )
    answer_text = response.content[0].text # Extract the text of Claude's answer from the response object

    latency_ms = round((time.time() - t_start) * 1000) # Calculate total latency for the entire process (retrieval + context assembly + LLM response) in milliseconds

    out = { # The final output dictionary that will be returned by the function, containing the answer, sources, routing info, latency, and token usage. If verbose=True, it will also include the raw retriever results and the full context text for debugging purposes.
        "answer":     answer_text,
        "sources":    context["sources"],
        "routing":    routing,
        "latency_ms": latency_ms,
        "tokens_in":  response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }

    if verbose: # If verbose mode is enabled, include the raw retriever results and the full context text in the output for debugging purposes
        out["raw_results"] = results
        out["context_text"] = context["context_text"]

    return out


if __name__ == "__main__": # Simple command-line interface for testing the answer_query function with different user questions, including list/browse queries. 
    # You can run this script with custom questions as command-line arguments, or it will default to a set of example questions if no arguments are provided.
    import sys
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "How much for a boiler installation?",
        "Show me PO-2026-P0042",
        "show me all POs",
    ]
    for q in queries:
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {q}")
        print('=' * 70)
        out = answer_query(q, verbose=False)
        print(f"\n{out['answer']}")
        print(f"\n--- routing: {out['routing']['intent']} via {out['routing']['method']} ---")
        print(f"--- {out['latency_ms']}ms / {out['tokens_in']} tokens in / {out['tokens_out']} tokens out ---")
