import os, sys, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "eval" / "results"
OUTPUT_FILE  = RESULTS_DIR / "human_eval_output.txt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from retrieve.orchestrator import answer_query

questions = [
    ("q01", "Show me PO-2026-P0058"),
    ("q02", "Get me invoice INV-2025-0001"),
    ("q03", "Look up customer cust_0005"),
    ("q04", "Find email msg_0001"),
    ("q05", "Show all purchase orders for Gerard Walsh"),
    ("q06", "Any customers complaining about water dripping?"),
    ("q10", "Need someone urgently, job is dangerous"),
    ("q11", "What has Gerard Walsh had done before?"),
    ("q16", "How much does boiler installation usually cost?"),
    ("q_mat", "What materials are needed for invoice INV-2025-0001?"),
    ("q_quote", "Get me a quote for a boiler installation"),
]

# For human evaluation, we will run the above queries, save the answers and sources to a text file, then read that file and score each answer on the following metrics:
# - Faithfulness  : are all claims backed by the sources shown?
# - Ans Relevancy : does the answer address the question asked?
# - Ctx Precision : were the retrieved sources actually useful?
# - Ctx Recall    : did sources contain everything needed to answer the question?   
# We will use a 0.0-1.0 scale for each metric, with the following targets:
# - Faithfulness > 0.90
# - Ans Relevancy > 0.85
# - Ctx Precision > 0.80
# - Ctx Recall > 0.85
# The final output file will have the question, answer, sources, and blank score fields for each query, as well as a summary table at the end for easy reporting.
# lines.append are used to build the output text file, which is saved to OUTPUT_FILE. After running this script, open that file and fill in your scores based on the answers and sources provided.
lines = []
lines.append("DACARag — Human Evaluation Output")
lines.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
lines.append("=" * 60)
lines.append("")
lines.append("Instructions:")
lines.append("  Read each ANSWER and SOURCES below.")
lines.append("  Score each metric 0.0-1.0 in the scorecard widget.")
lines.append("  Faithfulness  : are all claims backed by the sources shown?")
lines.append("  Ans Relevancy : does the answer address the question asked?")
lines.append("  Ctx Precision : were the retrieved sources actually useful?")
lines.append("  Ctx Recall    : did sources contain everything needed?")
lines.append("")

# Run each query, save the answer and sources, and leave blank score fields for human evaluation after reading the results.
for qid, query in questions:
    print("Running " + qid + ": " + query[:55] + "...")
    try:
        result  = answer_query(query)
        answer  = result.get("answer", "") # showing answer is optional, as some retrieval-only queries may not have an "answer" field, but we will show sources for all queries and evaluate based on that.
        sources = result.get("sources", []) # sources should be a list of dicts with at least 'source_id' and 'chunk_type' fields, but we will just show the whole dict for evaluation (some sources may have extra metadata that is useful for judging relevance and faithfulness)
        routing = result.get("routing", {}) #   routing info is optional, but may provide useful context for evaluation (e.g. if the system routed to a fallback or retrieval-only path, that may explain a less relevant answer)
        latency = result.get("latency_ms", "?") # latency is optional, but may provide useful context for evaluation (e.g. if latency is very high, that may explain a less relevant answer due to timeouts or fallback routing)

        lines.append("=" * 60)
        lines.append("[" + qid + "] " + query)
        lines.append("=" * 60)
        lines.append("")
        lines.append("ROUTING : " + str(routing.get("intent","?")) + " via " + str(routing.get("paths",[])))
        lines.append("LATENCY : " + str(latency) + "ms")
        lines.append("")
        lines.append("ANSWER:")
        lines.append(answer)
        lines.append("")
        lines.append("SOURCES (" + str(len(sources)) + " retrieved):")
        for i, s in enumerate(sources, 1):
            lines.append("  [" + str(i) + "] " + str(s))
        lines.append("")
        lines.append("YOUR SCORES (fill in after reading above):")
        lines.append("  Faithfulness   : ___")
        lines.append("  Ans Relevancy  : ___")
        lines.append("  Ctx Precision  : ___")
        lines.append("  Ctx Recall     : ___")
        lines.append("")

    except Exception as e:
        lines.append("=" * 60)
        lines.append("[" + qid + "] " + query)
        lines.append("ERROR: " + str(e))
        lines.append("")

lines.append("=" * 60)
lines.append("SUMMARY TABLE (copy into report)")
lines.append("=" * 60)
lines.append("")
lines.append("Q ID     | Question                                    | Faith | AnsRel | CtxPre | CtxRec")
lines.append("---------|---------------------------------------------|-------|--------|--------|-------")
for qid, query in questions:
    lines.append(qid.ljust(8) + " | " + query[:43].ljust(43) + " |  ___  |  ___   |  ___   |  ___")
lines.append("         | AVERAGE                                     |  ___  |  ___   |  ___   |  ___")
lines.append("")
lines.append("Targets: Faithfulness > 0.90 | Ans Relevancy > 0.85 | Ctx Precision > 0.80 | Ctx Recall > 0.85")

output = "\n".join(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(output)

print("")
print("Done. Output saved to:")
print(str(OUTPUT_FILE))
print("")
print("Open that file, read each answer and sources, then fill in your scores.")
