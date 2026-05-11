import os, sys, json, csv, time, argparse, statistics
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
RESULTS_DIR   = PROJECT_ROOT / "eval" / "results"
OUTPUT_CSV    = RESULTS_DIR / "latency.csv"
OUTPUT_JSON   = RESULTS_DIR / "latency.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from retrieve.orchestrator import answer_query
    print("OK orchestrator imported")
except ImportError as e:
    sys.exit("Cannot import orchestrator: " + str(e))


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return round(s[idx], 3)


def main():
    parser = argparse.ArgumentParser(description="Measure DACARag query latency")
    parser.add_argument("--runs",   type=int, default=5,
                        help="Passes over eval set. 5 passes x 9 questions = 45 queries (default: 5)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warm-up queries to discard — avoids cold-start skew (default: 2)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DACARag — Latency Measurement")
    print("="*60)

    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    annotated = [q for q in eval_set
                 if q.get("gold_doc_ids") or not q.get("ground_truth","").startswith("FILL IN")]
    print("\nQuestions: " + str(len(annotated)) + " from eval_set.json")
    print("Runs:      " + str(args.runs) + " passes x " + str(len(annotated)) + " questions = " + str(args.runs * len(annotated)) + " total queries")
    print("Warm-up:   " + str(args.warmup) + " queries discarded\n")

    # Warm-up: loads sentence-transformers model into memory (~2-3s first call)
    # These queries are NOT timed and NOT included in results
    if args.warmup > 0:
        print("Running " + str(args.warmup) + " warm-up queries (not timed)...")
        for i in range(min(args.warmup, len(annotated))):
            q = annotated[i]["question"]
            try:
                answer_query(q)
                print("  Warm-up " + str(i+1) + "/" + str(args.warmup) + " done")
            except Exception as e:
                print("  Warm-up " + str(i+1) + " error: " + str(e))
        print()

    # Measurement phase
    all_timings = []
    by_type     = {}
    rows        = []

    for run in range(1, args.runs + 1):
        print("Pass " + str(run) + "/" + str(args.runs) + "...")
        for q in annotated:
            qtype   = q["type"]
            query   = q["question"]
            qid     = q["id"]

            t0 = time.perf_counter()
            try:
                result  = answer_query(query)
                elapsed = time.perf_counter() - t0
                status  = "ok"
                routing = result.get("routing", {}).get("intent", "?")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                status  = "error: " + str(e)[:60]
                routing = "error"

            all_timings.append(elapsed)
            by_type.setdefault(qtype, []).append(elapsed)

            rows.append({
                "run":         run,
                "question_id": qid,
                "type":        qtype,
                "question":    query[:60],
                "routing":     routing,
                "latency_s":   round(elapsed, 3),
                "status":      status,
            })
            print("  [" + qid + "] " + str(round(elapsed, 2)) + "s  " + routing)

        print()

    # Save raw CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary
    summary = {}
    for qtype in ["exact_lookup", "semantic_paraphrase", "multi_hop_graph", "aggregation"]:
        times = by_type.get(qtype, [])
        if times:
            summary[qtype] = {
                "n":   len(times),
                "p50": percentile(times, 50),
                "p95": percentile(times, 95),
                "p99": percentile(times, 99),
                "max": round(max(times), 3),
                "mean": round(statistics.mean(times), 3),
            }

    summary["ALL"] = {
        "n":   len(all_timings),
        "p50": percentile(all_timings, 50),
        "p95": percentile(all_timings, 95),
        "p99": percentile(all_timings, 99),
        "max": round(max(all_timings), 3),
        "mean": round(statistics.mean(all_timings), 3),
    }

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw_rows": rows}, f, indent=2)

    # Print summary
    print("="*60)
    print("LATENCY SUMMARY  (" + str(args.runs) + " passes, " + str(len(all_timings)) + " total queries)")
    print("="*60)
    print("")
    header = "  {:<25} {:>4}  {:>6}  {:>6}  {:>6}  {:>6}".format(
        "Route type", "n", "p50", "p95", "p99", "max")
    print(header)
    print("  " + "-"*57)

    for qtype in ["exact_lookup", "semantic_paraphrase", "multi_hop_graph", "aggregation", "ALL"]:
        s = summary.get(qtype)
        if s:
            label = "ALL ROUTES" if qtype == "ALL" else qtype
            print("  {:<25} {:>4}  {:>5.2f}s  {:>5.2f}s  {:>5.2f}s  {:>5.2f}s".format(
                label, s["n"], s["p50"], s["p95"], s["p99"], s["max"]))

    print("")
    print("  Mean (all routes): " + str(summary["ALL"]["mean"]) + "s")

    # Flag slow queries above p95
    p95_all = summary["ALL"]["p95"]
    slow = [r for r in rows if r["latency_s"] > p95_all]
    if slow:
        print("\n  Slow queries (above p95 = " + str(p95_all) + "s):")
        shown = {}
        for r in slow:
            key = r["question_id"]
            if key not in shown:
                shown[key] = r
        for r in list(shown.values())[:5]:
            print("    [" + r["type"] + "] " + r["question"][:50] + "... " + str(r["latency_s"]) + "s")

    print("\nOK Results saved to:")
    print("  " + str(OUTPUT_CSV))
    print("  " + str(OUTPUT_JSON))
    print("")
    print("Copy the p50/p95 numbers into Section 4 of your report.")
    print("Target: p50 < 3s warm-state, p95 < 7s (excluding cold start).")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
