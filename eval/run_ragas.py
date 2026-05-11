"""
eval/run_ragas.py
=================
Measures answer *quality* using RAGAS — four metrics that check whether
Claude's generated answers are grounded, relevant, and complete.

RAGAS checks if the ANSWER TEXT is trustworthy.

Metrics:
  faithfulness      — Is every claim in the answer supported by the context?
  answer_relevancy  — Does the answer actually address the question asked?
  context_precision — Are the retrieved chunks actually useful?
  context_recall    — Does the context contain everything needed to answer?

Usage:
    cd C:\\Users\\Kasia\\DACARag
    venv\\Scripts\\activate
    python eval\\run_ragas.py
    python eval\\run_ragas.py --trials 3       (run 3 times, average scores)
    python eval\\run_ragas.py --limit 5        (quick test on first 5 questions)

Cost estimate: ~€0.05–0.10 per full run (20 questions × RAGAS LLM calls)
"""

import os, sys, json, csv, time, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Fix for RAGAS 0.2.x asyncio timeout errors on Windows
# RAGAS uses asyncio.timeout() which must run inside an active task.
# nest_asyncio patches the event loop to allow this on Windows.
import nest_asyncio
nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
RESULTS_DIR   = PROJECT_ROOT / "eval" / "results"
OUTPUT_CSV    = RESULTS_DIR / "ragas_scores.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import RAG pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from retrieve.orchestrator import answer_query
    print("✓ RAG orchestrator imported")
except ImportError as e:
    sys.exit(f"ERROR: Could not import RAG orchestrator: {e}\n"
             "Make sure you're running from the project root with venv active.")

# ---------------------------------------------------------------------------
# Import RAGAS + configure Claude as the judge LLM
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    print("✓ RAGAS imported (version check: ragas==0.2.10 expected)")
except ImportError:
    sys.exit("ERROR: RAGAS not installed.\n"
             "Run: pip install ragas==0.2.10 datasets langchain-anthropic")

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # Wrap Claude Haiku as the RAGAS judge LLM
    # RAGAS defaults to OpenAI — this overrides it with your existing Anthropic key
    ragas_llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    ))
    # Wrap the same sentence-transformers model you use for embeddings
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    print("✓ Claude Haiku configured as RAGAS judge LLM")
except ImportError:
    sys.exit(
        "ERROR: langchain-anthropic not installed.\n"
        "Run: pip install langchain-anthropic langchain-community"
    )

# ---------------------------------------------------------------------------
# Collect RAG answers and contexts for the eval set
# ---------------------------------------------------------------------------

def collect_rag_outputs(eval_set: list, limit: int = None) -> list:
    """
    For each question, run your RAG pipeline and collect:
      - question:     the original question text
      - answer:       what Claude generated
      - contexts:     the list of text chunks that were retrieved
      - ground_truth: the hand-written correct answer from eval_set.json

    This is the raw material RAGAS evaluates.
    """
    questions = eval_set if limit is None else eval_set[:limit]
    records   = []

    print(f"\nCollecting RAG outputs for {len(questions)} questions...")
    print("(This calls Claude once per question — ~€0.002 per question)\n")

    for i, q in enumerate(questions, 1):
        qid   = q["id"]
        query = q["question"]
        truth = q["ground_truth"]

        # Skip questions whose ground_truth still says "FILL IN"
        # You must hand-write the ground truth before RAGAS can score them
        if truth.startswith("FILL IN"):
            print(f"  [{i:02d}/{len(questions)}] {qid} — SKIPPED (ground_truth not filled in yet)")
            continue

        print(f"  [{i:02d}/{len(questions)}] {qid}: {query[:55]}...")

        try:
            t0     = time.perf_counter()
            result = answer_query(query)
            elapsed = time.perf_counter() - t0

            answer   = result.get("answer", "")
            sources  = result.get("sources", [])

            # Extract text from each source for the contexts list
            # RAGAS needs a list of strings, not a list of dicts
            context_texts = []
            for s in sources:
                if isinstance(s, dict):
                    text = s.get("text") or s.get("content") or str(s)
                else:
                    text = str(s)
                context_texts.append(text)

            # If orchestrator didn't return text contexts, try fetching them
            if not context_texts:
                context_texts = ["No context retrieved"]

            records.append({
                "question":     query,
                "answer":       answer,
                "contexts":     context_texts,
                "ground_truth": truth,
                "question_id":  qid,
                "question_type": q["type"],
                "latency_s":    round(elapsed, 2),
            })
            print(f"         answer={len(answer)} chars  contexts={len(context_texts)}  time={elapsed:.1f}s")

        except Exception as e:
            print(f"         ERROR: {e}")
            records.append({
                "question":     query,
                "answer":       f"ERROR: {e}",
                "contexts":     ["Error — no context retrieved"],
                "ground_truth": truth,
                "question_id":  qid,
                "question_type": q["type"],
                "latency_s":    0.0,
            })

    print(f"\n  Collected {len(records)} valid records for RAGAS")
    return records


def run_ragas_evaluation(records: list) -> dict:
    if not records:
        print("  No records to evaluate — skipping RAGAS.")
        return {}

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in records],
        "answer":       [r["answer"]       for r in records],
        "contexts":     [r["contexts"]     for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    print(f"\nRunning RAGAS on {len(records)} questions...")
    print("(RAGAS calls Claude to judge each answer — takes 2-5 minutes)\n")

    from ragas.run_config import RunConfig
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=None, max_retries=2, max_wait=60),
    )

    # EvaluationResult is not a dict — extract scores via to_pandas()
    # Each column is one metric; take the column mean, ignoring NaN (failed jobs)
    import pandas as pd
    df = result.to_pandas()
    print("\n  Raw per-question scores:")
    print(df[["faithfulness","answer_relevancy","context_precision","context_recall"]].to_string())

    scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            val = df[metric].dropna()
            scores[metric] = float(val.mean()) if len(val) > 0 else None
        else:
            scores[metric] = None
    return scores

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on DACARag")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of times to run RAGAS (results averaged). Default: 1")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only evaluate the first N questions. Good for a quick test.")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DACARag — RAGAS Evaluation")
    print("="*60)

    # Load eval set
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)
    print(f"\nLoaded {len(eval_set)} questions")

    # Collect RAG outputs once (reused across trials)
    records = collect_rag_outputs(eval_set, limit=args.limit)

    if not records:
        print("\nNo annotated questions found. Fill in the 'ground_truth' fields")
        print("in eval/eval_set.json for the questions that say 'FILL IN', then re-run.")
        sys.exit(0)

    # Run RAGAS for each trial
    trial_scores = []
    for trial in range(1, args.trials + 1):
        print(f"\n--- Trial {trial} of {args.trials} ---")
        scores = run_ragas_evaluation(records)
        if scores:
            trial_scores.append(scores)
            print(f"\n  Trial {trial} scores:")
            for metric in ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]:
                val = scores.get(metric) if isinstance(scores, dict) else None
                print(f"    {metric:<22} {val:.4f}" if isinstance(val, float) else f"    {metric:<22} {val}")

    if not trial_scores:
        print("No RAGAS scores returned. Check your ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Average across trials
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {}
    for m in metrics:
        vals = [s[m] for s in trial_scores if isinstance(s.get(m), float)]
        averages[m] = round(sum(vals) / len(vals), 4) if vals else None

    # Save results CSV
    rows = []
    for r in records:
        row = {"question_id": r["question_id"], "type": r["question_type"],
               "question": r["question"], "latency_s": r["latency_s"]}
        row.update({m: averages.get(m) for m in metrics})
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Final summary
    print("\n" + "="*60)
    print(f"RAGAS RESULTS  ({args.trials} trial(s) averaged)")
    print("="*60)
    target = {"faithfulness": 0.80, "answer_relevancy": 0.75,
              "context_precision": 0.70, "context_recall": 0.70}
    for m in metrics:
        val  = averages.get(m)
        tgt  = target[m]
        flag = "✓" if val and val >= tgt else "✗  (target ≥ {:.2f})".format(tgt)
        print(f"  {m:<25} {val:.4f}  {flag}" if val else f"  {m:<25} N/A")

    print(f"\n✓ Results saved to {OUTPUT_CSV}")
    print("\nIf faithfulness < 0.80:")
    print("  Add 'Only use the provided context. Do not use prior knowledge.' to your system prompt.")
    print("If context_precision < 0.70:")
    print("  Your vector search is returning too many irrelevant chunks. Reduce k or add a score threshold.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
