import os, sys, re, csv, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "eval" / "results" / "human_eval_output.txt"
OUTPUT_CSV   = PROJECT_ROOT / "eval" / "results" / "ragas_scores.csv"
OUTPUT_JSON  = PROJECT_ROOT / "eval" / "results" / "ragas_scores.json"

TARGETS = {
    "faithfulness":      0.90,
    "answer_relevancy":  0.85,
    "context_precision": 0.80,
    "context_recall":    0.85,
}

# ── Parse the output file into question blocks ───────────────────────────────

def parse_output_file(path):
    text = path.read_text(encoding="utf-8")
    blocks = []
    pattern = re.compile(
        r"\[(?P<qid>q\d+)\] (?P<question>.+?)\n"
        r"={60}\n\n"
        r"ROUTING.*?\n"
        r"LATENCY.*?\n\n"
        r"ANSWER:\n(?P<answer>.*?)\n\n"
        r"SOURCES \(\d+ retrieved\):\n(?P<sources>.*?)\n\n"
        r"YOUR SCORES",
        re.DOTALL
    )
    for m in pattern.finditer(text):
        sources_raw = m.group("sources").strip().split("\n")
        sources = [re.sub(r"^\s*\[\d+\]\s*", "", s).strip() for s in sources_raw if s.strip()]
        blocks.append({
            "qid":      m.group("qid").strip(),
            "question": m.group("question").strip(),
            "answer":   m.group("answer").strip(),
            "sources":  sources,
        })
    return blocks

# ── Claude Haiku judge — one metric at a time ────────────────────────────────

def judge(metric, question, answer, sources, ground_truth=""):
    ctx = "\n---\n".join(sources[:3]) if sources else "No context retrieved"

    prompts = {
        "faithfulness": (
            "You are a strict RAG evaluator scoring FAITHFULNESS.\n\n"
            "RETRIEVED CONTEXT:\n" + ctx + "\n\n"
            "SYSTEM ANSWER:\n" + answer + "\n\n"
            "Score: what fraction of claims in the answer are directly supported "
            "by the retrieved context? Ignore general common knowledge.\n"
            "1.0 = every claim is grounded in context\n"
            "0.5 = about half the claims are grounded\n"
            "0.0 = answer ignores context or invents facts\n\n"
            "Reply with ONLY a number between 0.0 and 1.0. Nothing else."
        ),
        "answer_relevancy": (
            "You are a strict RAG evaluator scoring ANSWER RELEVANCY.\n\n"
            "QUESTION: " + question + "\n\n"
            "SYSTEM ANSWER:\n" + answer + "\n\n"
            "Score: how directly and completely does the answer address the question?\n"
            "1.0 = perfectly addresses the question\n"
            "0.5 = partially answers it\n"
            "0.0 = answer is off-topic or empty\n\n"
            "Reply with ONLY a number between 0.0 and 1.0. Nothing else."
        ),
        "context_precision": (
            "You are a strict RAG evaluator scoring CONTEXT PRECISION.\n\n"
            "QUESTION: " + question + "\n\n"
            "RETRIEVED SOURCES:\n" + ctx + "\n\n"
            "Score: what fraction of the retrieved sources are actually relevant "
            "to answering this question?\n"
            "1.0 = all sources are useful\n"
            "0.5 = about half are useful\n"
            "0.0 = none of the sources are relevant\n\n"
            "Reply with ONLY a number between 0.0 and 1.0. Nothing else."
        ),
        "context_recall": (
            "You are a strict RAG evaluator scoring CONTEXT RECALL.\n\n"
            "QUESTION: " + question + "\n\n"
            "RETRIEVED SOURCES:\n" + ctx + "\n\n"
            "SYSTEM ANSWER:\n" + answer + "\n\n"
            "Score: does the retrieved context contain all the information needed "
            "to produce a complete and accurate answer?\n"
            "1.0 = context has everything needed\n"
            "0.5 = context has some but not all needed info\n"
            "0.0 = key information is missing from context\n\n"
            "Reply with ONLY a number between 0.0 and 1.0. Nothing else."
        ),
    }

    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompts[metric]}]
        )
        raw = r.content[0].text.strip().split()[0]
        return round(min(1.0, max(0.0, float(raw))), 2)
    except Exception as e:
        print("    Judge error (" + metric + "): " + str(e))
        return None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("DACARag — LLM Judge Evaluation (Claude Haiku)")
    print("="*60)

    if not INPUT_FILE.exists():
        sys.exit("ERROR: " + str(INPUT_FILE) + " not found.\nRun python eval\\run_human_eval.py first.")

    blocks = parse_output_file(INPUT_FILE)
    if not blocks:
        sys.exit("ERROR: Could not parse any questions from " + str(INPUT_FILE) +
                 "\nCheck the file was generated correctly.")

    print("\nParsed " + str(len(blocks)) + " questions from human_eval_output.txt")
    print("Sending each to Claude Haiku for scoring...\n")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = []

    for b in blocks:
        qid      = b["qid"]
        question = b["question"]
        answer   = b["answer"]
        sources  = b["sources"]

        print("[" + qid + "] " + question[:55] + "...")
        scores = {}
        for m in metrics:
            score = judge(m, question, answer, sources)
            scores[m] = score
            flag = str(score) if score is not None else "ERR"
            print("    " + m.ljust(22) + flag)

        rows.append({
            "question_id":       qid,
            "question":          question,
            "faithfulness":      scores.get("faithfulness"),
            "answer_relevancy":  scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
            "context_recall":    scores.get("context_recall"),
        })
        print()

    # Averages
    averages = {}
    for m in metrics:
        vals = [r[m] for r in rows if r[m] is not None]
        averages[m] = round(sum(vals) / len(vals), 4) if vals else None

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "question_id": "AVERAGE",
            "question":    "",
            "faithfulness":      averages.get("faithfulness"),
            "answer_relevancy":  averages.get("answer_relevancy"),
            "context_precision": averages.get("context_precision"),
            "context_recall":    averages.get("context_recall"),
        })

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"questions": rows, "averages": averages}, f, indent=2)

    # Print summary
    print("="*60)
    print("RAGAS LLM JUDGE RESULTS")
    print("="*60)
    for m in metrics:
        val = averages.get(m)
        tgt = TARGETS[m]
        if val is not None:
            flag = "OK  target >= " + str(tgt) if val >= tgt else "--- below target >= " + str(tgt)
            print("  " + m.ljust(25) + str(val) + "   " + flag)
        else:
            print("  " + m.ljust(25) + "N/A")

    print("\nOK Saved to:")
    print("  " + str(OUTPUT_CSV))
    print("  " + str(OUTPUT_JSON))
    print("="*60 + "\n")
    print("Copy the averages table into Section 4 of your report.")


if __name__ == "__main__":
    main()
