# Phase 8 — Evaluation Instructions

**Due: Wednesday 6 May. Today is Monday 4 May. You have today + tomorrow.**

---

## What's in this folder

| File | What it does |
|------|-------------|
| `eval_set.json` | 20 questions with gold-standard answers — your exam answer sheet |
| `run_baselines.py` | Compares MongoDB $text vs BM25 vs your full RAG pipeline |
| `run_ragas.py` | Measures answer quality (faithfulness, relevancy, precision, recall) |
| `latency.py` | Measures p50/p95/p99 query times across 100 queries |
| `results/` | Created automatically — CSVs land here |

---

## Step 1 — Fill in the gold labels in eval_set.json

**Do this first. Everything else depends on it.**

Open `eval/eval_set.json`. Find all questions where `gold_doc_ids` is an empty list `[]`
or `ground_truth` starts with `"FILL IN"`. There are 14 of them.

For each one:
1. Go to MongoDB Atlas Data Explorer (or run a quick Python query)
2. Run the question manually through your chatbot
3. Note which document IDs were correctly returned
4. Paste those IDs into `gold_doc_ids`
5. Write the correct answer in `ground_truth` (1–2 sentences)

**Example — filling in q06:**
```python
# In PowerShell with venv active, run this to find leak-related emails:
& "C:\Users\Kasia\AppData\Local\Python\bin\python.exe" -c "
from pymongo import MongoClient; from dotenv import load_dotenv; import os
load_dotenv()
db = MongoClient(os.getenv('MONGO_URI'))['trades_quotes']
leaks = list(db.emails.find({'body': {'$regex': 'leak|drip|water', '$options': 'i'}}, {'_id':1}))
print([str(x['_id']) for x in leaks[:5]])
"
```

The 6 questions that are already annotated (q01–q05, q16) are ready to go.
You can run the scripts with just these 6 to check everything works, then
add more gold labels and re-run.

---

## Step 2 — Run the baseline comparison

```powershell
cd C:\Users\Kasia\DACARag
venv\Scripts\activate
python eval\run_baselines.py
```

**What you'll see:** A table showing recall@5, precision@5, and MRR for each
of the three systems ($text, BM25, RAG) per question, then averages by type.

**What to look for:**
- RAG should clearly beat $text and BM25 on semantic paraphrase questions
- Exact lookup should score near 1.0 for all three (the regex path is deterministic)
- If RAG scores LOWER than BM25 on any type, check your router is classifying correctly

**Output file:** `eval/results/baselines.csv`

**Time:** ~5 minutes (BM25 corpus build + 20 RAG queries)

---

## Step 3 — Fill in ground_truth for the remaining questions

Before running RAGAS, you need `ground_truth` text for every question you want scored.
Run the chatbot manually for the 14 unannotated questions, note what the correct answer
*should* be, and write it into `eval_set.json`.

This is the most time-consuming part but there's no shortcut — RAGAS uses your
ground truth as the benchmark.

---

## Step 4 — Run RAGAS

```powershell
# Quick test on the 6 already-annotated questions (free, ~2 min):
python eval\run_ragas.py --limit 6

# Full run on all annotated questions, 3 trials averaged (~5 min, ~€0.05):
python eval\run_ragas.py --trials 3
```

**What you'll see:** Four scores between 0 and 1.

| Metric | What it means | Target |
|--------|--------------|--------|
| faithfulness | Are all claims in the answer backed by retrieved context? | ≥ 0.80 |
| answer_relevancy | Does the answer address the actual question? | ≥ 0.75 |
| context_precision | Are the retrieved chunks actually useful? | ≥ 0.70 |
| context_recall | Does the context contain everything needed? | ≥ 0.70 |

**If faithfulness < 0.80:** Add this to your system prompt in orchestrator.py:
```
"Only use information from the provided context. Do not use prior knowledge."
```

**If context_precision < 0.70:** Reduce `k` in your vector search from 5 to 3,
or add a cosine similarity threshold (e.g. only return chunks scoring > 0.6).

**Output file:** `eval/results/ragas_scores.csv`

---

## Step 5 — Run latency measurement

```powershell
# 5 passes × 20 questions = 100 queries, 2 warm-up discarded
python eval\latency.py --runs 5 --warmup 2
```

**What you'll see:** p50/p95/p99 broken down by route type.

**What to look for:**
- exact_lookup p50 should be ~1.5–2s (dominated by Claude API call, routing is instant)
- pricing_query p50 will be slowest (~3–4s, four sub-queries + Claude)
- p99 should be under 10s — if not, your AuraDB auto-recovery is adding latency

**Output file:** `eval/results/latency.csv`

**Time:** ~10 minutes for 100 queries

---

## Step 6 — Update Section 4 of your report

Copy these three tables into the report:

### Table 1 — Baseline comparison (from baselines.csv)

| System | Recall@5 | Precision@5 | MRR |
|--------|----------|-------------|-----|
| MongoDB $text | (fill in) | (fill in) | (fill in) |
| BM25 | (fill in) | (fill in) | (fill in) |
| DACARag (full) | (fill in) | (fill in) | (fill in) |

### Table 2 — RAGAS answer quality (from ragas_scores.csv)

| Metric | Score |
|--------|-------|
| Faithfulness | (fill in) |
| Answer Relevancy | (fill in) |
| Context Precision | (fill in) |
| Context Recall | (fill in) |

### Table 3 — Latency by route (from latency.csv)

| Route | p50 | p95 |
|-------|-----|-----|
| exact_lookup | (fill in) | (fill in) |
| semantic_paraphrase | (fill in) | (fill in) |
| multi_hop_graph | (fill in) | (fill in) |
| pricing_query | (fill in) | (fill in) |

### Failure taxonomy paragraph

Go through the questions where RAG scored 0 on recall@5.
Group the failures into buckets:
- **Retrieval miss** — router chose right intent, wrong documents returned
- **Routing error** — router chose wrong intent entirely
- **Missing data** — question asked about a sparse job type with few invoices
- **Ambiguous phrasing** — question was genuinely unclear
- **Generation drift** — right documents retrieved, Claude answered differently

Even 2–3 failures categorised shows Dr. Rizwan you understand your system's limits.

---

## Step 7 — Record backup demo video

Before you shut down for the night, record a 3–5 minute screen capture
of the Streamlit chatbot running these five queries:

1. `Show me PO-2026-P0042`              (exact lookup — fast regex path)
2. `How much for boiler installation?`  (pricing prediction — all three stores)
3. `Any complaints about water dripping?` (semantic vector search)
4. `What has Gerard Walsh had done before?` (Neo4j graph traversal)
5. `What are the top 5 most invoiced job types?` (aggregation)

Save to `docs/demo_backup.mp4`. Submit alongside the report.
If AuraDB auto-pauses on demo day, this video is your safety net.

---

## Commit when done

```powershell
git add eval\
git commit -m "Phase 8: baselines, RAGAS, latency, failure taxonomy"
git tag v0.5-eval
git push --tags
```
