#!/usr/bin/env python3
"""Evaluate the RAG retrieval step (TF-IDF over knowledge_base/). No API key or internet needed.

For every query in eval_data/retrieval_queries.csv we check whether one of the top-k retrieved rules
contains the expected keyword, and report Hit@1 / Hit@3 / Hit@k and MRR (mean reciprocal rank).

Usage (project root):  python evaluation/eval_retrieval.py [--k 5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import metrics as M  # noqa: E402
import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "eval_data" / "retrieval_queries.csv"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "report"))
    args = ap.parse_args()

    with open(args.queries, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    queries = list(csv.DictReader(lines))

    kb = P.KnowledgeBase()
    ranks, rows = [], []
    for q in queries:
        got = kb.retrieve(q["query"], k=args.k)
        kw = q["expected_keyword"].lower()
        rank = next((i + 1 for i, g in enumerate(got) if kw in g["text"].lower()), None)
        ranks.append(rank)
        rows.append({"query": q["query"], "expected_keyword": q["expected_keyword"], "rank": rank,
                     "top1_source": got[0]["source"] if got else None})

    n = len(ranks)
    hit = lambda k: sum(r is not None and r <= k for r in ranks)  # noqa: E731
    mrr = sum(1 / r for r in ranks if r) / n
    res = {"queries": n, "k": args.k,
           "hit@1": hit(1) / n, "hit@3": hit(3) / n, f"hit@{args.k}": hit(args.k) / n, "mrr": mrr,
           "failed_queries": [r["query"] for r in rows if r["rank"] is None]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_metrics.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=2), encoding="utf-8")
    lines = ["# RAG retrieval evaluation", "",
             f"Knowledge base chunks: {len(kb.chunks)} | Queries: {n} | k = {args.k}", "",
             "| Metric | Result |", "|---|---|",
             f"| Hit@1 | {M.fmt_rate(hit(1), n)} |", f"| Hit@3 | {M.fmt_rate(hit(3), n)} |",
             f"| Hit@{args.k} | {M.fmt_rate(hit(args.k), n)} |", f"| MRR | {mrr:.3f} |", ""]
    if res["failed_queries"]:
        lines += ["Queries where the right rule was NOT retrieved (improve the knowledge base text or the query):", ""]
        lines += [f"- {q}" for q in res["failed_queries"]]
    (out / "retrieval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Queries {n} | Hit@1 {res['hit@1']:.0%} | Hit@3 {res['hit@3']:.0%} | Hit@{args.k} {res[f'hit@{args.k}']:.0%} | MRR {mrr:.3f}")
    for q in res["failed_queries"]:
        print("  MISSED:", q)
    print(f"Written to {out}/retrieval_summary.md")


if __name__ == "__main__":
    main()
