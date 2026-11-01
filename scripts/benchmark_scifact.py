"""Evaluate document retrieval on a local BEIR SciFact zip; never download at import."""

import argparse
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np

from paper_retrieval.core import Chunk, EvidenceIndex


def rank_metrics(ranking, relevant, k=10):
    """Binary recall, reciprocal rank and nDCG for a single judged query."""
    hits = [int(doc in relevant) for doc in ranking[:k]]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "recall_at_10": sum(hits) / len(relevant),
        "mrr_at_10": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
        "ndcg_at_10": sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal,
    }


def benchmark(archive, output):
    start = time.perf_counter()
    with zipfile.ZipFile(archive) as z:
        # Read named members directly. Never extract arbitrary archive paths.
        corpus = [json.loads(line) for line in z.read("scifact/corpus.jsonl").splitlines()]
        queries = {
            row["_id"]: row["text"]
            for row in map(json.loads, z.read("scifact/queries.jsonl").splitlines())
        }
        qrels = {}
        for row in csv.DictReader(
            io.StringIO(z.read("scifact/qrels/test.tsv").decode()), delimiter="\t"
        ):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    documents = {doc["_id"] for doc in corpus}
    assert qrels and all(qid in queries and refs <= documents for qid, refs in qrels.items())
    index = EvidenceIndex()
    # SciFact provides abstracts, not PDFs. Page 1 is an internal adapter sentinel,
    # never exported as a scientific-paper page or used for citation evaluation.
    index.add(
        [Chunk(doc["_id"], doc["_id"], 1, doc["title"] + "\n" + doc["text"]) for doc in corpus]
    )
    report = {
        "dataset": "BEIR SciFact, official test qrels; no tuning on test labels",
        "source": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "documents": len(corpus),
        "test_queries": len(qrels),
        "protocol": "One title+abstract per document; existing ranker settings; top 10; binary qrels",
        "limitations": [
            "Document retrieval only; no PDF parsing, page citation, answer or abstention evaluation.",
            "Dense strategy uses 1024-dimensional lexical hashing, not semantic embeddings.",
            "No model training, tuning, reranking or external LLM; test data are not an untouched future validation set.",
        ],
        "results": {},
    }
    for strategy in ["bm25", "dense", "hybrid"]:
        started = time.perf_counter()
        rows = []
        for qid, relevant in sorted(qrels.items()):
            ranking = [h["chunk"]["id"] for h in index.search(queries[qid], strategy, k=10)]
            rows.append({"query_id": qid, **rank_metrics(ranking, relevant)})
        report["results"][strategy] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ["recall_at_10", "mrr_at_10", "ndcg_at_10"]
        } | {"query_seconds": time.perf_counter() - started, "per_query_metrics": rows}
    report["total_seconds"] = time.perf_counter() - start
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for strategy, result in report["results"].items():
        print(strategy, {k: v for k, v in result.items() if k != "per_query_metrics"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="results/scifact/metrics.json")
    args = parser.parse_args()
    benchmark(args.archive, args.output)
