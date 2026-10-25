import argparse
import json
from pathlib import Path
from time import perf_counter

from .core import Chunk, EvidenceIndex


def evaluate(corpus, output, backend="lexical-hash"):
    data = json.loads(Path(corpus).read_text())
    index = EvidenceIndex(backend)
    index.add([Chunk(**c) for c in data["chunks"]])
    results = {}
    for strategy in ["bm25", "dense", "hybrid"]:
        rows = []
        for q in data["questions"]:
            start = perf_counter()
            answer = index.answer(q["question"], strategy, k=3)
            seconds = perf_counter() - start
            retrieved = [h["chunk"]["id"] for h in answer["evidence"]]
            expected = set(q["evidence"])
            ranks = [retrieved.index(c) + 1 for c in expected if c in retrieved]
            cited = {c["chunk_id"] for c in answer["citations"]}
            valid = all(
                any(
                    c.id == cit["chunk_id"]
                    and c.document == cit["document"]
                    and c.page == cit["page"]
                    for c in index.chunks
                )
                for cit in answer["citations"]
            )
            rows.append(
                {
                    "question": q["question"],
                    "answerable": bool(expected),
                    "recall_at_3": len(expected & set(retrieved)) / len(expected)
                    if expected
                    else None,
                    "reciprocal_rank": 1 / min(ranks) if ranks else 0,
                    "evidence_coverage": len(expected & cited) / len(expected)
                    if expected
                    else None,
                    "citation_validity": valid,
                    "abstained": answer["abstained"],
                    "abstention_correct": answer["abstained"] == (not expected),
                    "latency_seconds": seconds,
                    "retrieved": retrieved,
                    "citations": answer["citations"],
                }
            )
        positive = [r for r in rows if r["answerable"]]
        results[strategy] = {
            "recall_at_3": sum(r["recall_at_3"] for r in positive) / len(positive),
            "mrr_at_3": sum(r["reciprocal_rank"] for r in positive) / len(positive),
            "evidence_coverage": sum(r["evidence_coverage"] for r in positive) / len(positive),
            "citation_validity": all(r["citation_validity"] for r in rows),
            "abstention_accuracy": sum(r["abstention_correct"] for r in rows) / len(rows),
            "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
            "queries": rows,
        }
    report = {
        "corpus": data["provenance"],
        "backend": backend,
        "semantic_model_executed": backend == "sentence-transformer",
        "results": results,
    }
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="experiments/evaluation.json")
    p.add_argument("--output", default="results/evaluation.json")
    p.add_argument(
        "--backend", choices=["lexical-hash", "sentence-transformer"], default="lexical-hash"
    )
    a = p.parse_args()
    print(json.dumps(evaluate(**vars(a)), indent=2))


if __name__ == "__main__":
    main()
