"""Benchmark: does a larger internal kNN candidate pool (KNN_HYBRID_CANDIDATES
in kb_common.hybrid_search) actually improve recall, and at what latency cost?

Usage:
    PYTHONPATH=src python scripts/eval/bench_hybrid_knn.py [--k 10] [--runs 10] [--candidates 10,50,150,300]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from kb_common import config, es_index, hybrid_search

CASES = [
    {
        "query": "Argo float in the Mediterranean sea",
        "ground_truth_field": "sea_area.keyword",
        "ground_truth_values": ["Mediterranean Sea - Eastern Basin", "Mediterranean Sea - Western Basin"],
    },
    {
        "query": "Argo float measuring dissolved oxygen in the Mediterranean sea",
        "ground_truth_field": "sea_area.keyword",
        "ground_truth_values": ["Mediterranean Sea - Eastern Basin", "Mediterranean Sea - Western Basin"],
    },
    {
        "query": "Argo float in the Labrador sea",
        "ground_truth_field": "sea_area.keyword",
        "ground_truth_values": ["Labrador Sea"],
    },
    {
        "query": "profiling float operated by Ifremer",
        "ground_truth_field": "owner.keyword",
        "ground_truth_values": ["Ifremer", "IFREMER"],
    },
]


def ground_truth_ids(client, index, field, values):
    resp = client.search(index=index, query={"terms": {field: values}}, size=1000, source=False)
    return {hit["_id"] for hit in resp["hits"]["hits"]}


def recall_at_k(hit_ids, truth_ids, k):
    if not truth_ids:
        return None
    return len(set(hit_ids[:k]) & truth_ids) / min(k, len(truth_ids))


def run_case(client, index, case, k, runs, candidates):
    orig = hybrid_search.KNN_HYBRID_CANDIDATES
    hybrid_search.KNN_HYBRID_CANDIDATES = candidates
    try:
        truth_ids = ground_truth_ids(client, index, case["ground_truth_field"], case["ground_truth_values"])
        latencies = []
        hits = None
        for _ in range(runs):
            t0 = time.perf_counter()
            hits = hybrid_search.search(index, case["query"], k=k)
            latencies.append((time.perf_counter() - t0) * 1000)
        hit_ids = [h["_id"] for h in hits]
        return {
            "recall": recall_at_k(hit_ids, truth_ids, k),
            "truth_count": len(truth_ids),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        }
    finally:
        hybrid_search.KNN_HYBRID_CANDIDATES = orig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.ES_INDEX)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--candidates", default="10,50,150,300")
    args = parser.parse_args()
    candidate_values = [int(c) for c in args.candidates.split(",")]

    client = es_index.get_client()

    for candidates in candidate_values:
        print(f"\n=== KNN_HYBRID_CANDIDATES={candidates} ===")
        for case in CASES:
            result = run_case(client, args.index, case, args.k, args.runs, candidates)
            recall = f"{result['recall']:.2f}" if result["recall"] is not None else "n/a"
            print(
                f"  {case['query']!r}\n"
                f"    recall@{args.k}={recall} (of {result['truth_count']} truth docs)  "
                f"mean={result['mean_ms']:.0f}ms median={result['median_ms']:.0f}ms p95={result['p95_ms']:.0f}ms"
            )


if __name__ == "__main__":
    main()
