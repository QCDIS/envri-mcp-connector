"""Dev tool: sanity-check semantic search quality, not a pass/fail test suite.

Runs hand-picked queries and prints the top hits so you can eyeball whether a
change improved retrieval. Default mode is hybrid (BM25 + kNN, summed in one
request rather than via the license-gated `retriever`/`rrf` API); --mode knn
or --mode bm25 compares against either half in isolation. Query logic lives
in kb_mcp.search (shared with the MCP server) - this is a thin CLI wrapper.

Usage:
    PYTHONPATH=src python scripts/eval_retrieval.py [--index kb-scratch] [--k 5] [--mode hybrid|knn|bm25]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kb_common import config
from kb_mcp import search

QUERIES = [
    "Argo float measuring dissolved oxygen in the Southern Ocean",
    "EMSO regional facility in the Mediterranean Sea",
    "moored surface buoy in the Black Sea",
    "profiling float operated by Ifremer",
    "organization involved in ocean observation in Portugal",
    "seafloor observatory monitoring cold-water coral habitats",
]


def run(index: str, k: int, mode: str):
    print(f"mode={mode}  index={index}  k={k}")
    for query in QUERIES:
        hits = search.search(index, query, k=k, mode=mode)
        print(f"\n=== {query!r} ===")
        for hit in hits:
            summary = hit["summary_text"] or ""
            preview = summary[:140] + ("..." if len(summary) > 140 else "")
            print(f"  [{hit['score']:.3f}] {hit['_id']}  {preview}")
            for fragment in hit.get("highlights") or []:
                print(f"      highlight: {fragment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.ES_INDEX)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "knn", "bm25"], default="hybrid")
    args = parser.parse_args()
    run(args.index, args.k, args.mode)
