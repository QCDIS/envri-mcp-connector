"""Relevance benchmark: ~50 queries, each backed by an exact-match ground
truth (a .keyword filter), scored with precision@k / recall@k / MRR instead
of eyeballing.

Metrics per query:
  precision@k = (relevant docs in top k) / k
  recall@k    = (relevant docs in top k) / min(k, |ground truth|)
  RR          = 1 / rank of first relevant doc in top k (0 if none found)

Usage:
    PYTHONPATH=src python scripts/eval/bench_relevance.py [--k 10] [--mode hybrid|knn|bm25]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from kb_common import config, es_index, hybrid_search

# Each case: (category, query, [(field, [values]), ...] ground-truth conditions ANDed together)
CASES = [
    # --- single sea ---
    ("sea", "Argo float in the North Pacific Ocean", [("sea_area.keyword", ["North Pacific Ocean"])]),
    ("sea", "Argo float in the South Atlantic Ocean", [("sea_area.keyword", ["South Atlantic Ocean"])]),
    ("sea", "Argo float in the Southern Ocean", [("sea_area.keyword", ["Southern Ocean"])]),
    ("sea", "Argo float in the Philippine Sea", [("sea_area.keyword", ["Philippine Sea"])]),
    ("sea", "Argo float in the Arabian Sea", [("sea_area.keyword", ["Arabian Sea"])]),
    ("sea", "Argo float in the Japan Sea", [("sea_area.keyword", ["Japan Sea"])]),
    ("sea", "Argo float in the Mediterranean sea",
     [("sea_area.keyword", ["Mediterranean Sea - Eastern Basin", "Mediterranean Sea - Western Basin"])]),
    ("sea", "Argo float in the Labrador sea", [("sea_area.keyword", ["Labrador Sea"])]),
    ("sea", "Argo float in the Gulf of Mexico", [("sea_area.keyword", ["Gulf of Mexico"])]),
    ("sea", "Argo float in the Arctic Ocean", [("sea_area.keyword", ["Arctic Ocean"])]),
    ("sea", "Argo float in the Black Sea", [("sea_area.keyword", ["Black Sea"])]),
    ("sea", "Argo float in the Ionian Sea", [("sea_area.keyword", ["Ionian Sea"])]),
    ("sea", "Argo float in the Adriatic Sea", [("sea_area.keyword", ["Adriatic Sea"])]),
    ("sea", "Argo float in the Aegean Sea", [("sea_area.keyword", ["Aegean Sea"])]),
    ("sea", "Argo float in the Red Sea", [("sea_area.keyword", ["Red Sea"])]),
    # --- broad ocean basin ---
    ("ocean", "Argo float in the Pacific Ocean", [("ocean_region.keyword", ["Pacific Ocean"])]),
    ("ocean", "Argo float in the Atlantic Ocean", [("ocean_region.keyword", ["Atlantic Ocean"])]),
    ("ocean", "Argo float in the Indian Ocean", [("ocean_region.keyword", ["Indian Ocean"])]),
    ("ocean", "Argo float in the Southern Ocean basin", [("ocean_region.keyword", ["Southern Ocean"])]),
    ("ocean", "Argo float in the Arctic Ocean basin", [("ocean_region.keyword", ["Arctic Ocean"])]),
    # --- data center ---
    ("data_center", "Argo float processed by AOML data center", [("data_center_name.keyword", ["AOML"])]),
    ("data_center", "Argo float processed by CORIOLIS data center", [("data_center_name.keyword", ["CORIOLIS"])]),
    ("data_center", "Argo float processed by JMA data center", [("data_center_name.keyword", ["JMA"])]),
    ("data_center", "Argo float processed by CSIRO data center", [("data_center_name.keyword", ["CSIRO"])]),
    ("data_center", "Argo float processed by MEDS data center", [("data_center_name.keyword", ["MEDS"])]),
    ("data_center", "Argo float processed by INCOIS data center", [("data_center_name.keyword", ["INCOIS"])]),
    # --- project ---
    ("project", "Argo float under the US ARGO PROJECT", [("project_name.keyword", ["US ARGO PROJECT"])]),
    ("project", "Argo float under the Argo Australia project", [("project_name.keyword", ["Argo Australia"])]),
    ("project", "Argo float under the Argo UK project", [("project_name.keyword", ["Argo UK"])]),
    ("project", "Argo float under the Argo Canada project", [("project_name.keyword", ["Argo Canada"])]),
    ("project", "Argo float under the Argo INDIA project", [("project_name.keyword", ["Argo INDIA"])]),
    ("project", "Argo float under the GO-BGC project", [("project_name.keyword", ["GO-BGC"])]),
    # --- sensor ---
    ("sensor", "Argo float equipped with a dissolved oxygen optode sensor",
     [("sensor_codes", ["OPTODE_DOXY"])]),
    ("sensor", "Argo float equipped with a chlorophyll fluorometer sensor",
     [("sensor_codes", ["FLUOROMETER_CHLA"])]),
    ("sensor", "Argo float equipped with a backscattering sensor at 700 nanometers",
     [("sensor_codes", ["BACKSCATTERINGMETER_BBP700"])]),
    ("sensor", "Argo float equipped with a nitrate spectrophotometer sensor",
     [("sensor_codes", ["SPECTROPHOTOMETER_NITRATE"])]),
    ("sensor", "Argo float equipped with a pH sensor",
     [("sensor_codes", ["TRANSISTOR_PH"])]),
    ("sensor", "Argo float equipped with a PAR radiometer sensor",
     [("sensor_codes", ["RADIOMETER_PAR"])]),
    # --- combined (location + org/project/data-center) ---
    ("combined", "Argo float under the Argo Canada project in the Labrador sea",
     [("project_name.keyword", ["Argo Canada"]), ("sea_area.keyword", ["Labrador Sea"])]),
    ("combined", "Argo float processed by CORIOLIS in the Mediterranean sea",
     [("data_center_name.keyword", ["CORIOLIS"]),
      ("sea_area.keyword", ["Mediterranean Sea - Eastern Basin", "Mediterranean Sea - Western Basin"])]),
    ("combined", "Argo float under the US ARGO PROJECT in the North Pacific Ocean",
     [("project_name.keyword", ["US ARGO PROJECT"]), ("sea_area.keyword", ["North Pacific Ocean"])]),
    ("combined", "Argo float under the GO-BGC project in the Southern Ocean",
     [("project_name.keyword", ["GO-BGC"]), ("sea_area.keyword", ["Southern Ocean"])]),
    ("combined", "Argo float processed by JMA in the Japan Sea",
     [("data_center_name.keyword", ["JMA"]), ("sea_area.keyword", ["Japan Sea"])]),
    ("combined", "Argo float processed by AOML in the North Pacific Ocean",
     [("data_center_name.keyword", ["AOML"]), ("sea_area.keyword", ["North Pacific Ocean"])]),
    ("combined", "Argo float under the Argo Australia project in the Southern Ocean",
     [("project_name.keyword", ["Argo Australia"]), ("sea_area.keyword", ["Southern Ocean"])]),
    ("combined", "Argo float processed by CSIRO in the Indian Ocean",
     [("data_center_name.keyword", ["CSIRO"]), ("ocean_region.keyword", ["Indian Ocean"])]),
    ("combined", "Argo float under the Argo UK project in the North Atlantic Ocean",
     [("project_name.keyword", ["Argo UK"]), ("sea_area.keyword", ["North Atlantic Ocean"])]),
    ("combined", "Argo float processed by INCOIS in the Arabian Sea",
     [("data_center_name.keyword", ["INCOIS"]), ("sea_area.keyword", ["Arabian Sea"])]),
    ("combined", "Argo float under the Argo INDIA project in the Bay of Bengal",
     [("project_name.keyword", ["Argo INDIA"]), ("sea_area.keyword", ["Bay of Bengal"])]),
    ("combined", "Argo float processed by MEDS in the Labrador sea",
     [("data_center_name.keyword", ["MEDS"]), ("sea_area.keyword", ["Labrador Sea"])]),
]


GROUND_TRUTH_FETCH_SIZE = 10000  # ES's default index.max_result_window


def ground_truth_ids(client, index, conditions):
    must = [{"terms": {field: values}} for field, values in conditions]
    resp = client.search(
        index=index,
        query={"bool": {"must": must}},
        size=GROUND_TRUTH_FETCH_SIZE,
        source=False,
        track_total_hits=True,
    )
    total = resp["hits"]["total"]["value"]
    ids = {hit["_id"] for hit in resp["hits"]["hits"]}
    if total > len(ids):
        print(f"    WARNING: ground truth truncated ({len(ids)} of {total} real matches fetched) - raise GROUND_TRUTH_FETCH_SIZE")
    return ids


def evaluate(hit_ids, truth_ids, k):
    if not truth_ids:
        return None
    top_k = hit_ids[:k]
    relevant_in_top_k = [1 if h in truth_ids else 0 for h in top_k]
    hits = sum(relevant_in_top_k)
    precision = hits / k
    recall = hits / min(k, len(truth_ids))
    rr = 0.0
    for i, rel in enumerate(relevant_in_top_k, start=1):
        if rel:
            rr = 1.0 / i
            break
    return {"precision": precision, "recall": recall, "rr": rr, "hits": hits, "truth_count": len(truth_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.ES_INDEX)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=["hybrid", "knn", "bm25"], default="hybrid")
    args = parser.parse_args()

    client = es_index.get_client()

    by_category = {}
    latencies = []
    skipped = 0

    print(f"mode={args.mode}  index={args.index}  k={args.k}  cases={len(CASES)}\n")

    for category, query, conditions in CASES:
        truth_ids = ground_truth_ids(client, args.index, conditions)

        t0 = time.perf_counter()
        hits = hybrid_search.search(args.index, query, k=args.k, mode=args.mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        hit_ids = [h["_id"] for h in hits]

        result = evaluate(hit_ids, truth_ids, args.k)
        if result is None:
            print(f"[{category:9s}] {query!r} -> SKIPPED (0 ground truth docs, filters don't intersect)")
            skipped += 1
            continue

        by_category.setdefault(category, []).append(result)
        print(
            f"[{category:9s}] {query!r}\n"
            f"    precision@{args.k}={result['precision']:.2f}  recall@{args.k}={result['recall']:.2f}  "
            f"RR={result['rr']:.2f}  ({result['hits']}/{args.k} relevant, {result['truth_count']} truth docs)"
        )

    print("\n=== Per-category averages ===")
    all_results = []
    for category, results in by_category.items():
        all_results.extend(results)
        p = statistics.mean(r["precision"] for r in results)
        r_ = statistics.mean(r["recall"] for r in results)
        mrr = statistics.mean(r["rr"] for r in results)
        print(f"  {category:9s} (n={len(results):2d})  precision@{args.k}={p:.2f}  recall@{args.k}={r_:.2f}  MRR={mrr:.2f}")

    print("\n=== Overall ===")
    if all_results:
        p = statistics.mean(r["precision"] for r in all_results)
        r_ = statistics.mean(r["recall"] for r in all_results)
        mrr = statistics.mean(r["rr"] for r in all_results)
        zero_hit = sum(1 for r in all_results if r["hits"] == 0)
        print(f"  scored={len(all_results)}  skipped={skipped}")
        print(f"  precision@{args.k}={p:.2f}  recall@{args.k}={r_:.2f}  MRR={mrr:.2f}")
        print(f"  queries with ZERO relevant hits in top {args.k}: {zero_hit}/{len(all_results)}")
    print(f"  mean query latency: {statistics.mean(latencies):.0f}ms  median: {statistics.median(latencies):.0f}ms")


if __name__ == "__main__":
    main()
