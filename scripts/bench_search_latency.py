"""Benchmark kb_api's /search endpoint using its Server-Timing response
header (see kb_common.timing / LOG_TIMING) to break latency down by stage -
per query, and under concurrent load.

Usage:
    python scripts/bench_search_latency.py [--base-url http://localhost:8080]
        [--repeats 5] [--limit 10] [--concurrency 1,4,8] [--queries "a,b,c"]
"""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.parse
import urllib.request

DEFAULT_QUERIES = [
    "seismologist",
    "Argo float temperature sensor Mediterranean",
    "Ifremer",
    "dissolved oxygen sensor",
    "Black Sea profiling float",
    "Antares project principal investigator",
    "float 6902919",
    "ocean salinity measurement platform",
]

STAGES = ["es_client_init", "embed", "es_query", "es_took", "format", "api_total"]


def one_request(base_url, query, limit):
    url = f"{base_url}/search?{urllib.parse.urlencode({'query': query, 'limit': limit})}"
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read()
        server_timing = resp.headers.get("Server-Timing", "")
    wall_ms = (time.perf_counter() - start) * 1000

    stages = {}
    for part in server_timing.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, dur = part.partition(";dur=")
        stages[name.strip()] = float(dur)
    return wall_ms, stages, len(json.loads(body))


def percentile(values, pct):
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * pct) - 1)]


def bench_per_query(base_url, queries, repeats, limit):
    print(f"\n=== Per-query latency (limit={limit}, {repeats} runs each, first run excluded as cold) ===")
    headers = ["wall avg", "wall p95", *STAGES]
    widths = [max(len(h), 8) for h in headers]
    col = "{:<42}" + "".join(f" {{:>{w}}}" for w in widths)
    print(col.format("query", *headers))
    print("-" * (42 + sum(widths) + len(widths)))

    for query in queries:
        runs = [one_request(base_url, query, limit) for _ in range(repeats)]
        warm = runs[1:] if len(runs) > 1 else runs
        walls = [w for w, _, _ in warm]
        stage_avgs = [statistics.mean(s.get(name, 0.0) for _, s, _ in warm) for name in STAGES]
        label = query if len(query) <= 42 else query[:39] + "..."
        print(col.format(
            label,
            f"{statistics.mean(walls):.1f}",
            f"{percentile(walls, 0.95):.1f}",
            *(f"{v:.1f}" for v in stage_avgs),
        ))


def bench_concurrency(base_url, queries, limit, concurrency_levels):
    print(f"\n=== Concurrency (limit={limit}, 3 requests per worker) ===")
    col = "{:<12} {:>12} {:>16} {:>16} {:>16} {:>16}"
    print(col.format("workers", "batch wall", "req wall avg", "req wall p95", "embed avg", "api_total avg"))
    print("-" * 92)

    for concurrency in concurrency_levels:
        requests = (queries * ((concurrency * 3 // len(queries)) + 1))[: concurrency * 3]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda q: one_request(base_url, q, limit), requests))
        batch_wall = (time.perf_counter() - t0) * 1000

        walls = [w for w, _, _ in results]
        embeds = [s.get("embed", 0.0) for _, s, _ in results]
        api_totals = [s.get("api_total", 0.0) for _, s, _ in results]
        print(col.format(
            concurrency,
            f"{batch_wall:.0f}",
            f"{statistics.mean(walls):.1f}",
            f"{percentile(walls, 0.95):.1f}",
            f"{statistics.mean(embeds):.1f}",
            f"{statistics.mean(api_totals):.1f}",
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080", help="kb_api base URL")
    parser.add_argument("--repeats", type=int, default=5, help="Runs per query for the per-query table")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit (k)")
    parser.add_argument("--concurrency", default="1,4,8", help="Comma-separated worker counts to test")
    parser.add_argument("--queries", default=None, help="Comma-separated queries (default: a built-in mixed set)")
    parser.add_argument("--skip-concurrency", action="store_true", help="Only run the per-query table")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    queries = [q.strip() for q in args.queries.split(",")] if args.queries else DEFAULT_QUERIES
    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    bench_per_query(base_url, queries, args.repeats, args.limit)
    if not args.skip_concurrency:
        bench_concurrency(base_url, queries, args.limit, concurrency_levels)


if __name__ == "__main__":
    main()
