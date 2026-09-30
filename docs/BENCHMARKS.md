# Benchmarks

All scripts live in `scripts/benchmarks`, need a live Elasticsearch (and, for the latency benchmark, a running `kb_api`) and print results, **none of them assert pass/fail.**

## Setup the reference numbers were measured on

| | |
| --- | --- |
| Embedding model | `Qwen3-Embedding-0.6B`, 1024 dimensions, GPU |
| Ingestion batch size | 64 (`EMBEDDING_BATCH_SIZE`) |
| Search | One request combining BM25 and kNN, scores summed (`kb_common/hybrid_search.py`) |
| kNN candidate pool | 150 (`KNN_HYBRID_CANDIDATES`) |
| Query batching | Up to 32 queries within an 8 ms window (`EMBED_QUERY_BATCH_MAX_SIZE`, `EMBED_QUERY_BATCH_WINDOW_MS`) |

| Benchmark | Script | Question it answers |
| --- | --- | --- |
| [Search methodology](#1-search-methodology-hybrid-vs-knn-vs-bm25) | `bench_retrieval.py` | Is hybrid search better than either half alone? |
| [Relevance](#2-relevance-precision-recall-mrr) | `bench_relevance.py` | How often are the right floats in the top *k*? |
| [Candidate pool](#3-knn-candidate-pool-size) | `bench_hybrid_knn.py` | Does a bigger kNN pool improve recall, and at what cost? |
| [Latency](#4-latency-and-query-batching) | `bench_search_latency.py` | How long does `/search` take, and where does the time go? |

---

## 1. Search methodology: hybrid vs. kNN vs. BM25

**What it is.** A qualitative check. `bench_retrieval.py` runs hand-picked natural-language queries and prints the top hits (score, document id, summary preview, highlighted fragments). Run it once per mode to compare:

```bash
PYTHONPATH=src python scripts/benchmarks/bench_retrieval.py --index ifremer-knowledge-base --k 5 --mode hybrid
PYTHONPATH=src python scripts/benchmarks/bench_retrieval.py --index ifremer-knowledge-base --k 5 --mode knn
PYTHONPATH=src python scripts/benchmarks/bench_retrieval.py --index ifremer-knowledge-base --k 5 --mode bm25
```

**The three modes.**

- **BM25** matches exact terms in the text. It's strong on names and identifiers ("Ifremer", "Black Sea") and weak on synonyms.
- **kNN** compares the meaning of the query to each document's 1024-dimension embedding (cosine similarity). It's strong on paraphrase and weak on precise entities such as geometry or proper names.
- **Hybrid** (adopted) runs both in one Elasticsearch request and sums the scores.

---

## 2. Relevance: precision, recall, MRR

**What it is.** The quantitative version of section 1. `bench_relevance.py` runs 50 queries, each with an **exact ground truth** built from a `.keyword` filter (for example, "Argo float in the Labrador Sea" -> every document with `sea_area.keyword = "Labrador Sea"`), and scores the top *k* results against it.

```bash
PYTHONPATH=src python scripts/benchmarks/bench_relevance.py --k 10 --mode hybrid
```

| Category | Queries | Ground-truth field |
| --- | --- | --- |
| `sea` | 15 | `sea_area.keyword` |
| `ocean` | 5 | `ocean_region.keyword` |
| `data_center` | 6 | `data_center_name.keyword` |
| `project` | 6 | `project_name.keyword` |
| `sensor` | 6 | `sensor_codes` |
| `combined` | 12 | Two conditions ANDed (e.g. project **and** sea) |

**Metrics, per query, at cut-off *k*:**

| Metric | Formula | Reads as |
| --- | --- | --- |
| precision@k | relevant docs in top *k* ÷ *k* | How clean the top *k* is |
| recall@k | relevant docs in top *k* ÷ min(*k*, number of ground-truth docs) | How much of what *could* be returned was returned |
| RR (reciprocal rank) | 1 ÷ rank of the first relevant doc (0 if none) | How quickly the first right answer appears |

Recall divides by min(*k*, ground-truth size), so a query with 4,000 matching floats and *k* = 10 can still reach 1.0: it only has to fill the ten slots. Averaged over queries, RR is **MRR**.

---

## 3. kNN candidate pool size

**What it is.** In hybrid mode, kNN considers a pool of candidates before its scores are merged with BM25's. A larger pool can surface documents that BM25 ranks highly but that sit outside kNN's nearest few; it also costs time. The deployed value is **150** (`KNN_HYBRID_CANDIDATES`). `bench_hybrid_knn.py` sweeps it:

```bash
PYTHONPATH=src python scripts/benchmarks/bench_hybrid_knn.py --k 10 --runs 10 --candidates 10,50,150,300
```

For each pool size and each of four queries (Mediterranean, Mediterranean + dissolved oxygen, Labrador Sea, Ifremer-operated), it prints recall@k against an exact ground truth plus mean, median and p95 latency over `--runs` repeats.

---

## 4. Latency and query batching

**What it is.** `bench_search_latency.py` calls `GET /search` and splits each request's time into stages using the `Server-Timing` header that `kb_api` adds (see `LOG_TIMING` env var to turn it on).

```bash
python scripts/benchmarks/bench_search_latency.py --base-url http://localhost:8080 --repeats 5 --limit 10 --concurrency 1,4,8
```

It produces two tables:

1. **Per query**: average and p95 wall time, plus each stage, for a fixed mix of queries (`"Ifremer"`, `"dissolved oxygen sensor"`, `"Black Sea profiling float"`, …). The first run of each query is dropped as a cold start.
2. **Concurrency**: *N* worker threads each send 3 requests at once (`--concurrency 1,4,8`), and the script reports batch wall time, per-request wall average and p95, and average embed and total server time.

**Stages** (the `Server-Timing` names):

| Stage | Covers |
| --- | --- |
| `es_client_init` | Getting the Elasticsearch client |
| `embed` | Turning the query into a 1024-dimension vector on the GPU (includes waiting for its batch) |
| `es_query` / `es_took` | The Elasticsearch request as timed by the API / by Elasticsearch itself |
| `format` | Shaping hits into the response |
| `api_total` | The whole handler, first byte in to last byte out |

- **Embed query (GPU)** = `embed`.
- **ES query + rest of server** = `api_total` − `embed`.
- **Outside the handler** = request real time − `api_total`: network, connection handling and queueing before the handler runs or after it returns.

**Why batching matters.** Encoding one query per call serializes the GPU behind per-call Python and tokenizer overhead, so concurrent requests wait in line. Instead, a background worker collects queries that arrive within an **8 ms window** (up to **32**) and encodes them in a single GPU call. Tune with:

| Variable | Default | Effect |
| --- | --- | --- |
| `EMBED_QUERY_BATCH_WINDOW_MS` | 8 | Longer window → bigger batches, but every query waits up to that long |
| `EMBED_QUERY_BATCH_MAX_SIZE` | 32 | Cap on queries per GPU call |

A single query pays at most the window (8 ms) in exchange for stable embedding cost under concurrency.

| Metric | Formula / Scope | Reads as |
| --- | --- | --- |
| `batch wall` | Total elapsed time for all workers | Total wall-clock time (ms) to run the entire batch |
| `req wall avg` | Sum of request wall times ÷ total requests | Average end-to-end request latency (ms) from client side |
| `req wall p95` | 95th percentile of individual request wall times | Tail latency (ms) experienced by the slowest 5% of requests |

---

## Running everything

| Goal | Command |
| --- | --- |
| Eyeball retrieval quality | `bench_retrieval.py --index ifremer-knowledge-base --k 5 --mode hybrid` |
| Score relevance | `bench_relevance.py --k 10 --mode hybrid` |
| Tune the candidate pool | `bench_hybrid_knn.py --k 10 --runs 10 --candidates 10,50,150,300` |
| Measure latency | `bench_search_latency.py --base-url http://localhost:8080` |
