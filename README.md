# Ifremer Knowledge Base

Euro-Argo floats + OSO ontology in one Elasticsearch index (`source: euro_argo` / `oso`), embedded for RAG, served via an HTTP API and an MCP server.

| Package | Role |
| --- | --- |
| `kb_common/` | Config, embedding, ES indexing, hybrid BM25+kNN search |
| `kb_argo/`, `kb_oso/` | Fetch → transform → embed → index, per source |
| `kb_api/` | HTTP API (only service needing `torch`/GPU at query time) |
| `kb_mcp/` | MCP tools; calls `kb_api` for search |

## Setup

```bash
cp .env.example .env
sudo sysctl -w vm.max_map_count=262144   # Linux only
docker compose up -d                     # Elasticsearch on :9200, API on :8080, MCP on :8765
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended (`EMBEDDING_DEVICE=cuda`); `cpu` works but it's slower.

## Ingest

| Step | Euro-Argo | OSO |
| --- | --- | --- |
| Fetch | `python -m kb_argo.pipeline fetch` (cached in `data/argo/raw/`) | `python -m kb_oso.pipeline fetch` (latest `OSO.owl` → `data/oso/oso.owl`) |
| Index | `python -m kb_argo.pipeline index` | `python -m kb_oso.pipeline index` |

Prefix with `PYTHONPATH=src`. Flags: `--limit N` (smoke test), `--force` (re-fetch).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /search?query=…&limit=…&source=…` | Public hybrid search |
| `POST /fetch?source=…` | Background fetch (`limit`, `force`) |
| `POST /index?source=…` | Background embed + index (`limit`) |
| `GET /stats` | Document counts per source |
| `GET /health` | Liveness |
| `POST /internal/search` | `kb_mcp` only |

`source` is `euro_argo` or `oso`. Start with `docker compose up -d kb-api` (:8080) or `kb-mcp` (:8765).

## MCP tools

| Tool | Purpose |
| --- | --- |
| `search_knowledge_base` | Hybrid semantic + keyword search (calls `kb_api`) |
| `get_argo_float`, `get_oso_entity` | Full record by WMO / OSO id |
| `find_argo_floats_near`, `find_argo_floats_in_box` | Floats by last known position |
| `list_argo_floats_by_sea` / `_by_ocean` / `_by_sensor` / `_by_organization` | Floats by region, sensor or operator |
| `list_oso_entities_by_type` | OSO entities of a given type |
| `list_seas`, `list_ocean_regions`, `list_oso_entity_types`, `list_field_values` | Discover valid filter values |
| `get_index_stats` | Document counts per source |

## Benchmarks

| Script | Measures |
| --- | --- |
| `scripts/eval_retrieval.py --index ifremer-knowledge-base` | Top hits for hand-picked queries |
| `scripts/bench_relevance.py --k 10 --mode hybrid` | precision@k, recall@k, MRR |
| `scripts/bench_hybrid_knn.py --k 10 --runs 10 --candidates 10,50,150,300` | Recall/latency vs. candidate pool |
