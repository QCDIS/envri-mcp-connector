# Ifremer Knowledge Base

Ingests Euro-Argo float metadata and the OSO ontology into one shared
Elasticsearch index (`source: euro_argo` / `oso`), embedded for RAG.
Ships an HTTP API and an MCP server for querying it.

## Architecture

| Package | Purpose |
| --- | --- |
| [`kb_common/`](src/kb_common) | Config, embedding, ES client/indexing, hybrid BM25+kNN search |
| [`kb_argo/`](src/kb_argo) | Fetch → transform → embed → index Euro-Argo floats |
| [`kb_oso/`](src/kb_oso) | Parse `.owl` → transform → embed → index OSO entities |
| [`kb_api/`](src/kb_api) | HTTP `/search` — only service needing `torch`/GPU at query time |
| [`kb_mcp/`](src/kb_mcp) | MCP tools, calls `kb_api` for the embedding-dependent one |

Docs are keyed `source:id` (`euro_argo:1900045`, `oso:ANTARES`) — re-ingest upserts.

## Prerequisites

- Docker + Docker Compose
- Python 3.10+
- Linux: `vm.max_map_count` raised (below)
- GPU recommended (`EMBEDDING_DEVICE=cuda`), CPU works too (`=cpu`, slower)

## Install

```bash
cp .env.example .env
sudo sysctl -w vm.max_map_count=262144   # Linux only
docker compose up -d                     # Elasticsearch on :9200

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full stack
# kb_mcp standalone only: pip install -r requirements-mcp.txt
```

## Adding data

```bash
# Euro-Argo (~21k floats)
PYTHONPATH=src python -m kb_argo.pipeline fetch   # cached to data/argo/raw/, resumable
PYTHONPATH=src python -m kb_argo.pipeline index

# OSO — place ontology at data/oso/oso.owl (or set OSO_OWL_PATH)
PYTHONPATH=src python -m kb_oso.pipeline index
```

`--limit N` on any subcommand for a smoke test.

**Cross-source linking**: `kb_argo/link_oso.py` matches an Argo float's
owner/data_center/institution to an OSO `Organization` (label matching) and
stores `oso_organization_id`.

```bash
PYTHONPATH=src python -m kb_argo.link_oso --report
```

## Querying

### MCP server

Needs `kb_api` reachable (`KB_API_URL`, default `localhost:8080`) — only
`search_knowledge_base` calls it (over HTTP, no embedding model loaded
locally); everything else hits Elasticsearch directly.

```bash
docker compose up -d kb-mcp   # brings up elasticsearch + kb-api too
```

Client → `http://localhost:8765`. Tools: `search_knowledge_base`,
`get_argo_float` / `get_oso_entity`, `find_argo_floats_near` /
`find_argo_floats_in_box`, `list_argo_floats_by_region` / `_by_sensor` /
`_by_organization`, `list_oso_entities_by_type`, discovery tools
(`list_ocean_regions` / `list_oso_entity_types` / `list_field_values`),
`get_index_stats`.

Direct run: `PYTHONPATH=src python -m kb_mcp.server` (needs `kb_api` running
separately). Geo tools need `last_cycle_geopoint` — reindex older indices
with `scripts/reindex.py`.

### HTTP API

```bash
docker compose up -d kb-api
curl "http://localhost:8080/search?query=profiling+float+operated+by+Ifremer&limit=5"
```

Result fields: `score`, `url`, `header`, `vector`, `summary`, `highlight`,
`last_modified`. Optional `source=euro_argo|oso` filter.

Direct run: `PYTHONPATH=src python -m kb_api.main`.

`POST /internal/search` is the unshaped internal endpoint `kb_mcp` uses —
not the public contract above.

## Benchmarking & evaluation

Needs a populated index (`ES_INDEX`). `bench_relevance.py`/`bench_hybrid_knn.py`
query Elasticsearch directly; `eval_retrieval.py` goes through `kb_mcp`/`kb_api`.

```bash
# Eyeball top hits for hand-picked queries
PYTHONPATH=src python scripts/eval_retrieval.py --index ifremer-knowledge-base

# precision@k / recall@k / MRR over ~50 ground-truth queries
PYTHONPATH=src python scripts/bench_relevance.py --k 10 --mode hybrid   # or knn | bm25

# Recall/latency vs. KNN_HYBRID_CANDIDATES pool size
PYTHONPATH=src python scripts/bench_hybrid_knn.py --k 10 --runs 10 --candidates 10,50,150,300
```
