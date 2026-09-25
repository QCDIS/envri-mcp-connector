# Ifremer Knowledge Base

Euro-Argo floats + OSO ontology in one Elasticsearch index (`source: euro_argo` / `oso`), embedded for RAG, served via an HTTP API and an MCP server.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for package roles and data flow.

## Security

- Bearer token required on every route except `/health` - `KB_READ_TOKENS` / `KB_ADMIN_TOKENS` in `.env` (`name:token`, generate with `python -c "import secrets; print(secrets.token_hex(32))"`)
- `KB_ADMIN_TOKENS` additionally gates `/fetch` and `/index` (real ingestion jobs)
- `MCP_ALLOWED_HOSTS` restricts which `Host` header `kb-mcp` accepts
- `KB_SECURITY_ENABLED=false` disables all of the above at once - local testing only

## Setup

```bash
cp .env.example .env    # set ELASTIC_PASSWORD, tokens, etc.
```

GPU recommended (`EMBEDDING_DEVICE=cuda`); `cpu` works but it's slower. Python ≥ 3.12 for local installs.

### Docker

Runs everything: Elasticsearch (:9200), API (:8080), MCP (:8765). `kb-api` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.

```bash
docker compose up -d --build
```

### pip

Start Elasticsearch with `docker compose up -d elasticsearch` (or point `ES_URL` at your own cluster), then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[api]   # kb-mcp alone only needs `pip install -e .` (no torch/GPU deps)
python -m kb_api.main   # API on :8080
python -m kb_mcp.server # MCP on :8765
```

For a specific CUDA build of torch, install it first: `pip install torch --index-url https://download.pytorch.org/whl/cu124`.

### uv

Same as pip, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[api]   # or `uv pip install -e .` for kb-mcp only
python -m kb_api.main
python -m kb_mcp.server
```

For a specific CUDA build of torch: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

## Ingest

| Step | Euro-Argo | OSO |
| --- | --- | --- |
| Fetch | `python -m kb_argo.pipeline fetch` (cached in `data/cache/argo/raw/`) | `python -m kb_oso.pipeline fetch` (latest `OSO.owl` → `data/cache/oso/oso.owl`) |
| Index | `python -m kb_argo.pipeline index` | `python -m kb_oso.pipeline index` |

Flags: `--limit N` (smoke test), `--force` (re-fetch).

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
| `scripts/eval/eval_retrieval.py --index ifremer-knowledge-base` | Top hits for hand-picked queries |
| `scripts/eval/bench_relevance.py --k 10 --mode hybrid` | precision@k, recall@k, MRR |
| `scripts/eval/bench_hybrid_knn.py --k 10 --runs 10 --candidates 10,50,150,300` | Recall/latency vs. candidate pool |
| `scripts/eval/bench_search_latency.py --base-url http://localhost:8080` | Per-stage `/search` latency (via `Server-Timing`), per query and under concurrent load |

Operational tools live in `scripts/ops/` (e.g. `scripts/ops/reindex.py`).

## Tests

```bash
pip install -e .[api,dev]
pytest
```

Unit tests under `tests/`, mirroring `src/`'s package layout. Unlike the benchmarks above, these assert pass/fail and hit no live Elasticsearch/network/model - see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

[Apache License 2.0](LICENSE).
