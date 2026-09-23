# Architecture

## Packages

| Package | Role |
| --- | --- |
| `kb_common/` | Config, embedding, ES indexing, hybrid BM25+kNN search |
| `kb_argo/`, `kb_oso/` | Fetch → transform → embed → index, per source |
| `kb_api/` | HTTP API (only service needing `torch`/GPU at query time) |
| `kb_mcp/` | MCP tools; calls `kb_api` for search |

`kb_argo` and `kb_oso` are structured identically — each has its own `config.py`, `fetch.py`, `transform.py`, `es_mapping.py`, `pipeline.py` (the `fetch` / `index` CLI entry points described in the README's Ingest table). The filenames repeat across the two packages on purpose: each pair is package-scoped (Euro-Argo vs. OSO), not a shared/duplicated module — there's no cross-imports between `kb_argo` and `kb_oso`. `kb_common` is the one package they both depend on, for anything source-agnostic (embedding, ES client/indexing, hybrid search, shared auth/rate-limiting).

`kb_api` (FastAPI, needs `torch`) and `kb_mcp` (MCP tool server, calls `kb_api` over HTTP rather than embedding/searching itself) are deployed as separate Docker services so only `kb_api` needs GPU/`torch` — see `pyproject.toml`'s `api` extra and `Dockerfile.api` / `Dockerfile.mcp`.

## Data layout

```
data/
  seas/World_Seas_IHO_v3.simplified.pkl   # tracked — no regeneration path in this repo
  reference/
    sea_to_ocean.json                      # tracked — small, hand-maintained lookup
    nerc_vocab_cache.json
  cache/                                   # gitignored — regenerated locally, never committed
    oso/oso.owl
    argo/raw/
```

- **`data/seas/`** — tracked and stays in place. `World_Seas_IHO_v3.simplified.pkl` is a simplified/pickled index built from the IHO World Seas v3 shapefile (marineregions.org).
- **`data/reference/`** — tracked. `sea_to_ocean.json` is a small, hand-maintained lookup with no fetch/build step. `nerc_vocab_cache.json` is committed rather than gitignored because regenerating it needs the NVS turtles.zip export manually downloaded from vocab.nerc.ac.uk first (`PYTHONPATH=src python -m kb_common.nerc_vocab --zip <turtles.zip>` — see `src/kb_common/nerc_vocab.py`); it's not a self-contained fetch, so it's kept in git as a working snapshot.
- **`data/cache/`** — gitignored. Everything here is reproducible from a documented command and is never committed:
  - `oso/oso.owl` — `PYTHONPATH=src python -m kb_oso.pipeline fetch` (downloads the latest OSO release asset).
  - `argo/raw/` — populated by `PYTHONPATH=src python -m kb_argo.pipeline fetch`; the largest artifact in the repo by far (multi-GB of per-float JSON).

## Scripts and tests

- `scripts/ops/` — operational tools run against a live deployment (e.g. `reindex.py`, for migrating to a new index mapping without recomputing embeddings).
- `scripts/eval/` — dev-quality tools: retrieval benchmarks and a manual eval script (see the README's Benchmarks table). These print results for a human to eyeball, not assertions, and need a live Elasticsearch/API to run against.
- `tests/` — the pytest suite, mirroring `src/`'s package layout. Unlike `scripts/eval/`, these assert pass/fail and are hermetic: no live Elasticsearch, network, or embedding model. Run with `pytest` from repo root (needs `pip install -e .[api,dev]`). There is no CI configured to run them automatically yet.
