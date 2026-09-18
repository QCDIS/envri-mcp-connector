# Ifremer Knowledge Base

Ingests Ifremer resources into Elasticsearch as embedded documents for use as
a retrieval knowledge base. Covers Euro-Argo float metadata and the OSO
ontology (Ocean Science Observation), each as its own `source` in one shared
index so both are searchable together.

## Setup

```bash
cp .env.example .env   # adjust ELASTIC_PASSWORD / ES_URL / credentials as needed

# Linux only: Elasticsearch needs this bumped on the host
sudo sysctl -w vm.max_map_count=262144

docker compose up -d   # starts a local single-node Elasticsearch on :9200

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # full stack: ingest pipelines, kb_api, kb_mcp
# Only running kb_mcp standalone (not via Docker)? It doesn't need the
# embedding stack at all - pip install -r requirements-mcp.txt is enough.
```

## Usage

### Euro-Argo

```bash
# 1. Fetch all ~21k float records from the Euro-Argo Fleet Monitoring API
#    (cached to data/argo/raw/, safe to re-run/resume)
PYTHONPATH=src python -m kb_argo.pipeline fetch

# 2. Transform to text summaries, embed locally (GPU), index into Elasticsearch
PYTHONPATH=src python -m kb_argo.pipeline index
```

### OSO

```bash
# Place the ontology at data/oso/oso.owl (or set OSO_OWL_PATH), then:
PYTHONPATH=src python -m kb_oso.pipeline index
```

Use `--limit N` on any subcommand for a quick smoke test before running the
full ingest.

### NERC vocabulary resolution (one-off, local only)

Both sources reference external NERC NVS vocabulary codes (Argo sensor `id`s
like `CTD_TEMP`; OSO's `hasPlatformCategory` etc.). If you have the NVS
Turtle export locally (`NERC_TURTLES_ZIP` in `.env`), build the resolution
cache once:

```bash
PYTHONPATH=src python -m kb_common.nerc_vocab --zip "$NERC_TURTLES_ZIP"
```

This extracts only the handful of collections actually referenced (not the
whole archive) and writes `data/nerc_vocab_cache.json` — a small file that
should travel with the rest of `data/` to wherever ingestion runs. Both
`transform.py` modules degrade gracefully (falling back to raw codes) if this
cache is absent, so it's optional but recommended.

### Sea/ocean classification (one-off, local, but the cache travels)

`kb_common/seas.py` tags each float's last-known position with a real named
sea/ocean (101 areas from the IHO World Seas v3 dataset — e.g. "Black Sea",
"Mediterranean Sea - Western Basin" — not just 5 broad ocean buckets).
Download the shapefile from
[marineregions.org](https://www.marineregions.org/downloads.php) (the full
"IHO Sea Areas" product, not the coarser "Global Oceans and Seas" one — it
should have ~101 records, check with `ogrinfo` or a quick `pyshp` read if
unsure) and place it at `data/seas/World_Seas_IHO_v3.shp` (plus its `.dbf`/
`.shx`/`.prj`/`.cpg`/`.qpj` siblings), or point `SEAS_SHAPEFILE_PATH` at it.

The first call simplifies the (very high-resolution, ~150MB) source
geometries and caches them to a small `.simplified.pkl` file next to the
shapefile — slow once (a few minutes, dominated by parsing the source file,
not the simplification), near-instant on every call after. That `.pkl` is
what needs to travel with the rest of `data/` to wherever ingestion runs; the
original `.shp` doesn't need to. `classify_sea()` degrades gracefully
(returns `None`) if the shapefile/cache isn't present at all.

### Cross-source organization linking

`kb_argo/link_oso.py` matches an Argo float's `owner`/`data_center_name`/
`institution` against OSO `Organization` individuals (conservative exact/
substring matching on normalized labels — e.g. Argo's `IFREMER` owner links to
OSO's `Ifremer` organization) and stores the result as `oso_organization_id`.
Review matches before trusting them at scale:

```bash
PYTHONPATH=src python -m kb_argo.link_oso --report
```

### Retrieval sanity check

`scripts/eval_retrieval.py` runs a handful of hand-picked natural-language
queries through kNN search and prints the top hits — a manual dev tool for
eyeballing whether an enrichment pass actually improved retrieval, not an
automated test suite:

```bash
PYTHONPATH=src python scripts/eval_retrieval.py --index ifremer-knowledge-base
```

### MCP server

`kb_mcp/` exposes the knowledge base as MCP tools over streamable-http, so an
AI (a chatbot, Claude, or any MCP-capable client) can search and browse it
directly instead of needing to know Elasticsearch query DSL.

**`kb_mcp` needs `kb_api` running and reachable** (`KB_API_URL`, default
`http://localhost:8080`) - its one embedding-dependent tool,
`search_knowledge_base`, calls `kb_api`'s `POST /internal/search` over HTTP
rather than loading the embedding model itself. This is why `kb_mcp` has its
own lightweight `requirements-mcp.txt`/`Dockerfile.mcp` with no
`torch`/`sentence-transformers` and no GPU reservation in `docker-compose.yml`
- every other tool (`get_argo_float`, `list_argo_floats_by_sensor`, the geo/
discovery tools) talks to Elasticsearch directly and never needed embeddings.

```bash
docker compose up -d kb-mcp   # brings up elasticsearch + kb-api too, via depends_on
```

Point an MCP client at `http://localhost:8765` (or wherever it's :ed).
Tools: `search_knowledge_base` (the primary hybrid search tool - each result
includes `highlights`, the matched fragment(s) of `summary_text`),
`get_argo_float` / `get_oso_entity` (fetch by id), `find_argo_floats_near` /
`find_argo_floats_in_box` (geo search on last known position),
`list_argo_floats_by_region` / `_by_sensor` / `_by_organization` and
`list_oso_entities_by_type` (structured filters), `list_ocean_regions` /
`list_oso_entity_types` / `list_field_values` (discovery — what values exist,
before filtering), and `get_index_stats`.

Run it directly instead of via Docker with
`PYTHONPATH=src python -m kb_mcp.server` (needs `kb_api` running separately -
`PYTHONPATH=src python -m kb_api.main` - and `KB_API_URL` pointing at it if
not the default `localhost:8080`; this also applies to
`scripts/eval_retrieval.py`, which goes through `kb_mcp.search`). Geo search
needs the `last_cycle_geopoint` field, added after the original mapping — if
your index predates it, migrate with `scripts/reindex.py` (same pattern as
the earlier `owner`/`data_center_name` mapping fix, see git history).

### HTTP API

`kb_api/` exposes the same hybrid search as a plain JSON HTTP endpoint for
non-MCP consumers - unlike the MCP tools, this one includes the raw embedding
vector per result, for a downstream system that wants to do its own re-ranking
or vector math rather than an LLM consuming the results as text.

```bash
docker compose up -d kb-api
curl "http://localhost:8080/search?query=profiling+float+operated+by+Ifremer&limit=5"
```

Each result: `score`, `url` (a real link to the authoritative source - the
Euro-Argo Fleet Monitoring page for floats, the entity's own ontology URI for
OSO), `header` (a short title), `vector` (the embedding), `summary`
(`summary_text`), `highlight` (matched fragment(s) of `summary`, empty if
nothing on the lexical side matched), and `last_modified` (`indexed_at`, set
automatically by `kb_common.es_index.bulk_index` on every index/reindex -
`null` for documents indexed before this field existed, until they're
re-ingested). Optional `source` query param (`euro_argo` or `oso`) restricts
to one source.

Run it directly instead of via Docker with `PYTHONPATH=src python -m kb_api.main`.

`kb_api` also exposes `POST /internal/search` - a thin, unshaped wrapper
around `kb_common.hybrid_search.search()` (raw `_id`/`score`/requested
`fields`/`highlights`, no reshaping). This is **not** the public contract
above; it exists so `kb_mcp` can search without loading the embedding model
itself (see the MCP server section). `kb_api` is the only service left that
needs `torch`/`sentence-transformers`/a GPU.

## How it works

`kb_common/` holds what both sources share: `config.py` (Elasticsearch +
embedding settings), `embed.py` (local embeddings via `sentence-transformers`,
`Qwen/Qwen3-Embedding-0.6B` by default — multilingual since OSO content
is in French, English, Portuguese, etc.), `es_index.py` (client + bulk
indexing; every document gets a base `source` + `summary_text` +
`embedding` + `indexed_at`, with each source layering its own extra mapped
fields), and `hybrid_search.py` (the BM25+kNN query itself — the one place
that logic lives; used by `kb_api`, which `kb_mcp` and
`scripts/eval_retrieval.py` reach indirectly over HTTP rather than importing
it directly — see the MCP server section).

- **`kb_argo/`** — the Euro-Argo pipeline.
  - `fetch.py` pulls the WMO code list from `GET /platformCodes`, then the
    full record for each float from `GET /floats/{wmo}` (Euro-Argo Fleet
    Monitoring API), caching raw JSON to disk.
  - `transform.py` converts each raw record into structured fields plus a
    natural-language `summary_text` (deployment, PI, project, resolved sensor
    names, mission length/cycle count, last known position/measurements tagged
    with a real named sea/ocean via `kb_common.seas`, status, data-quality
    caveats).
  - `link_oso.py` cross-links each float's operating organization to OSO.
  - `pipeline.py` ties fetch → transform → embed → index together.

- **`kb_oso/`** — the OSO ontology pipeline. OSO is RDF/OWL (SKOS-labelled
  individuals — sites, platforms, campaigns, organizations, people — linked
  by custom object properties), a completely different shape from Argo's
  JSON. Rather than a per-entity-type template, `transform.py` builds a
  generic label index from the graph and renders each individual's
  properties into a summary regardless of its `rdf:type`, so new OSO entity
  types added later don't need code changes.
  - `load.py` parses the `.owl` file with `rdflib`.
  - `transform.py` picks each individual's preferred label/definition
    (English, falling back to French, then whatever's available), and turns
    its remaining relations (e.g. `containsSite`, `isLedByOrganization`) into
    sentences by resolving the related entity's own label (external NERC
    vocab URIs resolve via `kb_common.nerc_vocab` when the cache is
    available). Identifier-style predicates (ROR, ORCID, EDMO, Wikidata
    `sameAs`) are kept as structured `external_ids` instead of noisy
    sentences.
  - `pipeline.py` ties transform → embed → index together (no fetch step —
    the ontology file is provided locally).

Documents from both sources are keyed `source:id` (e.g. `euro_argo:1900045`,
`oso:ANTARES`) so re-running ingestion upserts rather than duplicates.

- **`kb_mcp/`** — the MCP server (see above). `search.py`'s `search()` calls
  `kb_api`'s `POST /internal/search` over HTTP for the one embedding-dependent
  tool, plus its own direct Elasticsearch geo/terms-aggregation query builders
  for everything else (structured filters, discovery tools, fetch-by-id) -
  those never needed embeddings and stay untouched. `server.py` just wires
  those into MCP tool definitions. Ships as its own lightweight image
  (`Dockerfile.mcp`/`requirements-mcp.txt`) with no ML dependencies.

- **`kb_api/`** — the HTTP API (see above), and the only service that still
  imports `kb_common.hybrid_search`/`embed` for querying (`kb_argo`/`kb_oso`
  still use `kb_common.embed` directly at *indexing* time, separately).
  `format.py` shapes a hybrid-search hit into the fixed
  score/url/header/vector/summary/highlight/last_modified public result;
  `main.py` is the FastAPI app, exposing both that public `/search` and the
  unshaped `/internal/search` `kb_mcp` uses. Built from `Dockerfile.api`
  (keeps the CUDA `torch` install + GPU reservation).
