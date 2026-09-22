"""HTTP API exposing the knowledge base's hybrid search as plain JSON.

Two search routes, deliberately different contracts:
- `GET /search`: fixed public shape (score/url/header/vector/summary/
  highlight/last_modified) - e.g. for a downstream system doing its own
  vector re-ranking.
- `POST /internal/search`: unshaped passthrough to
  kb_common.hybrid_search.search(), used by kb_mcp so it can search without
  loading the embedding model itself.

`POST /fetch` and `POST /index` trigger the ingestion pipeline (fetch raw
Euro-Argo records, then embed and index cached records into Elasticsearch).
Both run in the background and return immediately - watch server logs for
progress. `GET /stats` reports document counts, overall and per source.

Run directly:
    PYTHONPATH=src python -m kb_api.main
"""
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, Query, Request
from pydantic import BaseModel, Field

from kb_api import config
from kb_api.format import FIELDS, format_hit
from kb_argo import fetch as argo_fetch
from kb_argo import pipeline as argo_pipeline
from kb_common import config as common_config
from kb_common import embed, es_index, hybrid_search, timing
from kb_oso import fetch as oso_fetch
from kb_oso import pipeline as oso_pipeline

app = FastAPI(title="Ifremer Knowledge Base API")


@app.on_event("startup")
def _warm_embedding_model() -> None:
    """Loads the model at startup instead of on the first search, so the
    model_load cold-start spike shows up in server logs, not in a user's
    first request."""
    embed.get_model()


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    if not common_config.LOG_TIMING:
        return await call_next(request)
    with timing.request() as stages:
        with timing.stage("api_total"):
            response = await call_next(request)
        response.headers["Server-Timing"] = ", ".join(
            f'{name};dur={elapsed * 1000:.1f}' for name, elapsed in stages.items()
        )
        return response

Source = Literal["euro_argo", "oso"]


class SearchHit(BaseModel):
    score: float
    url: str | None = Field(None, description="Fleet Monitoring page (euro_argo) or OSO ontology IRI (oso)")
    header: str = Field(description="'Argo float <wmo>', or the OSO pref_label / id")
    vector: list[float] | None = Field(None, description="Document embedding")
    summary: str | None = Field(None, description="Indexed summary text")
    highlight: list[str] = Field(description="Matched fragments of summary; empty if no lexical match")
    last_modified: str | None = Field(None, description="Index timestamp (ISO 8601); null until re-ingested")


class TaskStarted(BaseModel):
    status: Literal["started"]
    source: Source


class Stats(BaseModel):
    total: int
    by_source: dict[str, int]


class Health(BaseModel):
    status: Literal["ok"]


@app.get("/search", response_model=list[SearchHit])
def search(
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    source: Source | None = Query(
        None, description="Restrict results to one source: Euro-Argo float metadata or the OSO ontology"
    ),
) -> list[dict]:
    hits = hybrid_search.search(common_config.ES_INDEX, query, k=limit, source=source, fields=FIELDS)
    return [format_hit(hit) for hit in hits]


class InternalSearchRequest(BaseModel):
    query: str
    k: int = 10
    mode: str = "hybrid"
    source: str | None = None
    fields: list[str] = list(hybrid_search.DEFAULT_FIELDS)
    highlight_field: str | None = "summary_text"


@app.post("/internal/search")
def internal_search(req: InternalSearchRequest) -> list[dict]:
    """Used by kb_mcp only - not the public search contract, see module docstring."""
    return hybrid_search.search(
        common_config.ES_INDEX,
        req.query,
        k=req.k,
        mode=req.mode,
        source=req.source,
        fields=tuple(req.fields),
        highlight_field=req.highlight_field,
    )


@app.post("/fetch", response_model=TaskStarted)
def fetch(
    background_tasks: BackgroundTasks,
    source: Source = Query(..., description="Which source to fetch"),
    limit: int | None = Query(None, description="euro_argo only: fetch just the first N floats (testing)"),
    force: bool = Query(True, description="Re-fetch even if already cached on disk"),
) -> dict:
    """Fetch raw source data into the local cache: Euro-Argo float records
    from the upstream API, or the OSO ontology OWL file from its GitHub
    release. Runs in the background; see server logs for progress."""
    if source == "euro_argo":
        background_tasks.add_task(argo_fetch.fetch_all, limit=limit, force=force)
    else:
        background_tasks.add_task(oso_fetch.fetch_all, force=force)
    return {"status": "started", "source": source}


@app.post("/index", response_model=TaskStarted)
def index(
    background_tasks: BackgroundTasks,
    source: Source = Query(..., description="Which source to embed and index"),
    limit: int | None = Query(None, description="Only index the first N records (testing)"),
) -> dict:
    """Transform, embed and index cached records for one source into
    Elasticsearch. Runs in the background; see server logs for progress."""
    run_index = argo_pipeline.run_index if source == "euro_argo" else oso_pipeline.run_index
    background_tasks.add_task(run_index, limit=limit)
    return {"status": "started", "source": source}


@app.get("/stats", response_model=Stats)
def stats() -> dict:
    """Document counts in the knowledge base, overall and per source."""
    return es_index.index_stats(common_config.ES_INDEX)


@app.get("/health", response_model=Health)
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
