"""HTTP API exposing the knowledge base's hybrid search as plain JSON.

Two routes, deliberately different contracts:
- `GET /search`: fixed public shape (score/url/header/vector/summary/
  highlight/last_modified) - e.g. for a downstream system doing its own
  vector re-ranking.
- `POST /internal/search`: unshaped passthrough to
  kb_common.hybrid_search.search(), used by kb_mcp so it can search without
  loading the embedding model itself.

Run directly:
    PYTHONPATH=src python -m kb_api.main
"""
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel

from kb_api import config
from kb_api.format import FIELDS, format_hit
from kb_common import config as common_config
from kb_common import hybrid_search

app = FastAPI(title="Ifremer Knowledge Base API")


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    source: Literal["euro_argo", "oso"] | None = Query(
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
