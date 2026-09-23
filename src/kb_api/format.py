"""Formats a raw hybrid-search hit into the fixed API result shape:
score, url, header, vector, summary, highlight, last_modified.
"""

FIELDS = ("source", "summary_text", "embedding", "indexed_at", "wmo", "pref_label")


def _local_id(doc_id: str) -> str:
    return doc_id.split(":", 1)[1] if ":" in doc_id else doc_id


def format_hit(hit: dict) -> dict:
    source = hit.get("source")
    local_id = _local_id(hit["_id"])

    if source == "euro_argo":
        url = f"https://fleetmonitoring.euro-argo.eu/float/{local_id}"
        header = f"Argo float {local_id}"
    elif source == "oso":
        # A real, dereferenceable link to OSO's own ontology, not this project.
        url = f"https://w3id.org/earthsemantics/OSO#{local_id}"
        header = hit.get("pref_label") or local_id
    else:
        url = None
        header = local_id

    return {
        "score": hit["score"],
        "url": url,
        "header": header,
        "vector": hit.get("embedding"),
        "summary": hit.get("summary_text"),
        "highlight": hit.get("highlights") or [],
        "last_modified": hit.get("indexed_at"),
    }
