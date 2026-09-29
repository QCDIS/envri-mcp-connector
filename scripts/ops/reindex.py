"""Reindex into a new index with an updated mapping, carrying existing
documents - and their already-computed embeddings - across without
recomputing anything. Runs Elasticsearch's _reindex asynchronously and polls
for completion, since a synchronous wait can outlive the HTTP client's
timeout even though the task is still running fine server-side.

The destination mapping comes from kb_common.es_index.build_mapping, so it
includes the `lowercase` normalizer settings and every case-insensitive keyword
field; after copying, a check confirms case variants of a real value match the
same documents.

Usage:
    PYTHONPATH=src python scripts/ops/reindex.py --dest ifremer-knowledge-base-v2
    PYTHONPATH=src python scripts/ops/reindex.py --source foo --dest bar --dims 1024
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from kb_argo import es_mapping as argo_mapping
from kb_common import config, es_index
from kb_oso import es_mapping as oso_mapping

POLL_SECONDS = 5


def merged_extra_properties() -> dict:
    """Both sources share one index, so the new mapping needs both sources'
    extra fields, not just whichever pipeline prompted the change."""
    return {**argo_mapping.EXTRA_PROPERTIES, **oso_mapping.EXTRA_PROPERTIES}


def get_dims(client, source_index: str, explicit: int | None) -> int:
    if explicit:
        return explicit
    mapping = client.indices.get_mapping(index=source_index)
    props = next(iter(mapping.values()))["mappings"]["properties"]
    return props["embedding"]["dims"]


def get_source_embedding_model(client, source_index: str) -> str:
    """The embedding model recorded on the source index. The documents keep
    their existing embeddings, so the new index must claim the same model, not
    whatever EMBEDDING_MODEL happens to be set to here."""
    mapping = next(iter(client.indices.get_mapping(index=source_index).values()))["mappings"]
    model = (mapping.get("_meta") or {}).get("embedding_model")
    if model:
        return model
    print(f"WARNING: source index has no embedding metadata; recording the configured model "
          f"{config.EMBEDDING_MODEL!r} on the new index - make sure that's the model the embeddings came from.")
    return config.EMBEDDING_MODEL


def verify_case_insensitive(client, dest: str) -> bool:
    """Take a real sea_area value and confirm lower/upper/original casing all
    match the same number of documents on the normalized sea_area.keyword."""
    resp = client.search(
        index=dest,
        size=1,
        query={"exists": {"field": "sea_area"}},
        source=["sea_area"],
    )
    hits = resp["hits"]["hits"]
    if not hits:
        print("case-insensitivity check skipped: no documents with sea_area")
        return True
    original = hits[0]["_source"]["sea_area"]
    counts = {
        variant: client.count(index=dest, query={"term": {"sea_area.keyword": variant}})["count"]
        for variant in (original, original.lower(), original.upper())
    }
    ok = len(set(counts.values())) == 1 and next(iter(counts.values())) > 0
    print(f"case-insensitivity check on sea_area.keyword: {counts} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def reindex(source: str, dest: str, dims: int | None):
    client = es_index.get_client()

    if client.indices.exists(index=dest):
        print(f"destination index {dest!r} already exists - delete it first if this is a retry, "
              f"or you'll double-insert documents. Aborting.")
        sys.exit(1)

    resolved_dims = get_dims(client, source, dims)
    embedding_model = get_source_embedding_model(client, source)
    client.indices.create(
        index=dest,
        body=es_index.build_mapping(resolved_dims, merged_extra_properties(), embedding_model=embedding_model),
    )
    print(f"created {dest!r} (dims={resolved_dims}, model={embedding_model})")

    resp = client.reindex(source={"index": source}, dest={"index": dest}, wait_for_completion=False)
    task_id = resp["task"]
    print(f"reindex task: {task_id}")

    while True:
        status = client.tasks.get(task_id=task_id)
        s = status["task"]["status"]
        if status["completed"]:
            print(f"done: {s['created']}/{s['total']} created")
            failures = status.get("response", {}).get("failures")
            if failures:
                print(f"FAILURES ({len(failures)}):", failures[:5])
                sys.exit(1)
            break
        print(f"progress: {s['created']}/{s['total']}")
        time.sleep(POLL_SECONDS)

    src_count = client.count(index=source)["count"]
    dest_count = client.count(index=dest)["count"]
    print(f"source {source!r}: {src_count} docs")
    print(f"dest   {dest!r}: {dest_count} docs")
    if src_count != dest_count:
        print("WARNING: counts don't match - investigate before switching ES_INDEX over.")
    if not verify_case_insensitive(client, dest):
        print("WARNING: case variants returned different counts - the normalizer isn't applied; do not switch ES_INDEX over.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.ES_INDEX)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--dims", type=int, default=None, help="Defaults to the source index's existing embedding dims")
    args = parser.parse_args()
    reindex(args.source, args.dest, args.dims)
