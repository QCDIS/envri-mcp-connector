"""Reindex into a new index with an updated mapping, carrying existing
documents - and their already-computed embeddings - across without
recomputing anything. Runs Elasticsearch's _reindex asynchronously and polls
for completion, since a synchronous wait can outlive the HTTP client's
timeout even though the task is still running fine server-side.

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


def reindex(source: str, dest: str, dims: int | None):
    client = es_index.get_client()

    if client.indices.exists(index=dest):
        print(f"destination index {dest!r} already exists - delete it first if this is a retry, "
              f"or you'll double-insert documents. Aborting.")
        sys.exit(1)

    resolved_dims = get_dims(client, source, dims)
    client.indices.create(index=dest, body=es_index.build_mapping(resolved_dims, merged_extra_properties()))
    print(f"created {dest!r} (dims={resolved_dims})")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.ES_INDEX)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--dims", type=int, default=None, help="Defaults to the source index's existing embedding dims")
    args = parser.parse_args()
    reindex(args.source, args.dest, args.dims)
