"""End-to-end: fetch Euro-Argo floats, embed summaries, index into Elasticsearch.

Usage:
    python -m kb_argo.pipeline fetch [--limit N] [--force]
    python -m kb_argo.pipeline index [--limit N]
"""
import argparse
import itertools
import logging

from tqdm import tqdm

from kb_argo import es_mapping, fetch, transform
from kb_common import config, embed, es_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE = 256


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def run_index(limit: int | None = None):
    client = es_index.get_client()
    ensured = False

    records = transform.iter_records()
    if limit:
        records = itertools.islice(records, limit)

    total = 0
    for batch in tqdm(chunked(records, CHUNK_SIZE)):
        vectors = embed.embed_texts([r["summary_text"] for r in batch])
        for rec, vec in zip(batch, vectors):
            rec["embedding"] = vec

        if not ensured:
            es_index.ensure_index(client, dims=len(vectors[0]), extra_properties=es_mapping.EXTRA_PROPERTIES)
            ensured = True

        es_index.bulk_index(client, batch)
        total += len(batch)

    log.info("indexed %d documents into %s", total, config.ES_INDEX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch raw float records from Euro-Argo API")
    p_fetch.add_argument("--limit", type=int, default=None)
    p_fetch.add_argument("--force", action="store_true")

    p_index = sub.add_parser("index", help="Transform, embed and index cached float records")
    p_index.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.command == "fetch":
        fetch.fetch_all(limit=args.limit, force=args.force)
    elif args.command == "index":
        run_index(limit=args.limit)
