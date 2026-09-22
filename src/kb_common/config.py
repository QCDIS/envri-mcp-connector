import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_USERNAME = os.environ.get("ES_USERNAME", "elastic")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
ES_CA_CERT = os.environ.get("ES_CA_CERT", "")
ES_VERIFY_CERTS = os.environ.get("ES_VERIFY_CERTS", "true").lower() != "false"
ES_INDEX = os.environ.get("ES_INDEX", "ifremer-knowledge-base")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))

# Concurrent search requests' embed_query() calls are coalesced into one
# batched model.encode() call instead of hitting the GPU one at a time -
# see kb_common.embed. Window: how long to wait for more queries to join a
# batch once the first one arrives; max size: cap on how many join it.
EMBED_QUERY_BATCH_WINDOW_MS = int(os.environ.get("EMBED_QUERY_BATCH_WINDOW_MS", "8"))
EMBED_QUERY_BATCH_MAX_SIZE = int(os.environ.get("EMBED_QUERY_BATCH_MAX_SIZE", "32"))

# Per-stage latency logging (kb_mcp's end-of-tool log line, kb_api's
# Server-Timing header) - on by default, set to "false" to silence it.
LOG_TIMING = os.environ.get("LOG_TIMING", "true").lower() != "false"

# IHO World Seas v3 (marineregions.org) - used by kb_common.seas for real
# named-sea classification instead of a lat/lon bounding-box heuristic.
SEAS_SHAPEFILE_PATH = Path(os.environ.get("SEAS_SHAPEFILE_PATH", "data/seas/World_Seas_IHO_v3.shp"))
