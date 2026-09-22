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

# Per-stage latency logging (kb_mcp's end-of-tool log line, kb_api's
# Server-Timing header) - on by default, set to "false" to silence it.
LOG_TIMING = os.environ.get("LOG_TIMING", "true").lower() != "false"

# IHO World Seas v3 (marineregions.org) - used by kb_common.seas for real
# named-sea classification instead of a lat/lon bounding-box heuristic.
SEAS_SHAPEFILE_PATH = Path(os.environ.get("SEAS_SHAPEFILE_PATH", "data/seas/World_Seas_IHO_v3.shp"))
