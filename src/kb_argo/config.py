import os
from pathlib import Path

ARGO_API_BASE = os.environ.get("ARGO_API_BASE", "https://fleetmonitoring.euro-argo.eu")
ARGO_RAW_DIR = Path(os.environ.get("ARGO_RAW_DIR", "data/cache/argo/raw"))
ARGO_FETCH_WORKERS = int(os.environ.get("ARGO_FETCH_WORKERS", "8"))
