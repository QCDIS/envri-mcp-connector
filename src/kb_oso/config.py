import os
from pathlib import Path

OSO_OWL_PATH = Path(os.environ.get("OSO_OWL_PATH", "data/cache/oso/oso.owl"))
OSO_OWL_URL = os.environ.get(
    "OSO_OWL_URL", "https://github.com/emso-eric/oso-ontology/releases/latest/download/OSO.owl"
)
