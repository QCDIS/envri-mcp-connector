"""Sea/ocean classification via point-in-polygon lookup"""
import json
import pickle
from functools import lru_cache
from pathlib import Path

import shapefile
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from kb_common import config

# ~0.01 degree (~1km at the equator) - negligible for offshore float
# positions, drastically fewer vertices than the source's full resolution.
_SIMPLIFY_TOLERANCE = 0.01
_CACHE_SUFFIX = ".simplified.pkl"

SEA_TO_OCEAN_PATH = Path("data/sea_to_ocean.json")


class SeasIndex:
    def __init__(self, names: list, geoms: list):
        self.names = names
        self.geoms = geoms
        self.tree = STRtree(geoms)

    @classmethod
    def from_shapefile(cls, shp_path: Path) -> "SeasIndex":
        reader = shapefile.Reader(str(shp_path))
        names, geoms = [], []
        for sr in reader.shapeRecords():
            # preserve_topology=False is dramatically faster on this much
            # input detail; a slightly invalid polygon is fine for
            # .contains()/.distance() classification.
            geom = shape(sr.shape.__geo_interface__).simplify(_SIMPLIFY_TOLERANCE, preserve_topology=False)
            names.append(sr.record["NAME"])
            geoms.append(geom)
        return cls(names, geoms)

    def classify(self, lat: float, lon: float) -> str | None:
        if lat is None or lon is None:
            return None
        point = Point(lon, lat)
        for idx in self.tree.query(point):
            if self.geoms[idx].contains(point):
                return self.names[idx]
        # Rare fallback (coastal precision, boundary gaps): nearest polygon.
        nearest = min(range(len(self.geoms)), key=lambda i: self.geoms[i].distance(point))
        return self.names[nearest]


def _cache_path(shp_path: Path) -> Path:
    return shp_path.with_suffix(_CACHE_SUFFIX)


@lru_cache(maxsize=1)
def _get_index():
    shp_path = config.SEAS_SHAPEFILE_PATH
    cache_path = _cache_path(shp_path)

    if cache_path.exists():
        names, geoms = pickle.loads(cache_path.read_bytes())
        return SeasIndex(names, geoms)

    if not shp_path.exists():
        return None

    index = SeasIndex.from_shapefile(shp_path)
    cache_path.write_bytes(pickle.dumps((index.names, index.geoms)))
    return index


def classify_sea(lat: float, lon: float) -> str | None:
    """Returns the IHO sea/ocean name containing (lat, lon), or None if the
    shapefile isn't available (falls back gracefully, like nerc_vocab.py)."""
    index = _get_index()
    if index is None:
        return None
    return index.classify(lat, lon)


@lru_cache(maxsize=1)
def _load_sea_to_ocean() -> dict:
    if not SEA_TO_OCEAN_PATH.exists():
        return {}
    return json.loads(SEA_TO_OCEAN_PATH.read_text())


def ocean_for_sea(sea_name: str | None) -> str | None:
    return _load_sea_to_ocean().get(sea_name)


def classify(lat: float, lon: float) -> tuple[str | None, str | None]:
    """Returns (sea_name, ocean_name) - the specific IHO area and its parent
    ocean basin."""
    sea = classify_sea(lat, lon)
    return sea, ocean_for_sea(sea)
