from shapely.geometry import box

from kb_common.seas import SeasIndex, ocean_for_sea


def _sample_index():
    # Two adjacent squares: "Sea A" spans lon [0,10], "Sea B" spans lon [10,20].
    names = ["Sea A", "Sea B"]
    geoms = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    return SeasIndex(names, geoms)


def test_classify_point_inside_first_polygon():
    index = _sample_index()
    assert index.classify(lat=5, lon=5) == "Sea A"


def test_classify_point_inside_second_polygon():
    index = _sample_index()
    assert index.classify(lat=5, lon=15) == "Sea B"


def test_classify_none_coordinates_returns_none():
    index = _sample_index()
    assert index.classify(lat=None, lon=5) is None
    assert index.classify(lat=5, lon=None) is None


def test_classify_falls_back_to_nearest_polygon():
    index = _sample_index()
    # Far outside both boxes, closer to Sea B's edge (lon=20) than Sea A's (lon=10).
    assert index.classify(lat=5, lon=100) == "Sea B"


def test_ocean_for_sea_known(monkeypatch):
    import kb_common.seas as seas_module

    monkeypatch.setattr(seas_module, "_load_sea_to_ocean", lambda: {"Sea A": "Ocean X"})
    assert ocean_for_sea("Sea A") == "Ocean X"


def test_ocean_for_sea_unknown(monkeypatch):
    import kb_common.seas as seas_module

    monkeypatch.setattr(seas_module, "_load_sea_to_ocean", lambda: {"Sea A": "Ocean X"})
    assert ocean_for_sea("Nowhere") is None
    assert ocean_for_sea(None) is None
