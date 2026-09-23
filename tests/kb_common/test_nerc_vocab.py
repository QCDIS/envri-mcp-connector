import json
import zipfile

from kb_common import nerc_vocab

_TTL_ONE_CONCEPT = b"""
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

<http://vocab.nerc.ac.uk/collection/R25/current/TEMP/> a skos:Concept ;
    skos:prefLabel "Temperature"@en ;
    skos:definition "Sea water temperature"@en .
"""

_TTL_MULTILINGUAL_LABEL = b"""
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

<http://vocab.nerc.ac.uk/collection/R25/current/TEMP/> a skos:Concept ;
    skos:prefLabel "Temperature"@en, "Temperature (fr)"@fr .
"""


def test_discover_needed_collections_always_includes_base(tmp_path):
    missing = tmp_path / "missing.owl"
    assert nerc_vocab.discover_needed_collections(missing) == {"R25"}


def test_discover_needed_collections_finds_referenced_codes(tmp_path):
    owl = tmp_path / "oso.owl"
    owl.write_text("some xml with http://vocab.nerc.ac.uk/collection/L05/current/1234/ inside")
    assert nerc_vocab.discover_needed_collections(owl) == {"R25", "L05"}


def test_parse_collection_extracts_pref_label_and_definition():
    entries = nerc_vocab._parse_collection(_TTL_ONE_CONCEPT)
    assert entries == {
        "http://vocab.nerc.ac.uk/collection/R25/current/TEMP/": {
            "prefLabel": "Temperature",
            "definition": "Sea water temperature",
        }
    }


def test_parse_collection_prefers_english_label():
    entries = nerc_vocab._parse_collection(_TTL_MULTILINGUAL_LABEL)
    concept = entries["http://vocab.nerc.ac.uk/collection/R25/current/TEMP/"]
    assert concept["prefLabel"] == "Temperature"


def test_build_cache_writes_json_and_returns_entries(tmp_path):
    zip_path = tmp_path / "turtles.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("R25.ttl", _TTL_ONE_CONCEPT)

    out_path = tmp_path / "out.json"
    missing_owl = tmp_path / "missing.owl"  # -> discover_needed_collections falls back to {"R25"}

    entries = nerc_vocab.build_cache(zip_path, owl_path=missing_owl, out_path=out_path)

    assert entries == {
        "http://vocab.nerc.ac.uk/collection/R25/current/TEMP/": {
            "prefLabel": "Temperature",
            "definition": "Sea water temperature",
        }
    }
    assert json.loads(out_path.read_text()) == entries


def test_build_cache_skips_collections_missing_from_zip(tmp_path):
    zip_path = tmp_path / "turtles.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("OTHER.ttl", b"")  # R25 not present

    out_path = tmp_path / "out.json"
    missing_owl = tmp_path / "missing.owl"

    entries = nerc_vocab.build_cache(zip_path, owl_path=missing_owl, out_path=out_path)

    assert entries == {}
