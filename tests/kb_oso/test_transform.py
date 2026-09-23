import rdflib
from rdflib import RDF, RDFS, Literal, URIRef
from rdflib.namespace import OWL, SKOS

from kb_oso import transform

SCHEMA = rdflib.Namespace("http://schema.org/")
OSO_NS = rdflib.Namespace("https://w3id.org/earthsemantics/OSO#")


def test_local_name_with_fragment():
    assert transform.local_name(URIRef("https://w3id.org/earthsemantics/OSO#Organization")) == "Organization"


def test_local_name_with_trailing_slash_path():
    assert transform.local_name(URIRef("http://vocab.nerc.ac.uk/collection/R25/current/TEMP/")) == "TEMP"


def test_humanize_camel_case():
    assert transform.humanize("isLedByOrganization") == "is led by organization"


def test_humanize_snake_case():
    assert transform.humanize("some_snake_case") == "some snake case"


def test_humanize_all_caps_acronym():
    assert transform.humanize("hasROR") == "has ror"


def test_pick_prefers_english():
    assert transform._pick({"fr": "bonjour", "en": "hello"}) == "hello"


def test_pick_falls_back_to_any_language():
    assert transform._pick({"de": "hallo"}) == "hallo"


def test_pick_empty_returns_none():
    assert transform._pick({}) is None


def _graph_with_labels():
    g = rdflib.Graph()
    subject = URIRef("https://w3id.org/earthsemantics/OSO#Person1")
    g.add((subject, SKOS.prefLabel, Literal("Jane Doe", lang="en")))
    g.add((subject, SKOS.prefLabel, Literal("Jeanne Dupont", lang="fr")))
    return g, subject


def test_build_label_index_and_resolve_label_prefers_english():
    g, subject = _graph_with_labels()
    index = transform.build_label_index(g)
    assert transform.resolve_label(subject, index) == "Jane Doe"


def test_resolve_label_nerc_fallback(monkeypatch):
    g = rdflib.Graph()
    nerc_uri = URIRef("http://vocab.nerc.ac.uk/collection/R25/current/TEMP/")
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: "Temperature" if uri == str(nerc_uri) else None)

    assert transform.resolve_label(nerc_uri, {}) == "Temperature"


def test_resolve_label_humanizes_unknown_uri():
    unknown = URIRef("https://w3id.org/earthsemantics/OSO#someUnknownThing")
    assert transform.resolve_label(unknown, {}) == "some unknown thing"


def test_get_individuals_returns_only_named_individuals_sorted():
    g = rdflib.Graph()
    a = URIRef("https://w3id.org/earthsemantics/OSO#B")
    b = URIRef("https://w3id.org/earthsemantics/OSO#A")
    not_individual = URIRef("https://w3id.org/earthsemantics/OSO#NotOne")
    g.add((a, RDF.type, OWL.NamedIndividual))
    g.add((b, RDF.type, OWL.NamedIndividual))
    g.add((not_individual, RDF.type, OSO_NS.SomeClass))

    assert transform.get_individuals(g) == [b, a]  # sorted: "#A" < "#B"


def _build_sample_org_graph():
    g = rdflib.Graph()
    org = URIRef("https://w3id.org/earthsemantics/OSO#TestOrg")
    person = URIRef("https://w3id.org/earthsemantics/OSO#Person1")

    g.add((org, RDF.type, OWL.NamedIndividual))
    g.add((org, RDF.type, OSO_NS.Organisation))  # British spelling -> aliased to "Organization"
    g.add((org, SKOS.prefLabel, Literal("Test Organization", lang="en")))
    g.add((org, SKOS.prefLabel, Literal("Organisation de Test", lang="fr")))
    g.add((org, SKOS.altLabel, Literal("TestOrg", lang="en")))
    g.add((org, SKOS.definition, Literal("A fictional test organization.", lang="en")))
    g.add((org, RDFS.comment, Literal("Used only for tests.", lang="en")))
    g.add((org, OWL.sameAs, URIRef("http://example.org/other/TestOrg")))
    g.add((org, OSO_NS.isActive, Literal(True)))
    g.add((org, OSO_NS.isDissolved, Literal(False)))
    g.add((org, SCHEMA.member, person))

    g.add((person, SKOS.prefLabel, Literal("Jane Doe", lang="en")))

    return g, org


def test_build_record_structured_fields():
    g, org = _build_sample_org_graph()
    label_index = transform.build_label_index(g)

    record = transform.build_record(g, org, label_index)

    assert record["_id"] == "oso:TestOrg"
    assert record["source"] == "oso"
    assert record["oso_id"] == "TestOrg"
    assert record["entity_types"] == ["Organization"]
    assert record["pref_label"] == "Test Organization"
    assert record["pref_label_en"] == "Test Organization"
    assert record["pref_label_fr"] == "Organisation de Test"
    assert record["alt_labels"] == ["TestOrg"]
    assert record["definition_en"] == "A fictional test organization."
    assert record["external_ids"] == {"same_as": "http://example.org/other/TestOrg"}


def test_build_record_summary_text_includes_type_definition_flag_and_relation():
    g, org = _build_sample_org_graph()
    label_index = transform.build_label_index(g)

    record = transform.build_record(g, org, label_index)
    summary = record["summary_text"]

    assert "Test Organization is an organization." in summary
    assert "A fictional test organization." in summary
    assert "Used only for tests." in summary
    assert "is active" in summary  # boolean-true flag
    assert "Jane Doe" in summary  # resolved object-property relation
    assert "is dissolved" not in summary  # boolean-false flags are skipped
