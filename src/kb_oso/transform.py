"""Turn OSO ontology individuals into KB documents.

OSO is an RDF/OWL ontology (SKOS-labelled individuals connected by custom
object/data properties), a fundamentally different shape from the Argo JSON
records. Rather than a per-type template, this builds a generic label index
and renders each individual's properties into a natural-language summary,
regardless of its rdf:type.
"""
import argparse
import json
import re
from typing import Iterator

import rdflib
from rdflib import RDF, RDFS
from rdflib.namespace import OWL, SKOS

from kb_common import nerc_vocab
from kb_oso import load

SOURCE = "oso"

SCHEMA = rdflib.Namespace("http://schema.org/")
DCTERMS = rdflib.Namespace("http://purl.org/dc/terms/")
OSO_NS = rdflib.Namespace("https://w3id.org/earthsemantics/OSO#")

_NERC_PREFIX = "http://vocab.nerc.ac.uk/collection/"

# Identifier-style predicates: kept as structured `external_ids` metadata
# instead of noisy summary sentences (a ROR/ORCID/Wikidata code adds nothing
# to embedded text but is useful to keep around for lookups).
_IDENTIFIER_PREDICATES = {
    OWL.sameAs: "same_as",
    SKOS.exactMatch: "exact_match",
    SKOS.closeMatch: "close_match",
    OSO_NS.hasROR: "ror",
    OSO_NS.hasORCID: "orcid",
    OSO_NS.hasEDMO: "edmo",
    DCTERMS.identifier: "identifier",
}

# Some OSO revisions use the British spelling for a type that's otherwise
# American elsewhere in the same ontology; canonicalize so summaries don't
# read "is an organization, organisation."
_TYPE_ALIASES = {
    "Organisation": "Organization",
}

# Predicates handled explicitly (label/definition/type/identifiers) and
# excluded from the generic "remaining relations" pass below.
_HANDLED_PREDICATES = {
    RDF.type,
    SKOS.prefLabel,
    SKOS.altLabel,
    SKOS.definition,
    SKOS.note,
    RDFS.label,
    RDFS.seeAlso,
    RDFS.comment,
    SCHEMA.url,
    *_IDENTIFIER_PREDICATES,
}

_LANG_PREFERENCE = ("en", "fr")


def local_name(uri) -> str:
    s = str(uri).rstrip("/")
    return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]


def humanize(name: str) -> str:
    """CamelCase / snake_case -> lowercase words, e.g. 'isLedByOrganization' -> 'is led by organization'."""
    name = name.replace("_", " ")
    words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$|\s)", name)
    return " ".join(words).lower().strip() or name.lower()


def _lang_values(g: rdflib.Graph, subject, predicate) -> dict:
    values = {}
    for o in g.objects(subject, predicate):
        lang = getattr(o, "language", None) or "und"
        values.setdefault(lang, str(o))
    return values


def _pick(values: dict, prefer=_LANG_PREFERENCE):
    for lang in prefer:
        if lang in values:
            return values[lang]
    if values:
        return next(iter(values.values()))
    return None


def build_label_index(g: rdflib.Graph) -> dict:
    """subject URI -> {lang: label}, from skos:prefLabel falling back to rdfs:label."""
    labels: dict = {}
    for s, _, o in g.triples((None, SKOS.prefLabel, None)):
        lang = getattr(o, "language", None) or "und"
        labels.setdefault(s, {}).setdefault(lang, str(o))
    for s, _, o in g.triples((None, RDFS.label, None)):
        lang = getattr(o, "language", None) or "und"
        labels.setdefault(s, {}).setdefault(lang, str(o))
    return labels


def resolve_label(uri, label_index: dict) -> str:
    entry = label_index.get(uri)
    if entry:
        picked = _pick(entry)
        if picked:
            return picked
    if str(uri).startswith(_NERC_PREFIX):
        nerc_label = nerc_vocab.get_label(str(uri))
        if nerc_label:
            return nerc_label
    return humanize(local_name(uri))


def get_individuals(g: rdflib.Graph):
    return sorted(set(g.subjects(RDF.type, OWL.NamedIndividual)))


def build_summary_text(g: rdflib.Graph, subject, label_index: dict, pref_label: str, entity_types: list, definition: str) -> str:
    parts = []

    header = f"{pref_label} is"
    if entity_types:
        readable_types = [humanize(t) for t in entity_types]
        article = "an" if readable_types[0][:1] in "aeiou" else "a"
        header += f" {article} {', '.join(readable_types)}"
    parts.append(header.strip() + ".")

    if definition:
        parts.append(definition)

    # Group remaining object/data properties by predicate for a compact sentence each.
    grouped: dict = {}
    literals_by_lang: dict = {}
    flags = []
    for p, o in g.predicate_objects(subject):
        if p in _HANDLED_PREDICATES:
            continue
        if isinstance(o, rdflib.Literal):
            value = str(o)
            if value.lower() == "false":
                continue  # skip uninformative negative booleans
            if value.lower() == "true":
                flags.append(humanize(local_name(p)))
                continue
            lang = getattr(o, "language", None) or "und"
            literals_by_lang.setdefault(p, {}).setdefault(lang, []).append(value)
        else:
            grouped.setdefault(p, []).append(resolve_label(o, label_index))

    # A predicate tagged with several languages is free text translated
    # multiple times (e.g. rdfs:comment, dcterms:description) - keep only
    # one language's values rather than concatenating every translation.
    for p, lang_map in literals_by_lang.items():
        if len(lang_map) > 1:
            chosen_lang = next((l for l in _LANG_PREFERENCE if l in lang_map), next(iter(lang_map)))
            grouped[p] = lang_map[chosen_lang]
        else:
            grouped[p] = next(iter(lang_map.values()))

    if flags:
        parts.append(f"It is {', '.join(dict.fromkeys(flags))}.")

    for p, values in grouped.items():
        verb = humanize(local_name(p))
        if not verb.startswith(("is ", "has ", "contains ", "belongs")):
            verb = f"has {verb}"
        joined = ", ".join(dict.fromkeys(values))
        sep = " " if verb.startswith("is ") else ": "
        parts.append(f"It {verb}{sep}{joined}.")

    return " ".join(parts)


def build_record(g: rdflib.Graph, subject, label_index: dict) -> dict:
    entity_types = list(dict.fromkeys(
        _TYPE_ALIASES.get(local_name(t), local_name(t))
        for t in g.objects(subject, RDF.type)
        if t != OWL.NamedIndividual
    ))

    external_ids = {}
    for predicate, key in _IDENTIFIER_PREDICATES.items():
        values = [str(o) for o in g.objects(subject, predicate)]
        if values:
            external_ids[key] = values if len(values) > 1 else values[0]

    pref_labels = _lang_values(g, subject, SKOS.prefLabel) or _lang_values(g, subject, RDFS.label)
    pref_label = _pick(pref_labels) or humanize(local_name(subject))

    definitions = _lang_values(g, subject, SKOS.definition)
    comments = _lang_values(g, subject, RDFS.comment)
    definition = _pick(definitions)
    comment = _pick(comments)
    if comment and comment != definition:
        definition = f"{definition} {comment}" if definition else comment

    alt_labels = sorted({v for vals in [_lang_values(g, subject, SKOS.altLabel)] for v in vals.values()})

    summary_text = build_summary_text(g, subject, label_index, pref_label, entity_types, definition)

    return {
        "_id": f"{SOURCE}:{local_name(subject)}",
        "source": SOURCE,
        "oso_id": local_name(subject),
        "entity_types": entity_types,
        "pref_label": pref_label,
        "pref_label_en": pref_labels.get("en"),
        "pref_label_fr": pref_labels.get("fr"),
        "alt_labels": alt_labels,
        "definition_en": definitions.get("en"),
        "definition_fr": definitions.get("fr"),
        "external_ids": external_ids,
        "summary_text": summary_text,
    }


def iter_records(path=None) -> Iterator[dict]:
    g = load.load_graph(path)
    label_index = build_label_index(g)
    for subject in get_individuals(g):
        yield build_record(g, subject, label_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview generated OSO summaries")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for i, rec in enumerate(iter_records()):
        if i >= args.limit:
            break
        print(json.dumps(rec, indent=2, ensure_ascii=False))
