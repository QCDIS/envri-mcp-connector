"""Load the OSO ontology into an rdflib graph."""
import rdflib

from kb_oso import config


def load_graph(path=None) -> rdflib.Graph:
    """Loads the OSO ontology into an rdflib graph."""

    path = path or config.OSO_OWL_PATH
    g = rdflib.Graph()
    g.parse(str(path), format="xml")
    return g
