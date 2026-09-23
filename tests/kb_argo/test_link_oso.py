from kb_argo import link_oso


def test_normalize_lowercases_and_strips_punctuation():
    assert link_oso.normalize("IFREMER - French Research Institute!") == "ifremer french research institute"


def test_normalize_collapses_whitespace():
    assert link_oso.normalize("  Ifremer   Org  ") == "ifremer org"


def test_normalize_empty_input():
    assert link_oso.normalize("") == ""
    assert link_oso.normalize(None) == ""


def test_match_organization_exact_match():
    index = {"ifremer": "oso:ifremer"}
    assert link_oso.match_organization("Ifremer", index=index) == "oso:ifremer"


def test_match_organization_substring_match_candidate_contains_label():
    index = {"french research institute": "oso:ifremer"}
    result = link_oso.match_organization("Ifremer - French Research Institute", index=index)
    assert result == "oso:ifremer"


def test_match_organization_substring_match_label_contains_candidate():
    index = {"french research institute for the exploitation of the sea ifremer": "oso:ifremer"}
    result = link_oso.match_organization("Ifremer", index=index)
    assert result == "oso:ifremer"


def test_match_organization_tries_each_candidate_in_order():
    index = {"ifremer": "oso:ifremer"}
    result = link_oso.match_organization("Unknown Co", None, "Ifremer", index=index)
    assert result == "oso:ifremer"


def test_match_organization_too_short_candidate_is_skipped():
    index = {"abc": "oso:short"}  # below _MIN_MATCH_LEN, would never be indexed for real
    assert link_oso.match_organization("abc", index=index) is None


def test_match_organization_no_match_returns_none():
    index = {"ifremer": "oso:ifremer"}
    assert link_oso.match_organization("Completely Different Name", index=index) is None
