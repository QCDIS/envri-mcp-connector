from datetime import date

import pytest

from kb_common import filters


def test_status_values_inactive_does_not_include_active_codes():
    values = filters.status_values("inactive")
    assert {"I", "INACTIVE", "inactive"} <= set(values)
    assert not {"A", "ACTIVE", "O", "OPERATIONAL"} & set(values)


def test_status_values_is_case_and_whitespace_insensitive():
    assert filters.status_values("  ACTIVE ") == filters.status_values("active")


def test_status_values_unknown_passes_through_and_deduplicates():
    assert filters.status_values("X") == ["X", "x"]


def test_bare_year_bounds_cover_whole_years():
    clause = filters.date_range_clause("deployment_date", "2020", "2021")
    assert clause == {
        "range": {"deployment_date": {"format": "strict_date_optional_time", "gte": "2020", "lte": "2021"}}
    }


def test_date_objects_are_serialized():
    clause = filters.date_range_clause("deployment_date", date(2020, 6, 15))
    assert clause["range"]["deployment_date"]["gte"] == "2020-06-15"


def test_same_year_for_both_bounds_is_valid_and_means_that_year():
    filters.date_range_clause("deployment_date", "2015", "2015")


def test_month_precision_bounds_compare_by_period():
    filters.date_range_clause("deployment_date", "2020-03", "2020-03-01")
    with pytest.raises(ValueError):
        filters.date_range_clause("deployment_date", "2020-04", "2020-03-31")


def test_inverted_range_raises():
    with pytest.raises(ValueError):
        filters.date_range_clause("deployment_date", "2022", "2020")


@pytest.mark.parametrize("bad", ["last spring", "20", "2020-1", "2020-13-01x", "2020/01/01"])
def test_malformed_date_raises(bad):
    with pytest.raises(ValueError):
        filters.date_range_clause("deployment_date", bad)


def test_build_filters_deployment_window_targets_deployment_date():
    filters_ = filters.build_filters(source="euro_argo", status="inactive", deployed_after="2018", deployed_before="2019-06")
    assert filters_[0] == {"term": {"source": "euro_argo"}}
    assert filters_[1]["terms"]["status_code"][0] == "I"
    assert filters_[2]["range"]["deployment_date"]["gte"] == "2018"
    assert filters_[2]["range"]["deployment_date"]["lte"] == "2019-06"


def test_deployment_and_last_cycle_ranges_are_independent():
    result = filters.build_filters(date_min="2024-01-01", deployed_before="2020")
    assert [next(iter(c["range"])) for c in result] == ["last_cycle_date", "deployment_date"]


def test_build_filters_empty():
    assert filters.build_filters() == []
