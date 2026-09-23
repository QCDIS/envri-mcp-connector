from kb_common import timing


def test_stage_records_elapsed_inside_request():
    with timing.request() as stages:
        with timing.stage("es_query"):
            pass
        assert "es_query" in stages
        assert stages["es_query"] >= 0


def test_stage_accumulates_across_repeated_calls():
    with timing.request() as stages:
        with timing.stage("es_query"):
            pass
        with timing.stage("es_query"):
            pass
        assert stages["es_query"] >= 0


def test_record_adds_to_existing_stage_value():
    with timing.request() as stages:
        timing.record("es_took", 0.05)
        timing.record("es_took", 0.02)
        assert stages["es_took"] == 0.05 + 0.02


def test_stage_without_open_request_is_noop():
    # No timing.request() scope open - should not raise.
    with timing.stage("x"):
        pass
    timing.record("y", 1.0)


def test_request_scopes_are_isolated():
    with timing.request() as outer:
        timing.record("a", 1.0)
        with timing.request() as inner:
            timing.record("a", 2.0)
            assert inner["a"] == 2.0
        assert outer["a"] == 1.0
