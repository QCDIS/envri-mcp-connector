from kb_common.ratelimit import RateLimiter


def test_allow_under_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("caller") is True
    assert limiter.allow("caller") is True
    assert limiter.allow("caller") is True


def test_allow_blocks_over_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("caller") is True
    assert limiter.allow("caller") is True
    assert limiter.allow("caller") is False


def test_per_key_isolation():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("alice") is True
    assert limiter.allow("bob") is True
    assert limiter.allow("alice") is False
    assert limiter.allow("bob") is False


def test_window_expiry_allows_again(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    now = [1000.0]
    monkeypatch.setattr("kb_common.ratelimit.time.monotonic", lambda: now[0])

    assert limiter.allow("caller") is True
    assert limiter.allow("caller") is False

    now[0] += 10.1  # past the window
    assert limiter.allow("caller") is True
