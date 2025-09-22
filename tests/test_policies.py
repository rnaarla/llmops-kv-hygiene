import time
from tools.policies import evaluate_policies, check_ttl, check_reuse


def test_check_ttl_immediate_violation():
    created = time.time() - 10
    assert check_ttl(created, created, 0) is True
    assert check_ttl(created, None, 0) is True


def test_check_ttl_expired_after_use():
    created = time.time() - 100
    first_use = created + 1
    assert check_ttl(created, first_use, 10) is True


def test_check_ttl_not_expired():
    created = time.time() - 5
    first_use = created + 1
    assert check_ttl(created, first_use, 60) is False
    # Pre-first-use and not expired
    created2 = time.time() - 5
    assert check_ttl(created2, None, 60) is False


def test_check_reuse_over_limit():
    assert check_reuse(3, 2) is True
    assert check_reuse(2, 2) is False
    # Negative max_reuse treated as unlimited -> no violation
    assert check_reuse(10, -1) is False


def test_evaluate_combined():
    created = time.time() - 100
    first_use = created + 1
    decision = evaluate_policies(created_ts=created, first_use_ts=first_use, reuse_count=5, ttl_sec=10, max_reuse=2)
    assert decision.ttl_violation is True
    assert decision.reuse_violation is True
    assert decision.any() is True


def test_evaluate_none():
    now = time.time()
    decision = evaluate_policies(created_ts=now, first_use_ts=now, reuse_count=0, ttl_sec=1000, max_reuse=5)
    assert decision.any() is False
