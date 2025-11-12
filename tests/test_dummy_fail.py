"""Dummy failing test to trigger AI triage and auto-fix."""


def test_intentional_failure():
    """This test is intentionally failing to trigger AI analysis.

    Expected behavior:
    - AI triage should detect this failure
    - AI should analyze the error
    - If confidence > 90%, auto-fix should create a PR
    """
    expected = 42
    actual = 41  # Intentionally wrong value

    # This assertion will fail
    assert actual == expected, f"Expected {expected} but got {actual}"


def test_another_intentional_failure():
    """Another failing test with a clear fix."""
    result = calculate_sum(2, 2)
    assert result == 5, "2 + 2 should equal 5"  # Wrong expectation


def calculate_sum(a: int, b: int) -> int:
    """Simple addition function."""
    return a + b
