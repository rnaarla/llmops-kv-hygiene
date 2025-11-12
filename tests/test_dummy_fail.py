"""Dummy failing test to trigger AI triage and auto-fix."""


def test_intentional_failure():
    """This test is intentionally failing to trigger AI analysis.

    Expected behavior:
    - AI triage should detect this failure
    - AI should analyze the error
    - If confidence > 90%, auto-fix should create a PR
    
    FIX: Change the line below from 'actual = 41' to 'actual = 42'
    """
    expected = 42
    actual = 41  # BUG: This should be 42 to match expected value

    # This assertion will fail
    assert actual == expected, f"Expected {expected} but got {actual}"


def test_another_intentional_failure():
    """Another failing test with a clear fix.
    
    FIX: Change the assertion from '== 5' to '== 4'
    """
    result = calculate_sum(2, 2)
    # BUG: 2 + 2 equals 4, not 5
    assert result == 5, "2 + 2 should equal 5"  # Wrong: should be 4


def test_obvious_typo():
    """Test with an obvious typo in variable name.
    
    FIX: Change 'mesage' to 'message'
    """
    mesage = "Hello, World!"  # BUG: Typo - should be 'message'
    assert message == "Hello, World!"  # This will fail due to typo above


def calculate_sum(a: int, b: int) -> int:
    """Simple addition function."""
    return a + b
