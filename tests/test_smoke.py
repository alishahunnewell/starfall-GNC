"""Smoke tests: minimal checks that nothing catastrophic is broken."""


def test_import():
    """Verify that the starfall package can be imported."""
    import starfall
    assert starfall is not None


def test_basic_math():
    """If this fails, your computer has bigger problems than my code."""
    assert 1 + 1 == 2


def test_numpy_works():
    """Verify numpy is installed and usable in the test environment."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6