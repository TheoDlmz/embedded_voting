import pytest


def test_division():
    assert 10 / 2 == 5


def test_division_bis():
    """
    >>> 10 / 2
    5.0
    """
    pass


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        10 / 0


def test_division_by_zero_bis():
    """
        >>> 10 / 0
        Traceback (most recent call last):
        ZeroDivisionError: division by zero
    """
    pass
