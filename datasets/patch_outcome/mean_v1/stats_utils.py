"""Small statistics fixture with an intentionally lossy mean implementation."""


def mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty sequence."""
    return sum(values) // len(values)
