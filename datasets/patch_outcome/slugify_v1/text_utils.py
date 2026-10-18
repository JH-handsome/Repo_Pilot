"""Small text helper fixture with an intentionally incomplete slug implementation."""


def slugify(value: str) -> str:
    """Convert a label into a lowercase dash-separated slug."""
    return value.strip().lower().replace(" ", "-")
