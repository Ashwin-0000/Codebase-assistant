"""
Sample Python module for testing the ingestion pipeline.

This file is intentionally simple but covers the cases the parser must handle:
  - Module-level function with a docstring
  - Module-level function without a docstring
  - A class with a class docstring and two methods
  - A nested function inside a method
  - A decorated function
"""

import os
import sys
from typing import Optional

# Module-level constant (not a function — should NOT produce a chunk)
MAX_RETRIES = 3


def greet(name: str) -> str:
    """Return a personalised greeting.

    Args:
        name: The person's name.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    # No docstring — chunker should note its absence
    return a + b


def retry(func):
    """Decorator that retries *func* up to MAX_RETRIES times."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
    return wrapper


@retry
def fetch_data(url: str) -> Optional[str]:
    """Fetch data from *url* and return the response body."""
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


class Calculator:
    """A simple stateful calculator.

    Keeps a running total that can be reset.
    """

    def __init__(self, initial: int = 0) -> None:
        self.total = initial

    def add(self, value: int) -> "Calculator":
        """Add *value* to the running total and return self (fluent API)."""
        self.total += value
        return self

    def reset(self) -> None:
        # Reset the running total to zero — no docstring
        self.total = 0

    def _compute_internal(self, x: int, y: int) -> int:
        """Private helper with a nested function to test deep extraction."""

        def _clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, v))

        return _clamp(x + y, 0, 1000)


def main() -> None:
    """Entry point that exercises the other functions in this module.

    Used by the call-graph tests to verify edge extraction.
    Calls: greet, add (top-level), Calculator (constructor).
    """
    message = greet("World")
    total = add(1, 2)
    calc = Calculator(initial=total)
    calc.add(10)
    print(message, calc.total)
