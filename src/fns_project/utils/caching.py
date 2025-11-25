"""Simple caching utilities."""

from functools import lru_cache


def memoize(maxsize: int = 128):
    """Decorator for caching function results."""
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)
        return cached_func
    return decorator
