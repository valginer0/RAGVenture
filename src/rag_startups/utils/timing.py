"""Timing decorator for performance monitoring."""
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of functions.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            f"{func.__name__} took {end_time - start_time:.2f} seconds"
        )
        return result
    return wrapper
