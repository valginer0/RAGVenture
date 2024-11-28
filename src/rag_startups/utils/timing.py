"""Timing decorator for performance monitoring."""
import functools
import time
import logging
from ..utils.output_formatter import formatter
from typing import Any, Callable

logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of functions.
    
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
        duration = end_time - start_time
        
        # Add timing to our formatter
        formatter.add_timing(func.__name__, duration)
        
        return result
    return wrapper
