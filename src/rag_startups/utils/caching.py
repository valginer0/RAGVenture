"""Caching utilities for external API calls."""

import json
import logging
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, Union

import redis
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to connect to Redis, fall back to in-memory cache if unavailable
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )
    redis_client.ping()
    logger.info("Using Redis for caching")
    USING_REDIS = True
except (redis.ConnectionError, redis.ResponseError):
    logger.warning("Redis not available, using in-memory cache")
    USING_REDIS = False
    # Fallback to in-memory cache (1000 items, 24 hour TTL)
    memory_cache = TTLCache(maxsize=1000, ttl=24 * 60 * 60)


def cache_result(
    prefix: str,
    ttl: int = 24 * 60 * 60,  # 24 hours in seconds
    key_generator: Optional[Callable] = None,
) -> Callable:
    """Cache decorator for function results.

    Args:
        prefix: Prefix for cache key
        ttl: Time to live in seconds
        key_generator: Optional function to generate cache key

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = f"{prefix}:{key_generator(*args, **kwargs)}"
            else:
                # Default key from args and kwargs
                key_parts = [str(arg) for arg in args]
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{prefix}:{':'.join(key_parts)}"

            # Try to get from cache
            if USING_REDIS:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            else:
                if cache_key in memory_cache:
                    return memory_cache[cache_key]

            # Execute function if not cached
            result = func(*args, **kwargs)

            # Cache the result
            try:
                if USING_REDIS:
                    redis_client.setex(cache_key, ttl, json.dumps(result))
                else:
                    memory_cache[cache_key] = result
            except Exception as e:
                logger.error(f"Error caching result: {e}")

            return result

        return wrapper

    return decorator


def clear_cache(prefix: Optional[str] = None) -> None:
    """Clear cache entries.

    Args:
        prefix: Optional prefix to clear specific entries
    """
    try:
        if USING_REDIS:
            if prefix:
                keys = redis_client.keys(f"{prefix}:*")
                if keys:
                    redis_client.delete(*keys)
            else:
                redis_client.flushdb()
        else:
            if prefix:
                keys_to_delete = [
                    k for k in memory_cache.keys() if k.startswith(f"{prefix}:")
                ]
                for k in keys_to_delete:
                    del memory_cache[k]
            else:
                memory_cache.clear()
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    try:
        if USING_REDIS:
            info = redis_client.info()
            return {
                "type": "redis",
                "keys": redis_client.dbsize(),
                "used_memory": info["used_memory_human"],
                "hits": info["keyspace_hits"],
                "misses": info["keyspace_misses"],
            }
        else:
            return {
                "type": "memory",
                "keys": len(memory_cache),
                "max_size": memory_cache.maxsize,
                "currsize": memory_cache.currsize,
                "ttl": memory_cache.ttl,
            }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"error": str(e)}
