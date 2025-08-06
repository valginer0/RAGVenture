"""Shim to maintain backward compatibility.

This thin wrapper allows `import embed_master` to continue working after the
module was moved inside the `rag_startups` package. It simply re-exports
all public symbols from `rag_startups.embed_master`.
"""

from importlib import import_module as _import_module

_mod = _import_module("rag_startups.embed_master")

# Re-export everything except private names
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("_")})

__all__ = [k for k in globals().keys() if not k.startswith("_")]
