"""Legacy compatibility shim package `config`.

Older modules import constants via plain `config.config`.  This lightweight
package proxies those imports to the new implementation inside
`rag_startups.config` so that existing code keeps working without changes.
"""

from __future__ import annotations

import importlib
import sys as _sys

# Import actual implementation
_cfg_mod = importlib.import_module("rag_startups.config.config")

# Expose as submodule so that `import config.config` works
_sys.modules.setdefault("config.config", _cfg_mod)

# Re-export public names of the real config module
from rag_startups.config.config import *  # type: ignore # noqa: F401,F403,E402

__all__ = getattr(_cfg_mod, "__all__", [])
