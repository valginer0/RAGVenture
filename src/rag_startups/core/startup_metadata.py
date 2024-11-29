"""Startup metadata handling."""

from typing import Dict, List, Optional


class StartupLookup:
    """Class to handle startup metadata lookups."""

    def __init__(self, startup_data: List[Dict]):
        """Initialize with startup data."""
        self.startup_data = startup_data
        self._name_to_data = {
            startup["name"].lower(): startup
            for startup in startup_data
            if "name" in startup
        }

    def get_by_name(self, name: str) -> Optional[Dict]:
        """Get startup data by name."""
        return self._name_to_data.get(name.lower())

    def get_all_names(self) -> List[str]:
        """Get all startup names."""
        return list(self._name_to_data.keys())

    def get_by_description(self, description: str) -> Optional[Dict]:
        """Get startup data by matching long description."""
        if not description:
            return None

        description = description.lower()
        for startup in self.startup_data:
            if startup.get("long_desc", "").lower() == description:
                return startup
        return None
