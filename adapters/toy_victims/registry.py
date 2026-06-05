from typing import Any, Dict, List, Optional, Type

from adapters.base_victim import BaseVictim


class VictimRegistry:
    """Singleton registry for toy victim classes.
    
    Allows registration and lookup of victim types by name,
    with optional default configurations.
    """

    _instance: Optional["VictimRegistry"] = None

    def __new__(cls) -> "VictimRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def register(
        self,
        name: str,
        victim_class: Type[BaseVictim],
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._registry[name] = {
            "class": victim_class,
            "default_config": default_config or {},
        }

    def get(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseVictim:
        entry = self._registry.get(name)
        if entry is None:
            raise KeyError(f"Unknown victim '{name}'")
        victim_cls = entry["class"]
        resolved_config = {**entry["default_config"], **(config or {})}
        try:
            return victim_cls(**resolved_config)
        except TypeError as e:
            raise TypeError(
                f"Cannot instantiate victim '{name}' with config {resolved_config}: {e}"
            )

    def list_victims(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name, entry in self._registry.items():
            results.append({
                "name": name,
                "class": entry["class"].__name__,
                "default_config": entry["default_config"],
            })
        results.sort(key=lambda x: x["name"])
        return results
