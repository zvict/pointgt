from __future__ import annotations

from typing import Any, Iterator, Mapping


class ConfigNode(Mapping):
    """Read-only mapping addressable by attribute, with path-aware errors."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: Mapping[str, Any], path: str = "") -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name not in data:
            where = object.__getattribute__(self, "_path")
            dotted = f"{where}.{name}" if where else name
            raise AttributeError(f"config has no key {dotted!r}")
        return data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("ConfigNode is read-only; edit the mapping before wrapping it")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Recursively unwrap back into plain dicts."""
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, ConfigNode) else v for v in value]
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        where = object.__getattribute__(self, "_path") or "<root>"
        return f"ConfigNode({where}, keys={sorted(self._data)})"


def to_config_node(value: Any, path: str = "") -> Any:
    """Recursively wrap mappings, threading the dotted path for error messages."""
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, Mapping):
        wrapped = {
            key: to_config_node(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
        return ConfigNode(wrapped, path)
    if isinstance(value, list):
        return [to_config_node(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return value
