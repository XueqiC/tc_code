"""Benchmark adapters shipped with BFAS."""

from importlib import import_module
from typing import Any


__all__ = ["ALFWorldAdapter", "AppWorldAdapter", "BFCLAdapter", "Tau2Adapter"]

_MODULES = {
    "ALFWorldAdapter": ".alfworld",
    "AppWorldAdapter": ".appworld",
    "BFCLAdapter": ".bfcl",
    "Tau2Adapter": ".tau2_stub",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
