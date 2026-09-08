"""The selector's public interface. No broker, bank loader or payload imports.

Only these immutable values cross into a selector. JSON payloads live outside
Python's package tree. ``public_only`` also denies data reads and privileged
imports in the normal cooperative selector path; it is not an OS sandbox for
hostile Python. A future runner must isolate untrusted selector processes.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import builtins
import importlib
import math
import sys
from typing import Literal


@dataclass(frozen=True)
class PublicFeatures:
    projection: tuple[float, ...] | None = None
    source_logprob: float | None = None
    source_length: int | None = None

    def __post_init__(self):
        if self.projection is not None:
            object.__setattr__(self, "projection", tuple(self.projection))
            if len(self.projection) != 32 or not all(math.isfinite(x) for x in self.projection):
                raise ValueError("projection must have 32 finite frozen coordinates")
        if self.source_logprob is not None and (not math.isfinite(self.source_logprob) or self.source_logprob > 0):
            raise ValueError("source_logprob must be a full-sequence log probability")
        if self.source_length is not None and self.source_length < 1:
            raise ValueError("source length includes termination")


@dataclass(frozen=True)
class PublicQuerySpec:
    query_id: str
    state_hash: str
    features: PublicFeatures
    L: int
    cost_upper_bound: int
    cost_confidence: Literal["exact", "estimated"]
    cap_provenance: str

    def __post_init__(self):
        if not self.query_id or not self.state_hash or type(self.L) is not int or self.L < 1:
            raise ValueError("request identity, complete state and recorded L required")
        if type(self.features) is not PublicFeatures:
            raise ValueError("only pre-purchase PublicFeatures are accepted")
        if (type(self.cost_upper_bound) is not int or self.cost_upper_bound < 0
                or self.cost_confidence not in {"exact", "estimated"} or not self.cap_provenance):
            raise ValueError("public cap and cost confidence required")


@dataclass(frozen=True)
class StudentSnapshot:
    snapshot_id: str
    inner_parent_hashes: frozenset[str]
    # Only current-student/pre-purchase features, never teacher features.
    state_features: tuple[tuple[str, PublicFeatures], ...] = ()


_active = ContextVar("rtd_public_selector", default=None)


def _check_import(name, fromlist=()):
    trace = _active.get()
    parts = set(str(name).split(".")) | set(fromlist or ())
    if trace is not None and parts & {"broker", "bank", "ledger"}:
        trace.append({"kind": "denied_import", "resource": str(name)})
        raise PermissionError("selector cannot import privileged RTD modules")


def _audit(event, args):
    trace = _active.get()
    if trace is None:
        return
    if event == "import":
        _check_import(args[0])
    if event == "open":
        path, mode, flags = args
        # Imports may read code. Data, arbitrary descriptors and all writes are denied.
        readable_code = isinstance(path, (str, bytes)) and str(path).endswith((".py", ".pyc", ".so"))
        if not readable_code or (isinstance(mode, str) and any(c in mode for c in "wa+")) or flags & 3:
            trace.append({"kind": "denied_read", "resource": str(path)})
            raise PermissionError("selector cannot read payloads or source data")
    if event in {"os.listdir", "os.scandir", "subprocess.Popen", "os.system", "socket.connect"}:
        trace.append({"kind": "denied_access", "resource": event})
        raise PermissionError("selector only accepts public values")


sys.addaudithook(_audit)

@contextmanager
def public_only(trace: list[dict]):
    """Run a trusted selector on in-memory PublicQuerySpecs, tracing denials."""
    token = _active.set(trace)
    original_import, original_import_module = builtins.__import__, importlib.import_module

    # Import audit events alone miss already-cached modules. Both ordinary
    # import entrypoints are guarded for this synchronous selector invocation.
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        _check_import(name, fromlist)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        _check_import(name)
        return original_import_module(name, package)

    builtins.__import__, importlib.import_module = guarded_import, guarded_import_module
    try:
        yield
    finally:
        builtins.__import__, importlib.import_module = original_import, original_import_module
        _active.reset(token)


def select_public(candidates, choose):
    """Invoke a selector with public values only; None is the explicit replay choice.

    This is the T4 integration boundary, not an acquisition policy. Imports used
    by the policy should be initialized before entering the guarded data path.
    """
    candidates = tuple(candidates)
    if not all(type(c) is PublicQuerySpec for c in candidates):
        raise TypeError("selector receives only PublicQuerySpec values")
    trace = [{"kind": "public", "query_id": c.query_id,
              "fields": list(PublicQuerySpec.__dataclass_fields__)} for c in candidates]
    with public_only(trace):
        selected = choose(candidates)
    if selected is not None and selected not in {c.query_id for c in candidates}:
        raise ValueError("selector chose an unavailable request")
    return selected, trace


def select_public_batch(candidates, choose):
    """The same guarded boundary for a finite set, including the empty set."""
    candidates = tuple(candidates)
    if not all(type(c) is PublicQuerySpec for c in candidates):
        raise TypeError('selector receives only PublicQuerySpec values')
    trace = [dict(kind='public', query_id=c.query_id, fields=list(PublicQuerySpec.__dataclass_fields__))
             for c in candidates]
    with public_only(trace):
        selected = tuple(choose(candidates))
    if len(set(selected)) != len(selected) or not set(selected) <= {c.query_id for c in candidates}:
        raise ValueError('selector chose duplicate or unavailable requests')
    return selected, trace
