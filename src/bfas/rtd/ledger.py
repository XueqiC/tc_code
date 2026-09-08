"""Hard reservations and append-only, request-level replay accounting.

The currency is historical output tokens (exact or explicitly estimated).
Input/reasoning/money remain separate usage fields; unknown is never zero.
A reservation uses a public cap, not the sealed response's actual usage.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from threading import RLock


class BudgetError(ValueError):
    pass


class LedgerError(ValueError):
    pass


def _tokens(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError("token amounts must be nonnegative integers")
    return value


class Ledger:
    def __init__(self, budget: int, path: Path | None = None):
        self.budget = _tokens(budget)
        self.path = Path(path) if path is not None else None
        self.reservations: dict[str, int] = {}
        self.charges: dict[str, int] = {}
        self.events: list[dict] = []
        self.window = None
        self.lock = RLock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size:
                raise LedgerError("existing ledger: use Ledger.resume; refusing overwrite")

    @property
    def spent(self):
        return sum(self.charges.values())

    @property
    def remaining(self):
        return self.budget - self.spent - sum(self.reservations.values())

    @property
    def owned_ids(self):
        return frozenset(self.charges)

    @property
    def window_remaining(self):
        if self.window is None:
            return self.remaining
        return self.window['budget'] - (self.spent - self.window['start_spend']) - sum(self.reservations.values())

    @property
    def window_purchases(self):
        return 0 if self.window is None else len(self.charges) - self.window['start_count']

    def open_window(self, window_id, budget, max_packages):
        with self.lock:
            budget, max_packages = _tokens(budget), _tokens(max_packages)
            if self.window is not None:
                if (self.window['window_id'], self.window['budget'], self.window['max_packages']) != (window_id, budget, max_packages):
                    raise LedgerError('conflicting open window')
                return
            if self.reservations or budget > self.remaining or not window_id:
                raise BudgetError('window exceeds unreserved authorization')
            if any(e['kind'] == 'window_open' and e['window_id'] == window_id for e in self.events):
                raise LedgerError('closed window cannot be reopened')
            self._append('window_open', None, window_id=window_id, budget=budget, max_packages=max_packages)
            self.window = dict(window_id=window_id, budget=budget, max_packages=max_packages,
                               start_spend=self.spent, start_count=len(self.charges))

    def close_window(self, window_id):
        with self.lock:
            if self.window is None:
                if any(e['kind'] == 'window_close' and e['window_id'] == window_id for e in self.events):
                    return
                raise LedgerError('no open window')
            if self.window['window_id'] != window_id or self.reservations:
                raise LedgerError('wrong window or unsettled reservations')
            self._append('window_close', None, window_id=window_id,
                         cost=self.spent-self.window['start_spend'], purchases=self.window_purchases)
            self.window = None

    def _append(self, kind, query_id, **values):
        event = dict(sequence=len(self.events), timestamp=datetime.now(timezone.utc).isoformat(),
                     kind=kind, query_id=query_id, **values)
        event['previous_hash'] = self.events[-1]['event_hash'] if self.events else None
        event['event_hash'] = self.event_hash(event)
        if self.path is not None:
            import os
            with self.path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.events.append(event)

    @staticmethod
    def event_hash(event):
        return hashlib.sha256(json.dumps({k: v for k, v in event.items() if k != 'event_hash'},
                                        sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def resume(cls, budget, path):
        """Validate the entire durable journal before admitting owned evidence.

        A torn final write is preserved separately, then removed. No complete
        event is discarded. Charges survive checkpoints that lag behind reveal.
        """
        path = Path(path)
        ledger = cls(budget)
        raw = path.read_bytes() if path.exists() else b''
        if raw and not raw.endswith(b'\n'):
            cut = raw.rfind(b'\n') + 1
            with path.with_suffix('.torn').open('ab') as stream:
                stream.write(raw[cut:] + b'\n')
            with path.open('r+b') as stream:
                stream.truncate(cut)
                import os
                stream.flush(); os.fsync(stream.fileno())
            raw = raw[:cut]
        for line in raw.splitlines():
            e = json.loads(line)
            previous = ledger.events[-1]['event_hash'] if ledger.events else None
            if (e['sequence'] != len(ledger.events) or e.get('previous_hash') != previous
                    or e.get('event_hash') != cls.event_hash(e)):
                raise LedgerError('ledger hash/sequence mismatch')
            q, kind = e['query_id'], e['kind']
            if kind == 'reserve':
                ledger.reserve(q, e['cap'])
            elif kind == 'release':
                if q not in ledger.reservations:
                    raise LedgerError('release without reservation')
                ledger.release(q)
            elif kind == 'reveal':
                if q in ledger.charges:
                    raise LedgerError('duplicate reveal')
                ledger.settle(q, e['cost'], confidence=e['confidence'], usage=e['usage'],
                              dependencies=e['dependencies'])
            elif kind == 'authorize':
                ledger.authorize(e['budget'])
            elif kind == 'window_open':
                ledger.open_window(e['window_id'], e['budget'], e['max_packages'])
            elif kind == 'window_close':
                if (ledger.window is None or e['cost'] != ledger.spent-ledger.window['start_spend']
                        or e['purchases'] != ledger.window_purchases):
                    raise LedgerError('window closing totals mismatch')
                ledger.close_window(e['window_id'])
            else:
                raise LedgerError('unknown ledger event')
            if len(ledger.events) != e['sequence'] + 1:
                raise LedgerError('duplicate/non-transition event')
            ledger.events[-1] = e
        ledger.path = path
        return ledger

    def authorize(self, budget):
        """Increase a cumulative checkpoint ceiling; never reset actual spend."""
        budget = _tokens(budget)
        if budget < self.budget:
            raise BudgetError('cumulative authorization cannot decrease')
        if budget != self.budget:
            self._append('authorize', None, budget=budget)
            self.budget = budget

    def reserve(self, query_id: str, cap: int):
        with self.lock:
            cap = _tokens(cap)
            if query_id in self.charges:
                return False
            if query_id in self.reservations:
                if self.reservations[query_id] != cap:
                    raise LedgerError("conflicting reservation")
                return False
            if cap > self.remaining:
                raise BudgetError("public cap exceeds remaining hard budget")
            if self.window is not None and (cap > self.window_remaining or
                    self.window_purchases + len(self.reservations) >= self.window['max_packages']):
                raise BudgetError('public cap or package count exceeds hard window budget')
            self._append("reserve", query_id, cap=cap)
            self.reservations[query_id] = cap
            return True

    def reserve_chain(self, caps: dict[str, int]):
        """Atomically check the entire unpaid dependency chain before reserving."""
        with self.lock:
            unpaid = {q: _tokens(c) for q, c in caps.items() if q not in self.charges}
            for q, c in unpaid.items():
                if q in self.reservations and self.reservations[q] != c:
                    raise LedgerError("conflicting dependency reservation")
            needed = sum(c for q, c in unpaid.items() if q not in self.reservations)
            if needed > self.remaining:
                raise BudgetError("request plus unpaid dependencies exceeds hard budget")
            count = len(set(unpaid) | set(self.reservations))
            if self.window is not None and (needed > self.window_remaining or
                    self.window_purchases + count > self.window['max_packages']):
                raise BudgetError('dependency chain exceeds hard window budget or package limit')
            for q, c in unpaid.items():
                self.reserve(q, c)

    def release(self, query_id):
        with self.lock:
            if query_id in self.reservations:
                self._append("release", query_id)
                del self.reservations[query_id]

    def settle(self, query_id, cost: int, *, confidence: str, usage: dict, dependencies=()):
        """One durable charge/reveal event per package, independent of event count."""
        with self.lock:
            cost = _tokens(cost)
            if confidence not in {"exact", "estimated"}:
                raise LedgerError("cost confidence required")
            if query_id in self.charges:
                if cost != self.charges[query_id]:
                    raise LedgerError("conflicting duplicate charge")
                return False
            if not set(dependencies) <= self.owned_ids:
                raise LedgerError("dependencies must be purchased first")
            if query_id not in self.reservations:
                raise LedgerError("charge without reservation")
            if cost > self.reservations[query_id]:
                raise LedgerError("sealed usage exceeds reserved upper bound; bank error")
            self._append("reveal", query_id, cost=cost, confidence=confidence,
                         usage=usage, dependencies=list(dependencies))
            self.charges[query_id] = cost
            del self.reservations[query_id]
            return True
