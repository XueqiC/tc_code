"""Privileged sealed-cache broker. Selectors import selector.py only.

A bank has public/request metadata and a separate sealed/ JSON directory, never
inside src or on sys.path. Ledger banks buy individual teacher attempts, including
failures, in one seed-zero order across all parents. Public costs are reservation
bounds. Training eligibility remains subject to the active parent fold.
No teacher API exists on this path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys

from ..cc_pairs import digest
from .ledger import BudgetError, Ledger, LedgerError
from .selector import PublicFeatures, PublicQuerySpec, StudentSnapshot
from .transport import Behavior, FullState


class UnavailableError(ValueError):
    pass


@dataclass(frozen=True)
class RequestRecord:
    spec: PublicQuerySpec
    parent_hash: str
    dependencies: tuple[str, ...] = ()
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class PurchasedEvidencePackage:
    query_id: str
    dependencies: tuple[str, ...]
    behaviors: tuple[Behavior, ...]
    cost: int
    cost_confidence: str
    usage: dict
    provenance: dict
    historical_response: object
    historical_events: tuple[dict, ...] = ()


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    violations: tuple[str, ...]
    reveals: tuple[str, ...]


def _write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def seal_bank(directory, records: list[RequestRecord], payloads: dict[str, dict]):
    """Offline ingestion only; the selector never receives this callable.

    Payload hashes belong to the privileged integrity index, not selector features.
    Original costs/provenance remain archived in payload files. Certified state
    replay may disclose the recorded cost as a runtime reservation bound.
    """
    directory = Path(directory).resolve()
    source = Path(__file__).resolve().parents[2]
    if directory.is_relative_to(source) or directory in map(lambda p: Path(p or '.').resolve(), sys.path):
        raise ValueError("sealed bank must be outside the Python source/import tree")
    by_id = {r.spec.query_id: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(payloads):
        raise ValueError("each real request must have exactly one record and payload")
    for query_id, record in by_id.items():
        if not re.fullmatch(r"[a-f0-9]{64}", query_id):
            raise ValueError("request ids must be opaque SHA256 identifiers")
        if query_id in record.dependencies or not set(record.dependencies) <= by_id.keys():
            raise ValueError("missing/self request dependency")
    visiting, done = set(), set()
    def visit(q):
        if q in visiting:
            raise ValueError("cyclic request dependencies")
        if q not in done:
            visiting.add(q)
            for dep in by_id[q].dependencies:
                visit(dep)
            visiting.remove(q)
            done.add(q)
    for q in by_id:
        visit(q)
    # Refuse overwriting either an existing bank or a partial failed build.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "public").mkdir()
    (directory / "sealed").mkdir(mode=0o700)
    checksums = {}
    for q, payload in payloads.items():
        _write_new(directory / "sealed" / (q + ".json"), payload)
        checksums[q] = digest(payload)
    _write_new(directory / "public" / "requests.json", [asdict(r) for r in records])
    _write_new(directory / "sealed" / "integrity.json", checksums)
    return directory


class SealedReplayBroker:
    def __init__(self, directory, ledger: Ledger, *, inner_parent_hashes):
        self.directory, self.ledger = Path(directory).resolve(), ledger
        self.inner_parent_hashes = frozenset(inner_parent_hashes)
        self._records = {}
        for raw in json.loads((self.directory / "public/requests.json").read_text()):
            fields = raw["spec"]
            features = fields.pop("features")
            if features["projection"] is not None:
                features["projection"] = tuple(features["projection"])
            spec = PublicQuerySpec(**fields, features=PublicFeatures(**features))
            if not re.fullmatch(r"[a-f0-9]{64}", spec.query_id) or spec.query_id in self._records:
                raise ValueError("invalid/duplicate request identity")
            self._records[spec.query_id] = RequestRecord(spec, raw["parent_hash"],
                                                       tuple(raw["dependencies"]), raw["unavailable_reason"])
        self._integrity = json.loads((self.directory / "sealed/integrity.json").read_text())
        self.reservation_basis = 'public_class_cap'
        self.attempt_packages = {}
        self.purchase_order = ()
        self._load_recorded_reservations()
        if self.attempt_packages and any(q not in self.attempt_packages or cost != self.attempt_packages[q]['tokens']
                                      for q, cost in ledger.charges.items()):
            raise LedgerError('restored ledger uses different attempt purchase accounting')
        if self.attempt_packages:
            revealed = [e['query_id'] for e in ledger.events if e['kind'] == 'reveal']
            if revealed != list(self.purchase_order[:len(revealed)]):
                raise LedgerError('restored ledger is not a frozen attempt prefix')
        self._purchased: dict[str, PurchasedEvidencePackage] = {}
        self._offered: set[str] = set()
        self._ever_offered: set[str] = set()

    def _load_recorded_reservations(self):
        """Use ledger costs for certified, already-paid benchmark packages.

        Keep the archived certificate/class caps intact. Only the runtime public
        bound changes, so candidate filtering, cost prediction and reservation
        agree. Estimated historical usage stays estimated: the reservation is
        exact with respect to the recorded ledger, not the provider bill.
        Legacy banks and WebShop retain their public-class-cap convention.
        """
        from .bank_v11 import CERTIFICATE
        certificate = self.directory / CERTIFICATE
        if not certificate.exists():
            return
        core = json.loads(certificate.read_text())['core']
        if (core['version'] != 'rtd-v1.1.0-state-bank'
                or core['benchmark'] not in {'hotpotqa', 'alfworld', 'bfcl'}):
            return
        from .bank_build import validate_state_certificate
        validate_state_certificate(self.directory)
        from .task_packages import ATTEMPT_LEDGER, PURCHASE_BASIS, attempt_packages, frozen_attempt_order
        if ATTEMPT_LEDGER in core['public_artifacts']:
            summary = json.loads((self.directory / 'public' / ATTEMPT_LEDGER).read_text())
            self.attempt_packages = attempt_packages(summary, self._records)
            self.purchase_order = tuple(frozen_attempt_order(self._records))
            for q, attempt in self.attempt_packages.items():
                record = self._records[q]
                provenance = json.loads(record.spec.cap_provenance)
                provenance.update(basis=PURCHASE_BASIS, archived_class_cap=record.spec.cost_upper_bound,
                                  output_token_cap=attempt['tokens'], purchase_unit='teacher_attempt')
                self._records[q] = replace(record, spec=replace(record.spec,
                    cost_upper_bound=attempt['tokens'],
                    cap_provenance=json.dumps(provenance, sort_keys=True)))
            self.reservation_basis = PURCHASE_BASIS
            return
        total = 0
        for q, record in self._records.items():
            if record.unavailable_reason is not None:
                continue
            payload = self._read_payload(q)
            cost = payload['cost']
            if (type(cost) is not int or not 0 <= cost <= record.spec.cost_upper_bound
                    or payload['cost_confidence'] != record.spec.cost_confidence):
                raise LedgerError('invalid recorded package cost/confidence')
            total += cost
            provenance = json.loads(record.spec.cap_provenance)
            provenance.update(basis='recorded_package_cost',
                archived_class_cap=record.spec.cost_upper_bound, output_token_cap=cost)
            self._records[q] = replace(record, spec=replace(record.spec,
                cost_upper_bound=cost, cap_provenance=json.dumps(provenance, sort_keys=True)))
        if total != core['budget_denominator']:
            raise LedgerError('recorded package costs differ from certified denominator')
        self.reservation_basis = 'recorded_package_cost'

    def _read_payload(self, query_id):
        payload = json.loads((self.directory / 'sealed' / (query_id + '.json')).read_text())
        if digest(payload) != self._integrity[query_id]:
            raise LedgerError('sealed payload integrity failure')
        return payload

    def _read_attempt_payload(self, query_id):
        from .task_packages import attempt_summary
        attempt = self.attempt_packages[query_id]
        payload = self._read_payload(query_id)
        summary = attempt_summary(payload['historical_response'], confidence=payload['cost_confidence'])
        if (any(summary[k] != attempt[k] for k in summary)
                or payload['cost'] != attempt['tokens']
                or payload['provenance']['task_id'] != attempt['task_id']
                or len(payload.get('historical_attempts', [payload['historical_response']])) != 1):
            raise LedgerError('attempt ledger changed on reveal')
        return payload

    def frozen_unowned(self):
        """One global purchase prefix, independent of training seed and fold."""
        return [q for q in self.purchase_order if q not in self.ledger.owned_ids]

    def training_eligible(self, query_id):
        record = self._records[query_id]
        return record.unavailable_reason is None and record.parent_hash in self.inner_parent_hashes

    def set_inner_parents(self, parent_hashes):
        """Rotate folds before constructing features, candidates and D_inner."""
        self.inner_parent_hashes = frozenset(parent_hashes)
        self._offered.clear()

    def _legal(self, record):
        if self.attempt_packages:
            return True  # Paying for an attempt does not make its parent trainable.
        return (record.unavailable_reason is None and record.parent_hash in self.inner_parent_hashes
                and all(self._records[d].parent_hash in self.inner_parent_hashes
                        and self._records[d].unavailable_reason is None for d in record.dependencies))

    def list_candidates(self, student_snapshot: StudentSnapshot, owned_ids, remaining_budget):
        with self.ledger.lock:
            if frozenset(owned_ids) != self.ledger.owned_ids:
                raise LedgerError("caller ownership disagrees with authoritative ledger")
            if student_snapshot.inner_parent_hashes != self.inner_parent_hashes:
                raise ValueError("candidate snapshot uses a different inner fold")
            if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int) or remaining_budget < 0:
                raise BudgetError("remaining budget must be a nonnegative token count")
            budget = min(remaining_budget, self.ledger.remaining, self.ledger.window_remaining)
            features = dict(student_snapshot.state_features)
            candidates = []
            order = ((q, self._records[q]) for q in self.purchase_order) if self.attempt_packages else sorted(self._records.items())
            prefix_cost = 0
            for q, record in order:
                # Even state hashes/features of locked prefix requests are withheld.
                if (q in owned_ids or not self._legal(record)
                        or (self.ledger.window is not None and self.ledger.window_purchases +
                            len(self.ledger.reservations) >= self.ledger.window['max_packages'])
                        or not set(record.dependencies) <= self.ledger.owned_ids):
                    continue
                if record.spec.cost_upper_bound + prefix_cost > budget:
                    if self.attempt_packages:
                        break
                    continue
                spec = replace(record.spec, features=features.get(record.spec.state_hash, PublicFeatures()))
                candidates.append(spec)
                if self.attempt_packages:
                    prefix_cost += spec.cost_upper_bound
                    if self.ledger.window is not None and len(candidates) + self.ledger.window_purchases >= self.ledger.window['max_packages']:
                        break
            self._offered = {s.query_id for s in candidates}
            self._ever_offered.update(self._offered)
            return candidates

    def acquire(self, query_id):
        with self.ledger.lock:
            record = self._records.get(query_id)
            if record is None or not self._legal(record):
                raise UnavailableError("request unavailable in this inner fold")
            if query_id in self._purchased:
                return self._purchased[query_id]
            if not set(record.dependencies) <= self.ledger.owned_ids:
                raise UnavailableError("purchase prefix dependencies first")
            if query_id not in self._offered and query_id not in self.ledger.owned_ids:
                raise UnavailableError("request was not offered in the current candidate view")
            if self.attempt_packages and query_id not in self.ledger.owned_ids:
                if query_id != self.frozen_unowned()[0]:
                    raise UnavailableError('purchase must follow the frozen attempt order')
            self.ledger.reserve(query_id, record.spec.cost_upper_bound)
            try:
                payload = self._read_attempt_payload(query_id) if self.attempt_packages else self._read_payload(query_id)
                if (self.reservation_basis == 'recorded_package_cost'
                        and payload['cost'] != record.spec.cost_upper_bound):
                    raise LedgerError('recorded package cost changed on reveal')
                if payload["cost_confidence"] != record.spec.cost_confidence:
                    raise LedgerError("cost confidence changed on reveal")
                behaviors = tuple(Behavior(FullState(**r["state"]), r["text"]) for r in payload["behaviors"])
                if ((not behaviors and (not self.attempt_packages or self.attempt_packages[query_id]['usable']))
                        or any(b.state.parent_hash != record.parent_hash for b in behaviors)):
                    raise ValueError("missing/full-state parent mismatch in sealed behaviors")
                if self.attempt_packages:
                    if not self.attempt_packages[query_id]['usable']:
                        behaviors = ()
                    elif (payload['historical_response'].get('verified') is False
                          or payload.get('success') is False):
                        raise ValueError('unverified positive package')
                package = PurchasedEvidencePackage(query_id, record.dependencies, behaviors, payload["cost"],
                    payload["cost_confidence"], payload["usage"], payload["provenance"], payload["historical_response"],
                    tuple(payload.get("historical_events", ())))
                if query_id in self.ledger.owned_ids:
                    saved = next(e for e in self.ledger.events if e['kind'] == 'reveal' and e['query_id'] == query_id)
                    if (saved['cost'] != package.cost or saved['usage'] != package.usage or
                            saved['confidence'] != package.cost_confidence or
                            saved['dependencies'] != list(record.dependencies)):
                        raise LedgerError('restored reveal disagrees with sealed package')
                self.ledger.settle(query_id, package.cost, confidence=package.cost_confidence,
                                   usage=package.usage, dependencies=record.dependencies)
            except Exception:
                self.ledger.release(query_id)
                raise
            self._purchased[query_id] = package
            return package

    def assert_no_hidden_access(self, selector_trace):
        """Check cooperative-selector instrumentation against reveal order.

        ``kind=public`` is a PublicQuerySpec access. ``kind=purchased`` must carry
        an audit sequence AFTER that request's reveal. Unknown/denied reads fail.
        The actual selector boundary is selector.public_only, not a claimed trace.
        """
        violations = []
        revealed = {e["query_id"]: e["sequence"] for e in self.ledger.events if e["kind"] == "reveal"}
        fields = set(PublicQuerySpec.__dataclass_fields__)
        for event in selector_trace:
            kind, q = event.get("kind"), event.get("query_id")
            if kind == "public" and set(event.get("fields", ())) <= fields and q in self._ever_offered:
                continue
            if kind == "purchased" and q in revealed and event.get("sequence", -1) > revealed[q]:
                continue
            violations.append(f"hidden/unknown selector access: {event}")
        return AuditResult(not violations, tuple(violations), tuple(revealed))
