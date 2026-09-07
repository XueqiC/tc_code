"""Privileged CPU archive inventory for C26-A; no environment or teacher calls.

The public BFCL RequestRecord/PublicQuerySpec format is retained. All records
have the same pending-state reason until C26-B, including failures; success,
commands, legacy cost, event correspondence and integrity stay in sealed/.
A reset descriptor is NOT a FullState or a reconstructed environment observation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from ..broker import RequestRecord, seal_bank
from ..selector import PublicFeatures, PublicQuerySpec
from .. import selector
from ...cc_pairs import digest
from .alfworld_caps import CapConfiguration, cap_audit, public_cap
from .alfworld_state import canonical_hash, parent_fold, parent_hash, validate_request


LEDGER = "data/teacher_ledger/alfworld.jsonl"
SUPPORT = "results/bfas/alfworld/ours_s0/support_split.json"
PROBE = "data/alf_sft/probe_alf_base_correct_v1.jsonl"
CACHE = "results/bfas/alfworld/collect_shared/demos.json"
TOKENIZER_SNAPSHOT = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PENDING = "C26-B state/support audit pending"
EXPECTED = dict(attempts=219, task_ids=142, successful_attempts=107, failed_attempts=112,
                candidate_packages=107, command_turns=1377, legacy_token_estimate=189541,
                successful_legacy_token_estimate=36294, failed_legacy_token_estimate=153247,
                retained_command_tokens=8327, retained_prompt_tokens=672908)
LIMITATIONS = [
    "C26-A: zero environment replays; reset descriptors are requests, not verified FullState observations.",
    "107 is a historical candidate upper bound, not a usable inventory; C26-B must freeze support/world/environment identity.",
    "Survivorship bias: only successful attempts retained command payloads; failed responses are missing.",
    "payload_kind=extracted_teacher_commands, not original API response/reasoning; deployment prompts differ from teacher requests.",
    "Legacy tokens_spent uses sum(ceil(len(response)/4)); tokenizer recount measures retained strings only and never replaces cost.",
    "Input, reasoning, discarded output, complete retries and monetary costs unknown, not zero.",
    "L=40 and public caps are current-code retrospective conventions, not recorded-only request envelopes or provider guarantees.",
    "Event thought+ACTION is a derived transformation, not raw teacher output; r2 carries CE checkpoint/evidence dependencies and stays diagnostic.",
    "Proposed parent folds/exclusions are an audit, not the C26-B frozen support manifest; valid_seen is a development endpoint.",
    "Selector isolation is cooperative, not an OS sandbox; production import-guard integration is deferred to C26-F.",
]


def _privileged():
    # Existing selector import guard does not name alfworld_bank. Deny calls
    # through cached references too, without modifying/monkeypatching that guard.
    trace = selector._active.get()
    if trace is not None:
        trace.append(dict(kind="denied_access", resource="alfworld_bank"))
        raise PermissionError("selector cannot access privileged ALFWorld archive")


def file_hash(path):
    _privileged()
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def query_id(ledger_sha256, row, line):
    if not re.fullmatch(r"[a-f0-9]{64}", ledger_sha256) or type(line) is not int or line < 1:
        raise ValueError("ledger hash and physical line number required")
    parent_hash(row["task_id"])
    if type(row["attempt_index"]) is not int or row["attempt_index"] < 0 or not row["timestamp"]:
        raise ValueError("task/attempt/timestamp request identity required")
    return canonical_hash(dict(kind="alf_demo_episode", ledger_sha256=ledger_sha256,
                               task_id=row["task_id"], attempt_index=row["attempt_index"],
                               timestamp=row["timestamp"], line=line))


def public_record(q, request, *, configuration=CapConfiguration()):
    """No hidden argument: fixed class, declared L, independent reset request."""
    validate_request(request)
    if request["max_episode_steps"] != configuration.max_episode_steps:
        raise ValueError("reset horizon differs from public configuration")
    cap, provenance = public_cap("alf_demo_episode", configuration=configuration)
    return RequestRecord(PublicQuerySpec(q, canonical_hash(request), PublicFeatures(),
                                        configuration.max_episode_steps, cap, "estimated", provenance),
                         parent_hash(request["task_id"]), (), PENDING)


class _Sources:
    def __init__(self, root):
        self.root, self.files = Path(root).resolve(), {}

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("archive input escapes root")
        return path

    def read(self, name):
        path = self.path(name)
        raw = path.read_bytes()
        entry = dict(path=str(path.relative_to(self.root)), sha256=hashlib.sha256(raw).hexdigest(),
                     bytes=len(raw))
        if name in self.files and self.files[name] != entry:
            raise ValueError("source changed during inventory")
        self.files[name] = entry
        return raw

    def json(self, name):
        return json.loads(self.read(name))

    def rows(self, name):
        return [(i, json.loads(line), line) for i, line in enumerate(self.read(name).decode().splitlines(), 1)
                if line.strip()]


@dataclass
class Archive:
    records: list[RequestRecord]
    payloads: dict[str, dict]
    reset_requests: dict[str, dict]
    aliases: list[dict]
    summary: dict


def _reset_request(sources, tid, configuration, environment_hash):
    directory = f"envs/alfworld/data/json_2.1.1/train/{tid}"
    parent_hash(tid)  # reject traversal before reading files
    files = {}
    for name in ("game.tw-pddl", "traj_data.json", "initial_state.pddl"):
        raw = sources.read(f"{directory}/{name}")
        files[name] = hashlib.sha256(raw).hexdigest()
        if name == "game.tw-pddl":
            game = json.loads(raw)
    grammar = game["grammar"]
    if isinstance(grammar, str):
        # TextWorld appends other grammar definitions after this JSON object.
        grammar, _ = json.JSONDecoder().raw_decode(grammar[grammar.index("{"):])
    tasks = grammar["task"]
    if len(tasks) != 1 or not tasks[0]["rhs"].startswith("Your task is to: "):
        raise ValueError("unrecoverable unique train reset goal")
    goal = tasks[0]["rhs"].removeprefix("Your task is to: ").strip()
    # Never expose game walkthrough, solvable, trajectory plan or teacher prompt.
    return validate_request(dict(benchmark="alfworld", split="train", task_id=tid, goal=goal,
                                 world_hash=canonical_hash(files), world_files=files,
                                 environment_hash=environment_hash, state_kind="train_reset_request",
                                 max_episode_steps=configuration.max_episode_steps))


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    if isinstance(value, dict):
        if value.get("__bfas_dataclass__") == "tuple":
            return [_unpack(v) for v in value["items"]]
        return {k: _unpack(v) for k, v in value.items() if k != "__bfas_dataclass__"}
    return value


def _event_fields(event):
    p = event.get("provenance", {})
    return (event.get("task_id"), event.get("_event_turn", p.get("event_turn")),
            event.get("_event_good", p.get("teacher_command")),
            event.get("_prefix_len", p.get("prefix_len")))


def _token_totals(rows, count):
    if count is None:
        return dict(prompt_tokens=None, target_tokens=None, rejected_tokens=None)
    return dict(prompt_tokens=sum(count(r.get("prompt", "")) for r in rows),
                target_tokens=sum(count(r.get("response", "")) for r in rows),
                rejected_tokens=sum(count(r.get("_rejected", "")) for r in rows))


def collect_archive(root, *, ledger_paths=(LEDGER,), tokenizer_path=None, configuration=CapConfiguration()):
    """Read a consistent CPU snapshot. Only ledger attempts create packages.

    Duplicate byte-identical ledger copies deduplicate via their content-bound
    IDs. Same task/attempt names in different archives never merge. Ambiguous
    event origins remain unavailable instead of selecting a convenient request.
    """
    _privileged()
    sources = _Sources(root)
    count, tokenization = None, dict(status="missing", model="Qwen/Qwen3.5-4B",
                                    snapshot=TOKENIZER_SNAPSHOT, add_special_tokens=False,
                                    eos_added=False, truncation=False, provider_usage=False)
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        tokenizer_path = Path(tokenizer_path).resolve()
        raw = tokenizer_path.read_bytes()
        tok = Tokenizer.from_str(raw.decode())
        tok.no_truncation()
        tok.no_padding()
        count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
        tokenization.update(status="counted", path=str(tokenizer_path),
                            sha256=hashlib.sha256(raw).hexdigest(), library="tokenizers.Tokenizer")
    code_evidence = []
    for name in ("src/appworld_teacher.py", "src/bfas/adapters/alfworld.py", "tools/alf_event_mine.py"):
        if sources.path(name).is_file():
            sources.read(name)
            code_evidence.append(sources.files[name])
    env_hash = canonical_hash(dict(stage="c26-a-request-contract-only", sources=code_evidence))
    split = sources.json(SUPPORT)
    probe_rows = [r for _, r, _ in sources.rows(PROBE)]
    protected = {parent_hash(t) for t in split["calibration"]}
    protected.update(parent_hash(r["task_id"]) for r in probe_rows)
    demand = set(split["demand"])
    records, payloads, requests, by_task = [], {}, {}, defaultdict(list)
    cache_requests, source_rows = {}, []
    gaps = []
    attempts_by_task = defaultdict(set)
    for name in sorted(set(map(str, ledger_paths))):
        rows = sources.rows(name)
        sha = sources.files[name]["sha256"]
        source_rows.append(dict(path=name, rows=len(rows), sha256=sha))
        for line, row, raw_line in rows:
            q = query_id(sha, row, line)
            if q in payloads:
                payloads[q]["provenance"].setdefault("ledger_copies", []).append(name)
                continue
            tid = row["task_id"]
            if type(row.get("verified")) is not bool:
                raise ValueError("ledger verified must be bool")
            if tid not in cache_requests:
                cache_requests[tid] = _reset_request(sources, tid, configuration, env_hash)
            request = cache_requests[tid]
            record = public_record(q, request, configuration=configuration)
            records.append(record)
            requests[q] = request
            turns = row.get("demo", {}).get("turns", [])
            valid_commands = (isinstance(turns, list) and bool(turns) and
                              all(isinstance(t, dict) and isinstance(t.get("target"), str)
                                  and t["target"].strip() for t in turns))
            candidate = row["verified"] and valid_commands
            commands = [t["target"] for t in turns] if candidate else []
            cost = row.get("tokens_spent")
            reason = None if candidate else ("failed attempt: response/trajectory missing" if not row["verified"]
                                            else "successful attempt: command payload missing")
            if cost is None:
                candidate, reason = False, "historical cost missing"
            if commands and len(commands) > configuration.max_episode_steps:
                raise ValueError("command payload exceeds configured episode horizon")
            exclusions = []
            if record.parent_hash in protected:
                exclusions.append("protected probe/calibration parent")
            if tid not in demand:
                exclusions.append("outside historical demand")
            payloads[q] = dict(
                query_id=q, request_state_hash=record.spec.state_hash, dependencies=[],
                status="candidate" if candidate else "unavailable", unavailable_reason=reason,
                exclusion_reasons=exclusions, success=row["verified"],
                payload_kind="extracted_teacher_commands" if commands else None,
                commands=commands, behaviors=[], cost=cost, cost_basis="estimated", cost_confidence="estimated",
                usage=dict(output_tokens=None, estimated_output_tokens=cost, input_tokens=None,
                           reasoning_tokens=None, discarded_output_tokens=None, retry_tokens=None,
                           monetary_cost=None, estimation="legacy sum(ceil(len(response)/4))",
                           retained_command_tokens=sum(count(c) for c in commands) if count else None,
                           retained_prompt_tokens=sum(count(t["prompt"]) for t in turns) if count and commands else None),
                provenance=dict(kind="alf_demo_episode", ledger_path=name, ledger_sha256=sha, line=line,
                                task_id=tid, attempt_index=row["attempt_index"], timestamp=row["timestamp"],
                                recorded_L=None, replay_L=configuration.max_episode_steps,
                                event_aliases=[], missing_fields=["raw API request/reply", "reasoning", "per-call usage",
                                                                  "failed trajectory", "complete retry boundaries"],
                                state_validation="pending C26-B", tokenizer=tokenization),
                historical_response=row, raw_ledger_line=raw_line, historical_events=[])
            by_task[tid].append(q)
            attempts_by_task[tid].add(row["attempt_index"])
    for tid, indices in sorted(attempts_by_task.items()):
        missing = sorted(set(range(max(indices) + 1)) - indices)
        if missing:
            gaps.append(dict(kind="historical_attempt_gap", task_id=tid, missing_attempt_indices=missing,
                             cost=None, outcome=None))

    # Compare the cache's complete turns with the ledger, not just task names.
    demo_cache = sources.json(CACHE)
    cache_audit = []
    for tid, demo in sorted(demo_cache["demos"].items()):
        unpacked = _unpack(demo)
        matches = [q for q in by_task.get(tid, ()) if
                   payloads[q]["historical_response"].get("demo", {}).get("turns") == unpacked["turns"]]
        cache_audit.append(dict(task_id=tid, package_ids=matches,
                               status="copy" if len(matches) == 1 else "unavailable: ambiguous/mismatched cache"))
    aliases, inventory = [], []
    paths = sorted(sources.path("data/alf_sft").rglob("*.jsonl"))
    unified = sources.path("data/events_unified/alfworld_v1.jsonl")
    if unified.is_file():
        paths.append(unified)
    for path in paths:
        name = str(path.relative_to(sources.root))
        rows = [r for _, r, _ in sources.rows(name)]
        diagnostic = ("/r2/" in name or "r2ce" in name or "event_value" in name or "smoke" in name
                      or name == PROBE)
        mapped = 0
        for line, event, _ in sources.rows(name):
            tid, turn, command, prefix_len = _event_fields(event)
            matches = []
            if type(turn) is int and turn >= 0 and prefix_len == turn and isinstance(command, str):
                for q in by_task.get(tid, ()):
                    payload = payloads[q]
                    commands = payload["commands"]
                    if turn < len(commands) and commands[turn].strip() == command.strip():
                        matches.append(q)
            # Ambiguous requests never merge, even if prompt/target are identical.
            selected = matches[0] if len(matches) == 1 and not diagnostic else None
            alias = dict(path=name, line=line, package_id=selected,
                         status="alias" if selected else "unavailable/diagnostic",
                         candidate_package_ids=matches,
                         event_identity=canonical_hash([sources.files[name]["sha256"], line]),
                         state_validation="pending C26-B; origin match is not full-state verification",
                         transformation=("historical preference swap; response is not a teacher target"
                                         if "negteacher" in name else
                                         "derived student thought + substituted teacher ACTION"))
            aliases.append(alias)
            if selected:
                mapped += 1
                payloads[selected]["provenance"]["event_aliases"].append(alias)
                payloads[selected]["historical_events"].append(event)
        inventory.append(dict(path=name, rows=len(rows), task_ids=len({r.get("task_id") for r in rows}),
                              new_packages=0, alias_rows=mapped, diagnostic_only=diagnostic,
                              token_recount=_token_totals(rows, count)))
    # Inventory only: student rollouts, evaluation, atoms and diagnostics do not
    # create teacher requests. Arrays are inspected for shape, never as features.
    extra_patterns = ("results/alf_records/*.jsonl", "results/bfas/alfworld/ours_s0/pool.jsonl",
                      "results/bfas/alfworld/collect_s0/*.json", "data/atoms/alfworld*",
                      "data/fingerprints/*alfworld*", "results/analysis/transfer_matrix_alfworld*.npz")
    for pattern in extra_patterns:
        for path in sorted(sources.root.glob(pattern)):
            if not path.is_file():
                continue
            name = str(path.relative_to(sources.root))
            item = dict(path=name, new_packages=0)
            if path.suffix == ".jsonl":
                rows = [r for _, r, _ in sources.rows(name)]
                item.update(rows=len(rows), task_ids=len({r.get("task_id") for r in rows}),
                            teachers=dict(Counter(r.get("teacher", "absent") for r in rows)))
            elif path.suffix == ".npz":
                import io
                import numpy as np
                raw = sources.read(name)
                with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                    item["arrays"] = {key: list(arrays[key].shape) for key in arrays.files}
            else:
                value = sources.json(name)
                item["top_level_counts"] = {k: len(v) for k, v in value.items() if isinstance(v, (list, dict))}
            inventory.append(item)
    inventory.insert(0, dict(path="data/alf_records", exists=sources.path("data/alf_records").exists(), new_packages=0))
    if sources.path("data/alf_records").exists():
        for path in sorted(sources.path("data/alf_records").rglob("*.jsonl")):
            name = str(path.relative_to(sources.root))
            inventory.append(dict(path=name, rows=len(sources.rows(name)), new_packages=0))

    values = list(payloads.values())
    success = [p for p in values if p["success"]]
    failure = [p for p in values if not p["success"]]
    candidate = [p for p in values if p["status"] == "candidate"]
    all_parents = {parent_hash(t) for t in by_task}
    legal_tasks = sorted(t for t in by_task if t in demand and parent_hash(t) not in protected)
    legal_parents = {parent_hash(t) for t in legal_tasks}
    def total(items, field="cost"):
        return sum(p[field] for p in items if p[field] is not None)
    summary = dict(
        version="rtd-v1-alfworld-c26-a", mode="sealed_replay", scope="exploratory",
        attempts=len(values), task_ids=len(by_task), successful_attempts=len(success), failed_attempts=len(failure),
        candidate_packages=len(candidate), unavailable_packages=len(values) - len(candidate), usable_packages=0,
        command_turns=sum(len(p["commands"]) for p in values), legacy_token_estimate=total(values),
        legacy_cost_complete=all(p["cost"] is not None for p in values),
        successful_legacy_token_estimate=total(success), failed_legacy_token_estimate=total(failure),
        retained_command_tokens=sum(p["usage"]["retained_command_tokens"] for p in values) if count else None,
        retained_prompt_tokens=sum(p["usage"]["retained_prompt_tokens"] or 0 for p in values) if count else None,
        failed_attempts_on_successful_tasks=sum(p["provenance"]["task_id"] in {s["provenance"]["task_id"] for s in success}
                                               for p in failure),
        failed_cost_on_successful_tasks=sum(p["cost"] or 0 for p in failure
                                            if p["provenance"]["task_id"] in {s["provenance"]["task_id"] for s in success}),
        teacher_models=dict(Counter(p["historical_response"]["teacher"] for p in values)),
        tokenizer=tokenization, ledger_sources=source_rows, historical_attempt_gaps=gaps,
        proposed_support=dict(task_ids=len(legal_tasks), m=len(legal_parents), historical_parent_groups=len(all_parents),
                              fold_parent_counts={str(f): sum(parent_fold(h) == f for h in legal_parents) for f in (0, 1)},
                              calibration_task_ids=sorted(split["calibration"]),
                              probe_task_ids=sorted(r["task_id"] for r in probe_rows),
                              protected_parent_hashes=sorted(protected),
                              parents=[dict(task_id=t, parent_hash=parent_hash(t), fold=parent_fold(parent_hash(t)),
                                            protected=parent_hash(t) in protected) for t in sorted(by_task)],
                              demand_matches_ledger=demand == set(by_task), frozen=False),
        cache_audit=dict(demos=len(demo_cache["demos"]), completed_task_ids=len(demo_cache["completed_task_ids"]),
                         complete=demo_cache["complete"], new_packages=0, entries=cache_audit),
        inventory=inventory, source_files=list(sorted(sources.files.values(), key=lambda s: s["path"])),
        new_teacher_calls=0, new_teacher_tokens=0, limitations=LIMITATIONS,
        cap_configuration_audit=code_evidence)
    # Record name-level overlap without treating it as world equivalence.
    for split_name in ("valid_seen", "valid_unseen"):
        names = {p.name for p in sources.path(f"envs/alfworld/data/json_2.1.1/{split_name}").glob("*") if p.is_dir()}
        summary["proposed_support"][split_name + "_overlapping_parent_names"] = sorted(
            {t.split("/")[0] for t in by_task} & names)
    summary["cap_audit"] = cap_audit(records, payloads, configuration=configuration)
    summary["discrepancies"] = [dict(metric=k, expected=v, actual=summary[k])
                                for k, v in EXPECTED.items() if summary[k] != v]
    for name, expected_rows, expected_tasks in (("data/alf_sft/pool_A_all.jsonl", 385, 107),
                                               ("data/alf_sft/pool_C_conseq.jsonl", 85, 60)):
        found = next((i for i in inventory if i["path"] == name), {})
        for key, expected in (("rows", expected_rows), ("task_ids", expected_tasks)):
            if found.get(key) != expected:
                summary["discrepancies"].append(dict(metric=f"{name}:{key}", expected=expected, actual=found.get(key)))
        if found.get("alias_rows") != found.get("rows"):
            summary["discrepancies"].append(dict(metric=f"{name}:alias_rows", expected=found.get("rows"),
                                                 actual=found.get("alias_rows")))
    for item in cache_audit:
        if item["status"] != "copy":
            summary["discrepancies"].append(dict(metric="cached_demo_turn_correspondence", **item))
    if not summary["proposed_support"]["demand_matches_ledger"]:
        summary["discrepancies"].append(dict(metric="historical_demand_task_ids", expected=sorted(demand),
                                             actual=sorted(by_task)))
    return Archive(records, payloads, requests, aliases, summary)


def inventory_alfworld(root, **kwargs):
    return collect_archive(root, **kwargs).summary


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def build_alfworld_bank(root, directory, **kwargs):
    """Seal the C26-A candidate archive; never overwrite or mark candidates usable."""
    _privileged()
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    archive = collect_archive(root, **kwargs)
    seal_bank(directory, archive.records, archive.payloads)
    _write_new(directory / "public/reset_requests.json", archive.reset_requests)
    _write_new(directory / "sealed/audit.json", archive.summary)
    _write_new(directory / "sealed/event_aliases.json", archive.aliases)
    artifacts = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*.json"))}
    manifest = dict(version="rtd-v1-alfworld-c26-a", artifacts=artifacts,
                    source_files=archive.summary["source_files"], tokenizer=archive.summary["tokenizer"])
    _write_new(directory / "sealed/manifest.json", manifest)
    return archive.summary


def audit_bank(directory, *, root=None, expected_manifest_sha256=None):
    """Verify public, payload, alias and audit bytes and optionally source snapshot.

    A caller can pin the manifest hash externally. Hash consistency alone is not
    authentication against someone who can rewrite the entire sealed archive.
    """
    _privileged()
    directory = Path(directory).resolve()
    manifest_path = directory / "sealed/manifest.json"
    if expected_manifest_sha256 is not None and file_hash(manifest_path) != expected_manifest_sha256:
        raise ValueError("manifest integrity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != "rtd-v1-alfworld-c26-a":
        raise ValueError("unsupported archive version")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["artifacts"]) | {"sealed/manifest.json"}:
        raise ValueError("archive artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise ValueError(f"artifact integrity mismatch: {name}")
    requests = json.loads((directory / "public/reset_requests.json").read_text())
    raws = json.loads((directory / "public/requests.json").read_text())
    integrity = json.loads((directory / "sealed/integrity.json").read_text())
    summary = json.loads((directory / "sealed/audit.json").read_text())
    records, payloads = [], {}
    for raw in raws:
        q = raw["spec"]["query_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", q) or q in payloads:
            raise ValueError("invalid/duplicate request id")
        configuration = CapConfiguration(**json.loads(raw["spec"]["cap_provenance"])["configuration"])
        expected = public_record(q, requests[q], configuration=configuration)
        if json.loads(json.dumps(asdict(expected))) != raw:
            raise ValueError("public record contains non-public or inconsistent fields")
        payload = json.loads((directory / f"sealed/{q}.json").read_text())
        p = payload["provenance"]
        if digest(payload) != integrity[q] or query_id(p["ledger_sha256"], p, p["line"]) != q:
            raise ValueError("sealed identity/integrity mismatch")
        row = payload["historical_response"]
        if json.loads(payload["raw_ledger_line"]) != row:
            raise ValueError("raw ledger content mismatch")
        if query_id(p["ledger_sha256"], row, p["line"]) != q:
            raise ValueError("ledger row/request identity mismatch")
        if (payload["query_id"] != q or payload["request_state_hash"] != expected.spec.state_hash
                or p["task_id"] != requests[q]["task_id"]):
            raise ValueError("payload/reset request identity mismatch")
        if payload["success"] is not row["verified"] or payload["cost"] != row.get("tokens_spent"):
            raise ValueError("archived outcome/cost mismatch")
        if (payload["cost_basis"] != "estimated" or payload["cost_confidence"] != "estimated"
                or payload["usage"]["estimated_output_tokens"] != payload["cost"]):
            raise ValueError("cost basis mismatch")
        if payload["behaviors"] or payload["dependencies"] or payload["status"] not in {"candidate", "unavailable"}:
            raise ValueError("C26-A contains unverified usable evidence")
        turns = row.get("demo", {}).get("turns", [])
        valid = (bool(turns) and isinstance(turns, list) and all(
            isinstance(t, dict) and isinstance(t.get("target"), str) and t["target"].strip() for t in turns))
        commands = [t["target"] for t in turns] if row["verified"] and valid else []
        if payload["commands"] != commands:
            raise ValueError("command extraction differs from raw ledger")
        status = "candidate" if commands and payload["cost"] is not None else "unavailable"
        if payload["status"] != status:
            raise ValueError("candidate status differs from raw ledger")
        source = next((s for s in manifest["source_files"] if s["path"] == p["ledger_path"]), None)
        if source is None or source["sha256"] != p["ledger_sha256"]:
            raise ValueError("ledger source hash binding mismatch")
        records.append(expected)
        payloads[q] = payload
    if set(payloads) != set(integrity) or set(payloads) != set(requests):
        raise ValueError("public/sealed inventory mismatch")
    configurations = {r.spec.cap_provenance for r in records}
    if len(configurations) > 1:
        raise ValueError("C26-A requires uniform episode configuration")
    configuration = (CapConfiguration(**json.loads(records[0].spec.cap_provenance)["configuration"])
                     if records else CapConfiguration())
    checked = cap_audit(records, payloads, configuration=configuration)
    if checked != summary["cap_audit"]:
        raise ValueError("cap audit mismatch")
    if root is not None:
        root = Path(root).resolve()
        for source in manifest["source_files"]:
            path = (root / source["path"]).resolve()
            if not path.is_relative_to(root) or file_hash(path) != source["sha256"]:
                raise ValueError(f"historical source integrity mismatch: {source['path']}")
        tok = manifest["tokenizer"]
        if tok["status"] == "counted" and file_hash(tok["path"]) != tok["sha256"]:
            raise ValueError("tokenizer integrity mismatch")
    return dict(passed=True, attempts=len(records), usable_packages=0,
                source_hashes_verified=root is not None, manifest_sha256=file_hash(manifest_path))


def markdown_report(summary):
    lines = ["# C26-A ALFWorld archive audit", "", "CPU inventory only; no environment/model/teacher calls.", "",
             "| Metric | Observed | Readiness expectation |", "|---|---:|---:|"]
    for k, expected in EXPECTED.items():
        lines.append(f"| {k} | {summary[k]} | {expected} |")
    lines += [f"| usable_packages | {summary['usable_packages']} | 0 before C26-B |", "",
              "## Source inventory", "", "| Source | Rows / shapes | Task IDs | New packages | Aliases |",
              "|---|---|---:|---:|---:|"]
    for item in summary["inventory"]:
        quantity = item.get("rows", item.get("arrays", item.get("top_level_counts", item.get("exists", "unknown"))))
        lines.append(f"| `{item['path']}` | {quantity} | {item.get('task_ids', '')} | 0 | {item.get('alias_rows', '')} |")
    lines += ["", "## Budget and support", "", "```json", json.dumps(
        dict(cap_audit=summary["cap_audit"],
             support={k: v for k, v in summary["proposed_support"].items() if k not in
                      {"parents", "calibration_task_ids", "probe_task_ids", "protected_parent_hashes"}},
             historical_attempt_gaps=summary["historical_attempt_gaps"],
             discrepancies=summary["discrepancies"]), ensure_ascii=False, indent=2), "```", "",
        "## Limitations", ""]
    lines.extend("- " + limitation for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"
