"""Read-only, state-matched purchased evidence for PBSD — agent adaptation.

No tokenizer estimates, API clients, verifier calls, or writes to source pools.
The legacy BFCL pool omitted usage; resolve its exact archived result by task ID
AND response bytes (including the original JSON rendering for structured calls).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def response_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def recorded_tokens(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            raise ValueError("empty recorded output-token cost")
        return sum(recorded_tokens(x) for x in value)
    if type(value) is not int or value < 0:
        raise ValueError("recorded output-token cost must be a non-negative integer")
    return value


def output_cost(row: dict) -> tuple[int, str] | None:
    if row.get("cost_confidence", "exact") != "exact":
        raise ValueError("PBSD requires exact recorded output-token cost, not estimates")
    candidates = []
    for key in ("output_token_count", "output_tokens", "evidence_output_tokens", "completion_tokens"):
        if key in row and row[key] is not None:
            candidates.append((recorded_tokens(row[key]), key))
    if isinstance(row.get("usage"), dict):
        for key in ("output_tokens", "completion_tokens"):
            if row["usage"].get(key) is not None:
                candidates.append((recorded_tokens(row["usage"][key]), f"usage.{key}"))
    # Sealed records explicitly identify their cost as exact output-token usage.
    if "cost" in row and (row.get("cost_confidence") == "exact" or
                          row.get("cost_basis") == "output_tokens"):
        candidates.append((recorded_tokens(row["cost"]), "cost"))
    if not candidates:
        return None
    if len({c for c, _ in candidates}) != 1 or candidates[0][0] <= 0:
        raise ValueError("conflicting or zero recorded output-token costs")
    return candidates[0]


def read_cost_records(path: Path) -> list[tuple[dict, str]]:
    """Accept JSONL, JSON record/list, or a sealed directory; never unseal via API."""
    files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    records = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            values = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        for index, value in enumerate(values, 1):
            if isinstance(value, dict):
                records.append((value, f"{file}:{index}"))
    return records


class CostIndex:
    def __init__(self, records: list[tuple[dict, str]] = ()):
        self.records: dict[tuple[str, str], list[tuple[dict, str]]] = {}
        for record, location in records:
            if record.get("status") == "unavailable" or record.get("success") is False:
                continue
            raw = record.get("historical_response", record)
            if not isinstance(raw, dict):
                continue
            tid = raw.get("task_id", raw.get("id", record.get("provenance", {}).get("task_id")))
            response = raw.get("response", raw.get("result"))
            if tid is not None and response is not None:
                self.records.setdefault((str(tid), response_text(response)), []).append((record, location))

    def resolve(self, row: dict, source: str) -> tuple[int, str]:
        direct = output_cost(row)
        if direct:
            return direct[0], f"{source}#{direct[1]}"
        matches = self.records.get((str(row["task_id"]), row["response"]), [])
        costs = []
        for record, location in matches:
            raw = record.get("historical_response", record)
            if raw.get("prompt") is not None and raw["prompt"] != row["prompt"]:
                continue
            if raw.get("turn_index") is not None and raw["turn_index"] != row["turn_index"]:
                continue
            cost = output_cost(record)
            if cost is None and raw is not record:
                cost = output_cost(raw)
            if cost:
                costs.append((cost[0], f"{location}#{cost[1]}"))
        if not costs:
            raise ValueError(f"{source}: no exact cost for {row['task_id']}; supply "
                             "evidence_output_tokens or AW_PBSD_COST_RECORDS. token_hint is not usage.")
        if len({c for c, _ in costs}) != 1:
            raise ValueError(f"{source}: ambiguous recorded attempts; supply the exact cost record")
        return costs[0]


@dataclass(frozen=True)
class EvidenceUnit:
    task_id: str
    state: dict
    rows: tuple[dict, ...]
    costs: tuple[int, ...]
    sources: tuple[str, ...]

    @property
    def cost(self) -> int:
        return sum(self.costs)

    @property
    def key(self) -> str:
        return digest([self.task_id, self.state, [r["response"] for r in self.rows]])

    def manifest(self) -> dict:
        return dict(task_id=self.task_id, unit_id=self.key, state_sha256=digest(self.state),
                    evidence_output_tokens=self.cost,
                    rows=[dict(pool_index=r["_pool_index"], response_sha256=digest(r["response"]),
                               output_tokens=c, cost_source=s)
                          for r, c, s in zip(self.rows, self.costs, self.sources)])


def task_start_units(rows: list[dict], pool_path: Path, cost_index: CostIndex) -> list[EvidenceUnit]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["task_id"]), []).append(row)
    units = []
    for tid, group in groups.items():
        # Historical AppWorld turn_index counts message positions (first = 2).
        # Select minimum, then verify there is no prior assistant/tool turn.
        first = min(r["turn_index"] for r in group)
        initial = [r for r in group if r["turn_index"] == first]
        state = None
        costs, sources = [], []
        for row in initial:
            if row.get("verified") is False:
                raise ValueError(f"{tid}: evidence must be a verified purchased demonstration")
            if "deepseek" not in row["teacher"].lower() and row["teacher"].lower() != "ds":
                raise ValueError(f"{tid}: evidence must come from the purchased DeepSeek pool")
            messages = row.get("messages")
            if messages and any(m["role"] not in {"system", "user", "developer"} for m in messages):
                raise ValueError(f"{tid}: first available row is not the task-start state")
            if not messages:
                # Reject serialized histories; allow a final assistant generation marker.
                for marker in ("<|assistant|>", "<|im_start|>assistant"):
                    parts = row["prompt"].split(marker)
                    if len(parts) > 2 or (len(parts) == 2 and parts[-1].strip() not in {"", "<think>", "<think>\n\n</think>"}):
                        raise ValueError(f"{tid}: prompt contains an earlier assistant turn")
                if "<|tool|>" in row["prompt"] or "<|im_start|>tool" in row["prompt"]:
                    raise ValueError(f"{tid}: prompt contains a tool observation history")
            current = {"prompt": row["prompt"]}
            if messages:
                current["messages"] = messages
            if state is not None and current != state:
                raise ValueError(f"{tid}: demonstrations are not state-matched")
            state = current
            cost, source = cost_index.resolve(row, f"{pool_path}:row={row['_pool_index']}")
            costs.append(cost)
            sources.append(source)
        units.append(EvidenceUnit(tid, state, tuple(initial), tuple(costs), tuple(sources)))
    return units


def load_units(trainer, pool_path: Path, cost_path: str = "") -> list[EvidenceUnit]:
    rows = trainer.load_pool(pool_path)
    if cost_path:
        records = read_cost_records(Path(cost_path))
    elif pool_path.name == "pool_bfcl_ds_sft.jsonl":
        # This is the specific result directory used by tools/bfcl_demo_pool.py.
        # Do not glob later attempts: identical text can have different API costs.
        archive = trainer.ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC"
        records = []
        for file in sorted(archive.rglob("*_result.json")):
            records.extend(read_cost_records(file))
    else:
        records = []
    return task_start_units(rows, pool_path, CostIndex(records))


def select_units(units: list[EvidenceUnit], selection: str, budget: int, seed: int) -> list[EvidenceUnit]:
    import numpy as np
    if selection not in {"full", "random"}:
        raise ValueError("pbsd_agent supports full/random evidence selection only")
    order = range(len(units)) if selection == "full" else np.random.default_rng(seed).permutation(len(units))
    selected, used = [], 0
    for index in order:
        unit = units[int(index)]
        if used + unit.cost > budget:
            break
        selected.append(unit)
        used += unit.cost
    if not selected:
        raise ValueError("evidence budget cannot purchase the first task's demonstrations")
    return selected
