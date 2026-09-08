"""CPU-only Table 1 I/O and accounting. No model, environment, or API imports.

The inventory builder is privileged. Acquisition consumes only its frozen public
projection; pool construction opens only purchased snapshot files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/table1_audit"
BENCHMARKS = ("bfcl", "alfworld", "appworld")
ARMS = ("sft", "sad", "bbopd", "pbsd_insp", "ddpo", "star", "pbsd_agent")
INITIAL_CHECKPOINT = {"model": "Qwen/Qwen3.5-4B", "adapter": None}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if raw.strip():
                yield line, json.loads(raw)


def output_path(path):
    """Refuse data/envs, shared symlinks, other worktrees and control directories."""
    path = Path(path).absolute()
    resolved = path.resolve()
    forbidden = [ROOT / name for name in ("data", "envs", ".venv", ".git", ".agents", ".codex")]
    if any(path.is_relative_to(p) or resolved.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError(f"read-only output destination: {path}")
    if not (resolved.is_relative_to(ROOT) or resolved.is_relative_to(Path('/tmp'))):
        raise ValueError(f"output leaves this worktree: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = output_path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path = output_path(path)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_sealed(directory, package_id, owned_ids):
    """The sole pool-builder content reader; authorization precedes file access."""
    if package_id not in owned_ids:
        raise PermissionError('cannot reveal an unpurchased package')
    if len(package_id) != 64 or any(c not in '0123456789abcdef' for c in package_id):
        raise ValueError('invalid package id')
    return read_json(Path(directory) / 'sealed' / f'{package_id}.json')


def row_identity(row):
    return digest({k: row.get(k) for k in ('task_id', 'teacher', 'turn_index', 'prompt', 'messages', 'response')})


def cost_total(calls):
    """Unique call IDs, including failures; never charge training segments."""
    unique = {}
    for call in calls:
        q = call['call_id']
        cost = call['recorded_cost']
        if type(cost) is not int or cost < 0:
            raise ValueError('recorded cost must be a nonnegative integer')
        if q in unique and unique[q] != cost:
            raise ValueError('conflicting cost for the same call')
        unique[q] = cost
    return sum(unique.values())
