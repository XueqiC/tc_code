#!/usr/bin/env python3
"""Convert collected AppWorld trajectories into incremental-context SFT rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "data" / "appworld_traces"
OUTPUT_ROOT = ROOT / "data" / "appworld_sft"
POOL_PATH = OUTPUT_ROOT / "pool.jsonl"
EPISODES_PATH = OUTPUT_ROOT / "episodes.jsonl"

PYTHON_BLOCK_RE = re.compile(
    r"```[ \t]*(?:python|py)(?:[ \t]+[^\r\n]*)?\r?\n.*?```",
    re.IGNORECASE | re.DOTALL,
)


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AppWorld teacher trajectories to agent SFT JSONL."
    )
    parser.add_argument(
        "--min-turn",
        type=nonnegative_int,
        default=0,
        help="skip assistant turns whose zero-based trajectory index is smaller (default: 0)",
    )
    parser.add_argument(
        "--max-rows",
        type=nonnegative_int,
        default=0,
        help="maximum pool rows to write; 0 means no cap (default: 0)",
    )
    return parser.parse_args()


def stable_value_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


def serialize_prompt(turns: list[dict[str, Any]], stop: int) -> str:
    pieces: list[str] = []
    for turn in turns[:stop]:
        role = turn.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported turn role: {role!r}")
        content = turn.get("content")
        if not isinstance(content, str):
            raise ValueError("turn content must be a string")
        pieces.append(f"<|{role}|>\n{content}\n")
    pieces.append("<|assistant|>\n")
    return "".join(pieces)


def read_episodes(min_turn: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []

    for trace_path in sorted(INPUT_ROOT.glob("*/train.jsonl")):
        directory_teacher = trace_path.parent.name
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    trajectory = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{trace_path}:{line_number}: invalid JSON") from exc

                if not isinstance(trajectory, dict):
                    raise ValueError(
                        f"{trace_path}:{line_number}: trajectory must be an object"
                    )
                task_id = trajectory.get("task_id")
                teacher = trajectory.get("teacher", directory_teacher)
                attempt = trajectory.get("attempt", 0)
                turns = trajectory.get("turns")
                if not isinstance(turns, list):
                    raise ValueError(
                        f"{trace_path}:{line_number}: turns must be a list"
                    )

                rows: list[dict[str, Any]] = []
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        raise ValueError(
                            f"{trace_path}:{line_number}: turn {turn_index} must be an object"
                        )
                    if turn.get("role") != "assistant" or turn_index < min_turn:
                        continue
                    response = turn.get("content")
                    if not isinstance(response, str):
                        raise ValueError(
                            f"{trace_path}:{line_number}: turn {turn_index} content must be a string"
                        )
                    if not PYTHON_BLOCK_RE.search(response):
                        continue
                    rows.append(
                        {
                            "task_id": task_id,
                            "teacher": teacher,
                            "turn_index": turn_index,
                            "prompt": serialize_prompt(turns, turn_index),
                            "messages": [
                                {"role": turn["role"],
                                 "content": turn["content"]}
                                for turn in turns[:turn_index]
                            ],
                            "response": response,
                            "token_hint": len(response) // 4,
                        }
                    )

                episodes.append(
                    {
                        "task_id": task_id,
                        "teacher": teacher,
                        "attempt": attempt,
                        "rows": rows,
                    }
                )

    episodes.sort(
        key=lambda episode: (
            str(episode["teacher"]),
            stable_value_key(episode["task_id"]),
            stable_value_key(episode["attempt"]),
        )
    )
    return episodes


def apply_row_cap(episodes: list[dict[str, Any]], max_rows: int) -> None:
    remaining = max_rows
    if max_rows == 0:
        return

    for episode in episodes:
        rows = episode["rows"]
        episode["rows"] = rows[:remaining]
        remaining -= len(episode["rows"])
        if remaining == 0:
            remaining = 0


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def print_summary(episodes: list[dict[str, Any]]) -> None:
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "rows": 0, "response_chars": 0}
    )
    for episode in episodes:
        teacher = str(episode["teacher"])
        rows = episode["rows"]
        summary[teacher]["episodes"] += 1
        summary[teacher]["rows"] += len(rows)
        summary[teacher]["response_chars"] += sum(
            len(row["response"]) for row in rows
        )

    headers = ("teacher", "episodes", "rows", "response_chars")
    table_rows = [
        (teacher, values["episodes"], values["rows"], values["response_chars"])
        for teacher, values in sorted(summary.items())
    ]
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in table_rows))
        if table_rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:>{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}"
    )
    print("  ".join("-" * width for width in widths))
    for teacher, episode_count, row_count, response_chars in table_rows:
        print(
            f"{teacher:<{widths[0]}}  "
            f"{episode_count:>{widths[1]}}  "
            f"{row_count:>{widths[2]}}  "
            f"{response_chars:>{widths[3]}}"
        )


def main() -> None:
    args = parse_args()
    episodes = read_episodes(args.min_turn)
    apply_row_cap(episodes, args.max_rows)

    pool_rows = [row for episode in episodes for row in episode["rows"]]
    episode_rows = [
        {
            "task_id": episode["task_id"],
            "teacher": episode["teacher"],
            "n_rows": len(episode["rows"]),
            "total_response_chars": sum(
                len(row["response"]) for row in episode["rows"]
            ),
        }
        for episode in episodes
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(POOL_PATH, pool_rows)
    write_jsonl(EPISODES_PATH, episode_rows)
    print_summary(episodes)


if __name__ == "__main__":
    main()
