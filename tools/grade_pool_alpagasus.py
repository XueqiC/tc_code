#!/usr/bin/env python3
"""Grade the E3 candidate pool with the AlpaGasus accuracy rubric."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_PATH = ROOT / "data" / "alpagasus_scores.json"
TEACHER = "gpt-5.4"

SYSTEM_PROMPT = (
    "We would like to request your feedback on the performance of AI assistant "
    "in response to the instruction and the given input displayed following."
)
RATING_REQUEST = (
    "Please rate according to the accuracy of the response to the instruction "
    "and the input. Each assistant receives a score on a scale of 0 to 5, where "
    "a higher score indicates higher level of the accuracy. Please first output "
    "a single line containing value indicating the scores."
)
LEADING_FLOAT_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?![\d.])"
)
HASH_RE = re.compile(r"[0-9a-f]{16}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="E3 YAML config (default: src.e3_tier1.DEFAULT_CONFIG)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="maximum number of candidate-pool rows to consider",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load the pool and report pending grades without API calls or writes",
    )
    return parser.parse_args(argv)


def row_key(prompt: str, response: str) -> str:
    """Return the 16-hex prompt/response identity consumed by E3 tier-1."""
    return hashlib.sha256((prompt + response).encode("utf-8")).hexdigest()[:16]


def grading_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction: {prompt}\nResponse: {response}\n\n"
                f"{RATING_REQUEST}"
            ),
        },
    ]


def parse_leading_score(reply: str) -> float:
    """Parse and validate the leading numeric score from a teacher reply."""
    # The shared client can optionally request a separate reasoning section.
    if "</think>" in reply:
        reply = reply.split("</think>", 1)[1]
    match = LEADING_FLOAT_RE.match(reply)
    if match is not None:
        score = float(match.group(1))
    else:
        # tolerate formatting variance: first number anywhere in the reply
        any_match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", reply)
        if any_match is None:
            raise ValueError("teacher reply contains no numeric score")
        score = float(any_match.group(0))
    if not math.isfinite(score) or not 0.0 <= score <= 5.0:
        raise ValueError(f"teacher score must be between 0 and 5, got {score!r}")
    return score


def load_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as score_file:
        raw_scores = json.load(score_file)
    if not isinstance(raw_scores, dict):
        raise ValueError(f"{path} must contain a JSON object")

    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        if not isinstance(key, str) or HASH_RE.fullmatch(key) is None:
            raise ValueError(f"invalid row key in {path}: {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"score for {key} in {path} is not numeric")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 5.0:
            raise ValueError(f"score for {key} in {path} is outside [0, 5]")
        scores[key] = score
    return scores


def write_scores(path: Path, scores: dict[str, float]) -> None:
    """Atomically replace the score file after each completed grade."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as score_file:
            json.dump(scores, score_file, indent=2, sort_keys=True)
            score_file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_e3_module() -> ModuleType:
    """Load E3 while tolerating its matrix-label registry initialization order."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    existing = sys.modules.get("e3_tier1")
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location("e3_tier1", SRC / "e3_tier1.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {SRC / 'e3_tier1.py'}")
    module = importlib.util.module_from_spec(spec)
    # The current E3 module extends these registries before their declarations.
    # Pre-seeding them preserves its intended import behavior without editing it.
    module.COLORS = {}
    module.SHORT_LABELS = {}
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    return module


def load_pool(config_path: Path | None) -> list[dict[str, Any]]:
    """Load the pool through the exact configuration path used by E3 tier-1."""
    e3_tier1 = _load_e3_module()

    config = e3_tier1.load_config(config_path or e3_tier1.DEFAULT_CONFIG)
    pool = e3_tier1.load_candidate_pool(config)
    if hasattr(e3_tier1, "_maybe_extend_pool_with_evol"):
        pool = e3_tier1._maybe_extend_pool_with_evol(pool)
    return pool


def _pending_count(
    rows: Sequence[dict[str, Any]], scores: dict[str, float]
) -> int:
    known = set(scores)
    pending = 0
    for row in rows:
        key = row_key(row["prompt"], row["response"])
        if key not in known:
            known.add(key)
            pending += 1
    return pending


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pool = load_pool(args.config)
    rows = pool if args.limit is None else pool[: args.limit]
    scores = load_scores(OUTPUT_PATH)
    pending = _pending_count(rows, scores)

    print(
        f"pool_rows={len(pool)} considered={len(rows)} "
        f"pending={pending} existing_scores={len(scores)}",
        flush=True,
    )
    if args.dry_run:
        print(f"dry-run: no API calls or writes; output={OUTPUT_PATH}", flush=True)
        return 0
    if pending == 0:
        print(f"nothing to grade; output={OUTPUT_PATH}", flush=True)
        return 0

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from appworld_teacher import generate_reply, load_teacher_config

    teacher = load_teacher_config(TEACHER)
    completed = 0
    for index, row in enumerate(rows):
        key = row_key(row["prompt"], row["response"])
        if key in scores:
            continue
        try:
            reply = generate_reply(
                teacher, grading_messages(row["prompt"], row["response"])
            )
            score = parse_leading_score(reply)
        except Exception as exc:
            print(
                f"[skip] row={index} key={key} error={exc}",
                flush=True,
            )
            failures = getattr(main, "_failures", 0) + 1
            main._failures = failures
            if failures > 50:
                raise RuntimeError("too many grading failures") from exc
            scores[key] = None
            write_scores(OUTPUT_PATH, scores)
            completed += 1
            continue

        scores[key] = score
        write_scores(OUTPUT_PATH, scores)
        completed += 1
        print(
            f"graded={completed}/{pending} row={index} key={key} score={score:g}",
            flush=True,
        )

    print(
        f"complete: graded={completed} total_scores={len(scores)} "
        f"output={OUTPUT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
