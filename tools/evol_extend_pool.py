#!/usr/bin/env python3
"""Extend the controlled GSM8K-code pool with verified Evol-Instruct rows."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "data" / "evol_pool_v1.jsonl"
DOMAIN = "gsm8k-code"
TEACHER = "deepseek-v4-pro"
VERIFY_TIMEOUT_SECONDS = 10

# These templates and operations are kept verbatim (apart from newline style)
# from the authors' WizardLM Evol_Instruct/depth.py and breadth.py scripts.
DEPTH_BASE_PROMPT = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
{method}
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
{instruction}
#Rewritten Prompt#:
"""

DEPTH_METHODS = (
    (
        "add_constraints",
        "Please add one more constraints/requirements into #The Given Prompt#'",
    ),
    (
        "deepen",
        "If #The Given Prompt# contains inquiries about certain issues, the "
        "depth and breadth of the inquiry can be increased.",
    ),
    (
        "concretize",
        "Please replace general concepts with more specific concepts.",
    ),
    (
        "increase_reasoning_steps",
        "If #The Given Prompt# can be solved with just a few simple thinking "
        "processes, you can rewrite it to explicitly request multiple-step reasoning.",
    ),
)

BREADTH_PROMPT = """I want you act as a Prompt Creator.
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.
The #Created Prompt# must be reasonable and must be understood and responded by humans.
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#
#Given Prompt#:
{instruction}
#Created Prompt#:
"""

SOLUTION_PROMPT = """Solve by writing a Python function solution() that DERIVES the numeric answer step by step: one intermediate variable per arithmetic step, then return the final variable. Do NOT precompute the answer in your head — no bare `return <number>`. Output ONLY the code.

Problem:
{instruction}"""

CODE_TAG_RE = re.compile(r"<code(?:\s[^>]*)?>(.*?)</code>", re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?:(?:python|py)[ \t]*)?\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
TEXT_FENCE_RE = re.compile(
    r"^```[^\r\n]*\r?\n(?P<body>.*?)\r?\n```$", re.DOTALL
)
EVOL_LABEL_RE = re.compile(
    r"^\s*#(?:Rewritten|Created) Prompt#\s*:\s*", re.IGNORECASE
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=1,
        help="Evolution rounds per branch (default: 1)",
    )
    parser.add_argument(
        "--per-query",
        type=positive_int,
        default=2,
        help="Independent evolution branches per support query (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Use only the first N resolved support queries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and summarize work without API calls or file writes",
    )
    return parser.parse_args(argv)


def _add_src_to_path() -> None:
    src = str(SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_e3_task_resolver() -> tuple[Path, Callable[..., Any]]:
    """Import E3, tolerating unrelated early condition-registry population."""
    try:
        from e3_tier1 import DEFAULT_CONFIG, resolve_task_rows

        return DEFAULT_CONFIG, resolve_task_rows
    except NameError as exc:
        if getattr(exc, "name", None) not in {"COLORS", "SHORT_LABELS"}:
            raise

    # Some controlled-suite revisions extend these registries before their
    # literals are assigned. Pre-seeding them lets the module initialize while
    # leaving its resolver and eventual registry values unchanged.
    module_path = SRC / "e3_tier1.py"
    spec = importlib.util.spec_from_file_location("_evol_e3_tier1", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load E3 task resolver from {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.COLORS = {}
    module.SHORT_LABELS = {}
    spec.loader.exec_module(module)
    return module.DEFAULT_CONFIG, module.resolve_task_rows


def load_support_rows(limit: int | None = None) -> list[dict[str, Any]]:
    """Resolve the default E3 task support set through its canonical helper."""
    _add_src_to_path()
    import yaml

    default_config, resolve_task_rows = _load_e3_task_resolver()

    with default_config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("task_boundary"), dict):
        raise ValueError(f"invalid E3 config: {default_config}")

    # Do not inherit E3_TASK/E3_TASK_FILTER from the environment: this tool's
    # controlled-suite input is always the default GSM8K-code support set.
    config["task_boundary"]["domain"] = DOMAIN
    config["task_boundary"]["filter"] = None
    rows, _, _ = resolve_task_rows(config)
    selected = list(rows[:limit] if limit is not None else rows)
    for index, row in enumerate(selected):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"invalid support row at resolved index {index}")
    return selected


def evolution_strategy(
    parent_index: int, round_index: int, variant_index: int
) -> tuple[str, str | None]:
    """Choose breadth for odd branches and rotate depth methods for even ones."""
    if variant_index % 2:
        return "in_breadth", None
    depth_index = (parent_index + round_index + variant_index // 2) % len(
        DEPTH_METHODS
    )
    name, method = DEPTH_METHODS[depth_index]
    return name, method


def build_evolution_prompt(instruction: str, method: str | None) -> str:
    if method is None:
        return BREADTH_PROMPT.format(instruction=instruction)
    return DEPTH_BASE_PROMPT.format(method=method, instruction=instruction)


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def clean_evolved_prompt(reply: str) -> str:
    text = strip_think(reply).strip()
    fenced = TEXT_FENCE_RE.fullmatch(text)
    if fenced:
        text = fenced.group("body").strip()
    text = EVOL_LABEL_RE.sub("", text, count=1).strip()
    return text


def extract_python_code(reply: str) -> str:
    text = strip_think(reply).strip()
    tagged = CODE_TAG_RE.search(text)
    if tagged:
        return html.unescape(tagged.group(1)).strip()
    fenced = CODE_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text


def verify_solution(code: str) -> float | None:
    """Execute solution() in the repository's resource-limited subprocess."""
    _add_src_to_path()
    import verifier

    value = verifier.run_solution(code, timeout_s=VERIFY_TIMEOUT_SECONDS)
    return value if value is not None and math.isfinite(value) else None


def load_completed_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8", errors="replace") as output_file:
        for line in output_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                prompts.add(row["prompt"])
    return prompts


def needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as output_file:
        output_file.seek(-1, 2)
        return output_file.read(1) != b"\n"


def make_teacher_call() -> Callable[[str], str]:
    """Build the same OpenAI-compatible Ollama client used by AppWorld."""
    _add_src_to_path()
    from appworld_teacher import generate_reply, load_teacher_config

    config = load_teacher_config(TEACHER)

    def call(prompt: str) -> str:
        return generate_reply(config, [{"role": "user", "content": prompt}])

    return call


def _safe_failure(stage: str, parent_index: int, exc: BaseException) -> None:
    print(
        f"[skip] parent={parent_index} stage={stage} "
        f"error={type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def extend_pool(
    support_rows: list[dict[str, Any]],
    rounds: int,
    per_query: int,
    call_teacher: Callable[[str], str],
    output_path: Path = OUTPUT,
) -> dict[str, int]:
    completed = load_completed_prompts(output_path)
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leading_newline = needs_leading_newline(output_path)

    with output_path.open("a", encoding="utf-8") as output_file:
        if leading_newline:
            output_file.write("\n")
        for parent_index, row in enumerate(support_rows):
            branches = [row["prompt"]] * per_query
            for round_index in range(rounds):
                for variant_index, source_prompt in enumerate(branches):
                    strategy, method = evolution_strategy(
                        parent_index, round_index, variant_index
                    )
                    counts["attempted"] += 1
                    evolution_prompt = build_evolution_prompt(source_prompt, method)
                    try:
                        evolution_reply = call_teacher(evolution_prompt)
                    except Exception as exc:
                        counts["evolution_failed"] += 1
                        _safe_failure("evolve", parent_index, exc)
                        continue

                    evolved_prompt = clean_evolved_prompt(evolution_reply)
                    if not evolved_prompt or evolved_prompt == source_prompt:
                        counts["evolution_failed"] += 1
                        print(
                            f"[skip] parent={parent_index} stage=evolve "
                            "error=empty or unchanged prompt",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    branches[variant_index] = evolved_prompt
                    counts[f"strategy_{strategy}"] += 1

                    if evolved_prompt in completed:
                        counts["resumed"] += 1
                        continue

                    try:
                        solution_reply = strip_think(
                            call_teacher(
                                SOLUTION_PROMPT.format(instruction=evolved_prompt)
                            )
                        ).strip()
                    except Exception as exc:
                        counts["solution_failed"] += 1
                        _safe_failure("solve", parent_index, exc)
                        continue

                    code = extract_python_code(solution_reply)
                    value = verify_solution(code) if code else None
                    if value is None:
                        counts["verification_failed"] += 1
                        print(
                            f"[skip] parent={parent_index} stage=verify "
                            "error=solution did not print a number",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    result = {
                        "domain": DOMAIN,
                        "prompt": evolved_prompt,
                        "response": code,
                        "teacher": TEACHER,
                        "evol": 1,
                        "parent_index": parent_index,
                    }
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add(evolved_prompt)
                    counts["appended"] += 1
                    print(
                        f"[pass] parent={parent_index} round={round_index + 1} "
                        f"variant={variant_index + 1} strategy={strategy} value={value:g}",
                        flush=True,
                    )
    return dict(counts)


def dry_run_summary(
    support_rows: list[dict[str, Any]], rounds: int, per_query: int
) -> dict[str, Any]:
    strategies: Counter[str] = Counter()
    for parent_index in range(len(support_rows)):
        for round_index in range(rounds):
            for variant_index in range(per_query):
                strategy, _ = evolution_strategy(
                    parent_index, round_index, variant_index
                )
                strategies[strategy] += 1
    return {
        "support_queries": len(support_rows),
        "rounds": rounds,
        "per_query": per_query,
        "planned_variants": len(support_rows) * rounds * per_query,
        "strategies": dict(sorted(strategies.items())),
        "already_written": len(load_completed_prompts(OUTPUT)),
        "output": str(OUTPUT),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    support_rows = load_support_rows(args.limit)
    if args.dry_run:
        print(json.dumps(dry_run_summary(support_rows, args.rounds, args.per_query)))
        return 0

    call_teacher = make_teacher_call()
    planned = len(support_rows) * args.rounds * args.per_query
    print(
        f"support={len(support_rows)} rounds={args.rounds} "
        f"per_query={args.per_query} planned={planned} output={OUTPUT}",
        flush=True,
    )
    counts = extend_pool(
        support_rows,
        rounds=args.rounds,
        per_query=args.per_query,
        call_teacher=call_teacher,
    )
    print(json.dumps({"planned": planned, **counts}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
