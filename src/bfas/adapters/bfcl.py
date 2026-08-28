"""BFCL adapter using the official generator, evaluator, and prompt handler."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, TaskRef, Turn
from ..protocol import SAMPLING_TEMPERATURE


ROOT = Path(__file__).resolve().parents[3]
BFCL_ROOT = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
BFCL_DATA = BFCL_ROOT / "bfcl_eval/data"
BFCL_BIN = ROOT / "envs/bfcl/.venv/bin/bfcl"
MODEL_NAME = "Qwen/Qwen3.5-4B-FC"
GUIDE_HEADER = (
    "Worked example from an expert on this exact task. Study the approach, "
    "then solve the task yourself:\n"
)


class BFCLScoreError(RuntimeError):
    pass


def _score_category(path: Path) -> str:
    name = path.stem
    if name.startswith("BFCL_v4_"):
        name = name[len("BFCL_v4_"):]
    if name.endswith("_score"):
        name = name[:-len("_score")]
    return name


def _generated_by_category(
    generated: Mapping[str, str | Iterable[str]] | Sequence[TaskRef],
) -> dict[str, set[str]]:
    if isinstance(generated, Mapping):
        if all(isinstance(value, str) for value in generated.values()):
            out: dict[str, set[str]] = {}
            for task_id, category in generated.items():
                out.setdefault(str(category), set()).add(str(task_id))
            return out
        return {
            str(category): set(map(str, task_ids))
            for category, task_ids in generated.items()
        }
    out: dict[str, set[str]] = {}
    for task in generated:
        out.setdefault(task.category, set()).add(task.task_id)
    return out


def read_score_summaries(score_dir: Path) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for path in sorted(score_dir.rglob("*_score.json")):
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not lines or not isinstance(lines[0], Mapping):
            continue
        summary = lines[0]
        if "correct_count" not in summary or "total_count" not in summary:
            continue
        category = _score_category(path)
        if category in summaries:
            raise BFCLScoreError(f"duplicate BFCL score category {category!r}")
        summaries[category] = {
            "verified": int(summary["correct_count"]),
            "total": int(summary["total_count"]),
        }
    return summaries


def extract_verdicts(
    score_dir: Path | str,
    generated: Mapping[str, str | Iterable[str]] | Sequence[TaskRef],
) -> dict[str, bool]:
    """Admit passes only when a category's positive summary reconciles."""
    score_root = Path(score_dir)
    expected = _generated_by_category(generated)
    verdicts = {
        task_id: False for task_ids in expected.values() for task_id in task_ids
    }
    score_files = {
        _score_category(path): path
        for path in score_root.rglob("*_score.json")
    }
    for category, task_ids in expected.items():
        path = score_files.get(category)
        if path is None:
            continue
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not lines or not isinstance(lines[0], Mapping):
            raise BFCLScoreError(f"{path}: missing score summary")
        summary = lines[0]
        if "correct_count" not in summary or "total_count" not in summary:
            raise BFCLScoreError(f"{path}: incomplete score summary")
        total = int(summary["total_count"])
        correct = int(summary["correct_count"])
        if total != len(task_ids):
            raise BFCLScoreError(
                f"{category}: generated {len(task_ids)} tasks but checker summarized {total}"
            )
        failures: set[str] = set()
        explicit_passes: set[str] = set()
        for entry in lines[1:]:
            if not isinstance(entry, Mapping) or "id" not in entry:
                continue
            task_id = str(entry["id"])
            if entry.get("valid") is True:
                explicit_passes.add(task_id)
            else:
                failures.add(task_id)
        unknown = (failures | explicit_passes) - task_ids
        if unknown:
            raise BFCLScoreError(f"{category}: score contains unknown ids {sorted(unknown)}")
        inferred_passes = task_ids - failures
        if len(inferred_passes) != correct:
            raise BFCLScoreError(
                f"{category}: checker correct_count={correct}, reconciled={len(inferred_passes)}"
            )
        if explicit_passes and not explicit_passes <= inferred_passes:
            raise BFCLScoreError(f"{category}: contradictory explicit verdicts")
        for task_id in inferred_passes:
            verdicts[task_id] = True
    return verdicts


extract_bfcl_verdicts = extract_verdicts


class BFCLAdapter(BenchmarkAdapter):
    name = "bfcl"

    def __init__(self, seed: int = 0, port: int = 8901, mt_training: bool = False):
        self.seed = seed
        self.port = port
        self.mt_training = mt_training
        self._entries: dict[str, dict[str, Any]] | None = None
        self._categories: dict[str, str] | None = None
        self._handler_instance: Any = None

    def _load_entries(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        if self._entries is not None and self._categories is not None:
            return self._entries, self._categories
        entries: dict[str, dict[str, Any]] = {}
        categories: dict[str, str] = {}
        for path in sorted(BFCL_DATA.glob("BFCL_v4_*.json")):
            category = path.stem.replace("BFCL_v4_", "")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        entries_for_file = [tid for tid, cat in categories.items() if cat == category]
                        for task_id in entries_for_file:
                            entries.pop(task_id, None)
                            categories.pop(task_id, None)
                        break
                    if isinstance(row, dict) and isinstance(row.get("id"), str):
                        entries[row["id"]] = row
                        categories[row["id"]] = category
        self._entries, self._categories = entries, categories
        return entries, categories

    def _handler(self) -> Any:
        if self._handler_instance is None:
            if str(BFCL_ROOT) not in sys.path:
                sys.path.insert(0, str(BFCL_ROOT))
            from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

            self._handler_instance = QwenFCHandler(
                model_name="Qwen/Qwen3.5-4B-FC",
                temperature=0.001,
                registry_name=MODEL_NAME,
                is_fc_model=True,
            )
        return self._handler_instance

    def task_pool(self) -> list[TaskRef]:
        _, categories = self._load_entries()
        return [TaskRef(task_id, category) for task_id, category in sorted(categories.items())]

    def task_categories(self) -> dict[str, str]:
        return dict(self._load_entries()[1])

    def official_eval_split_disjoint(self) -> bool:
        return False

    def _messages(self, task_id: str, demo: Demo | None = None) -> list[dict[str, Any]]:
        entries, _ = self._load_entries()
        entry = entries[task_id]
        messages = copy.deepcopy(entry["question"][0])
        if demo is not None:
            for message in messages:
                if message.get("role") == "user":
                    message["content"] = (
                        f"{GUIDE_HEADER}Expert solution: {demo.worked_example}\n\n"
                        + str(message.get("content", ""))
                    )
                    break
        return messages

    def _functions(self, task_id: str) -> Any:
        entries, _ = self._load_entries()
        entry = entries[task_id]
        if "function" in entry:
            return entry["function"]
        if str(BFCL_ROOT) not in sys.path:
            sys.path.insert(0, str(BFCL_ROOT))
        from bfcl_eval.utils import populate_test_cases_with_predefined_functions

        populated = populate_test_cases_with_predefined_functions(
            [json.loads(json.dumps(entry))]
        )
        return populated[0].get("function")

    def _render(self, messages: Sequence[Mapping[str, Any]], functions: Any) -> str:
        return self._handler()._format_prompt(list(messages), functions)

    def _selective_file(self, task_ids: Sequence[str]) -> dict[str, list[str]]:
        categories = self.task_categories()
        by_category: dict[str, list[str]] = {}
        for task_id in task_ids:
            by_category.setdefault(categories[task_id], []).append(task_id)
        return by_category

    def _patch_guidance(
        self, guided_demos: Mapping[str, Demo]
    ) -> list[tuple[Path, bytes]]:
        if not guided_demos:
            return []
        backups: list[tuple[Path, bytes]] = []
        for path in sorted(BFCL_DATA.glob("BFCL_v4_*.json")):
            original = path.read_bytes()
            changed = False
            output: list[str] = []
            for line in original.decode("utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    output.append(line)
                    continue
                if not isinstance(row, dict) or row.get("id") not in guided_demos:
                    output.append(line)
                    continue
                demo = guided_demos[row["id"]]
                for message in row["question"][0]:
                    if message.get("role") == "user":
                        message["content"] = (
                            f"{GUIDE_HEADER}Expert solution: {demo.worked_example}\n\n"
                            + str(message.get("content", ""))
                        )
                        break
                output.append(json.dumps(row, ensure_ascii=False))
                changed = True
            if changed:
                backups.append((path, original))
                path.write_text("\n".join(output) + "\n")
        return backups

    def _official_pass(
        self,
        model_name: str,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
        local_server: bool = True,
    ) -> tuple[list[dict[str, Any]], Path, Path]:
        token = f"bfas_{uuid.uuid4().hex}"
        result_name = f"result_{token}"
        score_name = f"score_{token}"
        selection_path = BFCL_ROOT / "test_case_ids_to_generate.json"
        selection_backup = selection_path.read_bytes() if selection_path.exists() else None
        data_backups: list[tuple[Path, bytes]] = []
        env = os.environ.copy()
        env["LOCAL_SERVER_ENDPOINT"] = "localhost"
        env["LOCAL_SERVER_PORT"] = str(self.port)
        try:
            selection_path.write_text(
                json.dumps(self._selective_file(task_ids), indent=1) + "\n"
            )
            data_backups = self._patch_guidance(guided_demos or {})
            generate = [
                str(BFCL_BIN), "generate", "--model", model_name, "--run-ids",
                "--temperature", str(temperature), "--num-threads", "4",
                "--result-dir", result_name,
            ]
            if local_server:
                generate.append("--skip-server-setup")
            evaluate = [
                str(BFCL_BIN), "evaluate", "--model", model_name,
                "--result-dir", result_name, "--score-dir", score_name,
                "--partial-eval",
            ]
            subprocess.run(generate, cwd=ROOT / "envs/bfcl", env=env, check=True)
            subprocess.run(evaluate, cwd=ROOT / "envs/bfcl", env=env, check=True)
            result_dir = BFCL_ROOT / result_name
            score_dir = BFCL_ROOT / score_name
            results: list[dict[str, Any]] = []
            for path in result_dir.rglob("*_result.json"):
                for line in path.read_text().splitlines():
                    if line.strip():
                        results.append(json.loads(line))
            return results, result_dir, score_dir
        finally:
            for path, content in data_backups:
                path.write_bytes(content)
            if selection_backup is None:
                selection_path.unlink(missing_ok=True)
            else:
                selection_path.write_bytes(selection_backup)

    @staticmethod
    def _target(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping) and result.get("tool_calls"):
            calls = result["tool_calls"]
            return "\n".join(
                "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
                for call in calls
            )
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    def _rollouts_from_pass(
        self,
        task_ids: Sequence[str],
        results: Sequence[Mapping[str, Any]],
        score_dir: Path,
        guided_demos: Mapping[str, Demo] | None,
    ) -> list[Rollout]:
        categories = self.task_categories()
        expected = {task_id: categories[task_id] for task_id in task_ids}
        verdicts = extract_verdicts(score_dir, expected)
        by_id = {str(result["id"]): result for result in results}
        output: list[Rollout] = []
        for task_id in task_ids:
            result = by_id.get(task_id)
            training_excluded = (
                not self.mt_training
                and categories[task_id].startswith(("multi_turn", "web_search", "memory"))
            )
            turns: list[Turn] = []
            deployment_turns: list[Turn] = []
            if result is not None and not training_excluded:
                deployment_context = {
                    "messages": self._messages(task_id, None),
                    "functions": self._functions(task_id),
                }
                serving_context = {
                    "messages": self._messages(task_id, (guided_demos or {}).get(task_id)),
                    "functions": deployment_context["functions"],
                }
                prompt = self._render(
                    serving_context["messages"], serving_context["functions"]
                )
                turns.append(Turn(
                    prompt, self._target(result.get("result")), serving_context
                ))
                deployment_turns.append(Turn(
                    self._render(
                        deployment_context["messages"], deployment_context["functions"]
                    ),
                    self._target(result.get("result")),
                    deployment_context,
                ))
            output.append(Rollout(
                task_id,
                verdicts.get(task_id, False),
                turns,
                {
                    "checker_verified": verdicts.get(task_id, False),
                    "category": categories[task_id],
                    "training_excluded": training_excluded,
                    "result": result,
                    "guided": task_id in (guided_demos or {}),
                    "deployment_turns": deployment_turns,
                },
            ))
        return output

    def _official_full_evaluation(self) -> tuple[Path, Path]:
        token = f"bfas_{uuid.uuid4().hex}"
        result_name = f"result_{token}"
        score_name = f"score_{token}"
        env = os.environ.copy()
        env["LOCAL_SERVER_ENDPOINT"] = "localhost"
        env["LOCAL_SERVER_PORT"] = str(self.port)
        subprocess.run(
            [
                str(BFCL_BIN),
                "generate",
                "--model", MODEL_NAME,
                "--test-category", "all",
                "--temperature", "0",
                "--num-threads", "8",
                "--result-dir", result_name,
                "--skip-server-setup",
            ],
            cwd=ROOT / "envs/bfcl",
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(BFCL_BIN),
                "evaluate",
                "--model", MODEL_NAME,
                "--test-category", "all",
                "--result-dir", result_name,
                "--score-dir", score_name,
            ],
            cwd=ROOT / "envs/bfcl",
            env=env,
            check=True,
        )
        return BFCL_ROOT / result_name, BFCL_ROOT / score_name

    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        del policy
        results, result_dir, score_dir = self._official_pass(
            MODEL_NAME, task_ids, temperature, guided_demos, local_server=True
        )
        try:
            return self._rollouts_from_pass(task_ids, results, score_dir, guided_demos)
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(score_dir, ignore_errors=True)

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        model = os.environ.get("BFAS_BFCL_TEACHER", "deepseek-v4-pro-FC")
        remaining = list(task_ids)
        demos: dict[str, Demo] = {}
        for attempt in range(attempts):
            if not remaining:
                break
            results, result_dir, score_dir = self._official_pass(
                model,
                remaining,
                0.0 if attempt == 0 else SAMPLING_TEMPERATURE,
                local_server=False,
            )
            try:
                categories = self.task_categories()
                verdicts = extract_verdicts(
                    score_dir, {task_id: categories[task_id] for task_id in remaining}
                )
                for result in results:
                    task_id = str(result["id"])
                    if not verdicts.get(task_id) or task_id in demos:
                        continue
                    target = self._target(result.get("result"))
                    context = {
                        "messages": self._messages(task_id),
                        "functions": self._functions(task_id),
                    }
                    turns: list[Turn] = []
                    if not categories[task_id].startswith(("multi_turn", "web_search", "memory")):
                        turns.append(Turn(
                            self._render(context["messages"], context["functions"]),
                            target,
                            context,
                        ))
                    demos[task_id] = Demo(
                        task_id,
                        turns,
                        target[:4000],
                        {"attempt": attempt + 1, "checker_verified": True},
                    )
                remaining = [task_id for task_id in remaining if task_id not in demos]
            finally:
                shutil.rmtree(result_dir, ignore_errors=True)
                shutil.rmtree(score_dir, ignore_errors=True)
        return demos

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        del policy_ref
        result_dir, score_dir = self._official_full_evaluation()
        try:
            support = set(self.support_split().support)
            categories = self.task_categories()
            per_category: dict[str, float] = {}
            for path in score_dir.rglob("*_score.json"):
                lines = [json.loads(line) for line in path.read_text().splitlines() if line]
                if not lines:
                    continue
                category = _score_category(path)
                failures = {
                    str(row["id"]) for row in lines[1:]
                    if isinstance(row, Mapping) and "id" in row
                }
                category_ids = {
                    task_id for task_id, cat in categories.items()
                    if cat == category and task_id not in support
                }
                if category_ids:
                    per_category[category] = len(category_ids - failures) / len(category_ids)
            headline = sum(per_category.values()) / len(per_category) if per_category else 0.0
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics = {"headline": headline, "per_category": per_category}
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            source_csv = score_dir / "data_overall.csv"
            if source_csv.is_file():
                shutil.copy2(source_csv, out_dir / "data_overall_including_support.csv")
            return metrics
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(score_dir, ignore_errors=True)

    def serving_probe(self) -> None:
        with urllib.request.urlopen(
            f"http://localhost:{self.port}/v1/models", timeout=10
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"BFCL serving probe returned HTTP {response.status}")

    def generation_suffix(self) -> str:
        return "<|im_start|>assistant\n"

    def target_policy(self, category: str) -> str:
        if "irrelevance" in category or "relevance" in category:
            return "prose_ok"
        if category.startswith(("multi_turn", "web_search", "memory")):
            return "prose_ok"
        return "call_required"

    def is_call_target(self, target: str, category: str) -> bool:
        return "<tool_call>" in target and "</tool_call>" in target

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, Mapping):
            raise ValueError("row lacks BFCL render context")
        return self._render(context["messages"], context["functions"])
