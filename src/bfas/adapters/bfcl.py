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

from ..adapter import (
    BenchmarkAdapter,
    Demo,
    PolicyRef,
    Rollout,
    TaskRef,
    TeacherEpisode,
    Turn,
)
from ..protocol import SAMPLING_TEMPERATURE, SupportSplit, make_support_split


ROOT = Path(__file__).resolve().parents[3]
BFCL_ROOT = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
BFCL_DATA = BFCL_ROOT / "bfcl_eval/data"
BFCL_BIN = ROOT / "envs/bfcl/.venv/bin/bfcl"
VLLM_BIN_DIR = ROOT / "envs/vllm-serve/.venv/bin"
MERGE_EXPORT = ROOT / "tools/bfcl_hub_merge_export.py"
BFCL_CLI = ROOT / "tools/bfcl_cli.py"
MODEL_NAME = "Qwen/Qwen3.5-4B-FC"
# The official benchmark has no runnable "memory" category: BFCL_v4_memory.json
# is a group that expands into one runnable category per backend, with task ids
# rewritten (memory_3-x -> memory_kv_3-x) and a prerequisite write-phase chain
# hung off depends_on. Loading it as a flat file yields KeyError: MemoryAPI_.
MEMORY_BACKENDS = ("kv", "vector", "rec_sum")
GUIDE_HEADER = (
    "Worked example from an expert on this exact task. Study the approach, "
    "then solve the task yourself:\n"
)


class BFCLScoreError(RuntimeError):
    pass


def _completion_tokens(results: Sequence[Mapping[str, Any]]) -> int | None:
    def total(value: Any) -> int:
        if isinstance(value, list):
            return sum(total(item) for item in value)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"invalid BFCL completion token count: {value!r}")
        if int(value) != value:
            raise ValueError(f"non-integral BFCL completion token count: {value!r}")
        return int(value)

    if not results or any(row.get("output_token_count") is None for row in results):
        return None  # Historical result files may lack API usage.
    return sum(total(row["output_token_count"]) for row in results)


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
    needs_server = False

    def __init__(self, seed: int = 0, port: int = 8901, mt_training: bool = False):
        self.seed = seed
        self.port = port
        self.mt_training = mt_training
        self._entries: dict[str, dict[str, Any]] | None = None
        self._categories: dict[str, str] | None = None
        self._memory_prereqs: dict[str, list[str]] = {}
        self._prereq_ids: set[str] = set()
        self._handler_instance: Any = None

    def _load_entries(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        if self._entries is not None and self._categories is not None:
            return self._entries, self._categories
        entries: dict[str, dict[str, Any]] = {}
        categories: dict[str, str] = {}
        for path in sorted(BFCL_DATA.glob("BFCL_v4_*.json")):
            category = path.stem.replace("BFCL_v4_", "")
            if category == "memory":
                # handled below via the official per-backend expansion
                continue
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
        self._load_memory_entries(entries, categories)
        self._entries, self._categories = entries, categories
        return entries, categories

    def _load_memory_entries(
        self,
        entries: dict[str, dict[str, Any]],
        categories: dict[str, str],
    ) -> None:
        """Register memory tasks under their runnable per-backend categories.

        Also records each task's prerequisite write-phase chain so a selective
        generation run can carry it along; without the chain the task runs
        against an empty memory store and fails silently.
        """
        if str(BFCL_ROOT) not in sys.path:
            sys.path.insert(0, str(BFCL_ROOT))
        from bfcl_eval.utils import is_memory_prereq, load_dataset_entry

        self._memory_prereqs = {}
        self._prereq_ids = set()
        for backend in MEMORY_BACKENDS:
            category = f"memory_{backend}"
            try:
                rows = load_dataset_entry(category)
            except Exception:
                continue
            for row in rows:
                task_id = row.get("id")
                if not isinstance(task_id, str):
                    continue
                entries[task_id] = row
                categories[task_id] = category
                if is_memory_prereq(task_id):
                    self._prereq_ids.add(task_id)
                else:
                    self._memory_prereqs[task_id] = [
                        dep for dep in row.get("depends_on", []) if isinstance(dep, str)
                    ]

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
        return [
            TaskRef(task_id, category)
            for task_id, category in sorted(categories.items())
            if task_id not in self._prereq_ids
        ]

    def task_categories(self) -> dict[str, str]:
        return dict(self._load_entries()[1])

    def support_split(self) -> "SupportSplit":
        # BFCL categories differ in size by two orders of magnitude, so the
        # size-weighted default spends the probe budget on the axes the student
        # already handles and leaves memory / web search / parallel with too
        # few tasks to measure.  Measure every axis evenly; decide where to
        # spend from the measured deficit, not from category size.
        return make_support_split(self.task_pool(), coverage_floor=True)

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
            category = categories[task_id]
            selected = by_category.setdefault(category, [])
            # A memory task only makes sense together with the write-phase turns
            # it depends on; official loading filters the id file, so the chain
            # has to be listed explicitly or the store stays empty.
            for prereq in self._memory_prereqs.get(task_id, []):
                if prereq not in selected:
                    selected.append(prereq)
            if task_id not in selected:
                selected.append(task_id)
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

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("LOCAL_SERVER_ENDPOINT", None)
        env["LOCAL_SERVER_PORT"] = str(self.port)
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PATH"] = os.pathsep.join(
            part for part in (str(VLLM_BIN_DIR), env.get("PATH")) if part
        )
        return env

    @staticmethod
    def _trained_checkpoint(policy_ref: PolicyRef | None) -> Path | None:
        if policy_ref is None:
            return None
        checkpoint = Path(str(policy_ref))
        if not (checkpoint / "model.safetensors").is_file():
            return None
        return checkpoint.resolve()

    @staticmethod
    def _remove_merged_checkpoint(checkpoint: Path, merged: Path) -> None:
        expected = checkpoint.parent / "hub_merged"
        try:
            expected.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"refusing BFCL merged cleanup outside project: {merged}"
            ) from exc
        if merged != expected or merged.name != "hub_merged":
            raise RuntimeError(f"refusing unsafe BFCL merged cleanup: {merged}")
        if merged.is_symlink() or not merged.is_dir():
            raise RuntimeError(f"refusing non-directory BFCL merged cleanup: {merged}")
        shutil.rmtree(merged)

    def _export_trained_checkpoint(self, checkpoint: Path) -> Path:
        merged = checkpoint.parent / "hub_merged"
        if merged.exists() or merged.is_symlink():
            trash_root = ROOT / "_trash"
            trash_root.mkdir(parents=True, exist_ok=True)
            trash = trash_root / (
                f"hub_merged_{checkpoint.parent.name}_{uuid.uuid4().hex}"
            )
            shutil.move(str(merged), str(trash))
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(MERGE_EXPORT),
                "--adapter", str(checkpoint),
                "--out", str(merged),
                "--verify",
            ],
            cwd=ROOT,
            check=True,
        )
        if not (merged / "model.safetensors.index.json").is_file():
            raise RuntimeError(f"BFCL merge export did not produce a shard index: {merged}")
        return merged

    @staticmethod
    def _run_evaluate(
        command: Sequence[str], env: Mapping[str, str], score_dir: Path
    ) -> None:
        """Run the official evaluator, tolerating leaderboard-CSV failures.

        The official checkers write per-category ``*_score.json`` first and
        only then aggregate a leaderboard CSV.  On a single-task selective
        pass that aggregation raises (``stdev`` needs two latency points),
        which would otherwise discard a perfectly scored teacher episode.
        The scores themselves are still the official ones.
        """
        outcome = subprocess.run(command, cwd=BFCL_ROOT, env=dict(env))
        if outcome.returncode == 0:
            return
        if score_dir.exists() and any(score_dir.rglob("*_score.json")):
            return
        raise subprocess.CalledProcessError(outcome.returncode, command)

    def _run_generate(
        self,
        args: Sequence[str],
        policy_ref: PolicyRef | None = None,
    ) -> dict[str, str]:
        env = self._subprocess_env()
        model = str(args[args.index("--model") + 1]) if "--model" in args else ""
        is_azure = model.startswith("azure/")
        command = self._bfcl_command("generate", *args)
        # API-served teacher models (e.g. deepseek-v4-pro-FC) must not get
        # local vllm server args; they need the OpenAI-compatible creds for
        # the Ollama endpoint instead.
        is_api_model = is_azure or "deepseek" in model or model.startswith("gpt-")
        if is_api_model and not is_azure:
            env["OPENAI_BASE_URL"] = os.environ.get(
                "OLLAMA_BASE_URL", "https://ollama.com"
            ).rstrip("/") + "/v1"
            key = os.environ.get("OLLAMA_API_KEY", "")
            if not key:
                key_file = Path.home() / ".ollama_api_key2"
                if key_file.exists():
                    key = key_file.read_text().strip()
            env["OPENAI_API_KEY"] = key
        elif not is_api_model:
            command.extend([
                "--backend", "vllm",
                "--num-gpus", "1",
                "--gpu-memory-utilization", os.environ.get("GPU_UTIL") or "0.85",
            ])
        checkpoint = self._trained_checkpoint(policy_ref)
        merged = None
        if checkpoint is not None:
            merged = self._export_trained_checkpoint(checkpoint)
            command.extend(["--local-model-path", str(merged)])
        subprocess.run(command, cwd=BFCL_ROOT, env=env, check=True)
        if checkpoint is not None and merged is not None:
            self._remove_merged_checkpoint(checkpoint, merged)
        return env

    @staticmethod
    def _bfcl_command(*args: str) -> list[str]:
        if any(str(arg).startswith("azure/") for arg in args):
            return [str(BFCL_BIN.with_name("python")), str(BFCL_CLI), *args]
        return [str(BFCL_BIN), *args]

    def _official_pass(
        self,
        model_name: str,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
        policy_ref: PolicyRef | None = None,
    ) -> tuple[list[dict[str, Any]], Path, Path]:
        token = f"bfas_{uuid.uuid4().hex}"
        result_name = f"result_{token}"
        score_name = f"score_{token}"
        selection_path = BFCL_ROOT / "test_case_ids_to_generate.json"
        selection_backup = selection_path.read_bytes() if selection_path.exists() else None
        data_backups: list[tuple[Path, bytes]] = []
        try:
            selection_path.write_text(
                json.dumps(self._selective_file(task_ids), indent=1) + "\n"
            )
            data_backups = self._patch_guidance(guided_demos or {})
            generate = [
                "--model", model_name, "--run-ids",
                "--temperature", str(temperature), "--num-threads", "4",
                "--result-dir", result_name,
            ]
            evaluate = self._bfcl_command(
                "evaluate", "--model", model_name,
                "--result-dir", result_name, "--score-dir", score_name,
                "--partial-eval",
            )
            env = self._run_generate(generate, policy_ref)
            self._run_evaluate(evaluate, env, BFCL_ROOT / score_name)
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

    def _official_full_evaluation(self, policy_ref: PolicyRef) -> tuple[Path, Path]:
        token = f"bfas_{uuid.uuid4().hex}"
        result_name = f"result_{token}"
        score_name = f"score_{token}"
        env = self._run_generate(
            [
                "--model", MODEL_NAME,
                "--test-category", "all",
                "--temperature", "0",
                "--num-threads", "8",
                "--result-dir", result_name,
            ],
            policy_ref,
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
            cwd=BFCL_ROOT,
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
        results, result_dir, score_dir = self._official_pass(
            MODEL_NAME, task_ids, temperature, guided_demos, policy_ref=policy
        )
        try:
            return self._rollouts_from_pass(task_ids, results, score_dir, guided_demos)
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(score_dir, ignore_errors=True)

    def demo_from_result(self, result: Mapping[str, Any], attempt_index: int) -> Demo:
        """Construct a verified Demo offline, exactly as teacher acquisition does.

        The caller supplies the checker verdict. Stateful episodes deliberately
        have no training turns: the adapter does not reconstruct step contexts.
        """
        task_id = str(result["id"])
        target = self._target(result.get("result"))
        context = {"messages": self._messages(task_id), "functions": self._functions(task_id)}
        turns = []
        if not self.task_categories()[task_id].startswith(("multi_turn", "web_search", "memory")):
            turns.append(Turn(self._render(context["messages"], context["functions"]), target, context))
        return Demo(task_id, turns, target[:4000],
                    {"attempt": attempt_index + 1, "checker_verified": True})

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
                    demos[task_id] = self.demo_from_result(result, attempt)
                remaining = [task_id for task_id in remaining if task_id not in demos]
            finally:
                shutil.rmtree(result_dir, ignore_errors=True)
                shutil.rmtree(score_dir, ignore_errors=True)
        return demos

    def teacher_episode(
        self, task_id: str, attempt_index: int, temperature: float
    ) -> TeacherEpisode:
        model = os.environ.get("BFAS_BFCL_TEACHER", "deepseek-v4-pro-FC")
        results, result_dir, score_dir = self._official_pass(
            model, [task_id], temperature
        )
        try:
            category = self.task_categories()[task_id]
            verdicts = extract_verdicts(score_dir, {task_id: category})
            result = next(
                (item for item in results if str(item.get("id")) == task_id),
                None,
            )
            target = self._target(result.get("result")) if result else ""
            demo = None
            if verdicts.get(task_id) is True and result is not None:
                demo = self.demo_from_result(result, attempt_index)
            return TeacherEpisode(
                task_id=task_id,
                verified=demo is not None,
                demo=demo,
                response_texts=(target,) if target else (),
                # BFCL preserves completion usage as a scalar (single turn) or
                # nested turn/step lists. Charge every generated prerequisite
                # too, including failed attempts; never estimate visible text
                # when the provider reported its total (reasoning included).
                tokens_spent=_completion_tokens(results),
            )
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(score_dir, ignore_errors=True)

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        result_dir, score_dir = self._official_full_evaluation(policy_ref)
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
