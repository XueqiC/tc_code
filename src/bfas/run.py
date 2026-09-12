"""Run the unified BFAS pipeline for one benchmark and arm."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .adapter import BenchmarkAdapter, CollectionArtifacts, Demo, PolicyRef, Rollout
from .arms import ARM_NAMES, build_pool, trainer_environment
from .audit import AuditReport, audit_pool, write_decision_trace
from .ledger import acquire_demos, import_demos, load_ledger
from .protocol import (
    AdaptiveSampler,
    SAMPLING_TEMPERATURE,
    SEEDS,
    TEACHER_ATTEMPTS,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results/bfas"
DEFAULT_MODELS = {
    "appworld": "Qwen/Qwen3.5-4B",  # user decision 2026-09-02: 4B base, official scaffold
    "bfcl": "Qwen/Qwen3.5-4B",
    "alfworld": "Qwen/Qwen3.5-4B",  # paper protocol: one fixed 4B student (2B base scored 0/134, 2026-09-02)
    "tau2": "Qwen/Qwen3.5-4B",
    "webshop": "google/gemma-4-12B-it",
    "hotpotqa": "google/gemma-4-12B-it",
}
PHASE_CACHE_SCHEMA_VERSION = 1


def positive_seed_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= len(SEEDS):
        raise argparse.ArgumentTypeError(f"must be between 1 and {len(SEEDS)}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, choices=tuple(DEFAULT_MODELS)
    )
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--seeds", type=positive_seed_count, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_adapter(name: str, seed: int, port: int) -> BenchmarkAdapter:
    if name == "appworld":
        from .adapters.appworld import AppWorldAdapter

        if os.environ.get("BFAS_APPWORLD_SCAFFOLD", "official") == "legacy":
            return AppWorldAdapter(seed)
        # Default since 2026-09-02: the official stonybrooknlp scaffold (the
        # in-house harness scored the base at 0/40; see adapters/appworld_official).
        from .adapters.appworld_official import AppWorldOfficialAdapter

        return AppWorldOfficialAdapter(seed, port=port)
    if name == "bfcl":
        from .adapters.bfcl import BFCLAdapter

        return BFCLAdapter(seed, port=port, mt_training=False)
    if name == "alfworld":
        from .adapters.alfworld import ALFWorldAdapter

        return ALFWorldAdapter(seed, port=port)
    if name == "hotpotqa":
        from .adapters.hotpotqa import HotpotQAAdapter
        return HotpotQAAdapter(seed=seed, port=port)

    if name == "webshop":
        from .adapters.webshop import WebShopAdapter

        return WebShopAdapter(seed, port=port)
    from .adapters.tau2 import Tau2Adapter

    return Tau2Adapter(seed, port=port)


def collect_adaptive(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    task_ids: Sequence[str],
    guided_demos: Mapping[str, Demo] | None = None,
) -> tuple[list[Rollout], dict[str, float], dict[str, int]]:
    sampler = AdaptiveSampler(task_ids)
    collected: list[Rollout] = []
    while active := sampler.active():
        batch = adapter.rollout(
            policy,
            active,
            SAMPLING_TEMPERATURE,
            guided_demos=guided_demos,
        )
        by_id: dict[str, Rollout] = {}
        for rollout in batch:
            if rollout.task_id in by_id:
                raise RuntimeError(f"adapter returned duplicate rollout for {rollout.task_id}")
            by_id[rollout.task_id] = rollout
        missing = set(active) - set(by_id)
        extra = set(by_id) - set(active)
        if missing or extra:
            raise RuntimeError(
                f"adapter violated one-rollout-per-task contract: missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        for task_id in active:
            rollout = by_id[task_id]
            sampler.observe(task_id, rollout.verified)
            collected.append(rollout)
    return collected, sampler.p_hats(), sampler.rounds()


def _attach_guided_mu(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    rollouts: Sequence[Rollout],
) -> None:
    pending = [
        rollout for rollout in rollouts
        if rollout.verified and rollout.turns
        and isinstance(rollout.raw, dict)
        and rollout.raw.get("deployment_turns")
    ]
    if not pending:
        return
    import torch
    import appworld_train as trainer

    model = getattr(adapter, "_model", None)
    tokenizer = getattr(adapter, "_tokenizer", None)
    loaded_here = model is None or tokenizer is None
    if loaded_here:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(policy), trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            str(policy), dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
    try:
        for rollout in pending:
            fields: list[dict[str, float | int]] = []
            for turn in rollout.turns:
                input_ids, labels = trainer.encode(tokenizer, {
                    "prompt": turn.prompt,
                    "response": turn.target,
                })
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)
                with torch.inference_mode():
                    output = model(input_ids=input_ids, labels=labels)
                token_count = int((labels != -100).sum().item())
                fields.append({
                    "_mu_nll_sum": float(output.loss.item()) * token_count,
                    "_mu_ntok": token_count,
                })
            rollout.raw["mu_fields"] = fields
    finally:
        if loaded_here:
            del model
            torch.cuda.empty_cache()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class VLLMServer:
    def __init__(
        self,
        model: PolicyRef,
        gpu: str,
        port: int,
        log_path: Path,
        served_model_name: str,
        *,
        server_args: Sequence[str] | None = None,
    ):
        self.model = str(model)
        self.gpu = gpu
        self.port = port
        self.log_path = log_path
        self.served_model_name = served_model_name
        self.server_args = server_args
        self.process: subprocess.Popen[Any] | None = None
        self._log: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        command = [
            str(ROOT / "envs/vllm-serve/.venv/bin/vllm"),
            "serve", self.model,
            "--served-model-name", self.served_model_name,
            "--port", str(self.port),
            *(self.server_args if self.server_args is not None else [
                "--gpu-memory-utilization", os.environ.get("GPU_UTIL", "0.85"),
                "--max-model-len", "32768",
                # tau2's official harness sends tool_choice="auto"; vLLM
                # requires a tool-call parser for those requests.
                "--enable-auto-tool-choice",
                "--tool-call-parser", os.environ.get("BFAS_TOOL_PARSER", "hermes"),
            ]),
        ]
        # Evaluation workers redirect stdout to evaluate.log. Flush before
        # spawning so failed launches still leave the exact command behind.
        print("vLLM server command: " + shlex.join(command), flush=True)
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # setsid: vLLM workers share an owned group.
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{self.port}/v1/models", timeout=2
                ) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM server did not become ready on port {self.port}")

    def close(self) -> None:
        from .processes import stop_process_group
        try:
            if self.process is not None:
                stop_process_group(self.process)
        finally:
            self.process = None
            if self._log is not None:
                self._log.close()
                self._log = None


class PortRegistry:
    def __init__(self, port: int):
        self.port = port
        self.path = Path("/tmp/bfas-port-registry") / f"{port}.lock"
        self.handle: Any = None

    def __enter__(self) -> "PortRegistry":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise RuntimeError(f"BFAS port {self.port} is reserved by another lane") from exc
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                self.handle.close()
                raise RuntimeError(f"BFAS port {self.port} is already in use")
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


class CollectionLock:
    """Serialize cache creation when multiple arms start the same seed."""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir.with_name(f"{cache_dir.name}.lock")
        self.handle: Any = None

    def __enter__(self) -> "CollectionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


@contextmanager
def serving_lane(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    gpu: str,
    port: int,
    log_path: Path,
    *,
    served_model_name: str | None = None,
    server_args: Sequence[str] | None = None,
) -> Iterator[None]:
    if not getattr(adapter, "needs_server", True):
        yield
        return
    server_backed = (
        adapter.name in {"bfcl", "tau2"}
        or getattr(adapter, "server_backed", False)
        or (adapter.name == "alfworld" and os.environ.get("BFAS_NO_SERVER") != "1")
    )
    if not server_backed:
        yield
        return
    served_model_name = served_model_name or getattr(
        adapter,
        "served_model_name",
        "Qwen/Qwen3.5-4B" if adapter.name == "bfcl" else "bfas-policy",
    )
    with PortRegistry(port):
        server = VLLMServer(
            policy, gpu, port, log_path, served_model_name=served_model_name,
            server_args=server_args,
        )
        try:
            server.start()
            adapter.serving_probe()
            yield
        finally:
            server.close()


def _copy_base_policy(policy: PolicyRef, checkpoint: Path) -> None:
    source = Path(str(policy))
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    if source.is_dir():
        shutil.copytree(source, checkpoint)
        return
    from huggingface_hub import snapshot_download

    checkpoint.mkdir(parents=True)
    snapshot_download(repo_id=str(policy), local_dir=checkpoint)


def _train(
    benchmark: str,
    arm: str,
    seed: int,
    gpu: str,
    policy: PolicyRef,
    pool_path: Path,
    checkpoint: Path,
) -> None:
    tag = f"bfas_{benchmark}_{arm}_s{seed}"
    env = os.environ.copy()
    for name in (
        "AW_DISTILL",
        "AW_V3",
        "AW_UNIORPO",
        "AW_TOKSEL",
        "AW_TEACHER",
        "AW_DEMO_POOL",
    ):
        env.pop(name, None)
    env.update(trainer_environment(arm))
    env["AW_POOL_PATH"] = str(pool_path)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "src/appworld_train.py"),
        "--selection", "full",
        "--seed", str(seed),
        "--student", str(policy),
        "--tag", tag,
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    trained = ROOT / "results/appworld_students" / tag / "adapter"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    # The trainer saves a flattened causal-LM checkpoint; vLLM needs the Hub
    # layout (Qwen3_5ForConditionalGeneration), so overlay the trained weights
    # onto the Hub snapshot exactly as the BFCL campaign does (2026-09-02: the
    # first ALFWorld lane eval died with "no module named 'language_model'").
    export = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "tools/bfcl_hub_merge_export.py"),
            "--adapter", str(trained),
            "--out", str(checkpoint),
            "--model", str(policy),
            "--verify",
        ],
        cwd=ROOT,
        env=env,
    )
    if export.returncode != 0 or not (checkpoint / "config.json").exists():
        print("[bfas] hub-merge export failed; serving the flattened checkpoint", flush=True)
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        shutil.copytree(trained, checkpoint)


def _audit_json(report: AuditReport) -> dict[str, Any]:
    return {
        "row_count": report.row_count,
        "checked_round_trips": report.checked_round_trips,
        "should_train": report.should_train,
        "decision_trace": report.decision_trace,
    }


def _training_prompt_token_length(
    adapter: BenchmarkAdapter, policy: PolicyRef
) -> Callable[[str], int]:
    tokenizer = getattr(adapter, "_tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(policy), trust_remote_code=False
        )

    def token_length(prompt: str) -> int:
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    return token_length


def _collection_path(cache_dir: Path) -> Path:
    return cache_dir / "collection.json"


def _load_collection(cache_dir: Path) -> CollectionArtifacts | None:
    path = _collection_path(cache_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            return None
        return CollectionArtifacts.from_dict(value)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_collection(cache_dir: Path, artifacts: CollectionArtifacts) -> None:
    _write_atomic_json(_collection_path(cache_dir), artifacts.as_dict())


def _write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _phase_path(cache_dir: Path, phase: str) -> Path:
    return cache_dir / f"{phase}.json"


def _load_phase_json(cache_dir: Path, phase: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(_phase_path(cache_dir, phase).read_text(encoding="utf-8"))
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != PHASE_CACHE_SCHEMA_VERSION
        ):
            return None
        return value
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _empty_collection_dict() -> dict[str, Any]:
    return CollectionArtifacts(
        unguided_rollouts=(),
        p_hats={},
        unguided_rounds={},
        teacher_demos={},
        guided_rollouts=(),
        guided_rounds={},
        checker_summary={},
    ).as_dict()


def _write_unguided_phase(
    cache_dir: Path,
    rollouts: Sequence[Rollout],
    p_hats: Mapping[str, float],
    rounds: Mapping[str, int],
) -> None:
    encoded = CollectionArtifacts(
        unguided_rollouts=tuple(rollouts),
        p_hats=dict(p_hats),
        unguided_rounds=dict(rounds),
        teacher_demos={},
        guided_rollouts=(),
        guided_rounds={},
        checker_summary={},
    ).as_dict()
    _write_atomic_json(_phase_path(cache_dir, "unguided"), {
        "schema_version": PHASE_CACHE_SCHEMA_VERSION,
        "rollouts": encoded["unguided_rollouts"],
        "p_hats": encoded["p_hats"],
        "rounds": encoded["unguided_rounds"],
    })


def _load_unguided_phase(
    cache_dir: Path,
) -> tuple[list[Rollout], dict[str, float], dict[str, int]] | None:
    value = _load_phase_json(cache_dir, "unguided")
    if value is None:
        return None
    collection = _empty_collection_dict()
    collection.update({
        "unguided_rollouts": value.get("rollouts"),
        "p_hats": value.get("p_hats"),
        "unguided_rounds": value.get("rounds"),
    })
    try:
        artifacts = CollectionArtifacts.from_dict(collection)
    except (KeyError, TypeError, ValueError):
        return None
    return (
        list(artifacts.unguided_rollouts),
        artifacts.p_hats,
        artifacts.unguided_rounds,
    )


def _write_demos_phase(
    cache_dir: Path,
    demos: Mapping[str, Demo],
    completed_task_ids: set[str],
    *,
    complete: bool,
) -> None:
    encoded = CollectionArtifacts(
        unguided_rollouts=(),
        p_hats={},
        unguided_rounds={},
        teacher_demos=dict(demos),
        guided_rollouts=(),
        guided_rounds={},
        checker_summary={},
    ).as_dict()
    _write_atomic_json(_phase_path(cache_dir, "demos"), {
        "schema_version": PHASE_CACHE_SCHEMA_VERSION,
        "demos": encoded["teacher_demos"],
        "completed_task_ids": sorted(completed_task_ids),
        "complete": complete,
    })


def _load_demos_phase(
    cache_dir: Path,
) -> tuple[dict[str, Demo], set[str], bool] | None:
    value = _load_phase_json(cache_dir, "demos")
    if value is None:
        return None
    collection = _empty_collection_dict()
    collection["teacher_demos"] = value.get("demos")
    completed_value = value.get("completed_task_ids", [])
    if (
        not isinstance(completed_value, list)
        or not all(isinstance(task_id, str) for task_id in completed_value)
        or not isinstance(value.get("complete"), bool)
    ):
        return None
    try:
        artifacts = CollectionArtifacts.from_dict(collection)
    except (KeyError, TypeError, ValueError):
        return None
    completed = set(completed_value) | set(artifacts.teacher_demos)
    return artifacts.teacher_demos, completed, bool(value["complete"])


def _write_guided_phase(
    cache_dir: Path,
    rollouts: Sequence[Rollout],
    rounds: Mapping[str, int],
) -> None:
    encoded = CollectionArtifacts(
        unguided_rollouts=(),
        p_hats={},
        unguided_rounds={},
        teacher_demos={},
        guided_rollouts=tuple(rollouts),
        guided_rounds=dict(rounds),
        checker_summary={},
    ).as_dict()
    _write_atomic_json(_phase_path(cache_dir, "guided"), {
        "schema_version": PHASE_CACHE_SCHEMA_VERSION,
        "rollouts": encoded["guided_rollouts"],
        "rounds": encoded["guided_rounds"],
    })


def _load_guided_phase(
    cache_dir: Path,
) -> tuple[list[Rollout], dict[str, int]] | None:
    value = _load_phase_json(cache_dir, "guided")
    if value is None:
        return None
    collection = _empty_collection_dict()
    collection.update({
        "guided_rollouts": value.get("rollouts"),
        "guided_rounds": value.get("rounds"),
    })
    try:
        artifacts = CollectionArtifacts.from_dict(collection)
    except (KeyError, TypeError, ValueError):
        return None
    return list(artifacts.guided_rollouts), artifacts.guided_rounds


def _is_teacher_quota_error(exc: BaseException) -> bool:
    try:
        import appworld_teacher
    except ImportError:
        return False
    return (
        isinstance(exc, appworld_teacher.TeacherAPIError)
        and "HTTP 429" in str(exc)
    )


def _teacher_quota_number(name: str, default: str) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _wait_for_teacher_quota(waited: float) -> tuple[bool, float]:
    import appworld_teacher

    wait_s = _teacher_quota_number("BFAS_TEACHER_QUOTA_WAIT_S", "1800")
    deadline_s = _teacher_quota_number(
        "BFAS_TEACHER_QUOTA_DEADLINE_S", "86400"
    )
    while waited < deadline_s:
        sleep_s = min(wait_s, deadline_s - waited)
        print(
            f"[bfas][demo] teacher quota exhausted, sleeping {sleep_s:g}s",
            flush=True,
        )
        time.sleep(sleep_s)
        waited += sleep_s
        teacher_name = os.environ.get("BFAS_TEACHER", "gpt-5.4")
        config = appworld_teacher.load_teacher_config(teacher_name)
        try:
            appworld_teacher.generate_reply(
                config,
                [{"role": "user", "content": "Reply with OK."}],
                temperature=0.0,
            )
        except appworld_teacher.TeacherAPIError as exc:
            if _is_teacher_quota_error(exc):
                if sleep_s == 0:
                    waited = deadline_s
                continue
            raise
        return True, waited
    return False, waited


def _collect_teacher_demo_phase(
    adapter: BenchmarkAdapter,
    task_ids: Sequence[str],
    attempts: int,
    on_result: Callable[[str, Demo | None], None],
) -> None:
    incremental = getattr(adapter, "teacher_demo_incremental", None)
    if callable(incremental):
        incremental(task_ids, attempts, on_result)
        return

    quota_waited = 0.0
    for task_index, task_id in enumerate(task_ids):
        while True:
            try:
                found = adapter.teacher_demo([task_id], attempts)
            except Exception as exc:
                if _is_teacher_quota_error(exc):
                    try:
                        recovered, quota_waited = _wait_for_teacher_quota(
                            quota_waited
                        )
                    except Exception as probe_exc:
                        print(
                            f"[bfas][demo] task={task_id} failed: {probe_exc}",
                            flush=True,
                        )
                        on_result(task_id, None)
                        break
                    if recovered:
                        continue
                    for remaining_id in task_ids[task_index:]:
                        on_result(remaining_id, None)
                    return
                print(
                    f"[bfas][demo] task={task_id} failed: {exc}",
                    flush=True,
                )
                on_result(task_id, None)
                break
            on_result(task_id, found.get(task_id))
            break


def _migrate_legacy_seed_zero_demos(
    benchmark: str, benchmark_dir: Path
) -> None:
    legacy = _load_demos_phase(benchmark_dir / "collect_s0")
    if legacy is None or not legacy[0]:
        return
    import_demos(benchmark, legacy[0])


def _shared_teacher_demos(
    benchmark: str,
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    task_ids: Sequence[str],
    shared_dir: Path,
) -> dict[str, Demo]:
    """Materialize benchmark-wide demos from the authoritative ledger."""

    requested = set(task_ids)
    with CollectionLock(shared_dir):
        _migrate_legacy_seed_zero_demos(benchmark, shared_dir.parent)
        states = load_ledger(benchmark, attempts=TEACHER_ATTEMPTS)
        needs_purchase = any(
            task_id not in states
            or (
                states[task_id]["best_demo"] is None
                and states[task_id]["infeasible"] is False
            )
            for task_id in task_ids
        )
        if needs_purchase:
            adapter.prepare_renderer(policy)

        quota_waited = 0.0
        while True:
            try:
                acquired = acquire_demos(
                    benchmark, adapter, task_ids, TEACHER_ATTEMPTS
                )
                states = acquired.states
                demos = dict(acquired)
                break
            except Exception as exc:
                if not _is_teacher_quota_error(exc):
                    raise
                try:
                    recovered, quota_waited = _wait_for_teacher_quota(
                        quota_waited
                    )
                except Exception as probe_exc:
                    print(
                        f"[bfas][demo] teacher quota recovery failed: {probe_exc}",
                        flush=True,
                    )
                    recovered = False
                if recovered:
                    continue
                states = load_ledger(benchmark, attempts=TEACHER_ATTEMPTS)
                demos = {
                    task_id: state["best_demo"]
                    for task_id, state in states.items()
                    if task_id in requested and state["best_demo"] is not None
                }
                break

        completed = {
            task_id
            for task_id in requested
            if task_id in states
            and (
                states[task_id]["best_demo"] is not None
                or states[task_id]["infeasible"] is True
            )
        }
        _write_demos_phase(
            shared_dir,
            demos,
            completed,
            complete=requested.issubset(completed),
        )
        return demos


def _collect_seed_artifacts(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    task_ids: Sequence[str],
    gpu: str,
    port: int,
    cache_dir: Path,
    *,
    teacher_demos: Mapping[str, Demo] | None = None,
) -> CollectionArtifacts:
    refresh = os.environ.get("BFAS_REFRESH_COLLECT") == "1"
    unguided_phase = None if refresh else _load_unguided_phase(cache_dir)
    if unguided_phase is None:
        with serving_lane(
            adapter, policy, gpu, port, cache_dir / "vllm_unguided.log"
        ):
            unguided, p_hats, unguided_rounds = collect_adaptive(
                adapter, policy, task_ids
            )
            if getattr(adapter, "needs_server", True):
                adapter.serving_probe()
        _write_unguided_phase(cache_dir, unguided, p_hats, unguided_rounds)
    else:
        unguided, p_hats, unguided_rounds = unguided_phase

    if teacher_demos is not None:
        demos = dict(teacher_demos)
    else:
        # Compatibility path for callers of this private helper.  The CLI
        # always supplies the benchmark-wide ledger-backed demos.
        demos_phase = None if refresh else _load_demos_phase(cache_dir)
        if demos_phase is None:
            demos = {}
            completed_demo_ids: set[str] = set()
            demos_complete = False
        else:
            demos, completed_demo_ids, demos_complete = demos_phase

        requested_ids = set(task_ids)
        if not demos_complete or not requested_ids.issubset(completed_demo_ids):
            adapter.prepare_renderer(policy)
            remaining_ids = [
                task_id for task_id in task_ids if task_id not in completed_demo_ids
            ]

            def checkpoint_demo(task_id: str, demo: Demo | None) -> None:
                if demo is not None:
                    demos[task_id] = demo
                completed_demo_ids.add(task_id)
                _write_demos_phase(
                    cache_dir,
                    demos,
                    completed_demo_ids,
                    complete=False,
                )

            _collect_teacher_demo_phase(
                adapter,
                remaining_ids,
                TEACHER_ATTEMPTS,
                checkpoint_demo,
            )
            demos_complete = requested_ids.issubset(completed_demo_ids)
            _write_demos_phase(
                cache_dir,
                demos,
                completed_demo_ids,
                complete=demos_complete,
            )

    guided_ids = [
        task_id
        for task_id in task_ids
        if p_hats[task_id] < 0.5 and task_id in demos
    ]
    guided_phase = None if refresh else _load_guided_phase(cache_dir)
    if (
        guided_phase is not None
        and set(guided_phase[1]) != set(guided_ids)
    ):
        guided_phase = None
    if guided_phase is None:
        guided_demos = {task_id: demos[task_id] for task_id in guided_ids}
        if guided_ids:
            with serving_lane(
                adapter, policy, gpu, port, cache_dir / "vllm_guided.log"
            ):
                guided, _, guided_rounds = collect_adaptive(
                    adapter, policy, guided_ids, guided_demos
                )
        else:
            guided, guided_rounds = [], {}
        _attach_guided_mu(adapter, policy, guided)
        _write_guided_phase(cache_dir, guided, guided_rounds)
    else:
        guided, guided_rounds = guided_phase

    checker_summary = adapter.checker_summary([*unguided, *guided])
    return CollectionArtifacts(
        unguided_rollouts=tuple(unguided),
        p_hats=p_hats,
        unguided_rounds=unguided_rounds,
        teacher_demos=demos,
        guided_rollouts=tuple(guided),
        guided_rounds=guided_rounds,
        checker_summary=checker_summary,
    )


def _cached_collection(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    task_ids: Sequence[str],
    gpu: str,
    port: int,
    cache_dir: Path,
    *,
    teacher_demos: Mapping[str, Demo] | None = None,
) -> CollectionArtifacts:
    refresh = os.environ.get("BFAS_REFRESH_COLLECT") == "1"
    with CollectionLock(cache_dir):
        cached = None if refresh else _load_collection(cache_dir)
        if cached is not None and teacher_demos is not None:
            expected_guided_ids = {
                task_id
                for task_id in task_ids
                if cached.p_hats[task_id] < 0.5 and task_id in teacher_demos
            }
            if set(cached.guided_rounds) != expected_guided_ids:
                cached = None
        if cached is not None:
            if teacher_demos is None:
                return cached
            return CollectionArtifacts(
                unguided_rollouts=cached.unguided_rollouts,
                p_hats=cached.p_hats,
                unguided_rounds=cached.unguided_rounds,
                teacher_demos=dict(teacher_demos),
                guided_rollouts=cached.guided_rollouts,
                guided_rounds=cached.guided_rounds,
                checker_summary=cached.checker_summary,
            )
        if teacher_demos is None:
            artifacts = _collect_seed_artifacts(
                adapter, policy, task_ids, gpu, port, cache_dir
            )
            cached_artifacts = artifacts
        else:
            artifacts = _collect_seed_artifacts(
                adapter,
                policy,
                task_ids,
                gpu,
                port,
                cache_dir,
                teacher_demos=teacher_demos,
            )
            cached_artifacts = CollectionArtifacts(
                unguided_rollouts=artifacts.unguided_rollouts,
                p_hats=artifacts.p_hats,
                unguided_rounds=artifacts.unguided_rounds,
                teacher_demos={},
                guided_rollouts=artifacts.guided_rollouts,
                guided_rounds=artifacts.guided_rounds,
                checker_summary=artifacts.checker_summary,
            )
        _write_collection(cache_dir, cached_artifacts)
        return artifacts


def _append_log(
    benchmark: str, arm: str, seed: int, metrics: Mapping[str, Any], out_dir: Path
) -> None:
    headline = metrics.get("headline", metrics.get("success_rate", "?"))
    path = ROOT / "notes/exp_log.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"bfas {benchmark} {arm} s{seed} | unified pipeline | local | "
            f"headline={headline} | {out_dir.relative_to(ROOT)}\n"
        )


def run_seed(args: argparse.Namespace, seed: int) -> None:
    out_dir = RESULTS_ROOT / args.benchmark / f"{args.arm}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    port_base = int(os.environ.get("BFAS_PORT_BASE", "8900"))
    try:
        gpu_offset = int(str(args.gpu).split(",", 1)[0])
    except ValueError:
        gpu_offset = 0
    port = port_base + gpu_offset
    adapter = make_adapter(args.benchmark, seed, port)
    policy = os.environ.get(
        f"BFAS_{args.benchmark.upper()}_MODEL",
        DEFAULT_MODELS.get(args.benchmark, ""),
    )
    if not policy:
        raise RuntimeError(f"no base policy configured for {args.benchmark}")
    if args.benchmark == "webshop":
        # Retain the bounded deployment transcript during training as well.
        os.environ.setdefault("AW_MAX_PROMPT_TOKENS", "32768")
    if args.benchmark == "tau2":
        # Native tau2 policies and tool schemas are much longer than the
        # AppWorld-oriented default training cap.  Keep the full serving
        # transcript unless the operator explicitly selects another cap.
        os.environ.setdefault(
            "AW_MAX_PROMPT_TOKENS",
            os.environ.get("BFAS_TAU2_MAX_PROMPT_TOKENS", "16384"),
        )
    if args.benchmark == "appworld" and adapter.name == "appworld" and getattr(
        adapter, "server_backed", False
    ):
        # The official ReAct prompt carries API docs and a worked example
        # (~5k tokens before the task starts); the 4096 cap would truncate it.
        os.environ.setdefault(
            "AW_MAX_PROMPT_TOKENS",
            os.environ.get("BFAS_APPWORLD_MAX_PROMPT_TOKENS", "12288"),
        )

    split = adapter.support_split()
    _write_json(out_dir / "support_split.json", split.as_dict())

    if args.arm == "base":
        if args.dry_run:
            report = audit_pool([], adapter)
            _write_jsonl(out_dir / "pool.jsonl", [])
            _write_json(out_dir / "audit.json", _audit_json(report))
            write_decision_trace(out_dir / "decision_trace.json", report)
        else:
            with serving_lane(adapter, policy, args.gpu, port, out_dir / "vllm_eval.log"):
                metrics = adapter.evaluate(policy, out_dir)
            _append_log(args.benchmark, args.arm, seed, metrics, out_dir)
        return

    benchmark_dir = RESULTS_ROOT / args.benchmark
    demos = _shared_teacher_demos(
        args.benchmark,
        adapter,
        policy,
        split.demand,
        benchmark_dir / "collect_shared",
    )
    cache_dir = benchmark_dir / f"collect_s{seed}"
    collection = _cached_collection(
        adapter,
        policy,
        split.demand,
        args.gpu,
        port,
        cache_dir,
        teacher_demos=demos,
    )
    adapter.prepare_renderer(policy)
    unguided = collection.unguided_rollouts
    p_hats = collection.p_hats
    demos = collection.teacher_demos
    guided = collection.guided_rollouts

    rows = build_pool(
        args.arm,
        adapter,
        teacher_demos=demos,
        unguided_rollouts=unguided,
        guided_rollouts=guided,
        p_hats=p_hats,
    )
    all_rollouts = [*unguided, *guided]
    checker_summary = collection.checker_summary
    report = audit_pool(
        rows,
        adapter,
        rollouts=all_rollouts,
        checker_summary=checker_summary,
        prompt_token_length=_training_prompt_token_length(adapter, policy),
    )

    pool_path = out_dir / "pool.jsonl"
    _write_jsonl(pool_path, rows)
    _write_json(out_dir / "collection_cache.json", {
        "path": str(_collection_path(cache_dir).relative_to(ROOT)),
    })
    _write_json(out_dir / "audit.json", _audit_json(report))
    write_decision_trace(out_dir / "decision_trace.json", report)
    release_policy = getattr(adapter, "release_policy", None)
    if callable(release_policy):
        release_policy()
    if args.dry_run:
        return

    checkpoint = out_dir / "checkpoint"
    if report.should_train:
        _train(
            args.benchmark, args.arm, seed, args.gpu, policy, pool_path, checkpoint
        )
    else:
        _copy_base_policy(policy, checkpoint)
    with serving_lane(adapter, checkpoint, args.gpu, port, out_dir / "vllm_eval.log"):
        metrics = adapter.evaluate(checkpoint, out_dir)
    _append_log(args.benchmark, args.arm, seed, metrics, out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for seed in SEEDS[:args.seeds]:
        run_seed(args, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
