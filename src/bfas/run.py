"""Run the unified BFAS pipeline for one benchmark and arm."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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
from typing import Any, Iterator

from .adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, Turn
from .arms import ARM_NAMES, build_pool, trainer_environment
from .audit import AuditReport, audit_pool, write_decision_trace
from .protocol import (
    AdaptiveSampler,
    SAMPLING_TEMPERATURE,
    SEEDS,
    TEACHER_ATTEMPTS,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results/bfas"
DEFAULT_MODELS = {
    "appworld": "Qwen/Qwen3.5-2B",
    "bfcl": "Qwen/Qwen3.5-4B",
    "alfworld": "Qwen/Qwen3.5-2B",
}


def positive_seed_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= len(SEEDS):
        raise argparse.ArgumentTypeError(f"must be between 1 and {len(SEEDS)}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, choices=("appworld", "bfcl", "alfworld", "tau2")
    )
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--seeds", type=positive_seed_count, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_adapter(name: str, seed: int, port: int) -> BenchmarkAdapter:
    if name == "appworld":
        from .adapters.appworld import AppWorldAdapter

        return AppWorldAdapter(seed)
    if name == "bfcl":
        from .adapters.bfcl import BFCLAdapter

        return BFCLAdapter(seed, port=port, mt_training=False)
    if name == "alfworld":
        from .adapters.alfworld import ALFWorldAdapter

        return ALFWorldAdapter(seed)
    from .adapters.tau2_stub import Tau2Adapter

    return Tau2Adapter()


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


def _turn_json(turn: Turn) -> dict[str, Any]:
    return {"prompt": turn.prompt, "target": turn.target}


def _rollout_json(rollout: Rollout) -> dict[str, Any]:
    raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
    return {
        "task_id": rollout.task_id,
        "verified": rollout.verified,
        "turns": [_turn_json(turn) for turn in rollout.turns],
        "checker_verified": raw.get("checker_verified"),
        "category": raw.get("category"),
    }


def _demo_json(demo: Demo) -> dict[str, Any]:
    return {
        "task_id": demo.task_id,
        "worked_example": demo.worked_example,
        "turns": [_turn_json(turn) for turn in demo.turns],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class VLLMServer:
    def __init__(self, model: PolicyRef, gpu: str, port: int, log_path: Path):
        self.model = str(model)
        self.gpu = gpu
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[Any] | None = None
        self._log: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        self.process = subprocess.Popen(
            [
                str(ROOT / "envs/vllm-serve/.venv/bin/vllm"),
                "serve", self.model,
                "--served-model-name", "Qwen/Qwen3.5-4B",
                "--port", str(self.port),
                "--gpu-memory-utilization", os.environ.get("GPU_UTIL", "0.85"),
                "--max-model-len", "32768",
            ],
            cwd=ROOT,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
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
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log is not None:
            self._log.close()


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


@contextmanager
def serving_lane(
    adapter: BenchmarkAdapter,
    policy: PolicyRef,
    gpu: str,
    port: int,
    log_path: Path,
) -> Iterator[None]:
    if adapter.name != "bfcl":
        yield
        return
    with PortRegistry(port):
        server = VLLMServer(policy, gpu, port, log_path)
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
    shutil.copytree(trained, checkpoint)


def _audit_json(report: AuditReport) -> dict[str, Any]:
    return {
        "row_count": report.row_count,
        "checked_round_trips": report.checked_round_trips,
        "should_train": report.should_train,
        "decision_trace": report.decision_trace,
    }


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

    with serving_lane(adapter, policy, args.gpu, port, out_dir / "vllm_collect.log"):
        unguided, p_hats, unguided_rounds = collect_adaptive(
            adapter, policy, split.demand
        )
        adapter.serving_probe()
        demos = adapter.teacher_demo(split.demand, TEACHER_ATTEMPTS)
        guided_ids = [
            task_id for task_id in split.demand
            if p_hats[task_id] < 0.5 and task_id in demos
        ]
        guided_demos = {task_id: demos[task_id] for task_id in guided_ids}
        guided, _, guided_rounds = collect_adaptive(
            adapter, policy, guided_ids, guided_demos
        ) if guided_ids else ([], {}, {})
    _attach_guided_mu(adapter, policy, guided)

    rows = build_pool(
        args.arm,
        adapter,
        teacher_demos=demos,
        unguided_rollouts=unguided,
        guided_rollouts=guided,
        p_hats=p_hats,
    )
    all_rollouts = [*unguided, *guided]
    checker_summary = adapter.checker_summary(all_rollouts)
    report = audit_pool(
        rows,
        adapter,
        rollouts=all_rollouts,
        checker_summary=checker_summary,
    )

    pool_path = out_dir / "pool.jsonl"
    _write_jsonl(pool_path, rows)
    _write_json(out_dir / "teacher_demos.json", {
        task_id: _demo_json(demo) for task_id, demo in sorted(demos.items())
    })
    _write_json(out_dir / "rollouts.json", {
        "unguided": [_rollout_json(rollout) for rollout in unguided],
        "guided": [_rollout_json(rollout) for rollout in guided],
        "p_hat": p_hats,
        "unguided_rounds": unguided_rounds,
        "guided_rounds": guided_rounds,
        "checker_summary": checker_summary,
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
