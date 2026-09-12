#!/usr/bin/env python3
"""C26-B CPU support/verify/audit subcommands; outputs must be new paths.

Invoke with an absolute --root from /tmp. Only the existing adapter worker
invokes third-party environments; no downloader, policy, teacher or GPU calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_state import canonical_hash, parent_hash
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, FrozenRenderer, RealStepper, adapter, audit_verified_bank,
    environment_identity, freeze_support, markdown_report, seal_verified_bank,
    validate_support, verify_package,
)
from bfas.rtd.transport import FullState

ROOT = Path(__file__).resolve().parents[1]


def __getattr__(name):
    # Archived CPU replay callers retain their import; the model now comes
    # from the legacy run configuration, without a pinned tokenizer snapshot.
    if name == 'TOKENIZER':
        from bfas.rtd.benchmarks.alfworld_config import load_config, model_directory
        return model_directory(ROOT, load_config(ROOT / 'configs/rtd/v1_alfworld_c26.yaml'))
    raise AttributeError(name)


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bank._write_new(path, value)


def verify(args):
    root = args.root.resolve()
    if Path.cwd().resolve() == root:
        raise ValueError("run verification from /tmp, not the project root")
    if root != adapter.ROOT.resolve():
        raise ValueError("--root must match the imported existing adapter's data root")
    for path in (args.out, args.report):
        if path.exists():
            raise FileExistsError(path)
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    implementation_files = {name: bank.file_hash(root / name) for name in (
        "src/bfas/rtd/benchmarks/alfworld_support.py", "tools/rtd_alfworld_verify.py")}
    env = environment_identity(root, args.tokenizer)
    support = freeze_support(root, env)
    if (support["m"], len(support["historical_task_ids"]), support["historical_parent_groups"],
            len(support["training_task_ids"]), support["fold_parent_counts"]) != (135, 142, 139, 138, {"0": 79, "1": 56}):
        raise ValueError("C26-B historical support counts differ from readiness; review before freezing")
    if args.support.exists():
        if validate_support(json.loads(args.support.read_text())) != support:
            raise ValueError("existing support manifest differs; never overwrite")
    archive = bank.collect_archive(root, tokenizer_path=args.tokenizer / "tokenizer.json")
    if archive.summary["discrepancies"]:
        raise ValueError(f"C26-A inventory discrepancies: {archive.summary['discrepancies']}")
    candidates = [q for q, p in archive.payloads.items() if p["status"] == "candidate"]
    if len(candidates) != 107:
        raise ValueError("expected all 107 candidate packages")
    renderer = FrozenRenderer(args.tokenizer)
    roles = ALFWorldSupport(support)
    payloads, resets, measurements = {}, {}, []
    def stepper():
        return RealStepper(timeout=args.timeout, episode_timeout=args.episode_timeout,
                           environment_hash=env["environment_hash"])
    for index, (q, payload) in enumerate(archive.payloads.items(), 1):
        tid = payload["provenance"]["task_id"]
        request = support["tasks"][tid]["request"]
        if payload["status"] == "candidate":
            started = time.monotonic()
            verified = verify_package(payload, request, stepper(), renderer, support=roles)
            measurements.append(dict(query_id=q, task_id=tid, elapsed_seconds=time.monotonic() - started,
                                     status=verified["status"], reason=verified["unavailable_reason"]))
            if verified["verification"]["states"]:
                reset = verified["verification"]["states"][0]
                if tid in resets and resets[tid] != reset:
                    raise ValueError("same-task reset diverged between attempts")
                resets[tid] = reset
            payloads[q] = verified
            print(f"[{len(measurements)}/107] {verified['status']}: {tid}", flush=True)
        else:
            p = deepcopy(payload)
            p["provenance"]["original_request_state_hash"] = p["request_state_hash"]
            p["request_state_hash"] = canonical_hash(request)
            payloads[q] = p
    # Public reset coverage is independent of whether teacher responses survived.
    reset_errors = {}
    for tid, item in support["tasks"].items():
        if tid in resets:
            continue
        worker = stepper()
        try:
            _, obs = worker.reset(item["request"])
            history = [dict(role="user", index=0, content=obs.observation, admissible=list(obs.admissible), done=obs.done)]
            resets[tid] = asdict(FullState.create(item["request"], history,
                                                renderer(item["request"], history), parent_hash(tid)))
        except (ValueError, OSError, RuntimeError) as exc:
            reset_errors[tid] = dict(error_type=type(exc).__name__, reason=str(exc))
        finally:
            worker.close()
    # Check source/environment bytes again before any final output is published.
    if environment_identity(root, args.tokenizer) != env or freeze_support(root, env) != support:
        raise ValueError("environment/support source changed during replay")
    if any(bank.file_hash(root / name) != h for name, h in implementation_files.items()):
        raise ValueError("verification implementation changed during replay")
    for source in archive.summary["source_files"]:
        if bank.file_hash(root / source["path"]) != source["sha256"]:
            raise ValueError(f"historical source changed: {source['path']}")
    if not args.support.exists():
        _new(args.support, support)
    summary = seal_verified_bank(args.out, archive, support, payloads, resets)
    checked = audit_verified_bank(args.out)
    args.report.mkdir(parents=True, exist_ok=False)
    _new(args.report / "verification.json", summary)
    _new(args.report / "packages.json", measurements)
    _new(args.report / "reset_audit.json", dict(reset_states=len(resets), errors=reset_errors))
    _new(args.report / "integrity.json", checked)
    _new(args.report / "execution.json", dict(
        cwd=str(Path.cwd()), root=str(root), live_tree=True, cpu_only=True,
        timeout_seconds=args.timeout, episode_timeout_seconds=args.episode_timeout,
        worker_cwd="fresh /tmp/rtd-alfworld-c26b-* for each episode",
        worker_environment=dict(ALFWORLD_DATA=str(adapter.DATA.parent), ALFRED_DATA=str(adapter.DATA),
                                ALFWORLD_DATA_ROOT=str(adapter.DATA.parent), CUDA_VISIBLE_DEVICES=""),
        support_file_sha256=bank.file_hash(args.support),
        implementation_files=implementation_files,
    ))
    with (args.report / "verification.md").open("x") as stream:
        stream.write(markdown_report(summary))
    print(markdown_report(summary), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="replay all 107 candidates and seal the new bank")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=ROOT / "data/rtd/v1_alfworld_c26")
    p.add_argument("--report", type=Path, default=ROOT / "results/rtd_v1/alfworld_bank_audit/run_c26b")
    p.add_argument("--support", type=Path, default=ROOT / "configs/rtd/v1_alfworld_support_c26.json")
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtd/v1_alfworld_c26.yaml")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--episode-timeout", type=float, default=120.0)
    p = sub.add_parser("support", help="freeze parent/world/environment manifest without environment calls")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtd/v1_alfworld_c26.yaml")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "configs/rtd/v1_alfworld_support_c26.json")
    p = sub.add_parser("audit", help="verify sealed files and FullState semantics")
    p.add_argument("--bank", type=Path, default=ROOT / "data/rtd/v1_alfworld_c26")
    p.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    if args.command in {'verify', 'support'} and args.tokenizer is None:
        from bfas.rtd.benchmarks.alfworld_config import load_config, model_directory
        args.tokenizer = model_directory(args.root, load_config(args.config))
    if args.command == "verify":
        return verify(args)
    if args.command == "support":
        if args.out.exists():
            raise FileExistsError(args.out)
        result = freeze_support(args.root, environment_identity(args.root, args.tokenizer))
        _new(args.out, result)
        print(json.dumps({k: result[k] for k in ("m", "fold_parent_counts", "manifest_hash")}, indent=2))
    else:
        print(json.dumps(audit_verified_bank(args.bank, expected_manifest_sha256=args.expected_manifest_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
