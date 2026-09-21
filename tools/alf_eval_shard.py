#!/usr/bin/env python3
"""Produce one zero-based shard of an official ALFWorld campaign.

Tasks retain their frozen order: task index i belongs to shard i % N. Run all
shards, wait for them to exit successfully, then finalise with the existing
rtd_alfworld_evaluate.py run command using the same binding/output-root/tag.
This driver never aggregates or writes campaign.json.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile

from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks.alfworld_identity import campaign_identity, guard_manifest
from bfas.rtd.evaluation_lock import evaluation_lock
from bfas.rtd.hardware import checked_hardware, hardware_identity
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, fsync_directory, manifest_digest


class _ShardJournal(ComputeJournal):
    """Refresh the hash chain under the same short lock used by the finaliser."""

    def __init__(self, path, *, lock, tag, timeout):
        # Constructed while the caller holds the campaign lock.
        super().__init__(path, cuda=False)
        self.lock, self.tag, self.timeout = lock, tag, timeout

    def append(self, kind, **values):
        with evaluation_lock(self.lock, tag=self.tag, timeout=self.timeout):
            self.events = ComputeJournal(self.path, cuda=False).events
            return super().append(kind, **values)


def _saved_record(path, expected, identity):
    if path.is_symlink() or not path.is_file():
        raise ValueError("unexpected task artifact")
    envelope = json.loads(path.read_text())
    record = envelope["record"]
    if envelope.get("record_hash") != digest(record) or path.stem != digest(record["task_id"]):
        raise ValueError("corrupted task artifact")
    evaluation.validate_records(expected, [record], identity, complete=False)


def _publish_record(path, record, *, output_root):
    """Stage outside the artifact tree, then atomically publish without replacing.

    Use atomic_json for exactly the serial writer's bytes. A private directory
    on the output filesystem avoids shared .tmp names and incomplete artifacts
    even if a writer is killed. The caller holds the short campaign lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alfworld-shard-", dir=output_root) as temporary:
        staged = Path(temporary) / path.name
        atomic_json(staged, dict(record=record, record_hash=digest(record)))
        os.link(staged, path)  # atomic, and refuses to overwrite an existing record
        fsync_directory(path.parent)


def evaluate_shard(root, manifest, *, output_root, tag, shard, of, device="cuda:0",
                   hardware=None, backend_factory=None, env_factory=None, lock_timeout=None):
    """Write complete official episodes; factories are the serial CPU test seam."""
    if type(of) is not int or of < 1 or type(shard) is not int or not 0 <= shard < of:
        raise ValueError("require --of N >= 1 and zero-based 0 <= --shard I < N")
    lock = evaluation.tag_lock_path(root, tag)
    directory = Path(output_root) / tag
    hardware = manifest["hardware"] if hardware is None else hardware
    for name in ("data_root", "model_path", "tokenizer_path", "environment_root", "run_directory", "checkpoint"):
        value = manifest["paths"].get(name)
        if value and directory.resolve().is_relative_to(Path(value).resolve()):
            raise ValueError("campaign output must be separate from input/training directories")
    current, hashes = guard_manifest(root, manifest, hardware=hardware)
    identity = campaign_identity(current)
    expected = current["evaluation_harness"]["expected"]
    assigned = expected["task_ids"][shard::of]
    artifacts = directory / "artifacts"
    binding = dict(identity=identity, expected=expected, manifest_hash=manifest_digest(manifest))
    lock_tag = f"alfworld/{tag}/shard-{shard}-of-{of}"
    with evaluation_lock(lock, tag=lock_tag, timeout=lock_timeout):
        if directory.exists() and any(p.name not in {"artifacts", "campaign.json", "audit.jsonl"}
                                      or p.is_symlink() for p in directory.iterdir()):
            raise ValueError("existing directory is not an ALFWorld campaign; choose a new output")
        binding_path = artifacts / "binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("incomplete campaign belongs to another identity; refusing regeneration")
        else:
            if artifacts.exists() and any(artifacts.iterdir()):
                raise ValueError("unbound campaign artifacts")
            atomic_json(binding_path, binding)
        if (directory / "campaign.json").exists():
            evaluation.validate_evaluation(directory, identity, expected,
                audited_hashes=hashes, manifest_hash=manifest_digest(manifest))
        records = evaluation._records(artifacts, expected, identity, complete=False)
        done = {r["task_id"] for r in records}
        journal = _ShardJournal(directory / "audit.jsonl", lock=lock, tag=lock_tag, timeout=lock_timeout)

    backend, generated, skipped = None, [], []
    counts = dict(tag=tag, shard=shard, of=of, device=device, manifest_hash=manifest_digest(manifest))
    with ExitStack() as stack:
        measured = False
        try:
            for tid in assigned:
                if tid in done:
                    skipped.append(tid)
                    continue
                path = artifacts / "tasks" / f"{digest(tid)}.json"
                task_lock = lock.parent / "tasks" / digest(tid) / ".lock"
                # Different tasks run concurrently; a duplicate shard waits and
                # rechecks the record before constructing its backend.
                with evaluation_lock(task_lock, tag=lock_tag, timeout=lock_timeout):
                    if path.exists() or path.is_symlink():
                        _saved_record(path, expected, identity)
                        skipped.append(tid)
                        continue
                    if backend is None:
                        if backend_factory is None:
                            # The official hardware guard requires one visible GPU.
                            if device not in ("cuda", "cuda:0"):
                                raise ValueError("official shards require the single visible device cuda:0")
                            if checked_hardware(hardware_identity())["hard"] != hardware["hard"]:
                                raise ValueError("live evaluation hardware class differs")
                            journal.cuda = True
                        stack.enter_context(journal.measure("alfworld_evaluation_shard", **counts))
                        measured = True
                        backend = (evaluation.HFBackend(current, device=device) if backend_factory is None
                                   else backend_factory(current))
                    factory = env_factory or (lambda t: evaluation.EvaluationEnvBridge(t,
                        data_root=manifest["paths"]["data_root"],
                        environment_root=manifest["paths"]["environment_root"]))
                    record = evaluation.official_episode(tid, backend, env_factory=factory, identity=identity)
                    evaluation.validate_records(expected, [record], identity, complete=False)
                    with evaluation_lock(lock, tag=lock_tag, timeout=lock_timeout):
                        # A serial coordinator may have published while we ran.
                        if path.exists() or path.is_symlink():
                            _saved_record(path, expected, identity)
                            skipped.append(tid)
                        else:
                            _publish_record(path, record, output_root=directory.parent)
                            generated.append(tid)
            if not measured:
                # Retain a zero-GPU accounting entry for all-skipped/empty shards.
                stack.enter_context(journal.measure("alfworld_evaluation_shard", **counts))
            final, final_hashes = guard_manifest(root, manifest, hardware=hardware)
            if campaign_identity(final) != identity or final_hashes != hashes:
                raise ValueError("evaluation inputs changed during campaign")
        finally:
            if backend is not None:
                backend.close()
    return dict(tag=tag, shard=shard, of=of, generated=generated, skipped=skipped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--shard", type=int, required=True, help="zero-based shard index")
    parser.add_argument("--of", type=int, required=True, help="total shard count")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lock-timeout", type=float)
    args = parser.parse_args(argv)
    if args.of < 1 or not 0 <= args.shard < args.of:
        parser.error("require --of N >= 1 and zero-based 0 <= --shard I < N")
    manifest = json.loads(args.binding.read_text())
    result = evaluate_shard(args.root, manifest, output_root=args.output_root, tag=args.tag,
        shard=args.shard, of=args.of, device=args.device, lock_timeout=args.lock_timeout)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
