#!/usr/bin/env python3
"""Derive a frozen ALFWorld bank without purchases, replay, or ledger writes.

Select explicit package IDs (a JSON list or repeated --package-id), all usable
packages, or either half of the deterministic cabinet/drawer sweep ranking.
Optional --config registers the bank using the unchanged Kang pi1 recipe and
counts supervision with the cached student tokenizer and pi1's own encoder.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_support import (
    _signed, audit_verified_bank, seal_verified_bank,
)

SWEEP_PATTERN = r"^(go to|open|close) (cabinet|drawer) \d+$"
BANK_FIELDS = frozenset({"bank", "sealed_manifest_sha256", "support_size",
                         "demonstrations", "supervised_turns", "bank_supervised_tokens"})


def read_json(path):
    return json.loads(Path(path).read_text())


def usable_packages(source):
    """Read only auditor-accepted, usable full-ReAct packages."""
    source = Path(source).resolve()
    audit_verified_bank(source)
    packages = {}
    for record in read_json(source / "public/requests.json"):
        if record["unavailable_reason"] is None:
            qid = record["spec"]["query_id"]
            payload = read_json(source / "sealed" / f"{qid}.json")
            if payload.get("payload_kind") != "teacher_react_turns":
                raise ValueError("pi1 subsets require full teacher ReAct turns")
            packages[qid] = payload
    return packages


def sweep_split(packages):
    """Rank exact fractions, then opaque package IDs; split an even inventory."""
    if not packages or len(packages) % 2:
        raise ValueError("sweep split requires a nonempty even package count")
    rows = []
    for qid, payload in packages.items():
        commands = payload["commands"]
        if not commands:
            raise ValueError("sweep share requires nonempty commands")
        matches = sum(re.match(SWEEP_PATTERN, command) is not None for command in commands)
        rows.append(dict(package_id=qid, sweep_commands=matches, commands=len(commands),
                         sweep_share=matches / len(commands)))
    rows.sort(key=lambda r: (Fraction(r["sweep_commands"], r["commands"]), r["package_id"]))
    ids = [r["package_id"] for r in rows]
    half = len(ids) // 2
    return dict(pattern=SWEEP_PATTERN, sort=["sweep share ascending (exact fraction)",
                "package id ascending"], packages=rows, low_ids=ids[:half], high_ids=ids[half:])


def materialize_subset(source, output, package_ids, *, selection_rule=None,
                       expected_source_sha256=None):
    """Copy selected sealed bytes; regenerate all dependent manifests atomically.

    Support stays the full original frozen support, including tasks with no
    selected package. Historical inventory and collection identities remain
    source identities; training method is specified only in the pi1 config.
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source):
        raise ValueError("output must be outside the source bank")
    source_sha = bank.file_hash(source / "sealed/manifest.json")
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("source manifest differs from requested hash")
    packages = usable_packages(source)
    ids = list(package_ids)
    if (not ids or any(not isinstance(q, str) for q in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("package IDs must be nonempty, unique strings")
    if set(ids) - packages.keys():
        raise ValueError("selection contains unknown or unavailable package IDs")
    ids.sort()
    payloads = {qid: packages[qid] for qid in ids}
    support = _signed(read_json(source / "public/support.json"))
    records = [bank.public_record(qid, support["tasks"][p["provenance"]["task_id"]]["request"])
               for qid, p in payloads.items()]
    historical = read_json(source / "sealed/audit.json")["historical_inventory"]
    archive = bank.Archive(records, payloads, {},
                          read_json(source / "sealed/event_aliases.json"), historical)
    derivation = dict(source_bank=str(source), source_sealed_manifest_sha256=source_sha,
                      selection_rule=selection_rule or dict(kind="explicit package IDs"),
                      included_package_ids=ids, new_teacher_calls=0, new_teacher_tokens=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staged = Path(tmp) / "bank"
        seal_verified_bank(staged, archive, support, payloads,
                           read_json(source / "public/reset_states.json"))
        for qid in ids:
            # Never reserialize the source package: its original bytes are evidence.
            shutil.copyfile(source / "sealed" / f"{qid}.json", staged / "sealed" / f"{qid}.json")
        manifest_path = staged / "sealed/manifest.json"
        manifest = read_json(manifest_path)
        source_manifest = read_json(source / "sealed/manifest.json")
        for name in manifest["artifacts"]:
            manifest["artifacts"][name] = bank.file_hash(staged / name)
        for qid in ids:
            name = f"sealed/{qid}.json"
            if manifest["artifacts"][name] != source_manifest["artifacts"][name]:
                raise ValueError("source package changed during copy")
        manifest["derivation"] = derivation
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        audit = audit_verified_bank(staged)
        audit_verified_bank(source, expected_manifest_sha256=source_sha)
        # An empty destination is reserved exclusively to refuse concurrent writes.
        output.mkdir(exist_ok=False)
        try:
            staged.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return audit


def register_config(output, config_path, *, template=None, model_path=None):
    """Use load_bank + encode_teacher_turn, exactly as alf_pi1_train --preflight."""
    import yaml
    from bfas.rtd.baselines.pi1 import encode_teacher_turn, load_bank, load_config
    from bfas.rtd.benchmarks.alfworld_identity import tokenizer_identity
    from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer
    from tools.bfcl_hub_merge_export import _snapshot_for_model

    config_path = Path(config_path)
    if config_path.exists() or config_path.is_symlink():
        raise FileExistsError(config_path)
    template = Path(template or ROOT / "configs/rtd/pi1_alfworld_k32_kang.yaml")
    config = load_config(template)
    output = Path(output).resolve()
    audit = audit_verified_bank(output)
    summary = read_json(output / "sealed/audit.json")
    config.update(bank=str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
                  sealed_manifest_sha256=audit["manifest_sha256"], support_size=summary["m"],
                  demonstrations=summary["usable_packages"], supervised_turns=summary["command_turns"])
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    model = Path(model_path or _snapshot_for_model(config["student"])).resolve()
    renderer = FrozenRenderer(model)
    rows, identity = load_bank(output, config, renderer)
    tokenizer = renderer.adapter._tokenizer
    config["bank_supervised_tokens"] = sum(len(encode_teacher_turn(
        tokenizer, row, config["max_context_tokens"])["target_ids"]) for row in rows)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x") as stream:
        stream.write("# Frozen subset under the unchanged pre-registered Kang pi1 recipe.\n")
        yaml.safe_dump(config, stream, sort_keys=False)
    load_config(config_path)
    return dict(**identity, bank_supervised_tokens=config["bank_supervised_tokens"],
                authored_tokens=config["bank_supervised_tokens"] - len(rows), boundary_tokens=len(rows),
                tokenizer_path=str(model), tokenizer_hash=tokenizer_identity(model)["hash"],
                token_rule="sum(len(pi1.encode_teacher_turn(...)[target_ids])); full replies + <turn|> (106)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--package-ids", type=Path, help="JSON list of opaque package IDs")
    selection.add_argument("--package-id", action="append", help="repeat for each selected package")
    selection.add_argument("--selection", choices=("all-usable", "sweep-low", "sweep-high"))
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--config", type=Path, help="write a new pi1 registration")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--model-path", type=Path, help="local cached tokenizer snapshot")
    args = parser.parse_args(argv)
    if args.config and (args.config.exists() or args.config.is_symlink()):
        raise FileExistsError(args.config)
    # Pin the same source across selection, audit, and copying.
    source_sha = bank.file_hash(args.source / "sealed/manifest.json")
    if args.expected_source_sha256 and args.expected_source_sha256 != source_sha:
        raise ValueError("source manifest differs from requested hash")
    rule = dict(kind=args.selection or "explicit package IDs")
    report = {}
    if args.package_ids:
        ids = read_json(args.package_ids)
        if not isinstance(ids, list):
            raise ValueError("--package-ids requires a JSON list")
    elif args.package_id:
        ids = args.package_id
    else:
        packages = usable_packages(args.source)
        if args.selection == "all-usable":
            ids = sorted(packages)
            rule["predicate"] = "public unavailable_reason is null (audited usable full-ReAct package)"
        else:
            report["sweep_split"] = split = sweep_split(packages)
            half = args.selection.removeprefix("sweep-")
            ids = split[f"{half}_ids"]
            rule.update(pattern=SWEEP_PATTERN, denominator="all commands in package",
                        sort=split["sort"], half=half, source_packages=len(packages), take=len(ids))
    report["audit"] = materialize_subset(args.source, args.output, ids, selection_rule=rule,
                                          expected_source_sha256=source_sha)
    if args.config:
        report["registration"] = register_config(args.output, args.config,
                                                 template=args.template, model_path=args.model_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
