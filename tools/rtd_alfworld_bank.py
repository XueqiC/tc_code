#!/usr/bin/env python3
"""CPU-only C26-A inventory/build/audit. No environment, model or teacher calls.

Run in an isolated checkout while RTD jobs are active (readiness §7.1).
--root is read-only input; --out is an exclusively created output path.
Build archives candidates only: zero packages are usable before C26-B.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfas.rtd.benchmarks.alfworld_bank import (
    LEDGER, TOKENIZER_SNAPSHOT, audit_bank, build_alfworld_bank,
    inventory_alfworld, markdown_report,
)

ROOT = Path(__file__).resolve().parents[1]


def write_report(summary, out):
    out = Path(out)
    if out.suffix in {".json", ".md"}:
        paths = (out.with_suffix(".json"), out.with_suffix(".md"))
        if any(p.exists() for p in paths):
            raise FileExistsError("report output already exists")
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
        paths = (out / "inventory.json", out / "inventory.md")
    for path, text in zip(paths, (json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                  markdown_report(summary))):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "build"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=ROOT, help="read-only historical repository")
        p.add_argument("--ledger", action="append", help="JSONL ledger path relative to --root (repeatable)")
        p.add_argument("--tokenizer", type=Path, default=Path.home() / ".cache/huggingface/hub" /
                       "models--Qwen--Qwen3.5-4B/snapshots" / TOKENIZER_SNAPSHOT / "tokenizer.json")
        p.add_argument("--out", type=Path, default=ROOT / (
            "results/rtd_v1/alfworld_bank_audit" if command == "inventory" else "data/rtd/v1_alfworld_c26"))
    p = sub.add_parser("audit")
    p.add_argument("--bank", type=Path, default=ROOT / "data/rtd/v1_alfworld_c26")
    p.add_argument("--root", type=Path, help="also verify all historical source and tokenizer hashes")
    p.add_argument("--expected-manifest-sha256", help="optional external integrity anchor")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_bank(args.bank, root=args.root,
                                   expected_manifest_sha256=args.expected_manifest_sha256), indent=2))
        return 0
    kwargs = dict(ledger_paths=tuple(args.ledger or [LEDGER]), tokenizer_path=args.tokenizer)
    if args.command == "inventory":
        summary = inventory_alfworld(args.root, **kwargs)
        paths = write_report(summary, args.out)
        print("Reports: " + ", ".join(map(str, paths)))
    else:
        summary = build_alfworld_bank(args.root, args.out, **kwargs)
        print("Sealed candidate archive: " + str(args.out))
    print(json.dumps({k: summary[k] for k in (
        "attempts", "task_ids", "successful_attempts", "failed_attempts", "candidate_packages",
        "usable_packages", "command_turns", "legacy_token_estimate", "retained_command_tokens",
        "retained_prompt_tokens", "discrepancies")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
