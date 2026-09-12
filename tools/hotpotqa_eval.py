#!/usr/bin/env python3
"""Evaluate the fixed first 500 HotpotQA distractor dev questions via ReAct."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import hotpotqa as hp
from bfas.ledger import append_record


def evaluate(*, base_url, model, start=0, n=500, out, offline=False):
    if start < 0 or n < 0 or start + n > 500:
        raise ValueError("--start/--n must select a range within the first 500 dev questions")
    questions = hp.load_questions("dev")[start:start + n]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"base_url": base_url, "model": model, "start": start, "n": n,
              "split": "dev_distractor_first500", "max_steps": hp.MAX_STEPS,
              "max_tokens": hp.MAX_TOKENS, "temperature": 0.0, "prompt_version": hp.PROMPT_VERSION,
              "wiki_version": hp.WIKI_VERSION, "task_ids": [q["_id"] for q in questions]}
    if model.startswith("gpt-5.6-luna"):
        config.update(temperature=None, service_tier=os.environ.get("BFAS_OPENAI_SERVICE_TIER", "").strip() or None)
    with hp.file_lock(out / "evaluation.lock"):
        identity = out / "identity.json"
        if identity.exists():
            if json.loads(identity.read_text()) != config:
                raise ValueError("HotpotQA evaluation identity changed; use a new --out")
        elif (out / "records.jsonl").exists():
            raise ValueError("refusing records without an evaluation identity")
        else:
            hp.write_json(identity, config)
        records_path = out / "records.jsonl"
        records = {}
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                row = json.loads(line)
                if row["task_id"] in records or row["task_id"] not in config["task_ids"]:
                    raise ValueError("duplicate or outside evaluation record")
                records[row["task_id"]] = row
        pending = [q for q in questions if q["_id"] not in records]
        try:
            if pending:
                with hp.make_client(base_url) as client:
                    generate = hp.student_generator(client, model)
                    wiki = hp.Wikipedia(offline=offline)
                    for question in pending:
                        try:
                            record = hp.run_episode(question, wiki, generate)
                        except hp.OfflineCacheMiss as exc:
                            # Leave this question pending for an online/cache-warmed resume.
                            raise hp.OfflineCacheMiss(str(exc)) from exc
                        record["index"] = start + config["task_ids"].index(question["_id"])
                        append_record(records_path, record)
                        records[question["_id"]] = record
        finally:
            metrics = hp.compute_metrics([records[q["_id"]] for q in questions if q["_id"] in records], config)
            metrics.update(requested_n=n, complete=len(records) == n, offline=offline)
            hp.write_json(out / "metrics.json", metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(**vars(args))
    except (ValueError, FileNotFoundError, hp.OfflineCacheMiss) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps({k: metrics[k] for k in ("n", "em", "f1", "mean_steps", "complete")}, indent=2))


if __name__ == "__main__":
    main()
