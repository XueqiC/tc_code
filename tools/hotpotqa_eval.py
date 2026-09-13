#!/usr/bin/env python3
"""Evaluate HotpotQA or a frozen OOD multi-hop prefix through the same ReAct harness."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import hotpotqa as hp
from bfas import hotpotqa_audit as audit, multihop as mh
from bfas.ledger import append_record


class EvaluationFailed(RuntimeError):
    """An incomplete/errored campaign is not a benchmark result."""


def evaluate(*, base_url, model, start=0, n=None, out, offline=None,
             dataset="hotpotqa", data_dir=None, manifest_dir=mh.MANIFEST_DIR):
    if dataset == "hotpotqa":
        n = 500 if n is None else n
        if start < 0 or n < 0 or start + n > 500:
            raise ValueError("--start/--n must select a range within the first 500 dev questions")
        questions = (hp.load_questions("dev") if data_dir is None else
                     hp.load_questions("dev", data_dir=data_dir))[start:start + n]
        annotations = mh.hotpotqa_annotations(questions, data_dir=data_dir)
        selection = None  # Preserve the official HotpotQA dev selection.
        split = "dev_distractor_first500"
    else:
        if start != 0:
            raise ValueError("OOD multi-hop evaluation requires --start 0: first n stored IDs only")
        questions, annotations, selection = mh.select_questions(
            dataset, n, data_dir=mh.DATA if data_dir is None else data_dir, manifest_dir=manifest_dir)
        n, split = len(questions), selection["split"]
    offline = hp.retrieval_offline(offline)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"base_url": base_url, "model": model, "start": start, "n": n,
              "split": split, "max_steps": hp.MAX_STEPS,
              "max_tokens": hp.MAX_TOKENS, "temperature": 0.0, "prompt_version": hp.PROMPT_VERSION,
              "wiki_version": hp.WIKI_VERSION, "offline": offline,
              "task_ids": [q["_id"] for q in questions]}
    config.update(dataset=dataset, scorer_version=hp.SCORER_VERSION,
                  questions_sha256=mh.digest(questions), annotations_sha256=mh.digest(annotations),
                  step_audit=dict(version=audit.VERSION, match_rule=audit.MATCH_RULE, report_only=True,
                                  field_applicability=audit.FIELD_APPLICABILITY))
    if selection is not None:
        config["selection"] = selection
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
        failure = None
        try:
            if len(questions) != n:
                raise EvaluationFailed(f"HotpotQA requested {n} questions but loaded {len(questions)}")
            if any(r.get("error") for r in records.values()):
                raise EvaluationFailed("HotpotQA saved records contain episode errors; use a new --out")
            if pending:
                wiki = hp.Wikipedia(offline=offline)
                wiki.preflight()
                with hp.make_client(base_url) as client:
                    generate = hp.student_generator(client, model)
                    for question in pending:
                        try:
                            record = hp.run_episode(question, wiki, generate)
                        except hp.OfflineCacheMiss as exc:
                            # Leave this question pending for a cache-warmed resume.
                            raise hp.OfflineCacheMiss(str(exc)) from exc
                        audit.annotate_episode(record, annotations.get(question["_id"], {}), hp.parse_action)
                        record["index"] = start + config["task_ids"].index(question["_id"])
                        append_record(records_path, record)
                        records[question["_id"]] = record
                        if record.get("error"):
                            raise EvaluationFailed(f"HotpotQA episode {question['_id']} failed: {record['error']}")
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metrics = hp.compute_metrics([records[q["_id"]] for q in questions if q["_id"] in records], config)
            complete = failure is None and len(records) == n and not metrics["errored_episodes"]
            metrics.update(requested_n=n, complete=complete, offline=offline,
                           status="complete" if complete else "failed")
            metrics["annotation_status_counts"] = dict(Counter(
                r["supporting_facts_audit"]["status"] for r in records.values()))
            if not complete:
                metrics["error"] = failure or f"HotpotQA evaluated {len(records)} of {n} requested questions"
                # Retain diagnostic partial scores without exposing aggregateable results.
                scores = ("em", "f1", "headline", "mean_score", "per_category")
                metrics["partial_metrics"] = {k: metrics[k] for k in scores}
                metrics.update({k: None for k in scores})
            hp.write_json(out / "metrics.json", metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", *mh.DATASETS), default="hotpotqa")
    parser.add_argument("--data-dir", type=Path, help="already-local data; never downloaded or modified")
    parser.add_argument("--manifest-dir", type=Path, default=mh.MANIFEST_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, help="stored prefix length; default: full frozen inventory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", default=None)
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(**vars(args))
    except (ValueError, FileNotFoundError, hp.OfflineCacheMiss, hp.WikiError, EvaluationFailed) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps({k: metrics[k] for k in ("n", "em", "f1", "mean_steps", "complete")}, indent=2))


if __name__ == "__main__":
    main()
