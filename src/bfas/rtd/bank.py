"""CPU-only BFCL ingestion. This privileged module is not a selector dependency.

Build with: PYTHONPATH=src:. .venv/bin/python -m bfas.rtd.bank --out data/rtd/v1_bfcl
Original archives are read-only. No teacher, checker, model or tokenizer is called.
The BFCL prompt handler is reused; no training/evaluation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..cc_pairs import (Exclusions, digest, normalize_truth, question_content_hash,
                        read_jsonl, render_calls, render_prompt, render_truth, thinking_off)
from .broker import RequestRecord, seal_bank
from .selector import PublicFeatures, PublicQuerySpec
from .transport import FullState
from .caps import archived_cap_evidence, affordability, public_cap, limits_from_metadata

GEN_FILES = ("gen_pool_v3.jsonl", "gen_oos.jsonl")


def parent_hash(entry):
    return digest({k: v for k, v in entry.items() if k not in {"id", "ground_truth", "possible_answer"}})


def parent_fold(hash_value):
    return int(hash_value, 16) % 2


def _flat_tokens(value):
    if isinstance(value, list):
        return sum(_flat_tokens(v) for v in value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("missing or invalid exact historical usage")
    return value


def _response_text(result):
    if isinstance(result, str):
        return result
    if not isinstance(result, list) or not all(isinstance(r, dict) for r in result):
        raise ValueError("not a single complete historical FC action")
    calls = []
    for call in result:
        for name, arguments in call.items():
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise ValueError("unrecoverable historical FC arguments")
            calls.append({name: args})
    return render_calls(calls)


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allocate_estimate(rows, total):
    """Keep recorded file totals; per-item allocation is itself only estimated.

    Earlier usage was discarded. Weight by stored teacher-authored UTF-8 bytes,
    never by reused gen IDs; assign integer remainders in stable row order.
    This cannot recover rejected drafts, verification or repair calls.
    """
    weights = [max(1, len(json.dumps({k: r[k] for k in
               ("question", "ground_truth", "scenario", "initial_config", "why") if k in r},
               ensure_ascii=False).encode())) for r in rows]
    denominator = sum(weights)
    costs = [total * w // denominator for w in weights]
    for i in range(total - sum(costs)):
        costs[i % len(costs)] += 1
    return costs


def build_bfcl_bank(root, directory, *, entries=None, renderer=render_prompt):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    cap_evidence = archived_cap_evidence(root)
    data = root / "data/bfcl_sft"
    bfcl = root / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
    if entries is None:
        from ..adapters.bfcl import BFCLAdapter
        entries = BFCLAdapter()._load_entries()[0]
    split = json.loads((root / "configs/bfcl_support_split.json").read_text())
    demos = json.loads((data / "demos_ds.json").read_text())
    historical = json.loads((root / "results/analysis/bfcl_teacher_tokens.json").read_text())
    raw_gen = {name: read_jsonl(data / name) for name in GEN_FILES}
    events = read_jsonl(data / "pool_events_pref_v3t.jsonl")
    demand = set(split["demand"])
    seeds = {r["seed_task"] for rows in raw_gen.values() for r in rows}
    support_ids = demand | seeds | set(demos)
    missing = support_ids - entries.keys()
    if missing:
        raise ValueError(f"official support parent records missing: {sorted(missing)}")
    hashes = {tid: parent_hash(entries[tid]) for tid in sorted(support_ids)}
    parents = [dict(official_id=tid, parent_hash=h, fold=parent_fold(h),
                    demo_demand=tid in demand, generator_seed=tid in seeds) for tid, h in hashes.items()]

    exclusions = Exclusions()
    calibration = sorted(data.glob("calibration_ids*.json"))
    if not calibration:
        raise ValueError("calibration exclusion manifests required")
    for path in calibration:
        exclusions.load(path)
    for path in (root / "configs/support_split.json", root / "configs/bfcl_support_split.json"):
        exclusions.load(path, heldout_only=True)
    # Protect any previously registered probe content too, using cc_pairs rules.
    for folder in ("behavior_atom_v1", "bfcl_atom_v1"):
        for path in sorted((root / "data" / folder).glob("probes*.json")):
            exclusions.load(path, probe=True)

    prompts = {(name, i): renderer(row["question"], row["function"])
               for name, rows in raw_gen.items() for i, row in enumerate(rows)}
    exclusions.index_prompts([(r["id"], prompts[name, i]) for name, rows in raw_gen.items() for i, r in enumerate(rows)]
                              + [(r["task_id"], r["prompt"]) for r in events])
    records, payloads = [], {}
    requests_by_prompt = defaultdict(list)
    source_files = set(calibration) | {root / "configs/support_split.json", root / "configs/bfcl_support_split.json",
        data / "demos_ds.json", data / "pool_events_pref_v3t.jsonl", root / "results/analysis/bfcl_teacher_tokens.json"}
    source_files.update(data / name for name in GEN_FILES)

    def full_state(entry, h, prompt=None):
        # Only the currently observed turn; never flatten future observations.
        task = {k: entry[k] for k in ("function", "initial_config", "involved_classes", "scenario") if k in entry}
        task["question"] = entry["question"][0]
        raw_prompt = prompt if prompt is not None else renderer(entry["question"], entry.get("function", []))
        return FullState.create(task, entry["question"][0], thinking_off(raw_prompt), h)

    def add(q, state, h, cost, confidence, usage, provenance, response, behaviors,
            *, dependencies=(), unavailable=None):
        limits = limits_from_metadata(response, provenance["kind"])
        evidence = ([dict(source="archived request configuration envelope", fields="max-output/actions/finite retries")]
                    if limits is not None else cap_evidence[provenance["kind"]])
        cap, cap_provenance = public_cap(provenance["kind"], limits=limits, evidence=evidence)
        spec = PublicQuerySpec(q, state.state_hash if state else digest(["unavailable-state", q]),
                               PublicFeatures(), 1, cap, confidence, cap_provenance)
        records.append(RequestRecord(spec, h, tuple(dependencies), unavailable))
        payloads[q] = dict(cost=cost, cost_confidence=confidence, usage=usage, provenance=provenance,
                           historical_response=response, behaviors=behaviors)

    # Reconstruct true attempt identity from raw result files, not the 34-entry
    # verified/merged demo library. Keep all failed attempts and dependency costs.
    raw_attempts, raw_locations = {}, {}
    for attempt in (1, 2, 3):
        for path in sorted((bfcl / f"result_demos_deepseek_v4_pro_FC_a{attempt}").rglob("*_result.json")):
            source_files.add(path)
            for line, row in enumerate(read_jsonl(path), 1):
                key = (attempt, row["id"])
                if key in raw_attempts:
                    raise ValueError("duplicate raw attempt identity")
                raw_attempts[key] = row
                raw_locations[key] = dict(path=str(path.relative_to(root)), line=line)
    demo_qids = {key: digest(["bfcl-demo-attempt", *key]) for key in raw_attempts}
    actual_demo_total = sum(_flat_tokens(r["output_token_count"]) for r in raw_attempts.values())
    declared_demo_total = historical["demos_official_demand"]["total_output_tokens"]
    if actual_demo_total != declared_demo_total:
        raise ValueError("raw attempt total differs from frozen historical cost ledger")
    for key, row in sorted(raw_attempts.items()):
        attempt, tid = key
        q = demo_qids[key]
        entry = entries.get(tid)
        h = hashes.get(tid, parent_hash(entry) if entry else digest(["unresolved-parent", tid]))
        state, behaviors = None, []
        unavailable = None
        deps = tuple(demo_qids[attempt, dep] for dep in (entry or {}).get("depends_on", []) if (attempt, dep) in demo_qids)
        missing_deps = [dep for dep in (entry or {}).get("depends_on", []) if (attempt, dep) not in demo_qids]
        if tid not in support_ids:
            unavailable = "historical prerequisite/out-of-support request; retained in cost audit only"
        elif entry is None or missing_deps:
            unavailable = "missing official state or prerequisite request archive"
        elif tid.startswith(("multi_turn", "memory", "web_search")):
            unavailable = "stateful handler/request cap and exact bridge-state replay not yet reconstructed"
        else:
            try:
                prompt = renderer(entry["question"], entry.get("function", []))
                state = full_state(entry, h, prompt)
                if exclusions.matches(tid, tid, prompt):
                    unavailable = "protected calibration/heldout state"
                text = _response_text(row["result"])
                behaviors = [dict(state=asdict(state), text=text)]
                requests_by_prompt[prompt].append(q)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                unavailable = "unrecoverable complete historical action"
        cost = _flat_tokens(row["output_token_count"])
        usage = dict(output_tokens=cost, input_tokens=_flat_tokens(row["input_token_count"]),
                     reasoning_tokens=None, monetary_cost=None, raw_output_token_count=row["output_token_count"],
                     raw_input_token_count=row["input_token_count"], includes_reasoning="provider boundary unknown")
        add(q, state, h, cost, "exact", usage,
            dict(kind="demo_attempt", task_id=tid, attempt=attempt, **raw_locations[key],
                 dependency_chain=list(deps), missing_dependencies=missing_deps,
                 recorded_L=None if unavailable else 1, compact_demo_available=tid in demos,
                 compact_demo=demos.get(tid), event_aliases=[]), row, behaviors,
            dependencies=deps, unavailable=unavailable)

    for name, rows in raw_gen.items():
        total = historical["generation_pools"][name]["estimate"]
        costs = _allocate_estimate(rows, total)
        for i, (row, cost) in enumerate(zip(rows, costs)):
            # A row is the plan's historical item-level package; legacy id is
            # provenance only. No invented provider request UUID or usage.
            q = digest(["bfcl-generator-item", name, i + 1, question_content_hash(row["question"], row["function"], row["ground_truth"])])
            h, prompt = hashes[row["seed_task"]], prompts[name, i]
            content_hash = question_content_hash(row["question"], row["function"], row["ground_truth"])
            protected = exclusions.matches(row["id"], row["seed_task"], prompt, content_hashes=(content_hash,))
            state = full_state(row, h, prompt)
            # Generated question/truth are BOTH revealed at purchase. Before it,
            # only the official seed request state is known to the selector.
            request_state = full_state(entries[row["seed_task"]], h)
            unavailable, behaviors = None, []
            if protected:
                unavailable = "protected calibration/heldout content"
            try:
                text = render_truth(normalize_truth(row["ground_truth"]), row["function"], row.get("seed_category", "simple_python"))
                behaviors = [dict(state=asdict(state), text=text)]
            except (ValueError, TypeError, KeyError):
                unavailable = unavailable or "unrecoverable teacher-authored action"
            usage = dict(output_tokens=None, estimated_output_tokens=cost, input_tokens=None,
                         reasoning_tokens=None, monetary_cost=None,
                         estimation="UTF-8 authored-field proportional allocation of archived file token estimate",
                         missing_costs=["rejected drafts", "repair calls", "teacher verification", "provider reasoning boundary"])
            add(q, request_state, h, cost, "estimated", usage,
                dict(kind="generator_item", path=f"data/bfcl_sft/{name}", line=i + 1,
                     legacy_id=row["id"], parent_id=row["seed_task"], content_hash=content_hash,
                     prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), event_aliases=[],
                     boundary="archived item; provider batching/repair request IDs and L not recoverable",
                     replay_L=1, conclusion_scope="exploratory historical item replay"), row, behaviors,
                unavailable=unavailable)
            requests_by_prompt[prompt].append(q)

    # Events are aliases of paid evidence, never independently purchasable rows.
    # Match content AND rendered response, never gen id alone. Do not use event
    # correctness, Delta-U or rejected/student outputs for features or filtering.
    event_mapping = []
    for line, event in enumerate(events, 1):
        matches = requests_by_prompt.get(event["prompt"], [])
        exact = [q for q in matches if any(b["text"].strip() == event["response"].strip() for b in payloads[q]["behaviors"])]
        # Multiple attempts with identical responses are distinct paid requests;
        # attach provenance to all matching records, not a new package.
        selected = exact
        # An event can serialize a generator's authored truth differently (or
        # even contain an old formatting error). Its unchanged complete prompt
        # still identifies the originating item when the authored content is
        # unambiguous. Preserve that event for audit, without substituting it
        # for the independently archived authored action.
        generated = [q for q in matches if payloads[q]["provenance"]["kind"] == "generator_item"]
        if (not selected and event.get("teacher") in {"teacher_authored_gt", "teacher_authored_abstain"}
                and generated and len({payloads[q]["provenance"]["content_hash"] for q in generated}) == 1):
            selected = generated
        for q in selected:
            payloads[q]["provenance"]["event_aliases"].append(dict(path="data/bfcl_sft/pool_events_pref_v3t.jsonl", line=line,
                                                                 identical_target=q in exact))
            payloads[q].setdefault("historical_events", []).append(event)
        event_mapping.append(dict(line=line, package_ids=selected,
                                  status="alias" if selected else "unavailable: no matching full-state historical response"))

    # Keep dependencies even if their parent lies outside m; those chains are
    # unavailable, never silently treated as free prefix state.
    available = [r for r in records if r.unavailable_reason is None]
    summary = dict(version="rtd-v1-bfcl-bank", mode="sealed_replay", scope="exploratory",
        m=len(set(hashes.values())), parents=parents,
        fold_counts=dict(Counter(parent_fold(h) for h in set(hashes.values()))),
        demo_demand_tasks=len(demand), compact_verified_demos=len(demos), generator_seed_tasks=len(seeds),
        generator_new_parents=sorted(seeds - demand),
        demand_attempts=sum(tid in demand for _, tid in raw_attempts),
        demand_attempt_output_tokens=sum(_flat_tokens(r["output_token_count"]) for (_, tid), r in raw_attempts.items() if tid in demand),
        historical_demo_output_tokens_exact=actual_demo_total,
        historical_generation_output_tokens_estimated=historical["generation_total"]["estimate"],
        selected_generator_file_estimates={name: historical["generation_pools"][name]["estimate"] for name in GEN_FILES},
        total_archived_packages=len(records), available_packages=len(available),
        unavailable_reasons=dict(Counter(r.unavailable_reason for r in records if r.unavailable_reason)),
        bank_cost=sum(payloads[r.spec.query_id]["cost"] for r in available),
        bank_public_cap_sum=sum(r.spec.cost_upper_bound for r in available),
        protocol_version="1.0.1", budget_basis="usable_public_cap_sum",
        available_cost_by_confidence={c: sum(payloads[r.spec.query_id]["cost"] for r in available if r.spec.cost_confidence == c)
                                      for c in ("exact", "estimated")},
        generator_rows={name: len(rows) for name, rows in raw_gen.items()},
        event_rows=len(events), event_alias_rows=sum(bool(e["package_ids"]) for e in event_mapping),
        event_teacher_labels=dict(Counter(e["teacher"] for e in events)),
        recorded_demo_attempt_costs={f"a{a}": sum(_flat_tokens(r["output_token_count"]) for (attempt, _), r in raw_attempts.items()
                                                if attempt == a) for a in (1, 2, 3)},
        reused_generator_ids=sum(v > 1 for v in Counter(r["id"] for rows in raw_gen.values() for r in rows).values()),
        exclusions=exclusions.sources,
        source_files=[dict(path=str(p.relative_to(root)), sha256=_file_hash(p)) for p in sorted(source_files)],
        limitations=["generator package boundary is archived item, not recovered provider request; costs are lower estimates",
                     "stateful demos retained with exact costs/dependencies but unavailable for acquisition",
                     "v1.0.1 public replay convention: demo=65536/item=8192; archived calls omit caps and finite demo retries",
                     "bank costs and source audit are privileged research accounting, never selector inputs"])
    summary["cap_configuration_audit"] = cap_evidence
    summary["budget_affordability"] = affordability(records, payloads)
    seal_bank(directory, records, payloads)
    # Audit and event response correspondence are privileged, not public features.
    (directory / "sealed/audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (directory / "sealed/event_aliases.json").write_text(json.dumps(event_mapping, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_bfcl_bank(args.root, args.out)
    print(json.dumps({k: summary[k] for k in ("m", "demand_attempts", "total_archived_packages", "available_packages",
                                            "bank_cost", "unavailable_reasons", "event_alias_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
