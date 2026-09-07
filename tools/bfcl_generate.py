#!/usr/bin/env python3
"""Generate new BFCL queries aimed at the points the student is failing.

The few-shot support set is fixed at k=50; the distillation set is not. This
builds new queries from the ones the student actually fails, so the training
data covers the boundary instead of replaying the same fifty points.

Targeting is at QUERY granularity and uses no benchmark category label: the
suspicious thing is the individual query the student gets wrong. Each seed
query yields variations that must exercise the same call pattern with different
content, so the capability demand is preserved while the surface changes.

A generated item is only worth training on if its answer is known. The teacher
emits (query, ground-truth call) together; an independent solve pass that never
sees the ground truth must then reproduce it, judged by the benchmark's own AST
checker. Items that fail that reproduction are kept in the output and marked,
not silently dropped, so the yield is visible rather than hidden.

A failed check is also fed back: the writer is shown what the checker objected
to and asked to repair that item, for as many attempts as the teacher protocol
allows anywhere else in this project. APIGen-MT reports 70% blueprint yield
with such a loop against 28% without, and our own one-shot yield on hard seeds
was 33% -- the same order as their figure without feedback.

Usage: bfcl_generate.py --per-seed 4 [--max-seeds N] [--out data/bfcl_sft/gen_v1.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER_ATTEMPTS = 3  # same allowance the teacher gets everywhere else here
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

TEACHER = os.environ.get("BFAS_BFCL_TEACHER", "deepseek-v4-pro")
# The AST checker resolves this name in the benchmark's registry, and for models
# flagged underscore_to_dot it rewrites dots in the GROUND TRUTH function name
# (uber.ride -> uber_ride) to match what those APIs accept. We prompt the teacher
# with the raw schema instead of going through that path, so both sides must be
# judged under a config that leaves names alone -- the student's own registered
# name does that, and is also the name the trained rows will be scored under.
CHECKER_MODEL = os.environ.get("BFAS_BFCL_CHECKER_MODEL", "Qwen/Qwen3.5-4B-FC")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/") + "/v1"

WRITE_PROMPT = """You write evaluation items for a function-calling benchmark.

Available functions (JSON schemas):
{schemas}

Here is one existing item:
  user query: {query}
  correct call: {answer}

The example calls these functions, in this order: {names}. Write {n} NEW items
that call exactly those same functions, in that same order, about different
content and with different argument values. Use the function names exactly as
spelled above. Each query must be answerable by precisely the calls you give,
using only the schemas above, and must read as something a real person would ask.

For every argument give a LIST of all acceptable values, not one value. Free
text an assistant could word differently (addresses, titles, search phrases)
should list each acceptable wording; a value with only one correct form is a
one-element list. This is how the benchmark stores answers, and it is what
keeps a correct call from being marked wrong over a paraphrase.

Reply with JSON only, no prose:
{{"items": [{{"query": "...", "calls": [{{"name": "fn",
  "arguments": {{"param": ["value", "alternative wording"]}}}}]}}]}}"""

REPAIR_PROMPT = """You wrote this evaluation item:
  query: {query}
  calls: {calls}

An independent solver, shown only the query and the schemas, produced calls that
the benchmark checker judged different from yours:
  {why}

Either the query is ambiguous about an argument, or your answer disagrees with
what the query actually asks. Rewrite the item so the query determines the calls
unambiguously, keeping the same functions. Widen an argument's acceptable-value
list where a solver could reasonably word it differently.

Available functions (JSON schemas):
{schemas}

Reply with JSON only, no prose:
{{"query": "...", "calls": [{{"name": "fn", "arguments": {{"param": ["value"]}}}}]}}"""

REFUSE_PROMPT = """You write evaluation items for a function-calling benchmark.

These are the ONLY functions an assistant has:
{schemas}

Write {n} user queries that this function set CANNOT answer. Each query must be
close enough to the domain that a careless assistant would try to call one of
these functions anyway, but the functions genuinely cannot do what is asked --
a missing capability, an unsupported entity, an action outside their scope.
Do not write nonsense or off-topic questions; the point is the boundary, not
absurdity.

Reply with JSON only, no prose:
{{"items": [{{"query": "...", "why_out_of_scope": "..."}}]}}"""

SOLVE_PROMPT = """Available functions (JSON schemas):
{schemas}

User query: {query}

Reply with JSON only, no prose, giving the call(s) that answer the query:
{{"calls": [{{"name": "fn", "arguments": {{...}}}}]}}"""


def api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "")
    if key:
        return key
    for name in (".ollama_api_key2", ".ollama_api_key"):
        path = Path.home() / name
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
    raise SystemExit("no ollama api key available")


def ask(prompt: str, temperature: float, key: str, timeout: int = 180) -> str:
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps({
            "model": TEACHER,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"teacher HTTP {exc.code}: "
                           f"{exc.read(200).decode('utf-8', 'replace')}") from exc
    # usage accounting (added 2026-09-04): every teacher call is appended to the
    # ledger so the generation pools carry an exact output-token cost from now on
    try:
        import datetime, pathlib
        _ledger = pathlib.Path(__file__).resolve().parents[1] / "data/teacher_ledger/bfcl_generation_usage.jsonl"
        _ledger.parent.mkdir(parents=True, exist_ok=True)
        with _ledger.open("a") as _fh:
            _fh.write(json.dumps({"model": TEACHER, "usage": payload.get("usage"), "tool": pathlib.Path(__file__).name,
                                  "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
    except Exception as _exc:  # accounting must never break generation
        print(f"[usage-ledger] {_exc}", file=sys.stderr)
    return payload["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict | None:
    """Teachers wrap JSON in prose and fences often enough to be worth handling."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def as_ground_truth(calls: list[dict]) -> list[dict]:
    """Benchmark ground truth lists each parameter's acceptable values.

    The writer is asked for lists; a bare scalar still arrives sometimes and is
    promoted rather than rejected, since the shape is a formatting slip and not
    a wrong answer.
    """
    truth = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments") or {}
        if not name or not isinstance(args, dict):
            return []
        truth.append({
            name: {k: (v if isinstance(v, list) else [v]) for k, v in args.items()}
        })
    return truth


def reproduces(schemas: list, query: str, truth: list[dict],
               category: str, key: str) -> tuple[bool, str]:
    """An independent solve, never shown the answer, judged by the AST checker."""
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language

    raw = ask(SOLVE_PROMPT.format(
        schemas=json.dumps(schemas, ensure_ascii=False),
        query=query), 0.0, key)
    parsed = parse_json(raw)
    if not parsed or "calls" not in parsed:
        return False, "solve pass returned no parseable calls"
    model_output = [{c["name"]: c.get("arguments", {})}
                    for c in parsed["calls"] if isinstance(c, dict) and "name" in c]
    if not model_output:
        return False, "solve pass returned no calls"
    try:
        verdict = ast_checker(schemas, model_output, truth,
                              Language.PYTHON, category, CHECKER_MODEL)
    except Exception as exc:
        return False, f"checker raised {type(exc).__name__}: {exc}"
    return bool(verdict.get("valid")), str(verdict.get("error", ""))[:120]


def declines(schemas: list, query: str, key: str) -> tuple[bool, str]:
    """Mirror of `reproduces` for out-of-scope items.

    An in-scope item is accepted when an independent solve reproduces the
    answer. An out-of-scope item is accepted when an independent solve, shown
    only the query and the schemas, also declines to call anything. If the
    solver does call something, the query was answerable after all and the item
    would teach the model to refuse a request it should have served.
    """
    raw = ask(SOLVE_PROMPT.format(
        schemas=json.dumps(schemas, ensure_ascii=False), query=query), 0.0, key)
    parsed = parse_json(raw)
    calls = (parsed or {}).get("calls") or []
    named = [c for c in calls if isinstance(c, dict) and c.get("name")]
    if named:
        return False, f"solver called {named[0]['name']}; query is in scope"
    return True, ""


def seeds(limit: int | None) -> list[tuple[str, float]]:
    """Failing queries first — suspicion at query granularity, no labels."""
    phat = json.loads((ROOT / "results/analysis/phat_v4.json").read_text())
    ordered = sorted(phat.items(), key=lambda kv: kv[1])
    picked = [(t, p) for t, p in ordered if p < 1.0]
    return picked[:limit] if limit else picked


def generate_out_of_scope(args, entries, categories, key, out_path) -> int:
    """Queries the tools cannot serve, whose correct answer is no call at all.

    The pool otherwise contains only requests the functions can satisfy, so the
    model learns that a request always means a call. These items carry the other
    half of the boundary: where T stops.
    """
    seed_ids = ([args.only] if args.only
                else [t for t, _ in seeds(args.max_seeds)])
    kept = attempted = 0
    with out_path.open("w") as handle:
        for task_id in seed_ids:
            entry = entries.get(task_id)
            if entry is None or "function" not in entry:
                continue
            schemas = entry["function"]
            try:
                raw = ask(REFUSE_PROMPT.format(
                    schemas=json.dumps(schemas, ensure_ascii=False),
                    n=args.per_seed), 0.7, key)
            except RuntimeError as exc:
                print(f"[gen-oos] {task_id}: {exc}")
                break
            items = (parse_json(raw) or {}).get("items") or []
            print(f"[gen-oos] seed {task_id} -> {len(items)} drafted")
            for index, item in enumerate(items):
                attempted += 1
                query = str(item.get("query", "")).strip()
                if not query:
                    continue
                ok, why = declines(schemas, query, key)
                kept += ok
                handle.write(json.dumps({
                    "id": f"oos_{task_id}_{index}",
                    "seed_task": task_id,
                    "seed_category": categories.get(task_id, ""),
                    "question": [[{"role": "user", "content": query}]],
                    "function": schemas,
                    "ground_truth": [],
                    "out_of_scope": True,
                    "why": str(item.get("why_out_of_scope", ""))[:200],
                    "verified": ok,
                    "note": "" if ok else why,
                }, ensure_ascii=False) + "\n")
                print(f"    {'KEEP' if ok else 'drop'} {query[:66]}"
                      + ("" if ok else f"  ({why})"))
    print(f"\n[gen-oos] drafted {attempted}, both sides declined {kept} "
          f"-> {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", type=int, default=4)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--out-of-scope", action="store_true",
                        help="generate queries the function set CANNOT answer; "
                             "the correct behaviour is to decline without "
                             "calling anything. Without these the pool only "
                             "ever shows the model requests it can satisfy, so "
                             "it learns to call on everything -- which is what "
                             "collapsed Irrelevance Detection.")
    parser.add_argument("--only", default=None,
                        help="generate for this seed query alone (the loop "
                             "spends one round on one point)")
    parser.add_argument("--out", default="data/bfcl_sft/gen_v1.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    entries, categories = adapter._load_entries()
    answers: dict[str, list] = {}
    for path in (BFCL / "bfcl_eval/data/possible_answer").glob("BFCL_v4_*.json"):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    answers[row["id"]] = row["ground_truth"]
                except (json.JSONDecodeError, KeyError):
                    continue

    key = api_key()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = attempted = 0

    if args.out_of_scope:
        return generate_out_of_scope(args, entries, categories, key, out_path)
    with out_path.open("w") as handle:
        chosen = seeds(args.max_seeds)
        if args.only:
            phats = json.loads(
                (ROOT / "results/analysis/phat_v4.json").read_text())
            chosen = [(args.only, float(phats.get(args.only, 0.0)))]
        for task_id, phat in chosen:
            entry = entries.get(task_id)
            truth = answers.get(task_id)
            category = categories.get(task_id, "")
            # stateful items carry their demand in the turn stream, not in one
            # call, so variation generation for them needs its own design
            if entry is None or truth is None or "function" not in entry:
                print(f"[gen] skip {task_id} (no single-call ground truth)")
                continue
            schemas = entry["function"]
            allowed = {s.get("name") for s in schemas if isinstance(s, dict)}
            seed_names = [name for call in truth for name in call]
            question = entry["question"][0]
            query = next((m["content"] for m in question
                          if m.get("role") == "user"), "")
            try:
                raw = ask(WRITE_PROMPT.format(
                    schemas=json.dumps(schemas, ensure_ascii=False),
                    query=query,
                    answer=json.dumps(truth, ensure_ascii=False),
                    names=", ".join(seed_names),
                    n=args.per_seed), 0.7, key)
            except RuntimeError as exc:
                print(f"[gen] {task_id}: {exc}")
                break
            parsed = parse_json(raw)
            items = (parsed or {}).get("items") or []
            print(f"[gen] seed {task_id} (phat={phat:.2f}) -> {len(items)} drafted")
            for index, item in enumerate(items):
                attempted += 1
                new_query = str(item.get("query", "")).strip()
                new_truth = as_ground_truth(item.get("calls") or [])
                if not new_query or not new_truth:
                    continue
                # a generated call naming a function that is not in the schema
                # set can never be answered, so it is a drafting error rather
                # than a disagreement worth spending a solve pass on
                written = [name for call in new_truth for name in call]
                if not set(written) <= allowed:
                    print(f"    drop {new_query[:60]}  (invented function "
                          f"{sorted(set(written) - allowed)})")
                    continue
                ok, why = reproduces(schemas, new_query, new_truth, category, key)
                attempt = 1
                while not ok and attempt < TEACHER_ATTEMPTS:
                    attempt += 1
                    try:
                        repaired = parse_json(ask(REPAIR_PROMPT.format(
                            query=new_query,
                            calls=json.dumps(new_truth, ensure_ascii=False),
                            why=why,
                            schemas=json.dumps(schemas, ensure_ascii=False),
                        ), 0.7, key))
                    except RuntimeError as exc:
                        print(f"    repair aborted: {exc}")
                        break
                    if not repaired or not repaired.get("query"):
                        break
                    candidate_query = str(repaired["query"]).strip()
                    candidate_truth = as_ground_truth(repaired.get("calls") or [])
                    names = [n for call in candidate_truth for n in call]
                    if not candidate_truth or not set(names) <= allowed:
                        break
                    new_query, new_truth = candidate_query, candidate_truth
                    ok, why = reproduces(schemas, new_query, new_truth,
                                         category, key)
                    print(f"    repair {attempt}: "
                          f"{'reproduced' if ok else why[:60]}")
                kept += ok
                handle.write(json.dumps({
                    "id": f"gen_{task_id}_{index}",
                    "seed_task": task_id,
                    "seed_phat": phat,
                    "seed_category": category,
                    "question": [[{"role": "user", "content": new_query}]],
                    "function": schemas,
                    "ground_truth": new_truth,
                    "reproduced": ok,
                    "attempts": attempt,
                    "note": "" if ok else why,
                }, ensure_ascii=False) + "\n")
                print(f"    {'KEEP' if ok else 'drop'} {new_query[:70]}"
                      + ("" if ok else f"  ({why})"))
    print(f"\n[gen] drafted {attempted}, reproduced {kept} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
