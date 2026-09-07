#!/usr/bin/env python3
"""Generate new BFCL items as EPISODES -- one mechanism for every seed shape.

The generative object is always an episode: turns of (user utterance, action
sequence which may be empty, reply). Everything else is a degenerate case --
a single-turn item is a one-turn episode, a refusal is one turn with no
action, a silent turn is an empty action mid-episode, termination is the last
reply. Verification is always the benchmark's own judge, supplied by the seed's
shape: a stateful seed executes against the real backend and compares state; a
stateless seed reduces to independent-solve reproduction under the AST checker
(this script routes those to the stateless lane). The mechanism does not change
across shapes; only the judge specializes.

The single-call generator cannot draft for multi_turn or memory tasks, because a
stateful task's demand lives in the turn stream rather than in one call -- and
those are exactly the axes where the student loses most. This follows the
blueprint-then-verify shape APIGen-MT uses for multi-turn agent data, with one
simplification the benchmark hands us for free: BFCL ships the executable
backends (a file system, a memory store, and so on) and scores multi-turn items
by comparing final STATE, so no tool simulator and no simulated human are
needed. The user turns are scripted, so we write them too.

A blueprint is (per-turn user utterances, per-turn ground-truth call sequence)
over a seed's involved classes and initial state. It is accepted only if the
whole call sequence EXECUTES against a fresh instance of those real backends --
an answer that cannot run is not an answer. Execution errors are fed back to the
writer for repair, within the same attempt allowance the teacher gets elsewhere.

Usage: bfcl_generate_mt.py --only multi_turn_base_58 --per-seed 4
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
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

TEACHER = os.environ.get("BFAS_BFCL_TEACHER", "deepseek-v4-pro")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/") + "/v1"
TEACHER_ATTEMPTS = 3

WRITE_PROMPT = """You write multi-turn evaluation episodes for a function-calling
benchmark. The assistant works against a live backend whose starting state is:

{state}

It may call only these functions:
{schemas}

Here is an existing episode over the same backend:
{example}

Write {n} NEW episodes over this same starting state. Each episode is a list of
turns; each turn is what the user says, plus the exact calls that satisfy it.
Requirements:
- every call must be a real function above, written as executable Python, e.g.
  cd(folder='document') or mkdir(dir_name='temp')
- the calls in a turn must run in order against the state left by earlier turns
- ask for things this backend can actually do from THIS starting state
- vary what is asked; do not restate the example
- write {turns} turns or more. Short episodes leave the model untrained on the
  longer conversations it will actually meet, and its errors concentrate in the
  first turns of exactly those.
- SOME turns must need no call at all: give them "calls": []. A real
  conversation contains turns answerable from what is already on screen, and
  turns where a needed detail is missing and the assistant should ask for it
  rather than guess an argument. Roughly one turn in six should be one of
  these. An episode where every turn calls something teaches the model to call
  on every turn, which is what leaves it changing state it should not touch.
- give every turn a "reply": what the assistant says to the user after those
  calls come back. The LAST turn's reply must close the task -- report the
  result in plain words and stop. An episode of nothing but calls teaches a
  model never to finish, which is the single most damaging thing this data can
  do, so the closing reply is not optional.

Reply with JSON only, no prose:
{{"episodes": [{{"turns": [{{"user": "...", "calls": ["fn(arg='v')"],
  "reply": "..."}}]}}]}}"""

REPAIR_PROMPT = """You wrote this episode:
{episode}

Executing its calls against the real backend failed:
{why}

Rewrite the episode so every call runs. Keep the same backend and starting state;
drop or replace whatever cannot execute. Available functions:
{schemas}

Reply with JSON only, no prose:
{{"turns": [{{"user": "...", "calls": ["fn(arg='v')"]}}]}}"""


def api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "")
    if key:
        return key
    for name in (".ollama_api_key2", ".ollama_api_key"):
        path = Path.home() / name
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
    raise SystemExit("no ollama api key available")


def ask(prompt: str, temperature: float, key: str, timeout: int = 240) -> str:
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
    # usage accounting (added 2026-09-04): every teacher call is appended to the ledger
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
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def memory_config(involved: list[str], scenario: str, index: int,
                  workdir: Path) -> tuple[dict, str]:
    """A clean memory backend for one generated episode.

    Memory classes take their state from a snapshot folder rather than from an
    initial_config blob, and they start empty only for an id the harness reads
    as the first entry of a prerequisite chain. A generated episode writes and
    then reads within itself, so it wants exactly that clean start.
    """
    memory_classes = [c for c in involved if c.startswith("MemoryAPI_")]
    category = memory_classes[0].replace("MemoryAPI_", "memory_")
    test_id = f"{category}_prereq_{index}-gen-0"
    # the harness hands each class ITS OWN slice of initial_config, keyed by
    # class name, so a flat dict never reaches the backend
    per_class = {
        "model_result_dir": workdir,
        "test_id": test_id,
        "scenario": scenario or "gen",
    }
    return ({name: dict(per_class) for name in memory_classes}, test_id)


def executes(ground_truth: list[list[str]], initial_config: dict,
             involved: list[str], tag: str) -> tuple[bool, str]:
    """Run the whole answer against fresh backends; an answer must be runnable."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    flat = [call for turn in ground_truth for call in turn]
    if not flat:
        # an episode of nothing but silent turns exercises no tool at all
        return False, "episode makes no calls anywhere"
    try:
        results, _ = execute_multi_turn_func_call(
            flat, initial_config, involved, "Qwen/Qwen3.5-4B-FC", tag,
            long_context=False, is_evaL_run=False)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:220]
    for call, result in zip(flat, results):
        text = str(result)
        if "Error" in text or "error" in text.lower() and "Traceback" in text:
            return False, f"{call} -> {text}"[:220]
    return True, ""


def blueprint(turns: list[dict]) -> tuple[list, list[list[str]], list[str]]:
    question = [[{"role": "user", "content": str(t.get("user", "")).strip()}]
                for t in turns]
    truth = [[str(c) for c in (t.get("calls") or [])] for t in turns]
    # the spoken reply that closes each turn -- without it the training data
    # contains no example of stopping
    replies = [str(t.get("reply", "")).strip() for t in turns]
    return question, truth, replies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", required=True, help="seed multi-turn task id")
    parser.add_argument("--per-seed", type=int, default=4)
    # the benchmark's own multi-turn tasks average 4.2 turns; generated episodes
    # averaged 2.3, so the model never saw the horizon it is tested on
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--out", default="data/bfcl_sft/gen_mt.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    entries, categories = adapter._load_entries()
    entry = entries.get(args.only)
    if entry is None:
        raise SystemExit(f"unknown seed {args.only}")
    involved = entry.get("involved_classes") or []
    initial_config = entry.get("initial_config") or {}
    if not involved:
        # a stateless seed: the same mechanism with the trivial judge -- no
        # state to execute, so verification reduces to independent-solve
        # reproduction under the AST checker, which is the stateless lane
        import subprocess
        cmd = [sys.executable, str(ROOT / "tools/bfcl_generate.py"),
               "--only", args.only, "--per-seed", str(args.per_seed),
               "--out", args.out]
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))
    is_memory = any(c.startswith("MemoryAPI_") for c in involved)
    # WebSearchAPI reads show_snippet out of its config slice and raises without
    # it; the entries themselves carry initial_config: null, so it must be built
    if any(c == "WebSearchAPI" for c in involved) and not initial_config:
        initial_config = {"WebSearchAPI": {"show_snippet": True}}
    workdir = ROOT / "results/analysis/_genmt_memory"
    if is_memory:
        workdir.mkdir(parents=True, exist_ok=True)

    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    populated = populate_test_cases_with_predefined_functions(
        [json.loads(json.dumps(entry))])
    schemas = populated[0].get("function") or []
    official = {}
    for path in (BFCL / "bfcl_eval/data/possible_answer").glob("BFCL_v4_*.json"):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    official[row["id"]] = row["ground_truth"]
                except (json.JSONDecodeError, KeyError):
                    continue
    # a miss_func entry carries a turn with NO user message -- that absence is
    # the point of the category -- so the example builder must tolerate it
    example = json.dumps({
        "turns": [
            {"user": turn[0]["content"] if turn else "", "calls": calls}
            for turn, calls in zip(entry["question"],
                                   official.get(args.only, []))
        ]
    }, ensure_ascii=False)[:3000]

    key = api_key()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("a") as handle:
        raw = ask(WRITE_PROMPT.format(
            state=json.dumps(initial_config, ensure_ascii=False)[:3000],
            schemas=json.dumps(
                [{"name": s.get("name"), "description": s.get("description", "")[:200],
                  "parameters": s.get("parameters")} for s in schemas],
                ensure_ascii=False)[:6000],
            example=example, n=args.per_seed,
            turns=args.min_turns), 0.7, key)
        episodes = (parse_json(raw) or {}).get("episodes") or []
        print(f"[genmt] seed {args.only} -> {len(episodes)} episodes drafted")

        for index, episode in enumerate(episodes):
            turns = episode.get("turns") or []
            if not turns:
                continue
            question, truth, replies = blueprint(turns)
            tag = f"genmt_{args.only}_{index}"
            config, run_id = (
                memory_config(involved, entry.get("scenario") or "gen", index, workdir)
                if is_memory else (initial_config, tag))
            ok, why = executes(truth, config, involved, run_id)
            attempt = 1
            while not ok and attempt < TEACHER_ATTEMPTS:
                attempt += 1
                try:
                    repaired = parse_json(ask(REPAIR_PROMPT.format(
                        episode=json.dumps(episode, ensure_ascii=False)[:2500],
                        why=why,
                        schemas=json.dumps([s.get("name") for s in schemas],
                                           ensure_ascii=False)), 0.7, key))
                except RuntimeError as exc:
                    print(f"    repair aborted: {exc}")
                    break
                if not repaired or not repaired.get("turns"):
                    break
                episode = repaired
                question, truth, replies = blueprint(repaired["turns"])
                config, run_id = (
                    memory_config(involved, entry.get("scenario") or "gen",
                                  index, workdir)
                    if is_memory else (initial_config, tag))
                ok, why = executes(truth, config, involved, run_id)
                print(f"    repair {attempt}: "
                      + ("executes" if ok else why[:70]))
            kept += ok
            handle.write(json.dumps({
                "id": tag,
                "seed_task": args.only,
                "seed_category": categories.get(args.only, ""),
                "question": question,
                "initial_config": {} if is_memory else initial_config,
                "scenario": entry.get("scenario") or "gen",
                "involved_classes": involved,
                "ground_truth": truth,
                "replies": replies,
                "executes": ok,
                "attempts": attempt,
                "note": "" if ok else why,
            }, ensure_ascii=False) + "\n")
            print(f"    {'KEEP' if ok else 'drop'} {len(turns)} turns"
                  + ("" if ok else f"  ({why[:70]})"))
    print(f"\n[genmt] {kept}/{len(episodes)} episodes execute -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
