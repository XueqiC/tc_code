#!/usr/bin/env python3
"""Build the BFCL v3 (ours) training pool and the STaR pool from
verified student rollouts.

- Unguided verified rollouts: behavior policy equals the training
  policy and context, so the importance ratio is exactly one; rows
  carry no _mu fields.
- Guided verified rollouts: collected with the worked example in
  context but trained without it; rows carry _mu_nll_sum/_mu_ntok
  computed under the guided context with the trainer serialization
  (appworld_train.encode), so the trainer forms the clipped ratio.
- STaR pool: unguided verified rollouts only, plain rows.

Needs one GPU pass for guided mu. Run with the project venv.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

GUIDE_HEADER = (
    "Worked example from an expert on this exact task. Study the "
    "approach, then solve the task yourself:\n"
)


def serialize(messages: list[dict]) -> str:
    return "\n".join(
        f"<|{m['role']}|>\n{m['content']}" for m in messages
    ) + "\n<|assistant|>\n"


_HANDLER = None


def _handler():
    global _HANDLER
    if _HANDLER is None:
        from bfcl_eval.model_handler.local_inference.qwen_fc import (
            QwenFCHandler,
        )
        _HANDLER = QwenFCHandler(
            model_name="Qwen/Qwen3.5-4B-FC", temperature=0.001,
            registry_name="Qwen/Qwen3.5-4B-FC", is_fc_model=True)
    return _HANDLER


_MT_FUNCS: dict = {}


def _mt_functions(tid: str, entries: dict):
    if tid not in _MT_FUNCS:
        from bfcl_eval.utils import (
            populate_test_cases_with_predefined_functions,
        )
        entry = entries.get(tid)
        if entry is None:
            _MT_FUNCS[tid] = None
        else:
            try:
                pop = populate_test_cases_with_predefined_functions(
                    [json.loads(json.dumps(entry))])
                _MT_FUNCS[tid] = pop[0].get("function")
            except Exception:
                _MT_FUNCS[tid] = None
    return _MT_FUNCS[tid]


def load_verified(kind: str, repeats: int) -> dict[str, list[str]]:
    """id -> list of verified response strings across rollouts."""
    out: dict[str, list[str]] = {}
    for r in range(repeats):
        rdir = BFCL / f"result_roll_{kind}_r{r}"
        sdir = BFCL / f"score_roll_{kind}_r{r}"
        if not rdir.exists():
            continue
        failed: set[str] = set()
        for f in sdir.rglob("*_score.json"):
            lines = [json.loads(l) for l in f.open() if l.strip()]
            for e in lines[1:]:
                if isinstance(e, dict) and "id" in e:
                    failed.add(e["id"])
        for f in rdir.rglob("*_result.json"):
            for line in f.open():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["id"] in failed:
                    continue
                resp = e["result"]
                if not isinstance(resp, str):
                    # multi-turn results are call structures; keep the
                    # compact JSON form as the training target
                    resp = json.dumps(resp, ensure_ascii=False)
                out.setdefault(e["id"], []).append(resp)
    return out


def strip_guide_block(messages: list[dict]) -> list[dict]:
    """Remove the worked example the guided rollout saw.

    The rollout is collected with the demo injected into the first user turn,
    but the student must be trained on the context it will actually meet at
    evaluation time, so the block is taken back out of the training prompt
    while mu stays measured under the context that produced the response.
    """
    out = copy.deepcopy(messages)
    for m in out:
        content = m.get("content")
        if not isinstance(content, str) or GUIDE_HEADER not in content:
            continue
        head, _, rest = content.partition(GUIDE_HEADER)
        _, sep, tail = rest.partition("\n\n")
        m["content"] = head + (tail if sep else rest)
        break
    return out


def entry_messages(entry: dict, guided_demo: dict | None) -> list[dict]:
    from bfcl_eval.model_handler.utils import (
        system_prompt_pre_processing_chat_model,
    )
    e2 = copy.deepcopy(entry)
    if guided_demo is not None:
        calls = json.dumps(guided_demo["teacher_result"], ensure_ascii=False)
        if len(calls) > 4000:
            calls = calls[:4000] + " ..."
        for m in e2["question"][0]:
            if m.get("role") == "user":
                m["content"] = (
                    f"{GUIDE_HEADER}Expert solution: {calls}\n\n" + m["content"]
                )
                break
    e2["question"][0] = system_prompt_pre_processing_chat_model(
        e2["question"][0], e2["function"], e2["id"]
    )
    return list(e2["question"][0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tag", default="ds")
    ap.add_argument("--allow-empty-categories", action="store_true",
                    help="write the pool even when a scored category ends up "
                         "with no rows; without this the build aborts, because "
                         "a silent zero-row category costs its whole axis")
    ap.add_argument("--no-mt", action="store_true",
                    help="exclude multi-turn dump rows; every "
                         "construction of them degraded the stateful "
                         "categories, so the supported configuration "
                         "trains on single-turn advantage rows only")
    args = ap.parse_args()

    split = json.load((ROOT / "configs/bfcl_support_split.json").open())
    demand = set(split["demand"])
    demos = json.load((ROOT / f"data/bfcl_sft/demos_{args.tag}.json").open())

    # entries come from the adapter, not from raw data-file membership: a
    # memory task is stored under its group id and only exists under its
    # per-backend id once the group is expanded, so filename membership drops
    # the whole memory axis without saying anything
    from bfas.adapters.bfcl import BFCLAdapter

    all_entries = BFCLAdapter()._load_entries()[0]
    entries: dict[str, dict] = {
        task_id: row for task_id, row in all_entries.items() if task_id in demand
    }
    missing = demand - set(entries)
    if missing:
        print(f"[bfclv3pool] WARNING {len(missing)} demand ids have no entry: "
              f"{sorted(missing)[:5]}")

    # memory prerequisite write-phase entries ride along with every selective
    # run and are never scored, so load_verified counts them as passes; they are
    # setup turns, not tasks, and must not become training rows
    def demand_only(d: dict) -> dict:
        return {tid: v for tid, v in d.items() if tid in demand}

    unguided = demand_only(load_verified("base", args.repeats))
    guided = demand_only(load_verified("guided", args.repeats))
    # dedicated multi-turn rollout pass (message dumps enabled)
    # overrides the earlier base statistics for its tasks
    mt = demand_only(load_verified("mt", args.repeats))
    for tid, resps in mt.items():
        unguided[tid] = resps

    # empirical per-task success rate over the K unguided rollouts;
    # rows carry it as _task_phat and the trainer's advantage weight
    # (1 - p-hat) makes converged tasks contribute nothing
    n_attempts: dict[str, int] = {}
    mt_tasks: set[str] = set()
    for r in range(args.repeats):
        for kind in ("base", "mt"):
            rdir = BFCL / f"result_roll_{kind}_r{r}"
            if not rdir.exists():
                continue
            for f in rdir.rglob("*_result.json"):
                for line in f.open():
                    if line.strip():
                        tid = json.loads(line)["id"]
                        if kind == "mt":
                            if tid not in mt_tasks:
                                mt_tasks.add(tid)
                                n_attempts[tid] = 0
                            n_attempts[tid] += 1
                        elif tid not in mt_tasks:
                            n_attempts[tid] = n_attempts.get(tid, 0) + 1

    unguided_phat = {
        tid: len(resps) / max(n_attempts.get(tid, args.repeats), 1)
        for tid, resps in unguided.items()
    }
    rows: list[dict] = []
    star_rows: list[dict] = []
    for tid, resps in sorted(unguided.items()):
        # multi-turn and stateful entries carry no top-level function
        # docs; they stay out of the training rows for every method,
        # matching the baseline pools
        if "function" not in entries[tid]:
            continue
        phat = len(resps) / max(n_attempts.get(tid, args.repeats), 1)
        prompt = _handler()._format_prompt(
            list(entries[tid]["question"][0]), entries[tid]["function"])
        allow_prose = "irrelevance" in tid or "relevance" in tid
        for k, resp in enumerate(resps):
            if not allow_prose and "<tool_call>" not in resp:
                continue
            row = {
                "task_id": tid, "teacher": "self", "turn_index": 0,
                "prompt": prompt,
                "response": resp,
                "token_hint": max(len(resp) // 4, 1),
                "_task_phat": round(phat, 4),
                "_traj": f"{tid}#u{k}",
            }
            rows.append(row)
            star_rows.append(dict(row))

    # multi-turn rows come from message dumps written by the patched
    # handler (BFCL_DUMP_MESSAGES): the dumped stream is the exact
    # evaluation-time conversation, one training row per assistant turn
    dump_dir = ROOT / "data/bfcl_dumps"
    mt_rows = 0
    if dump_dir.exists() and not args.no_mt:
        # a dump row is admissible only if THAT rollout passed the
        # checker; task-level verification is not enough, since the
        # first dumped trajectory may be a failed attempt
        failed_by_r: dict[int, set[str]] = {}
        for r in range(args.repeats):
            failed: set[str] = set()
            for f in (BFCL / f"score_roll_mt_r{r}").rglob("*_score.json"):
                lines = [json.loads(l) for l in f.open() if l.strip()]
                for e in lines[1:]:
                    if isinstance(e, dict) and "id" in e:
                        failed.add(e["id"])
            failed_by_r[r] = failed
        generated_by_r: dict[int, set[str]] = {}
        for r in range(args.repeats):
            gen: set[str] = set()
            rdir = BFCL / f"result_roll_mt_r{r}"
            if rdir.exists():
                for f in rdir.rglob("*_result.json"):
                    for line in f.open():
                        if line.strip():
                            gen.add(json.loads(line)["id"])
            generated_by_r[r] = gen
        seen_dump: set[str] = set()
        for df in sorted(dump_dir.glob("roll_dump_r*.jsonl")):
            r = int(df.stem.rsplit("r", 1)[1])
            for line in df.open():
                if not line.strip():
                    continue
                d = json.loads(line)
                tid = d["id"]
                if tid not in demand or tid in seen_dump:
                    continue
                if tid not in generated_by_r.get(r, set()):
                    continue
                if tid in failed_by_r.get(r, set()):
                    continue
                seen_dump.add(tid)
                phat = len(unguided.get(tid, [])) / 4
                msgs = d["messages"]
                funcs = _mt_functions(tid, entries)
                if funcs is None:
                    continue
                for i, m in enumerate(msgs):
                    if m.get("role") != "assistant":
                        continue
                    # two admissible target kinds: call turns, rebuilt
                    # from the structured tool_calls field in the exact
                    # emission format, and TURN-FINAL prose turns (the
                    # next message is a user turn or the stream ends),
                    # which carry the termination skill; intermediate
                    # prose is skipped. Training calls only taught the
                    # model to never conclude, which zeroed the
                    # categories that require a final answer.
                    calls = m.get("tool_calls") or []
                    if calls:
                        content = "\n".join(
                            "<tool_call>\n"
                            + json.dumps(tc, ensure_ascii=False)
                            + "\n</tool_call>"
                            for tc in calls
                        )
                    else:
                        content = str(m.get("content", "")).strip()
                        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                        turn_final = nxt is None or nxt.get("role") == "user"
                        if not content or not turn_final:
                            continue
                    # render the context through the SERVING handler so
                    # training and evaluation see byte-identical prompts
                    prompt = _handler()._format_prompt(msgs[:i], funcs)
                    rows.append({
                        "task_id": tid, "teacher": "self",
                        "turn_index": i,
                        "prompt": prompt,
                        "response": content,
                        "token_hint": max(len(content) // 4, 1),
                        "_task_phat": round(phat, 4),
                        "_traj": f"{tid}#d",
                    })
                    mt_rows += 1
    if mt_rows:
        print(f"[bfclv3pool] dump-based multi-turn rows: {mt_rows}")

    guided_pending = [
        (tid, resps) for tid, resps in sorted(guided.items())
        if tid not in unguided and "function" in entries.get(tid, {})
    ]
    if guided_pending:
        import torch
        import appworld_train as tr
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.student)
        model = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for tid, resps in guided_pending:
            g_msgs = entry_messages(entries[tid], demos.get(tid))
            clean_msgs = entry_messages(entries[tid], None)
            for k, resp in enumerate(resps):
                mu_row = {
                    "messages": g_msgs,
                    "prompt": serialize(g_msgs),
                    "response": resp,
                }
                ids, labels = tr.encode(tok, mu_row)
                ids = ids.to(model.device)
                labels = labels.to(model.device)
                with torch.inference_mode():
                    out = model(input_ids=ids, labels=labels)
                ntok = int((labels != -100).sum())
                rows.append({
                    "task_id": tid, "teacher": "self", "turn_index": 0,
                    "messages": clean_msgs, "prompt": serialize(clean_msgs),
                    "response": resp,
                    "token_hint": max(len(resp) // 4, 1),
                    "_task_phat": 0.0,
                    "_mu_nll_sum": round(float(out.loss.item()) * ntok, 4),
                    "_mu_ntok": ntok,
                    "_traj": f"{tid}#g{k}",
                })

    # Guided rows for STATEFUL tasks. A stateful task carries its signal in the
    # turn stream rather than in one response, so it is emitted the way the mt
    # pass is -- one row per assistant turn. Without this branch a stateful task
    # that only succeeds under guidance can never become training rows, and a
    # category whose single draw is such a task contributes nothing at all.
    guided_stateful = sorted(
        tid for tid in guided
        if tid not in unguided and "function" not in entries.get(tid, {})
    )
    gs_rows = 0
    if guided_stateful:
        import torch
        import appworld_train as tr
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.student)
        model = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        seen_gs: set[str] = set()
        for r in range(args.repeats):
            dump_path = ROOT / f"data/bfcl_dumps/guided_dump_r{r}.jsonl"
            if not dump_path.exists():
                continue
            failed: set[str] = set()
            for f in (BFCL / f"score_roll_guided_r{r}").rglob("*_score.json"):
                lines = [json.loads(l) for l in f.open() if l.strip()]
                for e in lines[1:]:
                    if isinstance(e, dict) and "id" in e:
                        failed.add(e["id"])
            generated: set[str] = set()
            for f in (BFCL / f"result_roll_guided_r{r}").rglob("*_result.json"):
                for line in f.open():
                    if line.strip():
                        generated.add(json.loads(line)["id"])
            for line in dump_path.open():
                if not line.strip():
                    continue
                d = json.loads(line)
                tid = d["id"]
                if (tid not in guided_stateful or tid in seen_gs
                        or tid not in generated or tid in failed):
                    continue
                funcs = _mt_functions(tid, entries)
                if funcs is None:
                    continue
                seen_gs.add(tid)
                msgs = d["messages"]
                clean = strip_guide_block(msgs)
                for i, m in enumerate(msgs):
                    if m.get("role") != "assistant":
                        continue
                    calls = m.get("tool_calls") or []
                    if calls:
                        content = "\n".join(
                            "<tool_call>\n"
                            + json.dumps(tc, ensure_ascii=False)
                            + "\n</tool_call>"
                            for tc in calls
                        )
                    else:
                        content = str(m.get("content", "")).strip()
                        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                        turn_final = nxt is None or nxt.get("role") == "user"
                        if not content or not turn_final:
                            continue
                    guided_prompt = _handler()._format_prompt(msgs[:i], funcs)
                    clean_prompt = _handler()._format_prompt(clean[:i], funcs)
                    ids, labels = tr.encode(
                        tok,
                        {"prompt": guided_prompt, "response": content},
                    )
                    ids = ids.to(model.device)
                    labels = labels.to(model.device)
                    with torch.inference_mode():
                        out = model(input_ids=ids, labels=labels)
                    ntok = int((labels != -100).sum())
                    rows.append({
                        "task_id": tid, "teacher": "self", "turn_index": i,
                        "prompt": clean_prompt,
                        "response": content,
                        "token_hint": max(len(content) // 4, 1),
                        "_task_phat": 0.0,
                        "_mu_nll_sum": round(float(out.loss.item()) * ntok, 4),
                        "_mu_ntok": ntok,
                        "_traj": f"{tid}#gs{r}",
                    })
                    gs_rows += 1
        if gs_rows:
            print(f"[bfclv3pool] guided stateful rows: {gs_rows} "
                  f"from {len(seen_gs)} tasks")

    # Coverage invariant, checked on REALIZED ROWS rather than on sampled tasks.
    # A category can be sampled, rolled out and still contribute nothing -- if its
    # draw is a task the student never solves and the guided path cannot rescue --
    # and nothing downstream says so. That is how multi_turn_base reached the
    # trainer with zero rows and took Multi Turn from 44.88 to 4.38.
    from bfas.adapters.bfcl import BFCLAdapter as _Adapter

    categories = _Adapter().task_categories()
    demand_categories = {categories[tid] for tid in demand}
    covered = {categories[r["task_id"]] for r in rows}
    empty = sorted(demand_categories - covered)
    if empty:
        print("[bfclv3pool] COVERAGE INVARIANT VIOLATED — "
              f"{len(empty)} scored categories have no rows:")
        for category in empty:
            tasks = sorted(t for t in demand if categories[t] == category)
            detail = ", ".join(
                f"{t} (phat={unguided_phat.get(t, 0.0):.2f}, "
                f"demo={'yes' if t in demos else 'no'})" for t in tasks
            )
            print(f"    {category}: {detail}")
        if not args.allow_empty_categories:
            print("[bfclv3pool] aborting; pass --allow-empty-categories to "
                  "write the pool anyway")
            return 1

    out_dir = ROOT / "data/bfcl_sft"
    pool = out_dir / f"pool_bfcl_{args.tag}_v3.jsonl"
    star = out_dir / f"pool_bfcl_star.jsonl"
    with pool.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with star.open("w") as fh:
        for r in star_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_g = sum(1 for r in rows if "_mu_nll_sum" in r)
    print(f"[bfclv3pool] rows={len(rows)} (guided {n_g}) "
          f"tasks={len({r['task_id'] for r in rows})} star_rows={len(star_rows)} "
          f"-> {pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
