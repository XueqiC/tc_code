#!/usr/bin/env python3
"""Evaluate a served student on WebShop with one fixed ReAct-style example.

Run with envs/webshop/venv/bin/python (Python 3.8, openai>=1.0), for example:
    tools/webshop_eval.py --base-url http://localhost:8000/v1 --model student \
        --out results/webshop/student

Reuse an output directory only for the same model and evaluation settings.
Completed sessions are skipped; metrics cover the requested session range.
Select --prompt-version v1 (default) or v2, or set WEBSHOP_PROMPT_VERSION.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Optional


WEBSHOP_REPO = Path(__file__).resolve().parents[1] / "envs/webshop/repo"
MAX_STEPS = 15
OBS_CHARS = 6000
HISTORY_OBS_CHARS = 600
MAX_PROMPT_CHARS = 60000
SYSTEM_PROMPT = """You are shopping in WebShop. Follow the user's shopping instruction
and buy the product that best matches all requested attributes, options, and price.
You can take two forms of action:
search[<query>] searches for products when a search bar is available.
click[<button text>] clicks a visible product ID, button, or option. Use the text
shown on the page. Select the required options before clicking Buy Now.
Respond with a single search[...] or click[...] action on its own line.
You may put one short Thought: line before the action. Do not invent observations.
"""

# One authored demonstration, fixed across sessions and runs.
WORKED_EXAMPLE = """Worked example:
Observation: WebShop [SEP] Instruction: Find a blue insulated stainless steel water bottle, 24 oz, under $30. [SEP] Search
Action: search[blue insulated stainless steel water bottle 24 oz]
Observation: Results [SEP] B0BOTTLE24 [SEP] Trail Bottle, insulated stainless steel, $24.99 [SEP] B09GLASS12 [SEP] Glass bottle, $18.00
Action: click[B0BOTTLE24]
Observation: Trail Bottle [SEP] Price: $24.99 [SEP] Color: blue, black [SEP] Size: 18 oz, 24 oz [SEP] Buy Now
Action: click[blue]
Observation: Color selected: blue [SEP] Size: 18 oz, 24 oz [SEP] Buy Now
Action: click[24 oz]
Observation: Color selected: blue [SEP] Size selected: 24 oz [SEP] Price: $24.99 [SEP] Buy Now
Action: click[Buy Now]
"""

# v1 constants above remain unchanged for existing callers and snapshots.
# v2 follows docs/webshop_prompt_v2_draft.md in the main tree.
SYSTEM_PROMPT_V2 = """You are shopping in WebShop. Follow the user's shopping instruction
and buy the product that best matches all requested attributes, options, and price.
You can take two forms of action:
search[<query>] searches for products when a search bar is available.
click[<button text>] clicks a visible product ID, button, or option. Use the text
shown on the page. Select the required options before clicking Buy Now.
Before clicking Buy Now, verify every required attribute, option, and price
against the instruction on the item page. Select every required option.
If the item page lacks a required option or the price exceeds the limit,
use click[< Prev] or click[Back to Search] and try another candidate.
Never buy from the search page; open an item page and check it first.
Respond with a single search[...] or click[...] action on its own line.
You may put one short Thought: line before the action. Do not invent observations.
"""

WORKED_EXAMPLE_V2 = """Worked example:
Observation: WebShop [SEP] Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars [SEP] Search
Thought: I need 3 ounce, bright citrus, for sensitive skin, and under $50.
Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: [Back to Search] [SEP] Page 1 (Total results: 50) [SEP] [Next >] [SEP] B078GWRC1J [SEP] Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B078GTKVXY [SEP] Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B08KBVJ4XN [SEP] Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) [SEP] $15.95
Thought: B078GWRC1J is bright citrus, 3 ounce, sensitive skin, $10.99 < $50. It matches everything; I will check its options.
Action: click[B078GWRC1J]
Observation: [Back to Search] [SEP] [< Prev] [SEP] scent [SEP] assorted scents [SEP] bright citrus [SEP] calming lavender [SEP] ginger fresh [SEP] simply non-scents [SEP] size [SEP] travel set (4-pack) [SEP] 3 ounce (pack of 1) [SEP] 3-ounce (2-pack) [SEP] Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] Price: $10.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
Thought: The instruction needs scent bright citrus and size 3 ounce; both options exist, so I select them before buying.
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Thought: Scent, size and price all match the instruction. Buy.
Action: click[Buy Now]
"""

PROMPT_VERSIONS = ("v1", "v2")


def resolve_prompt_version(prompt_version: Optional[str] = None) -> str:
    """Explicit selection wins over the environment; resolve once per run."""
    version = (os.environ.get("WEBSHOP_PROMPT_VERSION", "v1")
               if prompt_version is None else prompt_version)
    if version not in PROMPT_VERSIONS:
        raise ValueError("WEBSHOP_PROMPT_VERSION / --prompt-version must be v1 or v2")
    return version


def get_prompts(prompt_version: Optional[str] = None):
    if resolve_prompt_version(prompt_version) == "v2":
        return SYSTEM_PROMPT_V2, WORKED_EXAMPLE_V2
    return SYSTEM_PROMPT, WORKED_EXAMPLE


def default_obs_chars(prompt_version: Optional[str] = None) -> int:
    return 6000 if resolve_prompt_version(prompt_version) == "v2" else OBS_CHARS


ACTION_PATTERN = re.compile(r"(?:search|click)\[([^\[\]\r\n]+)\]")


def parse_action(response: Optional[str]) -> Optional[str]:
    """Take the last complete action line, allowing an optional Action: label."""
    for line in reversed((response or "").splitlines()):
        line = line.strip()
        if line.startswith("Action:"):
            line = line[len("Action:"):].strip()
        match = ACTION_PATTERN.fullmatch(line)
        if match and match.group(1).strip():
            return line
    return None


def build_messages(history, observation: str, obs_chars: int,
                   history_obs_chars: int = HISTORY_OBS_CHARS,
                   max_prompt_chars: int = MAX_PROMPT_CHARS,
                   prompt_version: Optional[str] = None):
    """Rebuild bounded messages without modifying the caller's history."""
    messages, _ = _build_messages_with_drops(
        history, observation, obs_chars, history_obs_chars, max_prompt_chars, prompt_version
    )
    return messages


def _build_messages_with_drops(history, observation, obs_chars, history_obs_chars,
                               max_prompt_chars, prompt_version=None):
    system_prompt, worked_example = get_prompts(prompt_version)
    prefix = worked_example.rstrip() + "\n\nLive episode:\n"
    turns = []
    for previous_observation, response in history:
        head = previous_observation[:history_obs_chars]
        if len(previous_observation) > history_obs_chars:
            head += " ..."
        turns.append(
            "Observation: {}\nAction: {}".format(head, response[:300])
        )
    current = "Observation: {}\nAction:".format(observation[:obs_chars])
    prompt_chars = len(prefix) + sum(len(turn) + 1 for turn in turns) + len(current)
    dropped = 0
    # Keep the instruction pair and the three latest pairs, even if this
    # protected minimum exceeds the requested character budget.
    while prompt_chars > max_prompt_chars and len(turns) > 4:
        prompt_chars -= len(turns.pop(1)) + 1
        dropped += 1
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prefix + "\n".join(turns + [current])},
    ]
    return messages, dropped


def make_env(num_products=None):
    """Import/register WebShop only when actually evaluating; never edit its files."""
    sys.path.insert(0, str(WEBSHOP_REPO))
    import gym
    import web_agent_site.envs  # noqa: F401 -- registers WebAgentTextEnv-v0
    from web_agent_site.engine import engine

    # Upstream defaults point at the 1k preview, even for num_products=None.
    # Override the attribute path in memory during construction and pass the
    # product path explicitly. The small download also supports 100/1000 runs.
    data = WEBSHOP_REPO / "data"
    suffix = ""
    if num_products in (100, 1000) and not (data / "items_shuffle.json").is_file():
        suffix = "_1000"
    previous_attr_path = engine.DEFAULT_ATTR_PATH
    engine.DEFAULT_ATTR_PATH = str(data / ("items_ins_v2" + suffix + ".json"))
    try:
        return gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=num_products,
            file_path=str(data / ("items_shuffle" + suffix + ".json")),
        )
    finally:
        engine.DEFAULT_ATTR_PATH = previous_attr_path


def make_client(base_url: str):
    from openai import OpenAI

    # Retry here at the episode level so the SDK cannot multiply our attempts.
    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                  max_retries=0)


def run_episode(env, client, session: int, model: str, max_steps: int, obs_chars: int,
                history_obs_chars: int = HISTORY_OBS_CHARS,
                max_prompt_chars: int = MAX_PROMPT_CHARS,
                prompt_version: Optional[str] = None):
    """Steps count model turns, including failed turns, but exclude API retries."""
    from openai import APIError, BadRequestError

    prompt_version = resolve_prompt_version(prompt_version)
    started = time.monotonic()
    initial = env.reset(session=session)
    observation = initial[0] if isinstance(initial, tuple) else initial
    history = []
    reward = 0.0
    steps = 0
    format_failures = 0
    consecutive_failures = 0
    final_action = None
    dropped_history = 0
    error = None
    for steps in range(1, max_steps + 1):
        messages, dropped = _build_messages_with_drops(
            history, observation, obs_chars, history_obs_chars, max_prompt_chars, prompt_version
        )
        del history[1:1 + dropped]
        dropped_history += dropped
        # One initial attempt plus three retries, with a fixed two-second backoff.
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=128,
                    stop=["\nObservation", "Observation:"],
                )
                break
            except APIError as exc:
                if isinstance(exc, BadRequestError) and re.search(
                    r"context[\s_-]+length", str(exc), re.IGNORECASE
                ):
                    error = "context_length"
                    break
                if attempt == 3:
                    error = "{}: {}".format(type(exc).__name__, " ".join(str(exc).split()))[:200]
                    break
                time.sleep(2)
        if error is not None:
            break
        response = completion.choices[0].message.content or ""
        history.append((observation, response))
        final_action = parse_action(response)
        if final_action is None:
            # A format failure leaves the page unchanged and consumes one step.
            format_failures += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        consecutive_failures = 0
        observation, reward, done, _ = env.step(final_action)
        if done:
            break
    reward = float(reward)
    return {
        "session": session,
        "prompt_version": prompt_version,
        "reward": reward,
        "success": reward == 1.0,
        "steps": steps,
        "format_failures": format_failures,
        "final_action": final_action,  # null if the last attempt was a no-op
        "elapsed": time.monotonic() - started,
        "dropped_history": dropped_history,
        "error": error,
    }


def load_records(path: Path):
    """Index completed sessions, repairing only an interrupted final JSONL write."""
    records = {}
    if not path.exists():
        return records
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                stream.seek(offset)
                stream.truncate()
                break
            records[record["session"]] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return records


def compute_metrics(records, config):
    records = list(records)
    n = len(records)
    return {
        "n": n,
        "score": 100 * sum(record["reward"] for record in records) / n if n else 0.0,
        "success_rate": sum(record["reward"] == 1.0 for record in records) / n if n else 0.0,
        "mean_steps": sum(record["steps"] for record in records) / n if n else 0.0,
        "errored_episodes": sum(bool(record.get("error")) for record in records),
        "config": dict(config),
    }


def evaluate(args):
    prompt_version = resolve_prompt_version(getattr(args, "prompt_version", None))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    records = load_records(records_path)
    # Pre-versioning records used v1. Refuse a mixed resume before any API call,
    # including completed/empty runs and records outside the requested range.
    versions = {record.get("prompt_version", "v1") for record in records.values()}
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        versions.add(saved.get("config", {}).get("prompt_version", "v1"))
    if versions - {prompt_version}:
        raise ValueError("WebShop prompt_version mismatch; use a separate output directory")
    sessions = range(args.start, args.start + args.n)
    pending = [session for session in sessions if session not in records]
    # Opening the log also creates it for an empty run. A completed resume needs
    # neither environment dependencies nor a running model server.
    with records_path.open("a", encoding="utf-8") as stream:
        if pending:
            with make_client(args.base_url) as client:
                env = make_env(args.num_products)
                try:
                    for completed, session in enumerate(pending, 1):
                        record = run_episode(
                            env, client, session, args.model, args.max_steps, args.obs_chars,
                            args.history_obs_chars, args.max_prompt_chars,
                            prompt_version=prompt_version,
                        )
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        records[session] = record
                        if completed % 25 == 0:
                            running = compute_metrics(
                                (records[s] for s in sessions if s in records), {}
                            )
                            print(
                                "WebShop progress n={}/{} score={:.2f}".format(
                                    running["n"], args.n, running["score"]
                                ),
                                flush=True,
                            )
                finally:
                    env.close()
    config = dict(vars(args))
    config["prompt_version"] = prompt_version
    config["out"] = str(out)
    metrics = compute_metrics((records[s] for s in sessions), config)
    temporary = out / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    print(
        "WebShop tag={} prompt_version={} n={} score={:.2f} success_rate={:.4f} mean_steps={:.2f}".format(
            args.tag, prompt_version, metrics["n"], metrics["score"],
            metrics["success_rate"], metrics["mean_steps"],
        ),
        flush=True,
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Chat API base URL, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=None,
                        help="Overrides WEBSHOP_PROMPT_VERSION (default: v1)")
    parser.add_argument("--obs-chars", type=int, default=None,
                        help="Current observation limit (default: 6000 for both prompt versions)")
    parser.add_argument("--history-obs-chars", type=int, default=HISTORY_OBS_CHARS,
                        help="Keep this many characters of each past observation, plus ' ...' if cut")
    parser.add_argument("--max-prompt-chars", type=int, default=MAX_PROMPT_CHARS,
                        help="User-message budget; always retain the first and last three history pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-products", type=int, choices=[100, 1000, 100000], default=None,
                        help="Omit for all products; subsets require the corresponding index")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=0, help="Recorded only; unused")
    args = parser.parse_args(argv)
    try:
        args.prompt_version = resolve_prompt_version(args.prompt_version)
    except ValueError as exc:
        parser.error(str(exc))
    if args.obs_chars is None:
        args.obs_chars = default_obs_chars(args.prompt_version)
    if args.start < 0 or args.n < 0:
        parser.error("--start and --n must be nonnegative")
    if min(args.max_steps, args.obs_chars, args.history_obs_chars, args.max_prompt_chars) < 1:
        parser.error("--max-steps, --obs-chars, --history-obs-chars and --max-prompt-chars must be positive")
    return args


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
