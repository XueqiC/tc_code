#!/usr/bin/env python3
"""Evaluate a served student on WebShop with one fixed ReAct-style example.

Run with envs/webshop/venv/bin/python (Python 3.8, openai>=1.0), for example:
    tools/webshop_eval.py --base-url http://localhost:8000/v1 --model student \
        --out results/webshop/student

Reuse an output directory only for the same model and evaluation settings.
Completed sessions are skipped; metrics cover the requested session range.
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


def build_messages(history, observation: str, obs_chars: int):
    """Rebuild two fresh messages; retain every prior turn and each observation head."""
    turns = [WORKED_EXAMPLE.rstrip(), "\nLive episode:"]
    for previous_observation, response in history:
        turns.append(
            "Observation: {}\nAction: {}".format(
                previous_observation[:obs_chars], response
            )
        )
    turns.append("Observation: {}\nAction:".format(observation[:obs_chars]))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(turns)},
    ]


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

    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY")


def run_episode(env, client, session: int, model: str, max_steps: int, obs_chars: int):
    """Steps count model attempts, including no-ops; elapsed is in seconds."""
    started = time.monotonic()
    initial = env.reset(session=session)
    observation = initial[0] if isinstance(initial, tuple) else initial
    history = []
    reward = 0.0
    steps = 0
    format_failures = 0
    consecutive_failures = 0
    final_action = None
    for steps in range(1, max_steps + 1):
        completion = client.chat.completions.create(
            model=model,
            messages=build_messages(history, observation, obs_chars),
            temperature=0,
            max_tokens=128,
            stop=["\nObservation", "Observation:"],
        )
        response = (completion.choices[0].message.content or "").strip()
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
        "reward": reward,
        "success": reward == 1.0,
        "steps": steps,
        "format_failures": format_failures,
        "final_action": final_action,  # null if the last attempt was a no-op
        "elapsed": time.monotonic() - started,
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
        "config": dict(config),
    }


def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    records = load_records(records_path)
    sessions = range(args.start, args.start + args.n)
    pending = [session for session in sessions if session not in records]
    # Opening the log also creates it for an empty run. A completed resume needs
    # neither environment dependencies nor a running model server.
    with records_path.open("a", encoding="utf-8") as stream:
        if pending:
            with make_client(args.base_url) as client:
                env = make_env(args.num_products)
                try:
                    for session in pending:
                        record = run_episode(
                            env, client, session, args.model, args.max_steps, args.obs_chars
                        )
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        records[session] = record
                finally:
                    env.close()
    config = dict(vars(args))
    config["out"] = str(out)
    metrics = compute_metrics((records[s] for s in sessions), config)
    metrics_path = out / "metrics.json"
    temporary = out / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    print(
        "WebShop tag={} n={} score={:.2f} success_rate={:.4f} mean_steps={:.2f}".format(
            args.tag, metrics["n"], metrics["score"],
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
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--obs-chars", type=int, default=2500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-products", type=int, choices=[100, 1000, 100000], default=None,
                        help="Omit for all products; subsets require the corresponding index")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=0, help="Recorded only; unused")
    args = parser.parse_args(argv)
    if args.start < 0 or args.n < 0:
        parser.error("--start and --n must be nonnegative")
    if args.max_steps < 1 or args.obs_chars < 1:
        parser.error("--max-steps and --obs-chars must be positive")
    return args


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
