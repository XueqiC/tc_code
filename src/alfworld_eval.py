#!/usr/bin/env python3
"""ALFWorld unseen-split evaluation for a HF causal LM student.

Official protocol: eval_out_of_distribution games, task success rate.
The agent sees the textworld observation, the goal, and the admissible
commands, and must output one command per step. A command not in the
admissible list counts as a no-op look action.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import yaml

PROMPT = (
    "You are an agent in a household. Complete the task by issuing one "
    "command at a time from the admissible commands.\n\nTask context:\n"
    "{obs}\n\nAdmissible commands:\n{cmds}\n\nHistory:\n{hist}\n\n"
    "Reply with exactly one admissible command and nothing else."
)


def pick_command(text: str, admissible: list[str]) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    line = re.sub(r"^[->*\s`\"']+|[`\"']+$", "", line).strip()
    if line in admissible:
        return line
    lowered = line.lower()
    for c in admissible:
        if c.lower() == lowered:
            return c
    for c in admissible:
        if lowered and (lowered in c.lower() or c.lower() in lowered):
            return c
    return "look"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-games", type=int, default=134)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import alfworld.agents.environment as environment

    config_path = Path(os.environ["ALFWORLD_CONFIG"])
    with config_path.open() as fh:
        config = yaml.safe_load(fh)
    config["env"]["type"] = "AlfredTWEnv"

    env_cls = environment.get_environment("AlfredTWEnv")
    env = env_cls(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    torch.manual_seed(args.seed)

    out_dir = Path("results/alfworld") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    n_games = min(args.num_games, env.num_games)
    for game_i in range(n_games):
        obs, info = env.reset()
        obs0 = obs[0]
        hist: list[str] = []
        won = False
        for step in range(args.max_steps):
            admissible = list(info["admissible_commands"][0])
            prompt = PROMPT.format(
                obs=obs0[-2000:],
                cmds="\n".join(admissible),
                hist="\n".join(hist[-8:]) or "(start)",
            )
            messages = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=32, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            cmd = pick_command(reply, admissible)
            obs, scores, dones, info = env.step([cmd])
            obs0 = obs[0]
            hist.append(f"> {cmd}\n{obs0[:300]}")
            if dones[0]:
                won = bool(info["won"][0])
                break
        records.append({"game": game_i, "won": won, "steps": step + 1})
        with (out_dir / "records.jsonl").open("a") as fh:
            fh.write(json.dumps(records[-1]) + "\n")
        if (game_i + 1) % 10 == 0:
            sr = sum(r["won"] for r in records) / len(records)
            print(f"[alfworld] {game_i+1}/{n_games} success={sr:.3f}", flush=True)
    sr = sum(r["won"] for r in records) / len(records)
    json.dump(
        {"tag": args.tag, "model": args.model, "n": len(records),
         "success_rate": sr},
        (out_dir / "metrics.json").open("w"), indent=1,
    )
    print(f"[alfworld] FINAL {args.tag} success_rate={sr:.4f} n={len(records)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
