#!/usr/bin/env python3
"""Atom-directed generation: for each high-demand capability atom, show the
teacher the support queries that load most on that atom as exemplars, ask it
to generate NEW problems exercising the same underlying skill, then have it
solve each with a verified `def solution()` program.

Combines Self-Instruct's generation engine (teacher creates supply beyond the
candidate pool) with our per-skill targeting (which skills need supply)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from evol_extend_pool import (  # noqa: E402
    DOMAIN,
    TEACHER,
    extract_python_code,
    load_completed_prompts,
    load_support_rows,
    make_teacher_call,
    needs_leading_newline,
    strip_think,
    verify_solution,
)

OUTPUT = ROOT / "data" / "atomgen_pool_v1.jsonl"

GEN_PROMPT = """Here are {n_ex} math word problems. They all exercise the same \
underlying reasoning skill, even though their surface topics differ.

{exemplars}

Generate {n_new} NEW math word problems that exercise the SAME underlying \
reasoning skill. Requirements:
- Each problem must be self-contained, solvable, and have a single numeric answer.
- Vary the surface story (names, objects, quantities) so no problem resembles \
the examples or each other.
- Do not solve them.
Output exactly {n_new} problems as a numbered list, one problem per line."""

SOLVE_PROMPT = """Solve this problem by writing a Python function.

{problem}

Write a complete Python function `def solution():` that returns the numeric \
answer. Output only the code."""


def parse_numbered_list(reply: str) -> list[str]:
    items = []
    for line in strip_think(reply).splitlines():
        m = re.match(r"\s*\d+[\.)]\s+(.*\S)", line)
        if m and len(m.group(1)) > 40:
            items.append(m.group(1).strip())
    return items


def rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--per-atom", type=int, default=15)
    parser.add_argument("--exemplars", type=int, default=3)
    parser.add_argument("--student", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    support = load_support_rows()
    print(f"[atomgen] support rows: {len(support)}", flush=True)

    import numpy as np
    import features as _features
    from sklearn.decomposition import MiniBatchDictionaryLearning

    grad = _features.extract_features(
        args.student, support, proj_dim=64,
        max_prompt_tok=1024, max_resp_tok=512, device=args.device,
    ).astype(np.float64)
    dico = MiniBatchDictionaryLearning(
        n_components=int(min(64, max(8, len(support) // 2))),
        alpha=0.05, transform_algorithm="lasso_lars", transform_alpha=0.05,
        random_state=0, max_iter=100, batch_size=16,
    )
    dico.fit(grad)
    codes = np.abs(dico.transform(grad))
    demand = codes.mean(axis=0)
    top_atoms = [int(a) for a in np.argsort(-demand)[: args.atoms]]
    print(f"[atomgen] top atoms by demand: {top_atoms}", flush=True)

    if args.dry_run:
        for a in top_atoms:
            ex = np.argsort(-codes[:, a])[: args.exemplars]
            print(f"atom {a}: exemplar prompts {[int(i) for i in ex]}")
        return 0

    done = load_completed_prompts(OUTPUT)
    support_prompts = {row["prompt"] for row in support}
    call = make_teacher_call()
    appended = 0
    mode_newline = needs_leading_newline(OUTPUT)
    with OUTPUT.open("a", encoding="utf-8") as out:
        if mode_newline:
            out.write("\n")
        for a in top_atoms:
            ex_idx = np.argsort(-codes[:, a])[: args.exemplars]
            exemplars = "\n\n".join(
                f"Problem {k+1}: {support[int(i)]['prompt']}"
                for k, i in enumerate(ex_idx)
            )
            try:
                reply = call(GEN_PROMPT.format(
                    n_ex=args.exemplars, exemplars=exemplars,
                    n_new=args.per_atom,
                ))
            except Exception as exc:
                print(f"[atomgen] atom {a} gen failed: {exc}", flush=True)
                continue
            problems = parse_numbered_list(reply)
            print(f"[atomgen] atom {a}: {len(problems)} problems parsed",
                  flush=True)
            for prob in problems:
                if prob in done or prob in support_prompts:
                    continue
                try:
                    sol_reply = call(SOLVE_PROMPT.format(problem=prob))
                except Exception as exc:
                    print(f"[atomgen] solve failed: {exc}", flush=True)
                    continue
                code = extract_python_code(sol_reply)
                if "def solution" not in code:
                    continue
                if verify_solution(code) is None:
                    continue
                row = {
                    "prompt": prob,
                    "response": code,
                    "teacher": TEACHER,
                    "domain": DOMAIN,
                    "source": f"atomgen_a{a}",
                    "token_hint": rough_tokens(code),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                done.add(prob)
                appended += 1
    print(f"[atomgen] appended {appended} verified rows -> {OUTPUT}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
