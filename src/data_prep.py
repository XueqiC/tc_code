"""Build pilot domains as jsonl: {domain, prompt, response}.

T1  gsm8k-code : GSM8K questions, solutions mechanically converted to Python
                 from the <<a op b = c>> calculator annotations, exec-verified.
T1p gsm8k-cot  : the SAME questions, original chain-of-thought text solutions.
T2  pandas     : DS-1000 Pandas subset (prompt -> reference_code).
T3  sql        : Spider (question -> SQL query).
gen alpaca     : general instruction pool (out-of-domain reservoir).

T1/T1p share identical prompts (question only, no format instruction) so that
query-only features cannot separate them by construction.
"""
import json
import random
import re
import signal
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "pilot"
N_PER_DOMAIN = 120
SEED = 0

NUM = r"[-+]?\d[\d,]*\.?\d*%?"


def norm_num(s: str):
    s = s.strip().replace(",", "").replace("$", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100 if pct else v


def gsm8k_to_code(question: str, cot: str, gold: float):
    """Turn <<expr=result>> annotations into a python solution(); verify by exec."""
    calcs = re.findall(r"<<([^<>=]+)=([^<>=]+)>>", cot)
    if len(calcs) < 2:
        return None
    lines, result_vars = [], {}
    for i, (expr, res) in enumerate(calcs):
        expr = expr.replace(",", "").replace("$", "").replace("%", "/100")
        # substitute earlier results with their variables (longest match first)
        for lit, var in sorted(result_vars.items(), key=lambda kv: -len(kv[0])):
            expr = re.sub(rf"(?<![\d.\w]){re.escape(lit)}(?![\d.\w])", var, expr)
        if not re.fullmatch(r"[\w\s+\-*/().]+", expr):
            return None
        lines.append(f"    v{i} = {expr}")
        r = res.strip().replace(",", "").replace("$", "").replace("%", "")
        result_vars[r] = f"v{i}"
    lines.append(f"    return v{len(calcs) - 1}")
    code = "def solution():\n" + "\n".join(lines)
    try:
        env = {}
        signal.alarm(2)
        exec(code, env)  # noqa: S102 - our own generated arithmetic
        out = env["solution"]()
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        return None
    if out is None or abs(out - gold) > 1e-4:
        return None
    return code


def build_gsm8k(rng):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    code_rows, cot_rows = [], []
    for i in idx:
        ex = ds[i]
        gold = norm_num(ex["answer"].split("####")[-1])
        if gold is None:
            continue
        cot = ex["answer"].split("####")[0].strip()
        code = gsm8k_to_code(ex["question"], ex["answer"], gold)
        if code is None:
            continue
        prompt = ex["question"].strip()
        code_rows.append({"domain": "gsm8k-code", "prompt": prompt, "response": code})
        cot_rows.append({"domain": "gsm8k-cot", "prompt": prompt,
                         "response": re.sub(r"<<[^<>]*>>", "", cot)})
        if len(code_rows) >= N_PER_DOMAIN:
            break
    return code_rows, cot_rows


def build_pandas():
    ds = load_dataset("xlangai/DS-1000", split="test")
    rows = [{"domain": "pandas", "prompt": ex["prompt"].strip(),
             "response": ex["reference_code"].strip()}
            for ex in ds if ex["metadata"]["library"] == "Pandas"]
    return rows[:N_PER_DOMAIN]


def build_sql(rng):
    ds = load_dataset("xlangai/spider", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx[: N_PER_DOMAIN * 2]:
        ex = ds[i]
        rows.append({"domain": "sql",
                     "prompt": f"Database: {ex['db_id']}\n{ex['question'].strip()}",
                     "response": ex["query"].strip()})
        if len(rows) >= N_PER_DOMAIN:
            break
    return rows


def build_alpaca(rng):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        ex = ds[i]
        if ex["input"].strip() or "code" in ex["instruction"].lower():
            continue
        rows.append({"domain": "alpaca", "prompt": ex["instruction"].strip(),
                     "response": ex["output"].strip()})
        if len(rows) >= N_PER_DOMAIN:
            break
    return rows


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    code_rows, cot_rows = build_gsm8k(rng)
    domains = {
        "gsm8k-code": code_rows,
        "gsm8k-cot": cot_rows,
        "pandas": build_pandas(),
        "sql": build_sql(rng),
        "alpaca": build_alpaca(rng),
    }
    for name, rows in domains.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
