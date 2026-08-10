"""pilot-distill-v1: does distilling T move the student's capability C into T,
measurably, in the same spaces we used for the boundary?

Train: LoRA SFT of Qwen2.5-1.5B-Instruct on gsm8k-code trajectories (train split).
Measure before vs after, on held-out test splits of every domain:
  loss      : teacher-forced mean loss (capability proxy, cheap)
  gradnorm  : per-sample gradient norm w.r.t. the student's LoRA params
              (optimization residual: small = already learned)
  exec-acc  : gsm8k-code only — generate greedily, exec solution(), compare gold;
              also "any-answer" accuracy via last-number fallback (captures
              solving in the wrong action space, e.g. CoT before SFT)
Stable splits via sha256 (fixes the salted-hash issue in boundary.py).
"""
import json
import re
import signal
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from grad_features import MODEL, MAX_PROMPT_TOK, MAX_RESP_TOK, stable_seed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pilot_distill"
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
N_TEST = 30
EPOCHS = 3
LR = 2e-4
ACCUM = 8
SEED = 0


def rows_of(dom):
    return [json.loads(l) for l in open(ROOT / "data" / "pilot" / f"{dom}.jsonl")]


def split(dom):
    rows = rows_of(dom)
    perm = np.random.default_rng(stable_seed("split-" + dom)).permutation(len(rows))
    return [rows[i] for i in perm[:-N_TEST]], [rows[i] for i in perm[-N_TEST:]]


def encode(tok, r, dev):
    msgs = [{"role": "user", "content": r["prompt"]}]
    ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pids = tok(ptxt, add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOK]
    rids = tok(r["response"], add_special_tokens=False)["input_ids"][:MAX_RESP_TOK] \
        + [tok.eos_token_id]
    ids = torch.tensor([pids + rids], device=dev)
    labels = torch.tensor([[-100] * len(pids) + rids], device=dev)
    return ids, labels


def exec_solution(code, timeout=2):
    try:
        env = {}
        signal.alarm(timeout)
        exec(code, env)  # noqa: S102 - pilot sandbox, our own data
        v = env["solution"]()
        signal.alarm(0)
        return float(v)
    except Exception:
        signal.alarm(0)
        return None


def last_number(text):
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None


@torch.no_grad()
def eval_loss(model, tok, tests, dev):
    model.eval()
    out = {}
    for dom, rows in tests.items():
        ls = []
        for r in rows:
            ids, labels = encode(tok, r, dev)
            ls.append(model(input_ids=ids, labels=labels).loss.item())
        out[dom] = ls
    return out


def eval_gradnorm(model, tok, tests, dev):
    model.train()
    out = {}
    for dom, rows in tests.items():
        ns = []
        for r in rows:
            ids, labels = encode(tok, r, dev)
            model.zero_grad(set_to_none=True)
            model(input_ids=ids, labels=labels).loss.backward()
            g2 = sum(p.grad.float().pow(2).sum().item()
                     for _, p in model.named_parameters()
                     if p.grad is not None)
            ns.append(g2 ** 0.5)
        out[dom] = ns
    model.zero_grad(set_to_none=True)
    return out


@torch.no_grad()
def eval_gen(model, tok, rows, dev):
    model.eval()
    execs, anys, fmt = 0, 0, 0
    for r in rows:
        gold = exec_solution(r["response"])
        msgs = [{"role": "user", "content": r["prompt"]}]
        ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(ptxt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=320, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        pred = exec_solution(text) if "def solution" in text else None
        if "def solution" in text:
            fmt += 1
        if pred is not None and gold is not None and abs(pred - gold) < 1e-4:
            execs += 1
        pa = pred if pred is not None else last_number(text)
        if pa is not None and gold is not None and abs(pa - gold) < 1e-4:
            anys += 1
    n = len(rows)
    return {"exec_acc": execs / n, "any_acc": anys / n, "format_rate": fmt / n}


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    torch.manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)

    train, test_code = split("gsm8k-code")
    tests = {"gsm8k-code": test_code}
    for d in DOMAINS[1:]:
        tests[d] = split(d)[1]
    print(f"train={len(train)} test={N_TEST}/domain", flush=True)

    metrics = {"before": {}}
    metrics["before"]["loss"] = eval_loss(model, tok, tests, dev)
    metrics["before"]["gradnorm"] = eval_gradnorm(model, tok, tests, dev)
    metrics["before"]["gen"] = eval_gen(model, tok, test_code, dev)
    print("before:", metrics["before"]["gen"], flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = np.random.default_rng(SEED)
    step = 0
    for ep in range(EPOCHS):
        order = rng.permutation(len(train))
        for j, i in enumerate(order):
            ids, labels = encode(tok, train[i], dev)
            loss = model(input_ids=ids, labels=labels).loss / ACCUM
            loss.backward()
            if (j + 1) % ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        print(f"epoch {ep + 1}/{EPOCHS} done ({step} steps)", flush=True)
    opt.zero_grad(set_to_none=True)

    metrics["after"] = {}
    metrics["after"]["loss"] = eval_loss(model, tok, tests, dev)
    metrics["after"]["gradnorm"] = eval_gradnorm(model, tok, tests, dev)
    metrics["after"]["gen"] = eval_gen(model, tok, test_code, dev)
    print("after:", metrics["after"]["gen"], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT / "adapter")
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    for phase in ["before", "after"]:
        for dom in DOMAINS:
            l = np.mean(metrics[phase]["loss"][dom])
            g = np.median(metrics[phase]["gradnorm"][dom])
            print(f"{phase:6s} {dom:12s} loss={l:.3f} gradnorm_med={g:.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
