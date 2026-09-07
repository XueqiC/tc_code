#!/usr/bin/env python3
"""Empirical single-event transfer matrix on the BFCL r2 preference pool (CRCD Block I, T6).

For every event i of ``data/bfcl_sft/pool_events_pref_v2.jsonl`` (prompt s_i, teacher continuation
y^T_i = ``response``, student continuation y^S_i = ``_rejected``):

  1. reset the LoRA adapter to its initialisation (A = init draw, B = 0  ->  policy == base),
  2. take S optimizer steps (default 3) of the trainer's base-centred pairwise loss
     (``src/appworld_train.py`` AW_DISTILL=ddpo:  -logsigmoid(beta * (policy_margin - base_margin)),
     summed response log-probs, AdamW, lr = AW_DDPO_LR default 5e-6, beta = AW_DDPO_BETA default 0.1)
     on event i alone (batch = that one event),
  3. re-score every event j:  m_j = mean-per-token log pi(y^T_j | s_j) - mean-per-token log pi(y^S_j | s_j)
     (and the summed version), and store  T[i, j] = m_j(after i) - m_j(base).

Encoding reuses the trainer's ``encode`` (prompt tail-truncated to AW_MAX_PROMPT_TOKENS=2048, response
capped at 512 tokens + EOS).  LoRA: r=8, alpha=16, dropout 0, the trainer's target modules, seed 0
(the fingerprint configuration of data/fingerprints/bfcl_r2_v1_meta.npz; the trainer itself uses r=16).
Measurement forward passes are batched (length-sorted, fixed batch composition for before/after so that
the only difference between the two passes is the adapter delta), bf16, no grad, only the response
positions go through the LM head.

Outputs (npz, resumable; checkpointed every --ckpt-every rows):
  T_mean (n, n) float32   per-token margin change, rows = updated event, cols = affected event
  T_sum  (n, n) float32   summed-log-prob margin change
  logp_T_after, logp_S_after (n, n)   summed log-probs after the update of row i
  base_logp_T, base_logp_S, n_tok_T, n_tok_S (n,)   base (before) values
  done (n,) bool          rows measured;  row_order (n,) the processing order (seed 0 permutation)
  row_seconds (n,)        wall time per row;  self_effect = diag(T_mean)
  loss_trace (n, S)       the pair loss at each step of row i
  meta (json string)      configuration

Step size.  The default --lr is the trainer's 5e-6, but at that size 3 Adam steps move the LoRA-B
coordinates by ~1.5e-5 and the resulting margin changes sit at the bf16 forward-pass rounding floor
(smoke, 2026-09-04: --noise-check random-sign perturbation gives per-token margin sd ~0.007 regardless of
size; at lr 5e-6 T_ii was +0.014 / -0.005 and rows at lr 5e-6 correlate ~0.1 with the same rows at
larger lr).  results/analysis/transfer_matrix_bfcl_r2.npz was measured with --lr 1e-4 (T_ii >> floor,
rows at 5e-5 and 2e-4 correlate 0.76-0.97), a 120-row random subset (seed 0) of the 210 events because
the 3 h budget rule fired (68 s/row), all 210 columns.

Usage (rai, GPU 0 by UUID):
  CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=src .venv/bin/python tools/bfcl_transfer_matrix.py \
      --out results/analysis/transfer_matrix_bfcl_r2.npz --log logs/t6_transfer_matrix.log
  --rows 2 --out <smoke.npz>   smoke run (2 rows x all columns), prints T_ii
  --check-batching             compare batched vs single-sequence log-probs on a few events
  --no-subset                  resume a subset checkpoint and measure the remaining rows (lifts the stored subset)
ALFWorld (T9, 2026-09-04): the same protocol on data/alf_sft/events_v1.jsonl (385 events, prompt = state,
response = teacher command y^T, _rejected = student continuation y^S) ->
results/analysis/transfer_matrix_alfworld_v1.npz; the pool's _seed_category / _event_turn / _event_good /
_event_dU are stored in meta (category / event_turn / teacher_command / dU) for the label predictors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The trainer reads these at call time; set before importing so every helper sees the same caps.
os.environ.setdefault("AW_MAX_PROMPT_TOKENS", "2048")
os.environ.setdefault("AW_TRUNCATE_SIDE", "tail")

POOL = ROOT / "data" / "bfcl_sft" / "pool_events_pref_v2.jsonl"
STUDENT = "Qwen/Qwen3.5-4B"
LORA_R, LORA_ALPHA, LORA_SEED = 8, 16, 0
STEPS = 3
LR = 5e-6            # appworld_train.distillation_config(): AW_DDPO_LR default
BETA = 0.1           # AW_DDPO_BETA default
TIME_BUDGET_S = 3 * 3600
SUBSET_ROWS = 120


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ----------------------------------------------------------------------------------------------
# pure bookkeeping (CPU-testable)
# ----------------------------------------------------------------------------------------------
def row_order(n: int, seed: int = 0) -> np.ndarray:
    """Processing order: a seeded permutation, so that a time-budget truncation to the first k rows
    is a uniformly random subset (seed 0)."""
    return np.random.default_rng(seed).permutation(n)


def make_batches(lengths: list[int], token_budget: int, max_batch: int = 64) -> list[list[int]]:
    """Length-sorted batches under a padded-token budget.  Deterministic; the same batches are used
    for the before and after passes so padding is identical."""
    order = sorted(range(len(lengths)), key=lambda k: (lengths[k], k))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for k in order:
        L = lengths[k]
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > token_budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(k)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def empty_state(n: int, steps: int, order: np.ndarray) -> dict:
    return {
        "T_mean": np.full((n, n), np.nan, dtype=np.float32),
        "T_sum": np.full((n, n), np.nan, dtype=np.float32),
        "logp_T_after": np.full((n, n), np.nan, dtype=np.float32),
        "logp_S_after": np.full((n, n), np.nan, dtype=np.float32),
        "base_logp_T": np.full(n, np.nan, dtype=np.float32),
        "base_logp_S": np.full(n, np.nan, dtype=np.float32),
        "n_tok_T": np.zeros(n, dtype=np.int32),
        "n_tok_S": np.zeros(n, dtype=np.int32),
        "done": np.zeros(n, dtype=bool),
        "row_order": np.asarray(order, dtype=np.int64),
        "row_seconds": np.full(n, np.nan, dtype=np.float32),
        "loss_trace": np.full((n, steps), np.nan, dtype=np.float32),
    }


def margins(logp_T: np.ndarray, logp_S: np.ndarray, n_T: np.ndarray, n_S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-token margin, summed margin)."""
    return logp_T / n_T - logp_S / n_S, logp_T - logp_S


def record_row(state: dict, i: int, logp_T: np.ndarray, logp_S: np.ndarray, seconds: float,
               losses: list[float] | None = None) -> None:
    """Fill row i of the matrices from the after-update log-probs of all events."""
    m_after, s_after = margins(logp_T, logp_S, state["n_tok_T"], state["n_tok_S"])
    m_base, s_base = margins(state["base_logp_T"], state["base_logp_S"], state["n_tok_T"], state["n_tok_S"])
    state["T_mean"][i] = (m_after - m_base).astype(np.float32)
    state["T_sum"][i] = (s_after - s_base).astype(np.float32)
    state["logp_T_after"][i] = logp_T
    state["logp_S_after"][i] = logp_S
    state["row_seconds"][i] = seconds
    if losses is not None:
        state["loss_trace"][i, :len(losses)] = losses
    state["done"][i] = True


def save_state(state: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    out = dict(state)
    out["self_effect"] = np.diag(state["T_mean"]).astype(np.float32)
    out["meta"] = np.array(json.dumps(meta))
    np.savez(tmp, **out)
    os.replace(tmp, path)


def load_state(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=True) as f:
        state = {k: f[k] for k in f.files if k not in ("meta", "self_effect")}
        meta = json.loads(str(f["meta"]))
    return state, meta


def projected_total(row_seconds: np.ndarray, done_rows: int, total_rows: int, base_seconds: float) -> float:
    per_row = float(np.nanmean(row_seconds))
    return base_seconds + per_row * total_rows


# ----------------------------------------------------------------------------------------------
# model side
# ----------------------------------------------------------------------------------------------
def load_pool(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if not isinstance(r.get("_rejected"), str) or not r["_rejected"]:
            raise ValueError(f"event {r.get('task_id')} has no _rejected continuation")
    return rows


def build_model(seed: int):
    import torch
    from peft import LoraConfig, get_peft_model
    import appworld_train as at

    base = at.build_base_model(STUDENT, seed)
    torch.manual_seed(LORA_SEED)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=list(at.LORA_TARGET_MODULES))
    model = get_peft_model(base, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def lora_params(model) -> list[tuple[str, "torch.nn.Parameter"]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def snapshot(params) -> list["torch.Tensor"]:
    return [p.detach().clone() for _, p in params]


def restore(params, snap) -> None:
    import torch
    with torch.no_grad():
        for (_, p), s in zip(params, snap):
            p.copy_(s)


def encode_all(tokenizer, rows: list[dict]):
    """Token ids for (prompt + teacher) and (prompt + student) per event, via the trainer's encode."""
    import appworld_train as at
    seqs_T, seqs_S, nP_T, nP_S = [], [], [], []
    for r in rows:
        ids, labels = at.encode(tokenizer, r, device="cpu")
        seqs_T.append(ids[0].tolist()); nP_T.append(int((labels[0] == -100).sum()))
        ids, labels = at.encode(tokenizer, at.rejected_row(r), device="cpu")
        seqs_S.append(ids[0].tolist()); nP_S.append(int((labels[0] == -100).sum()))
    return seqs_T, nP_T, seqs_S, nP_S


def batched_logprobs(model, backbone, head, seqs: list[list[int]], n_prompt: list[int],
                     batches: list[list[int]], pad_id: int, device: str, head_chunk: int = 1024) -> np.ndarray:
    """Summed continuation log-prob of every sequence (fixed batches, right padding, bf16, no grad)."""
    import torch
    out = np.full(len(seqs), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in batches:
            L = max(len(seqs[k]) for k in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for b, k in enumerate(batch):
                ids[b, :len(seqs[k])] = torch.tensor(seqs[k]); mask[b, :len(seqs[k])] = 1
            ids, mask = ids.to(device), mask.to(device)
            hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            # gather (position, target) pairs of the continuation tokens of each sequence
            pos_b, pos_t, tgt = [], [], []
            for b, k in enumerate(batch):
                n = len(seqs[k])
                p = torch.arange(n_prompt[k] - 1, n - 1)
                pos_b.append(torch.full_like(p, b)); pos_t.append(p)
                tgt.append(torch.tensor(seqs[k][n_prompt[k]:]))
            pos_b, pos_t, tgt = (torch.cat(pos_b).to(device), torch.cat(pos_t).to(device),
                                 torch.cat(tgt).to(device))
            h = hidden[pos_b, pos_t]                       # (n_cont, d)
            tok_lp = torch.empty(h.shape[0], dtype=torch.float64, device=device)
            for s in range(0, h.shape[0], head_chunk):
                logits = head(h[s:s + head_chunk]).float()
                tok_lp[s:s + head_chunk] = (logits.gather(-1, tgt[s:s + head_chunk, None]).squeeze(-1)
                                            - torch.logsumexp(logits, dim=-1)).double()
            off = 0
            for b, k in enumerate(batch):
                n = len(seqs[k]) - n_prompt[k]
                out[k] = float(tok_lp[off:off + n].sum()); off += n
    return out


def single_logprob(model, tokenizer, row: dict, device: str) -> float:
    """Trainer code path (unbatched) for the batching check."""
    import torch
    import appworld_train as at
    model.eval()
    with torch.no_grad():
        ids, labels = at.encode(tokenizer, row, device=device)
        return float(at.completion_log_prob(model, ids, labels))


def train_on_event(model, tokenizer, row: dict, ref_margin: float, steps: int, lr: float, beta: float,
                   device: str) -> list[float]:
    """S AdamW steps of the trainer's pair loss on one event; returns the loss trace."""
    import torch
    import torch.nn.functional as F
    import appworld_train as at
    params = [p for _, p in lora_params(model)]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses = []
    chosen = at.encode(tokenizer, row, device=device)
    rejected = at.encode(tokenizer, at.rejected_row(row), device=device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        policy_margin = at.completion_log_prob(model, *chosen) - at.completion_log_prob(model, *rejected)
        loss = -F.logsigmoid(beta * (policy_margin - ref_margin))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    opt.zero_grad(set_to_none=True)
    del opt
    model.eval()
    return losses


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--out", type=Path, default=ROOT / "results/analysis/transfer_matrix_bfcl_r2.npz")
    ap.add_argument("--log", type=Path, default=ROOT / "logs/t6_transfer_matrix.log")
    ap.add_argument("--rows", type=int, default=None, help="only the first N rows of the seeded order (smoke)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=12288, help="padded tokens per measurement batch")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET_S)
    ap.add_argument("--subset-rows", type=int, default=SUBSET_ROWS)
    ap.add_argument("--no-subset", action="store_true", help="never truncate to a subset")
    ap.add_argument("--check-batching", action="store_true")
    ap.add_argument("--null-check", action="store_true", help="re-measure after a reset with 0 steps (must be 0)")
    ap.add_argument("--noise-check", action="store_true", help="measure the effect of a random-sign B perturbation of size lr*steps")
    args = ap.parse_args(argv)

    import torch
    from bfas.fingerprint import _backbone_and_head
    import appworld_train as at

    args.log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.log, "a")
    device = "cuda"
    rows = load_pool(args.pool)
    n = len(rows)
    order = row_order(n, args.seed)
    meta = {"pool": str(args.pool), "student": STUDENT, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_seed": LORA_SEED, "steps": args.steps, "lr": args.lr, "beta": args.beta,
            "loss": "-logsigmoid(beta*(policy_margin-base_margin)), summed logp, AdamW (torch defaults)",
            "max_prompt_tokens": int(os.environ["AW_MAX_PROMPT_TOKENS"]), "truncate_side": os.environ["AW_TRUNCATE_SIDE"],
            "max_response_tokens": at.MAX_RESPONSE_TOKENS, "dtype": "bfloat16", "seed": args.seed,
            "prompt_hash": [prompt_hash(r["prompt"]) for r in rows], "traj": [r["_traj"] for r in rows],
            "task_id": [r["task_id"] for r in rows], "category": [r.get("_seed_category") for r in rows],
            "side": [1 if "<tool_call>" in r["response"] else 0 for r in rows],
            # generic event labels, present for event-derived pools (ALFWorld: task type = category,
            # game/task id, teacher command, turn depth, consequential dU); None where the pool lacks them
            "state_hash": [hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest() for r in rows],
            "event_turn": [r.get("_event_turn") for r in rows],
            "teacher_command": [r.get("_event_good") for r in rows],
            "dU": [r.get("_event_dU") for r in rows],
            "rows_planned": n, "subset": None}
    if args.out.is_file():
        state, old_meta = load_state(args.out)
        if old_meta.get("steps") != args.steps or old_meta.get("lr") != args.lr:
            raise SystemExit(f"{args.out} was produced with different steps/lr; refusing to resume")
        meta["subset"] = old_meta.get("subset"); meta["rows_planned"] = old_meta.get("rows_planned", n)
        if args.no_subset and meta["subset"] is not None:
            log(f"--no-subset: lifting the stored subset of {meta['subset']} rows, all {n} rows planned", fh)
            meta["subset"] = None; meta["rows_planned"] = n
        log(f"resuming {args.out}: {int(state['done'].sum())}/{n} rows done", fh)
    else:
        state = empty_state(n, args.steps, order)
    log(f"config: {json.dumps({k: v for k, v in meta.items() if not isinstance(v, list)})}", fh)

    tokenizer = at.load_tokenizer(STUDENT)
    model = build_model(args.seed)
    backbone, head = _backbone_and_head(model)
    if backbone is None:
        raise RuntimeError("could not locate backbone / lm_head")
    params = lora_params(model)
    init = snapshot(params)
    log(f"model ready: {sum(p.numel() for _, p in params)} trainable LoRA params in {len(params)} tensors; "
        f"gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", fh)

    seqs_T, nP_T, seqs_S, nP_S = encode_all(tokenizer, rows)
    seqs = seqs_T + seqs_S
    nP = nP_T + nP_S
    batches = make_batches([len(s) for s in seqs], args.token_budget)
    pad_id = tokenizer.pad_token_id
    log(f"encoded {n} events -> {len(seqs)} sequences, {len(batches)} batches, "
        f"{sum(len(s) for s in seqs)} tokens (prompt tokens {sum(nP)})", fh)

    def measure() -> tuple[np.ndarray, np.ndarray]:
        lp = batched_logprobs(model, backbone, head, seqs, nP, batches, pad_id, device)
        return lp[:n], lp[n:]

    if args.check_batching:
        restore(params, init)
        lpT, lpS = measure()
        diffs = []
        for k in list(range(0, n, max(1, n // 6)))[:6]:
            sT = single_logprob(model, tokenizer, rows[k], device)
            sS = single_logprob(model, tokenizer, at.rejected_row(rows[k]), device)
            diffs.append((k, lpT[k], sT, lpS[k], sS))
            log(f"  batching check event {k}: batched T={lpT[k]:.4f} single T={sT:.4f} | "
                f"batched S={lpS[k]:.4f} single S={sS:.4f}", fh)
        log(f"  max |batched - single| = {max(max(abs(a - b), abs(c - d)) for _, a, b, c, d in diffs):.4f} nats "
            f"(bf16; sequences are compared only within a fixed batching, so this is not the measurement noise)", fh)

    # base (before) pass, once
    t0 = time.time()
    restore(params, init)
    if np.isnan(state["base_logp_T"]).any():
        lpT, lpS = measure()
        state["base_logp_T"], state["base_logp_S"] = lpT.astype(np.float32), lpS.astype(np.float32)
        state["n_tok_T"] = np.array([len(s) - p for s, p in zip(seqs_T, nP_T)], dtype=np.int32)
        state["n_tok_S"] = np.array([len(s) - p for s, p in zip(seqs_S, nP_S)], dtype=np.int32)
        base_seconds = time.time() - t0
        log(f"base pass: {base_seconds:.1f}s; mean per-token margin "
            f"{np.mean(margins(lpT, lpS, state['n_tok_T'], state['n_tok_S'])[0]):+.4f}", fh)
    else:
        base_seconds = 0.0
        log("base pass: reused from checkpoint", fh)
    base_T64 = state["base_logp_T"].astype(np.float64); base_S64 = state["base_logp_S"].astype(np.float64)

    if args.null_check:
        restore(params, init)
        lpT, lpS = measure()
        d = np.abs(lpT - base_T64).max(), np.abs(lpS - base_S64).max()
        log(f"null check (reset, 0 steps): max |delta logp| T={d[0]:.3e} S={d[1]:.3e} (must be 0)", fh)

    if args.noise_check:
        # resolution floor: a random-sign perturbation of the LoRA-B tensors of the same magnitude as
        # `steps` Adam steps at this lr (Adam's early steps move every coordinate by ~lr).  The resulting
        # margin changes are bf16 rounding noise plus the effect of a random direction, i.e. the scale
        # below which a measured T_ij cannot be distinguished from a random update of the same size.
        g = torch.Generator(device="cpu").manual_seed(args.seed + 11)
        restore(params, init)
        with torch.no_grad():
            for name, p in params:
                if "lora_B" in name:
                    p.add_(args.lr * args.steps * torch.sign(torch.randn(p.shape, generator=g)).to(p.device, p.dtype))
        lpT, lpS = measure()
        dm = margins(lpT, lpS, state["n_tok_T"], state["n_tok_S"])[0] - margins(base_T64, base_S64, state["n_tok_T"], state["n_tok_S"])[0]
        log(f"noise check (random-sign B of size {args.lr * args.steps:.1e}): per-token margin delta mean {dm.mean():+.5f} "
            f"sd {dm.std():.5f} |max| {np.abs(dm).max():.5f}; frac>0 {np.mean(dm > 0):.2f}", fh)
        state["noise_check_delta"] = dm.astype(np.float32)

    planned = order if meta["subset"] is None else order[:int(meta["subset"])]
    if args.rows is not None:
        planned = order[:args.rows]
    todo = [int(i) for i in planned if not state["done"][i]]
    log(f"rows to measure: {len(todo)} (planned {len(planned)}, order seed {args.seed})", fh)
    n_done_this_run = 0
    t_run = time.time()
    k = 0
    while k < len(todo):        # `todo` may be truncated by the time-budget rule below
        i = todo[k]; k += 1
        t1 = time.time()
        restore(params, init)
        ref_margin = float(base_T64[i] - base_S64[i])
        losses = train_on_event(model, tokenizer, rows[i], ref_margin, args.steps, args.lr, args.beta, device)
        lpT, lpS = measure()
        record_row(state, i, lpT.astype(np.float32), lpS.astype(np.float32), time.time() - t1, losses)
        n_done_this_run += 1
        row = state["T_mean"][i]
        others = np.delete(row, i)
        log(f"row {i:3d} ({n_done_this_run}/{len(todo)}) {state['row_seconds'][i]:.1f}s "
            f"loss {losses[0]:.4f}->{losses[-1]:.4f} T_ii={row[i]:+.5f} "
            f"offdiag mean {others.mean():+.5f} sd {others.std():.5f} frac>0 {np.mean(others > 0):.2f} "
            f"|max| {np.abs(others).max():.4f} cat={meta['category'][i]}", fh)
        if n_done_this_run == 3 and args.rows is None and meta["subset"] is None:
            done_rows = state["row_seconds"][~np.isnan(state["row_seconds"])]
            proj = projected_total(done_rows, len(done_rows), n, base_seconds)
            log(f"projection from first 3 rows: {np.mean(done_rows):.1f}s/row -> {proj / 3600:.2f}h for {n} rows "
                f"(budget {args.time_budget / 3600:.1f}h)", fh)
            if proj > args.time_budget and not args.no_subset:
                meta["subset"] = args.subset_rows
                planned = order[:args.subset_rows]
                todo = [int(j) for j in planned if not state["done"][j]]
                k = 0       # the rebuilt list starts at its first unmeasured row (T9 fix: k=3 skipped three rows)
                log(f"projection exceeds budget: reducing to a random subset of {args.subset_rows} rows "
                    f"(seed {args.seed}; all {n} columns kept); {len(todo)} rows remaining", fh)
        if n_done_this_run % args.ckpt_every == 0:
            save_state(state, args.out, meta)
            log(f"checkpoint: {int(state['done'].sum())} rows -> {args.out}", fh)
    meta["rows_planned"] = int(len(planned))
    meta["wall_seconds_this_run"] = time.time() - t_run
    save_state(state, args.out, meta)
    done = state["done"]
    diag = np.diag(state["T_mean"])[done]
    log(f"finished: {int(done.sum())} rows measured, T_ii>0 for {int((diag > 0).sum())}/{len(diag)}, "
        f"median T_ii {np.median(diag):+.5f}, total row time {np.nansum(state['row_seconds']) / 3600:.2f}h -> {args.out}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
