#!/usr/bin/env python3
"""Offline acquisition replay on the BFCL event pool (unified CRCD brief §4 Block II / §9 BF-6).

The pool ``data/bfcl_sft/pool_events_pref_v3t.jsonl`` is fully mined: every row already carries
the teacher continuation y^T (``response``), the student continuation y^S (``_rejected``), the
branch outcomes (``_event_u_plus`` / ``_event_u_minus`` / ``_event_dU``) and a provenance
(``teacher``).  An acquisition policy therefore reduces to *subset selection under a teacher-output
token budget*: the policy sees the state, the atom loadings and the cost of every candidate, and
"reveals" y^T / the outcome only for the rows it selects.  The selected subsets are written as
training pools so they can be trained and evaluated later as arms at equal teacher tokens.

Cost model (teacher OUTPUT tokens per row; docs/METHOD.md §9)
  * generated rows (``teacher`` = teacher_authored_gt / teacher_authored_abstain): the teacher's
    output for the originating generated task (question + ground_truth [+ scenario /
    initial_config / why]) re-tokenised with the Qwen3.5 tokenizer, x PROSE_FACTOR (1.3) - the
    same reconstruction as tools/bfcl_teacher_tokens.py; rejected drafts and repair calls were not
    stored, so this is a LOWER BOUND.  Rows are matched to ``data/bfcl_sft/gen*.jsonl`` by
    ``task_id == id``; an id stored in several gen files is averaged (they are copies).
  * deepseek demo rows (``teacher`` = deepseek-v4-pro-FC): the exact ``output_token_count`` summed
    over every attempt of that official task (tools/bfcl_teacher_tokens.demos() per_task).
  * several pool rows share one task (the same teacher artefact re-mined at different student
    states in rounds 2-4).  The teacher was paid once per task, so the per-row cost is the task
    cost divided by the number of pool rows of that task (``share``); selecting all rows of a task
    sums to exactly the task cost.  Every subset also reports ``tokens_unique_tasks`` (the full
    cost of each distinct task touched: the exact bill if the subset were acquired online) and a
    "campaign-amortised" figure that charges each demo task the demo campaign's cost per verified
    demo (1,233,607 / 34 = 36,283 tokens) instead of its own per-task count.

Residual model (Block I/III quantities, no training)
  * loadings zbar (n, K) from ``data/atoms/bfcl_r3t_K32.npz`` (rows sum to 1; aligned to the pool
    by the unique event key ``_traj|sha1(prompt)[:16]|sha1(_rejected)[:16]``);
  * base competence v_i = pi_0-restricted mass on the best candidate,
    p_0(y | s_i; {y^S, y^T}) = softmax(logp0 / length) from the cached base scores
    ``results/analysis/mirror_bf3_base_scores.json`` (exactly the trainer's J_0 definition,
    ``j_mode = best_mass``); fallback for rows without a cached score: v_i = ``_task_phat``;
  * J_k(pi_0) = sum_i zbar_ki v_i / sum_i zbar_ki (mirror.atom_J_from_events), target rho_k = 1,
    residual r_k = [1 - J_k(pi_0)]_+ ;
  * predicted first-order transfer between events k(i, j) = max(psi_i . psi_j, 0) with psi the
    unit-norm Fisher-whitened fingerprints of ``data/fingerprints/bfcl_r3t_v1.npz`` (the r2
    transfer matrix validated psi . psi as a predictor of transfer sign and rank,
    docs/2026-09-04-transfer-matrix.md).

Greedy residual-reduction-per-token (policy ``residual_per_token``)
  state: residuals r (K,), selected set S, spent tokens.
  one selected event i moves every atom residual by the fixed step
        r_k <- r_k (1 - eta zbar_ki)                       (eta = 0.25 by default),
  so its predicted reduction of the posterior residual risk  R = sum_k r_k^2  is
        delta_i(r) = sum_k r_k^2 [1 - (1 - eta zbar_ki)^2] = sum_k r_k^2 (2 eta zbar_ki - eta^2 zbar_ki^2).
  The event-level overlap with what is already selected discounts it,
        disc_i(S) = 1 / (1 + gamma sum_{j in S} k(i, j))    (gamma = 1 by default),
  and the score is  delta_i(r) * disc_i(S) / share_i .  At each step the candidate with the
  highest score among those that still fit the remaining budget is selected (ties: lower cost,
  then lower pool row - deterministic); r is updated with the step above; stop when no candidate
  fits.  ``residual_per_token_no_overlap`` is the same rule with disc = 1 (ablation).

Other policies (all fill the budget in their order, skipping rows that no longer fit):
  random (seed 0) | cheapest_first (share ascending, ties random seed 0) | consequential_first (|dU|
  descending, ties by cost then row - the established selection rule) | first_divergence
  (turn_index ascending, ties random seed 0; 288/302 rows are turn 0, so it is nearly random) |
  du_positive_filter (keep dU > 0, then random seed 0 - the rejected hard filter, as a control;
  every row of v3t has dU > 0, so it coincides with random).

Usage:
    PYTHONPATH=src .venv/bin/python tools/acquisition_replay.py \
        --pool data/bfcl_sft/pool_events_pref_v3t.jsonl --budgets 0.25,0.5,0.75 \
        --out-json results/analysis/acquisition_replay_bfcl.json --out-dir data/bfcl_sft
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import mirror  # noqa: E402

PROSE_FACTOR = 1.3
GEN_FIELDS = ("question", "ground_truth", "scenario", "initial_config", "why")
DEMO_TEACHER = "deepseek-v4-pro-FC"
DEMO_CAMPAIGN_TOTAL = 1_233_607
DEMO_CAMPAIGN_VERIFIED = 34
DEMO_CAMPAIGN_PER_VERIFIED = DEMO_CAMPAIGN_TOTAL / DEMO_CAMPAIGN_VERIFIED
POLICIES = ("random", "cheapest_first", "consequential_first", "first_divergence",
            "residual_per_token", "residual_per_token_no_overlap", "du_positive_filter")
DU_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)


def sha16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def event_key_for_row(row: dict) -> str:
    return f"{mirror.event_id_for_row(row)}|{sha16(row['prompt'])}|{sha16(row['_rejected'])}"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_loadings(path: Path, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """zbar (n, K) aligned to ``rows`` by the unique event key; fingerprint row per pool row."""
    L = np.load(path, allow_pickle=True)
    keys = [str(k) for k in L["event_key"]]
    pos = {k: i for i, k in enumerate(keys)}
    Z = np.zeros((len(rows), int(L["K"])))
    fp_row = np.full(len(rows), -1, dtype=np.int64)
    missing = []
    for i, row in enumerate(rows):
        j = pos.get(event_key_for_row(row))
        if j is None:
            missing.append(i)
            continue
        Z[i] = L["Z"][j]
        fp_row[i] = int(L["fingerprint_row"][j])
    if missing:
        raise ValueError(f"{len(missing)} pool rows have no loadings (first: {missing[:5]})")
    return Z, fp_row


def load_kernel(path: Path, fp_row: np.ndarray) -> np.ndarray:
    """k(i, j) = max(psi_i . psi_j, 0) on unit-normalised fingerprints."""
    psi = np.load(path)["psi"].astype(np.float64)[fp_row]
    psi /= np.maximum(np.linalg.norm(psi, axis=1, keepdims=True), 1e-12)
    return np.maximum(psi @ psi.T, 0.0)


def base_values(rows: list[dict], base_scores: dict | None) -> tuple[np.ndarray, dict]:
    """v_i = pi_0 mass on the best candidate of Y_i (trainer's j_mode=best_mass), fallback _task_phat."""
    by_suffix: dict[str, dict] = {}
    for k, v in (base_scores or {}).items():
        parts = k.split("|")
        by_suffix["|".join(parts[-3:])] = v  # event_id | sha16(prompt) | owner:sha16(text)
    v = np.zeros(len(rows))
    n_cached = 0
    for i, row in enumerate(rows):
        ev = mirror.candidates_from_row(row, "auto")
        hits = [by_suffix.get(f"{ev.event_id}|{sha16(ev.prompt)}|{c.key()}") for c in ev.candidates]
        if all(h is not None for h in hits):
            for c, h in zip(ev.candidates, hits):
                c.logp0, c.length = float(h["logp0"]), int(h["length"])
            p0 = mirror.restricted_policy(ev.scores0())
            v[i] = mirror.event_J_value(p0, ev, "best_mass")
            n_cached += 1
        else:
            v[i] = float(row.get("_task_phat", 0.0))
    return v, {"n_rows": len(rows), "n_from_base_scores": n_cached,
               "n_from_task_phat": len(rows) - n_cached}


def atom_residuals(Z: np.ndarray, v: np.ndarray, rho: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    J0 = mirror.atom_J_from_events(Z, v)
    J0 = np.where(np.isnan(J0), 0.0, J0)
    return J0, np.clip(rho - J0, 0.0, 1.0)


def load_demo_per_task() -> dict[str, int]:
    spec = importlib.util.spec_from_file_location("bfcl_teacher_tokens", ROOT / "tools/bfcl_teacher_tokens.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: int(t) for k, t in m.demos()["per_task"].items()}


def gen_task_tokens(gen_files: list[Path], tokenize) -> dict[str, dict]:
    """Teacher-authored content tokens per generated task id (mean over copies), x PROSE_FACTOR."""
    per_id: dict[str, list[int]] = collections.defaultdict(list)
    for f in gen_files:
        for r in read_jsonl(f):
            text = "\n".join(json.dumps(r[k], ensure_ascii=False) for k in GEN_FIELDS if k in r)
            per_id[r.get("id")].append(int(tokenize(text)))
    return {k: {"content_tokens": statistics.mean(v), "copies": len(v),
                "estimate": statistics.mean(v) * PROSE_FACTOR} for k, v in per_id.items()}


def row_costs(rows: list[dict], gen_tok: dict[str, dict], demo_tok: dict[str, int]) -> list[dict]:
    """Per row: task cost, share = task cost / rows of that task, kind, campaign-amortised task cost."""
    n_rows_task = collections.Counter(r["task_id"] for r in rows)
    out = []
    for r in rows:
        tid = r["task_id"]
        if r.get("teacher") == DEMO_TEACHER:
            if tid not in demo_tok:
                raise KeyError(f"no exact demo token count for official task {tid}")
            task_cost, kind, amort = float(demo_tok[tid]), "demo_exact", DEMO_CAMPAIGN_PER_VERIFIED
        else:
            if tid not in gen_tok:
                raise KeyError(f"generated task {tid} not found in gen*.jsonl")
            task_cost, kind, amort = float(gen_tok[tid]["estimate"]), "generated_estimate", float(gen_tok[tid]["estimate"])
        out.append({"task_id": tid, "kind": kind, "task_cost": task_cost,
                    "share": task_cost / n_rows_task[tid], "rows_of_task": n_rows_task[tid],
                    "task_cost_campaign_amortised": amort})
    return out


# --------------------------------------------------------------------------
# selection rules
# --------------------------------------------------------------------------

def fill_in_order(order: list[int], cost: np.ndarray, budget: float, mask: np.ndarray | None = None) -> list[int]:
    """Take rows in ``order`` while they fit; skip rows that no longer fit (never exceeds budget)."""
    sel, spent = [], 0.0
    for i in order:
        if mask is not None and not mask[i]:
            continue
        if spent + cost[i] <= budget + 1e-9:
            sel.append(int(i))
            spent += cost[i]
    return sel


def predicted_reduction_step(r: np.ndarray, zbar_i: np.ndarray, eta: float) -> float:
    """delta_i(r) = sum_k r_k^2 [1 - (1 - eta zbar_ki)^2]  (reduction of R = sum_k r_k^2)."""
    f = 1.0 - eta * zbar_i
    return float((r ** 2 * (1.0 - f ** 2)).sum())


def greedy_residual(Z: np.ndarray, r0: np.ndarray, kernel: np.ndarray | None, cost: np.ndarray,
                    budget: float, eta: float = 0.25, gamma: float = 1.0) -> tuple[list[int], list[dict]]:
    """Greedy expected-residual-reduction-per-token (see module docstring). Deterministic."""
    n = Z.shape[0]
    r = r0.astype(np.float64).copy()
    overlap = np.zeros(n)          # sum_{j in S} k(i, j)
    selected: list[int] = []
    chosen = np.zeros(n, dtype=bool)
    spent = 0.0
    trace = []
    while True:
        best, best_key = -1, None
        for i in range(n):
            if chosen[i] or spent + cost[i] > budget + 1e-9:
                continue
            delta = predicted_reduction_step(r, Z[i], eta)
            disc = 1.0 / (1.0 + gamma * overlap[i]) if kernel is not None else 1.0
            score = delta * disc / max(cost[i], 1e-9)
            key = (-score, cost[i], i)
            if best_key is None or key < best_key:
                best, best_key = i, key
        if best < 0:
            break
        selected.append(best)
        chosen[best] = True
        spent += cost[best]
        r_before = float((r ** 2).sum())
        r *= (1.0 - eta * Z[best])
        if kernel is not None:
            overlap += kernel[:, best]
        trace.append({"row": best, "score": -best_key[0], "cost": float(cost[best]),
                      "risk_before": r_before, "risk_after": float((r ** 2).sum()), "spent": spent})
    return selected, trace


def replay_residual(Z: np.ndarray, r0: np.ndarray, sel: list[int], eta: float) -> dict:
    """Residual model evaluated on any subset (order-invariant product of the fixed steps)."""
    r = r0.astype(np.float64).copy()
    for i in sel:
        r *= (1.0 - eta * Z[i])
    return {"risk_before": float((r0 ** 2).sum()), "risk_after": float((r ** 2).sum()),
            "risk_reduction": float((r0 ** 2).sum() - (r ** 2).sum()),
            "residual_l1_before": float(r0.sum()), "residual_l1_after": float(r.sum()),
            "residual_after": [round(float(x), 5) for x in r]}


def select(policy: str, budget: float, rows: list[dict], cost: np.ndarray, Z: np.ndarray, r0: np.ndarray,
           kernel: np.ndarray, seed: int = 0, eta: float = 0.25, gamma: float = 1.0) -> list[int]:
    n = len(rows)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if policy == "random":
        return fill_in_order(list(rng.permutation(n)), cost, budget)
    if policy == "cheapest_first":
        tie = rng.permutation(n)   # rows of one task share a cost; random tie-break (seed) not row order
        return fill_in_order(sorted(idx, key=lambda i: (cost[i], tie[i])), cost, budget)
    if policy == "consequential_first":
        du = np.abs(np.array([float(r["_event_dU"]) for r in rows]))
        return fill_in_order(sorted(idx, key=lambda i: (-du[i], cost[i], i)), cost, budget)
    if policy == "first_divergence":
        turn = np.array([int(r.get("turn_index", 0)) for r in rows])
        tie = rng.permutation(n)
        return fill_in_order(sorted(idx, key=lambda i: (turn[i], tie[i])), cost, budget)
    if policy == "du_positive_filter":
        du = np.array([float(r["_event_dU"]) for r in rows])
        return fill_in_order(list(rng.permutation(n)), cost, budget, mask=du > 0)
    if policy == "residual_per_token":
        return greedy_residual(Z, r0, kernel, cost, budget, eta, gamma)[0]
    if policy == "residual_per_token_no_overlap":
        return greedy_residual(Z, r0, None, cost, budget, eta, gamma)[0]
    raise ValueError(f"unknown policy {policy!r}")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def du_hist(du: np.ndarray) -> dict:
    counts, _ = np.histogram(du, bins=np.array(DU_BINS))
    return {f"[{DU_BINS[b]:.1f},{min(DU_BINS[b + 1], 1.0):.1f})": int(c) for b, c in enumerate(counts)}


def summarise(sel: list[int], rows: list[dict], costs: list[dict], Z: np.ndarray, r0: np.ndarray,
              kernel: np.ndarray, eta: float, reference: list[int] | None = None) -> dict:
    sub = [rows[i] for i in sel]
    du = np.array([float(r["_event_dU"]) for r in sub]) if sub else np.zeros(0)
    tasks = {}
    for i in sel:
        tasks.setdefault(costs[i]["task_id"], costs[i])
    demo_tasks = [t for t, c in tasks.items() if c["kind"] == "demo_exact"]
    n_demo_rows = sum(1 for i in sel if costs[i]["kind"] == "demo_exact")
    K = kernel[np.ix_(sel, sel)] if len(sel) > 1 else np.zeros((1, 1))
    off = K[~np.eye(len(sel), dtype=bool)] if len(sel) > 1 else np.zeros(0)
    out = {
        "n_rows": len(sel),
        "tokens_share": float(sum(costs[i]["share"] for i in sel)),
        "tokens_unique_tasks": float(sum(c["task_cost"] for c in tasks.values())),
        "tokens_campaign_amortised": float(sum(c["task_cost_campaign_amortised"] for c in tasks.values())),
        "n_tasks": len(tasks), "n_demo_rows": n_demo_rows, "n_demo_tasks": len(demo_tasks),
        "n_turn_gt0": int(sum(1 for r in sub if int(r.get("turn_index", 0)) > 0)),
        "categories": dict(collections.Counter(r.get("_seed_category") for r in sub).most_common()),
        "teacher": dict(collections.Counter(r.get("teacher") for r in sub).most_common()),
        "dU_hist": du_hist(du), "dU_mean": float(du.mean()) if len(du) else None,
        "mean_pairwise_kernel": float(off.mean()) if len(off) else None,
        "predicted": replay_residual(Z, r0, sel, eta),
        "pool_rows": sorted(int(i) for i in sel),
        "selection_order": [int(i) for i in sel],
    }
    if reference is not None:
        a, b = set(sel), set(reference)
        out["jaccard_vs_random"] = len(a & b) / max(len(a | b), 1)
    return out


def write_pool(rows: list[dict], sel: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i in sorted(sel):               # original pool order, rows verbatim
            fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/bfcl_sft/pool_events_pref_v3t.jsonl")
    ap.add_argument("--loadings", default="data/atoms/bfcl_r3t_K32.npz")
    ap.add_argument("--fingerprints", default="data/fingerprints/bfcl_r3t_v1.npz")
    ap.add_argument("--base-scores", default="results/analysis/mirror_bf3_base_scores.json")
    ap.add_argument("--gen-glob", default="data/bfcl_sft/gen*.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--budgets", default="0.25,0.5,0.75", help="fractions of the pool's total cost")
    ap.add_argument("--cost-modes", default="share,uniform",
                    help="'share' (teacher tokens; pools pool_acq_<policy>_b<pct>) and/or 'uniform' "
                         "(1 per row, equal-row-count control; pools written at 50%% only, _u50)")
    ap.add_argument("--eta", type=float, default=0.25, help="fixed residual step per selected event")
    ap.add_argument("--gamma", type=float, default=1.0, help="overlap discount strength")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="results/analysis/acquisition_replay_bfcl.json")
    ap.add_argument("--out-dir", default="data/bfcl_sft", help="pool_acq_<policy>_b<pct>.jsonl go here")
    ap.add_argument("--pool-budgets", default="all", help="'all' or comma list of pct to write pools for")
    args = ap.parse_args(argv)
    t0 = time.time()

    rows = read_jsonl(ROOT / args.pool)
    Z, fp_row = load_loadings(ROOT / args.loadings, rows)
    kernel = load_kernel(ROOT / args.fingerprints, fp_row)
    bs_path = ROOT / args.base_scores
    base = json.loads(bs_path.read_text()) if bs_path.is_file() else None
    v, v_info = base_values(rows, base)
    J0, r0 = atom_residuals(Z, v)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    gen_tok = gen_task_tokens(sorted(Path(p) for p in glob.glob(str(ROOT / args.gen_glob))),
                              lambda s: len(tok(s)["input_ids"]))
    demo_tok = load_demo_per_task()
    costs = row_costs(rows, gen_tok, demo_tok)
    share = np.array([c["share"] for c in costs])
    total = float(share.sum())
    print(f"[acq] rows={len(rows)} total_share_tokens={total:.0f} "
          f"(gen est {sum(c['share'] for c in costs if c['kind'] != 'demo_exact'):.0f} + demo exact "
          f"{sum(c['share'] for c in costs if c['kind'] == 'demo_exact'):.0f}); "
          f"base values: {v_info}; R0={float((r0 ** 2).sum()):.4f}", flush=True)

    budgets = [float(b) for b in args.budgets.split(",")]
    pool_pcts = None if args.pool_budgets == "all" else {int(p) for p in args.pool_budgets.split(",")}
    results: dict[str, dict] = {}
    pools: dict[str, str] = {}
    for mode in args.cost_modes.split(","):
        # 'share': budget = fraction of the pool's teacher tokens, cost_i = share_i (the acquisition
        # question proper).  'uniform': cost_i = 1, budget = fraction of the rows - an equal-row-count
        # control that isolates the value model from the 300x spread of per-row token costs.
        cost = share if mode == "share" else np.ones(len(rows))
        tag = "b" if mode == "share" else "u"
        for frac in budgets:
            B = frac * float(cost.sum())
            pct = int(round(frac * 100))
            ref = select("random", B, rows, cost, Z, r0, kernel, args.seed, args.eta, args.gamma)
            for pol in POLICIES:
                sel = ref if pol == "random" else select(pol, B, rows, cost, Z, r0, kernel, args.seed, args.eta, args.gamma)
                s = summarise(sel, rows, costs, Z, r0, kernel, args.eta, reference=ref)
                s["budget"] = B
                s["cost_mode"] = mode
                results.setdefault(pol, {})[f"{tag}{pct}"] = s
                if (pool_pcts is None or pct in pool_pcts) and (mode == "share" or pct == 50):
                    p = ROOT / args.out_dir / f"pool_acq_{pol}_{tag}{pct}.jsonl"
                    write_pool(rows, sel, p)
                    pools[f"{pol}_{tag}{pct}"] = str(p.relative_to(ROOT))
                print(f"[acq] {pol:30s} {tag}{pct:<3d} rows={s['n_rows']:3d} tokens={s['tokens_share']:8.0f} "
                      f"tasks={s['n_tasks']:2d} demo_rows={s['n_demo_rows']:2d} risk {s['predicted']['risk_before']:.3f}->"
                      f"{s['predicted']['risk_after']:.3f} J_rand={s['jaccard_vs_random']:.2f}", flush=True)

    # hyper-parameter sensitivity of the main rule at the 50% budget (Jaccard vs the default)
    B50 = 0.5 * total
    default = set(select("residual_per_token", B50, rows, share, Z, r0, kernel, args.seed, args.eta, args.gamma))
    sens = {}
    for eta in (0.1, 0.25, 0.5):
        for gamma in (0.5, 1.0, 2.0):
            s = set(greedy_residual(Z, r0, kernel, share, B50, eta, gamma)[0])
            sens[f"eta{eta}_gamma{gamma}"] = {"n_rows": len(s), "jaccard_vs_default": len(s & default) / len(s | default)}

    n_rows_task = collections.Counter(c["task_id"] for c in costs)
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pool": args.pool, "n_rows": len(rows),
        "loadings": args.loadings, "fingerprints": args.fingerprints, "base_scores": args.base_scores,
        "tokenizer": args.tokenizer, "prose_factor": PROSE_FACTOR,
        "cost_model": {
            "generated": "mean Qwen3.5 tokens of question+ground_truth(+scenario/initial_config/why) of the gen row with id == task_id, x1.3 (lower bound)",
            "demo": "exact output_token_count summed over all attempts of the official task (tools/bfcl_teacher_tokens.demos)",
            "share": "task cost / number of pool rows of that task (teacher paid once per task)",
            "campaign_amortised_per_verified_demo": DEMO_CAMPAIGN_PER_VERIFIED,
            "demo_campaign_total": DEMO_CAMPAIGN_TOTAL, "demo_campaign_verified": DEMO_CAMPAIGN_VERIFIED,
        },
        "pool_cost": {
            "total_share_tokens": total,
            "generated_rows": int(sum(1 for c in costs if c["kind"] != "demo_exact")),
            "demo_rows": int(sum(1 for c in costs if c["kind"] == "demo_exact")),
            "generated_tasks": len({c["task_id"] for c in costs if c["kind"] != "demo_exact"}),
            "demo_tasks": len({c["task_id"] for c in costs if c["kind"] == "demo_exact"}),
            "generated_tokens_estimate": float(sum(c["share"] for c in costs if c["kind"] != "demo_exact")),
            "demo_tokens_exact": float(sum(c["share"] for c in costs if c["kind"] == "demo_exact")),
            "demo_tokens_campaign_amortised": float(sum({c["task_id"]: c["task_cost_campaign_amortised"]
                                                         for c in costs if c["kind"] == "demo_exact"}.values())),
            "share_quantiles": {q: float(np.percentile(share, q)) for q in (0, 25, 50, 75, 100)},
            "rows_per_task_max": max(n_rows_task.values()),
            "per_task": {c["task_id"]: {"cost": c["task_cost"], "rows": c["rows_of_task"], "kind": c["kind"]}
                         for c in costs},
        },
        "residual_model": {
            "rho": 1.0, "eta": args.eta, "gamma": args.gamma, "K": int(Z.shape[1]),
            "base_values": v_info, "J0": [round(float(x), 5) for x in J0],
            "r0": [round(float(x), 5) for x in r0], "R0": float((r0 ** 2).sum()),
            "kernel": "max(psi_i.psi_j, 0), unit-norm Fisher-whitened fingerprints",
            "kernel_offdiag_mean": float(kernel[~np.eye(len(rows), dtype=bool)].mean()),
        },
        "policies": {
            "random": "uniform permutation, seed 0, fill while fits",
            "cheapest_first": "share ascending, ties random seed 0 (== random under uniform cost)",
            "consequential_first": "|dU| descending, ties by cost then row (established rule)",
            "first_divergence": "turn_index ascending, ties random seed 0",
            "residual_per_token": "greedy delta_i(r) * 1/(1+gamma*sum_{j in S} k(i,j)) / share_i; r_k <- r_k(1-eta*zbar_ki) after each pick",
            "residual_per_token_no_overlap": "same greedy without the overlap discount",
            "du_positive_filter": "keep dU > 0, then random seed 0 (rejected hard filter; control)",
        },
        "notes": [
            "all 302 rows of v3t have dU > 0: du_positive_filter selects the same rows as random",
            "288/302 rows are turn_index 0: first_divergence is random within turn-0 rows (turn>0 rows go last)",
            "predicted risk reduction is the residual model's own quantity; residual_per_token optimises it, so it is a sanity check, not evidence",
            "budget fractions refer to the pool's total share cost (generated estimate + exact demo tokens) for b<pct>, and to the row count for u<pct> (uniform cost control; cheapest_first == random there)",
            "per-row share costs span ~3 to ~1220 tokens (rows of a 10-row generated task cost ~4 each; a demo task with 3 attempts costs its full bill): under token budgets every value-aware rule converges to 'buy the cheap rows', see results[*][b*].jaccard and docs/2026-09-04-acquisition-replay.md",
        ],
        "sensitivity_b50": sens,
        "results": results, "pools": pools, "seconds": round(time.time() - t0, 1),
    }
    out_path = ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[acq] wrote {out_path} and {len(pools)} pools in {out['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
