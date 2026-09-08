# Offline acquisition replay on the BFCL event pool (2026-09-04, T11)

Brief: `docs/2026-09-04-crcd-next-tasks-from-user.md` §4 Block II (offline replay first) and §9
**BF-6** "Acquisition and teacher-token efficiency" (the task was labelled BF-4 in the hand-off;
BF-4 in the brief is atom-wise constraint behaviour — this document is the acquisition item).
Method reference: `docs/METHOD.md` §4 (acquisition) and §9 (teacher-token accounting).

Tool: `tools/acquisition_replay.py` (CPU, 9 s). Tests: `tests/test_acquisition_replay.py` (16,
all pass). Output: `results/analysis/acquisition_replay_bfcl.json`, pools
`data/bfcl_sft/pool_acq_<policy>_{b25,b50,b75,u50}.jsonl` (28 files), log `logs/t11_replay.log`,
tables `logs/t11_tables.log`.

## 1. What is replayed

`data/bfcl_sft/pool_events_pref_v3t.jsonl` (302 rows, 70 tasks) is fully mined: every row has
y^T (`response`), y^S (`_rejected`), the branch outcomes (`_event_u_plus/_minus/_dU`) and the
provenance (`teacher`: 249 `teacher_authored_gt`, 38 `teacher_authored_abstain`, 15
`deepseek-v4-pro-FC` demo rows on 9 official demand tasks). An acquisition policy is therefore
replayed as **subset selection under a teacher-output-token budget**: the policy sees the state,
the atom loadings, the fingerprint and the cost of every candidate; y^T and the outcome are
"revealed" only for selected rows. Nothing is trained here — the subsets are written as pools so
the caller can train them as arms at equal teacher tokens (§5).

ΔU is used exactly as the brief prescribes: it is a derived random variable; no policy filters,
weights, discards ΔU=0 or reverses ΔU<0 (the hard-filter control is included for the record and,
because every v3t row has ΔU>0 — min 0.167, max 0.833 — it selects the same rows as random).

## 2. Cost model (teacher output tokens per row)

| row type | cost | source |
|---|---|---|
| generated rows (287 rows, 61 tasks) | Qwen3.5 tokens of the originating gen row's `question + ground_truth (+scenario/initial_config/why)`, ×1.3 prose factor | same reconstruction as `tools/bfcl_teacher_tokens.py`; row ↔ gen row by `task_id == id` over `data/bfcl_sft/gen*.jsonl` (ids stored in several files are copies; averaged). **Lower bound**: rejected drafts and repair calls were never stored. |
| demo rows (15 rows, 9 official tasks) | exact `output_token_count` summed over all attempts of that task | official result files via `bfcl_teacher_tokens.demos()['per_task']` (236 – 2,440 tokens per task) |

Several pool rows share one task (the same teacher artefact re-mined at different student states in
rounds 2–4; up to 31 rows per task). The teacher was paid once per task, so the per-row cost is the
**share** = task cost / rows of that task (selecting every row of a task sums to the task cost).
Every subset also reports `tokens_unique_tasks` — the full cost of each distinct task touched,
i.e. the exact bill if the subset were acquired online — and `tokens_campaign_amortised`, which
charges each demo task the demo campaign's cost per verified demo instead of its own count:
1,233,607 tokens bought 34 verified demos → **36,283 tokens per verified demo** (the 9 demo tasks
here: 6,748 exact vs 326,543 campaign-amortised). Under campaign-amortised accounting no budget
below the whole pool would ever buy a demo row; the b-series below uses exact per-task counts.

Pool totals: 15,977 share tokens = 9,229 (generated, estimate) + 6,748 (demo, exact). Per-row
share is extremely skewed: quantiles 3.3 / 3.8 / 17.9 / 50.2 / 1,220 (0/25/50/75/100 %). The
rows of a 10-row `simple_python` task cost ≈4 tokens each; a demo task with 3 attempts costs its
full bill. This skew decides most of what follows.

## 3. Residual model (no training)

* loadings z̄ (302×32) from `data/atoms/bfcl_r3t_K32.npz`, aligned by the unique event key
  `_traj|sha1(prompt)[:16]|sha1(_rejected)[:16]` (302/302 matched);
* base competence v_i = π₀-restricted mass on the best candidate of {y^S, y^T},
  softmax(logp0/length), from `results/analysis/mirror_bf3_base_scores.json` (302/302 cached;
  identical to the trainer's J₀ with `j_mode = best_mass`; fallback `_task_phat` unused);
* J_k(π₀) = Σ_i z̄_ki v_i / Σ_i z̄_ki (`mirror.atom_J_from_events`), 0.37–0.49 across atoms;
  target ρ_k = 1; residual r_k = [1 − J_k]_+ ∈ [0.51, 0.63]; posterior residual risk
  R = Σ_k r_k² = 10.89;
* predicted first-order transfer k(i,j) = max(ψ_i·ψ_j, 0) on the unit-norm Fisher-whitened
  fingerprints (`data/fingerprints/bfcl_r3t_v1.npz`; off-diagonal mean 0.18, 67 % positive). The
  r2 transfer matrix validated ψ·ψ as a predictor of transfer rank/sign (ρ 0.68, sign AUROC 0.80,
  `docs/2026-09-04-transfer-matrix.md`); it is used here as given, not re-validated.

## 4. Policies (exact rules)

All rules fill the budget in their order and skip rows that no longer fit, so the budget is never
exceeded and no remaining row fits (tested).

1. **random** — permutation, seed 0.
2. **cheapest_first** — share ascending; ties (rows of one task) random, seed 0.
3. **consequential_first** — |ΔU| descending, ties by cost then row (the established rule).
4. **first_divergence** — `turn_index` ascending, ties random seed 0. 288/302 rows are turn 0, so
   it is random over turn-0 rows with the 14 turn>0 rows last (a proxy: BFCL rows carry no state
   depth beyond the turn).
5. **residual_per_token** — greedy. State: residuals r, selected set S. One selected event moves
   every atom residual by the fixed step r_k ← r_k (1 − η z̄_ki), η = 0.25, so its predicted
   reduction of R is δ_i(r) = Σ_k r_k² [1 − (1 − η z̄_ki)²]. Overlap with what is already selected
   discounts it: disc_i(S) = 1 / (1 + γ Σ_{j∈S} k(i,j)), γ = 1. Score = δ_i(r) · disc_i(S) / cost_i;
   pick the highest-scoring candidate that still fits (ties: lower cost, then lower row —
   deterministic); apply the step; repeat until nothing fits. Sensitivity at b50: η ∈ {0.1,
   0.25, 0.5} × γ ∈ {0.5, 1, 2} all give Jaccard ≥ 0.97 with the default selection.
6. **residual_per_token_no_overlap** — the same with disc ≡ 1 (ablation).
7. **du_positive_filter** — keep ΔU > 0, then random seed 0 (rejected rule; control). Equals
   random on v3t.

Two budget definitions are replayed, B ∈ {25, 50, 75} % each:
* **b** = share of the pool's teacher tokens (the acquisition question proper; pools `_b25/_b50/_b75`);
* **u** = share of the rows, cost ≡ 1 (equal-row-count control that isolates the value model from
  the 300× cost spread; cheapest_first ≡ random there; pools written at 50 % only, `_u50`).

## 5. Results

"pred. risk" is the residual model's own R after the selected steps (R₀ = 10.89); policies 5/6
optimise it, so it is a sanity check, not evidence. "cats" = distinct `_seed_category`.

### Token budgets

| budget | policy | rows | tokens (share) | tokens (unique tasks) | tasks | demo rows | cats | turn>0 | mean ΔU | pred. risk | Jaccard vs random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 (3,994) | random | 87 | 3,992 | 8,419 | 41 | 5 | 16 | 5 | 0.680 | 3.46 | 1.00 |
| | cheapest | 243 | 3,960 | 3,960 | 36 | 0 | 14 | 0 | 0.671 | 0.72 | 0.25 |
| | consequential | 184 | 3,994 | 5,291 | 35 | 2 | 13 | 0 | 0.788 | 1.39 | 0.27 |
| | first-div | 106 | 3,992 | 6,415 | 41 | 4 | 14 | 0 | 0.657 | 2.83 | 0.21 |
| | **residual/tok** | 200 | 3,993 | 5,558 | 43 | 0 | 16 | 4 | 0.662 | 0.98 | 0.25 |
| | residual/tok no-overlap | 241 | 3,967 | 4,219 | 40 | 0 | 14 | 0 | 0.665 | 0.71 | 0.26 |
| | ΔU>0 filter | = random | | | | | | | | | |
| b50 (7,989) | random | 155 | 7,987 | 13,092 | 55 | 9 | 18 | 7 | 0.656 | 1.55 | 1.00 |
| | cheapest | 285 | 7,867 | 7,867 | 58 | 5 | 19 | 9 | 0.657 | 0.40 | 0.49 |
| | consequential | 217 | 7,988 | 11,267 | 48 | 6 | 19 | 9 | 0.759 | 0.89 | 0.44 |
| | first-div | 158 | 7,988 | 11,590 | 54 | 7 | 19 | 0 | 0.665 | 1.52 | 0.39 |
| | **residual/tok** | 278 | 7,936 | 8,717 | 58 | 2 | 17 | 13 | 0.658 | 0.42 | 0.49 |
| | residual/tok no-overlap | 284 | 7,851 | 8,402 | 59 | 3 | 19 | 11 | 0.658 | 0.40 | 0.50 |
| | ΔU>0 filter | = random | | | | | | | | | |
| b75 (11,983) | random | 207 | 11,981 | 14,797 | 60 | 13 | 19 | 10 | 0.660 | 0.86 | 1.00 |
| | cheapest | 297 | 11,527 | 11,527 | 66 | 11 | 20 | 14 | 0.653 | 0.33 | 0.67 |
| | consequential | 262 | 11,981 | 13,230 | 57 | 12 | 20 | 12 | 0.701 | 0.51 | 0.64 |
| | first-div | 285 | 11,915 | 13,379 | 64 | 12 | 20 | 0 | 0.656 | 0.40 | 0.65 |
| | **residual/tok** | 297 | 11,737 | 12,014 | 67 | 10 | 20 | 14 | 0.654 | 0.32 | 0.67 |
| | residual/tok no-overlap | 297 | 11,549 | 11,549 | 66 | 10 | 20 | 14 | 0.653 | 0.32 | 0.67 |
| | ΔU>0 filter | = random | | | | | | | | | |

### Equal row counts (uniform cost control)

| budget | policy | rows | tokens (share) | tokens (unique tasks) | tasks | demo rows | cats | turn>0 | mean ΔU | pred. risk | Jaccard vs random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| u25 (75 rows) | random | 75 | 3,493 | 7,758 | 38 | 5 | 15 | 4 | 0.664 | 4.06 | 1.00 |
| | consequential | 75 | 1,865 | 4,546 | 26 | 2 | 6 | 0 | 0.833 | 4.49 | 0.14 |
| | first-div | 75 | 4,594 | 8,235 | 34 | 5 | 12 | 0 | 0.662 | 4.11 | 0.14 |
| | **residual/tok** | 75 | 10,798 | 13,552 | 50 | 12 | 20 | 13 | 0.609 | 3.39 | 0.18 |
| | residual/tok no-overlap | 75 | 11,173 | 13,436 | 48 | 12 | 20 | 12 | 0.589 | 3.35 | 0.15 |
| u50 (151 rows) | random | 151 | 7,892 | 13,092 | 55 | 9 | 18 | 7 | 0.656 | 1.62 | 1.00 |
| | consequential | 151 | 4,512 | 7,238 | 30 | 3 | 9 | 0 | 0.833 | 2.03 | 0.32 |
| | first-div | 151 | 7,939 | 11,590 | 54 | 7 | 19 | 0 | 0.660 | 1.63 | 0.37 |
| | **residual/tok** | 151 | 14,487 | 15,909 | 69 | 15 | 21 | 14 | 0.631 | 1.19 | 0.33 |
| | residual/tok no-overlap | 151 | 14,295 | 15,839 | 68 | 15 | 21 | 14 | 0.619 | 1.17 | 0.35 |
| u75 (226 rows) | random | 226 | 13,026 | 15,008 | 62 | 14 | 20 | 11 | 0.662 | 0.68 | 1.00 |
| | consequential | 226 | 12,087 | 13,459 | 54 | 13 | 19 | 12 | 0.756 | 0.73 | 0.63 |
| | first-div | 226 | 12,183 | 13,261 | 62 | 13 | 20 | 0 | 0.665 | 0.72 | 0.62 |
| | **residual/tok** | 226 | 15,647 | 15,977 | 70 | 15 | 21 | 14 | 0.653 | 0.56 | 0.62 |
| | residual/tok no-overlap | 226 | 15,273 | 15,977 | 70 | 15 | 21 | 14 | 0.643 | 0.54 | 0.64 |

(cheapest_first and ΔU>0 filter coincide with random in the u-series and are omitted.)

Pairwise Jaccard, b50: residual/tok vs cheapest 0.93, vs no-overlap 0.95, vs consequential 0.71,
vs random 0.49. u50: residual/tok vs cheapest 0.37, vs consequential 0.22, vs no-overlap 0.81,
vs random 0.33.

### What the replay says before any training

1. **Under token budgets, every value-aware rule converges to "buy the cheap rows".** The 300×
   spread of per-row cost dwarfs the ≤ 2–3× spread of predicted per-event value, so at b50 the
   residual rule is 93 % identical to cheapest-first and at b75 both are the full pool minus the
   five most expensive demo rows (rows 148/194 `live_irrelevance_433-107-1` at 2,440 tokens,
   250 `parallel_199`, 167 `live_multiple_923-191-11`, 168 `live_simple_20-4-0`). The
   token-budget series is therefore most discriminative at **b25** (Jaccard 0.21–0.76 between
   policies; residual/tok 200 rows / 43 tasks vs cheapest 243 rows / 36 tasks vs consequential
   184 rows / 35 tasks vs random 87 rows / 41 tasks).
2. **At equal row counts the residual rule is a coverage rule.** It spreads over 69/70 tasks and
   all 21 categories at u50 (random: 55 tasks / 18 categories), takes every demo row and every
   turn>0 row, and lowers the share of `simple_python` from 48 % (random) to 11 %. Its subset is
   also the most *expensive* one at equal rows (14.5k vs 7.9k tokens), i.e. the value model and
   the cost model pull in opposite directions on this pool.
3. **Consequential-first is the narrowest selector.** At u50 it keeps only ΔU ≥ 0.8 rows from 30
   tasks / 9 categories (81 `simple_python`, no abstain rows beyond 3, no turn>0 rows); at token
   budgets it also concentrates (48 tasks at b50 vs 58 for residual/tok). The mean ΔU of the
   residual selection (0.63–0.66) is at or slightly below random — the rule does not chase ΔU.
4. The overlap term changes little (Jaccard 0.95 at b50, 0.81 at u50): the fixed-step residual
   update already produces most of the diminishing returns; ψ·ψ overlap matters mainly under
   uniform cost, where it swaps ~30 rows for lower-kernel neighbours.
5. The rejected ΔU>0 hard filter is vacuous on v3t (all rows positive); it is kept as a control
   only so the ledger records that its selection equals random here.

## 6. What to train (caller submits; single seed, base student, pref-only 1 epoch = BF-1 anchor)

Train each pool with the same recipe as the BF-1 pairwise anchor (base Qwen3.5-4B student,
pref-only, 1 epoch, seed 0, the established base-centred pairwise objective) and score with the
official harness (Overall + Relevance/Irrelevance/multi-turn/memory axes + mastered-prompt KL).
Suggested order:

* **P0 — token budget, most discriminative:** `pool_acq_{random,consequential_first,residual_per_token,cheapest_first}_b25.jsonl`
  (≈4.0k teacher tokens each; 87 / 184 / 200 / 243 rows).
* **P1 — equal rows, value model only:** `pool_acq_{random,consequential_first,residual_per_token}_u50.jsonl`
  (151 rows each; note residual/tok costs 14.5k vs 7.9k tokens here — this arm answers "does the
  residual model pick better rows", not "per token").
* **P2 — token budget 50 %:** `pool_acq_{random,consequential_first,residual_per_token}_b50.jsonl`,
  plus `residual_per_token_no_overlap_{b25,u50}` for the overlap ablation.
* b75 pools are ≈ the full pool for every rule except random (297 vs 302 rows) and need not be
  trained; the full-pool BF-1 anchor already covers that point.

Each arm's teacher-token figure is `results[<policy>][<tag>].tokens_share` (replay accounting);
report `tokens_unique_tasks` alongside as the online-equivalent bill, and follow METHOD §9 for the
demo campaign (any arm containing a demo row carries the 1,233,607-token campaign in its
cumulative account; b25 residual/tok and cheapest contain none).

## 7. How to read the outcome (Go / No-Go)

Metric: official Overall at matched teacher tokens (b-series) and at matched rows (u-series), with
the per-axis breakdown; single-seed noise on BFCL Overall is about ±1.3 points, so differences
inside that band are not decisions.

* **Go** (residual acquisition supported): at b25 (and b50 if run) residual/tok beats **both**
  random and consequential-first on Overall by more than the noise band, with no single-axis
  collapse (Irrelevance/Relevance boundary, multi-turn), **and** at u50 residual/tok beats random.
  The u50 comparison separates a value effect from a cost effect.
* **Cost-only** (not evidence for the residual model): residual/tok beats random/consequential at
  b25 but cheapest-first does as well or better, and u50 shows residual/tok ≤ random. Then the
  finding is "more cheap rows beat fewer expensive rows" — report it as such and do not claim
  atom-residual acquisition.
* **No-Go** (negative, keep): residual/tok ≤ random or ≤ consequential-first at equal tokens.
  Record in the ledger; the offline replay then does not license online acquisition (brief §4.3:
  online acquisition only after replay passes).
* Secondary reads: consequential-first vs random at u50 tests whether ΔU-driven narrowing hurts
  coverage (the brief's warning that raw ΔU is not a valid selector); residual/tok vs its
  no-overlap ablation tests whether the ψ·ψ transfer term adds anything beyond atom residuals.

## 8. Caveats

* Generated-row costs are a re-tokenisation lower bound (no rejected drafts/repairs; Qwen3.5
  tokenizer as a proxy for deepseek's).
* The residual model is first-order and static (fixed step η, target ρ = 1, base residuals only);
  it does not re-estimate reachability or residuals between rounds (Block II §4.4). Sequential
  re-planning would change the b50/b75 selections little (they are near the full pool) but could
  change b25/u50.
* BFCL rows have no state depth beyond `turn_index`, so first-divergence is a weak proxy here.
* The predicted-risk column is the selector's own objective; only the trained arms decide.
