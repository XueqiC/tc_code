# Block IV — certification of the BFCL teacher-consistent arm (crcd_r3_union_t_s0)

Date 2026-09-04 (T10). Code: `src/bfas/certify.py` (bound primitives), `tools/certify_bfcl.py`
(orchestration), `tests/test_certify.py` (15 CPU tests, all pass). Output:
`results/analysis/certify_bfcl_r3t.json`, hash map `results/analysis/certify_bfcl_hashmap.json`.
Nothing here was tuned on the held-out data; every constant (alpha = 0.05, delta = 0.05,
tau grid, min n_eff = 5, UCB threshold 0.1) is fixed a priori by the brief.

Reference model = `results/bfcl_std/base_hpg` (fetched from hpg, overall 46.06 %, the METHOD §7
anchor). The older local `results/bfcl_std/base` is a different evaluation run (46.27 %) and is not used.

## 1. Representation coverage certificate

**Definition.** Basis `U` (256 x 32) = uncentred PCA-32 of the L2-normalised Fisher-whitened r3t
fingerprints (`data/atoms/bfcl_r3t_K32.npz`, 302 rows, 198 unique state hashes). Per event
`c_i = ||P_U psi_i||^2 / ||psi_i||^2` (P_U = orthogonal projector on span(U); psi are unit-norm so
this is `1 - min_z ||psi_i - U z||^2 / ||psi_i||^2`); set coverage = spec §6.1
`1 - min_Z ||Psi - U Z||_F^2 / ||Psi||_F^2` (energy-weighted mean of `c_i`).
**Guarantee.** Descriptive: percentile-bootstrap 95 % CI over events (2000 resamples, seed 0).
References: 20 random orthonormal 32-bases (expected K/d = 0.125), PCA-32 refit on the evaluated set
itself (in-sample upper reference), and a 5-fold *group* cross-fit (state_hash groups never split).

| set | n | per-event mean [95 % CI] | set | random basis | self-PCA in-sample | group cross-fit |
|---|---|---|---|---|---|---|
| bfcl_r2_v1 minus 7 calibration rows ("spec") | 203 | 0.732 [0.706, 0.754] | 0.732 | 0.127 | 0.775 | 0.594 |
| bfcl_r2_v1 strict held-out (state_hash not in fit set) | 9 | 0.392 [0.257, 0.536] | 0.392 | 0.130 | – | – |
| bfcl_r3_v1 minus 4 calibration rows ("spec") | 144 | 0.709 [0.680, 0.736] | 0.709 | 0.129 | 0.779 | 0.558 |
| bfcl_r3_v1 strict held-out | 7 | 0.370 [0.231, 0.523] | 0.370 | 0.128 | – | – |
| bfcl_r3t_v1 5-fold group cross-fit (honest held-out) | 302 | 0.603 [0.573, 0.631] | fold mean 0.604 (0.555–0.650) | 0.127 | in-sample 0.759 | = |
| ALFWorld psi on the BFCL basis (cross-benchmark) | 385 | 0.110 [0.109, 0.112] | 0.110 | 0.133 | 0.903 | 0.853 |

**Reading.** (i) The brief's "held-out" sets are almost entirely in-sample: 194/203 r2 rows and
137/144 r3 rows share a state hash with the r3t fit set, so 0.73/0.71 is in-sample coverage. (ii) The
honest number is the group cross-fit on r3t: **0.60 [0.57, 0.63]** per event (set 0.60), 4.7x the
random-basis reference 0.127 and well below the in-sample 0.76; the 9 + 7 strictly new states give
0.39 / 0.37 (wide CIs, n tiny) — new states are covered at roughly half the in-sample rate.
(iii) ALFWorld fingerprints are *not* covered by the BFCL dictionary: 0.110, i.e. at or slightly below
a random 32-basis (0.133) — a different Fisher and a different capability subspace; the ALFWorld set has
its own 32-d structure (self-PCA 0.90, cross-fit 0.85). Coverage is reported per §6.1 together with the
transfer results elsewhere; it is not a success guarantee on its own.

## 2. Atom-wise residual certificate (from the primal-dual traces, job 41122834)

Traces fetched from hpg: `results/analysis/mirror_bf3_{rho1,rhoJ0,k1}_trace.jsonl` (38 steps x 8
events = 304 visits, each of the 302 pool rows visited once, plus summaries).

**Per-event term (src/bfas/mirror.py).** `JEstimator.add(zbar_i, v_i)` accumulates
`J_k = sum_i zbar_ki v_i / sum_i zbar_ki` with `zbar_ki = |z_ki| / sum_j |z_ji|` (per-event loadings
summing to 1) and `v_i = event_J_value(p_theta, event, "best_mass") = p_theta[best candidate]`, the
restricted-policy mass on the best-utility candidate, hence `v_i in [0, 1]`. The trainer's own
`J_final` is the last 32-event window; the certificate uses all 302 rows with each row's `v_i` taken
at its (single) visit.

**Guarantee.** For fixed weights `w_ki = zbar_ki / sum_i zbar_ki` and independent `v_i in [0,1]`,
Hoeffding gives `P(J_k - E J_k <= -t) <= exp(-2 t^2 / sum_i w_ki^2)`; with Bonferroni over K atoms
(per-atom level delta/K, delta = 0.05) all K bounds hold jointly with probability >= 0.95:
`LCB[J_k] = J_k - sqrt(ln(K/delta) sum_i w_ki^2 / 2)`, `UCB[r_k] = [rho_k - LCB[J_k]]_+`,
`n_k = sum_i zbar_ki`, `n_eff,k = 1 / sum_i w_ki^2`. Atoms with `n_eff < 5` would be marked
unresolved (none are). A seed-task cluster bootstrap 5 % quantile of `J_k` is reported as a
diagnostic (not a guarantee) because the independence assumption is doubtful (below).

| trace | K | rho | mean J (trainer window J_final) | mean r | mean UCB[r] (max) | median half-width | median n_eff | n_k range | atoms with UCB[r] > 0.1 | certified satisfied | lambda_final at cap 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rho1 (rho = 1, eps 0.05) | 32 | 1.0 | 0.564 (0.66) | 0.436 | 0.579 (0.696) | 0.137 | 172 | 5.8–51.5 | **32 / 32** | 0 | 32 / 32 |
| rhoJ0 (rho = J0 + 0.3, eps 0) | 32 | 0.67–0.79 | 0.523 (0.59) | 0.194 | 0.338 (0.414) | 0.137 | 172 | 5.8–51.5 | **32 / 32** | 0 | 0 / 32 (final 3.6–4.3, still rising) |
| k1 (single atom) | 1 | 1.0 | 0.437 (0.448) | 0.563 | 0.633 | 0.070 | 302 | 302 | 1 / 1 | 0 | 1 / 1 |

Per-atom rows (rhoJ0): J 0.48–0.61 versus J0 0.37–0.49 (+0.10 to +0.12 on every atom during the
single pass), LCB[J] 0.27–0.46, UCB[r] 0.32–0.41, n_k from 51.5 (atom 0, the 31 % variance component)
down to 5.8 (atoms 27–29), n_eff 72–195. lambda trajectories: rho1 — every multiplier hits the cap
kappa = 5 (first cap hits at steps 20/24/28/32; 16–47 % of the 38 steps spent at cap; 5–8 dual
increases, 0 decreases), i.e. every constraint is binding and the slack xi is nonzero on all atoms;
rhoJ0 — multipliers rise monotonically at all 9 dual updates (0 decreases) to 3.6–4.3 without
reaching the cap. Full tables in the JSON
(`atoms.<tag>.atoms[k]`: J, rho, r, LCB_J, UCB_r, n_weighted, n_eff, n_events, J0_trainer,
J_trainer_window_final, cluster_bootstrap_q05_J, lambda{init,final,max,n_increase,frac_steps_at_cap,first_step_at_cap}).

**What is certified.** With rho_k = 1 no atom is certified satisfied and every atom's residual is
certified above 0.1 (UCB[r] >= 0.49): the rho = 1 constraint set is provably unmet after 38 steps.
With rho_k = J0 + 0.3, the certified residual is 0.32–0.41 on all atoms (point residual 0.17–0.21):
none satisfied even at point estimate (`J >= rho - eps` fails on all 32).

**Caveats (all reduce the strength of the claim).** (a) Independence: the 302 rows are generated
variants of only 80 event ids / 20 seed tasks, so `v_i` are clustered; the cluster-bootstrap 5 %
quantile (0.36–0.49 for rhoJ0) is close to the Hoeffding LCB but is not a bound. (b) The per-event
values were scored under a *drifting* policy during one training pass (early rows near pi_0, late rows
near the final policy); the certificate therefore bounds the pass-average J, not the final policy's J.
The trainer's last-window J_final is 0.05–0.10 higher. A final-policy certificate requires one
re-scoring pass of all 302 events with the saved adapter (GPU, not done here). (c) Weights `zbar` are
fixed PCA loadings, not posterior relevances (no q_phi yet), so `n_k` is a loading mass, not a
probe count. (d) `v_i` is the mirror-target mass on the best candidate among two (teacher/student
continuations), not a task success probability; rho = 1 is the strictest reading of "match the best
continuation" and is unrealistic under gamma_perp/KL regularisation.

## 3. Held-out risk / selective-deployment certificate

### 3a. What the calibration set actually is
`data/bfcl_sft/calibration_ids.json` holds 60 hashes `sha1(rendered prompt)[:16]`. Rendering every
official single-turn entry (3641) and every generated task exactly as the miner does
(`BFCLAdapter._render` on `populate_test_cases_with_predefined_functions` output) maps
**0 of 60 hashes to official prompts and 60 of 60 to generated tasks: all are `simple_python`
variants from `gen_pool_v3.jsonl`, seeded from one official id (which is in the derived, not the
exact, training set).** They are not official BFCL prompts, so the official score files contain no
outcome for them. Per-prompt outcomes exist only for the base model, from the gate-v2 pass-rate files
(`data/bfcl_sft/gate_v2/rates_base.json`, k = 8 sampled replies per prompt).

### 3b. Official single-turn set (13 categories, 3641 ids; per-id correctness recovered)
Per-id correctness = id in the official test file and not in the category score file's failure list
(`total_count` = number of official ids and `correct_count` = ids - failures verified for every
category and both models). Contamination: 9 official ids appear verbatim in the r3t training pool
(support-demand tasks with verified teacher demos), and generated training rows were seeded from 20
official ids; tables are given for all ids / excluding the 9 / excluding the 20+9 — the numbers move
by < 0.001 and the conclusions are identical, so the "all" table is quoted.

Pooled failure rate (exact Clopper–Pearson 95 %): student 741/3641 = **0.2035 [0.1905, 0.2170]**,
base 726/3641 = 0.1994 [0.1865, 0.2128]. Paired (McNemar exact): student-only-correct b = 78,
base-only-correct c = 93, p = 0.28; paired difference -0.0041 [-0.0112, +0.0029] (Wald). The student
is not distinguishable from base on the single-turn set overall; per category it is significantly
better on `parallel` (+0.090, b = 20, c = 2, p = 1.2e-4) and significantly worse on
`live_irrelevance` (-0.067, b = 5, c = 64, p = 4e-14) — the call/abstain boundary moved, matching
METHOD §7's "one coupled boundary".

Per-category one-sided UCB at level alpha/13 (Bonferroni, so all 13 hold jointly at 95 %):

| category | n | student fail / UCB | base fail / UCB |
|---|---|---|---|
| simple_python | 400 | 0.085 / 0.129 | 0.095 / 0.141 |
| multiple | 200 | 0.085 / 0.151 | 0.075 / 0.139 |
| parallel_multiple | 200 | 0.135 / 0.211 | 0.135 / 0.211 |
| irrelevance | 240 | 0.146 / 0.216 | 0.133 / 0.202 |
| live_multiple | 1053 | 0.204 / 0.239 | 0.214 / 0.249 |
| parallel | 200 | 0.175 / 0.257 | 0.265 / 0.356 |
| live_simple | 258 | 0.217 / 0.293 | 0.236 / 0.314 |
| live_irrelevance | 884 | 0.279 / 0.321 | 0.213 / 0.252 |
| simple_java | 100 | 0.380 / 0.518 | 0.430 / 0.568 |
| live_relevance | 16 | 0.188 / 0.546 | 0.250 / 0.610 |
| live_parallel_multiple | 24 | 0.292 / 0.582 | 0.333 / 0.623 |
| simple_javascript | 50 | 0.420 / 0.616 | 0.500 / 0.690 |
| live_parallel | 16 | 0.375 / 0.723 | 0.438 / 0.772 |

Category-level risk–coverage (deploy categories with UCB <= tau; certified risk of the deployed pool
= n-weighted mean UCB, valid jointly with the table):

| tau | student: coverage / certified / empirical (deployed) | base: coverage / certified / empirical |
|---|---|---|
| 0.1 | 0 / – / – (none) | 0 / – / – |
| 0.2 | 0.165 / 0.136 / 0.085 (simple_python, multiple) | 0.165 / 0.140 / 0.088 |
| 0.3 | 0.701 / 0.217 / 0.164 (+ parallel_multiple, irrelevance, live_multiple, parallel, live_simple) | 0.818 / 0.221 / 0.176 (base also deploys live_irrelevance, not parallel/live_simple) |
| 0.4 | 0.943 / 0.244 / 0.194 (+ live_irrelevance) | 0.943 / 0.236 / 0.186 |
| 0.5 | 0.943 / 0.244 / 0.194 | 0.943 / 0.236 / 0.186 |

The five small or hard categories (simple_java, simple_javascript, live_parallel,
live_parallel_multiple, live_relevance; 206 ids) are never certifiable below tau = 0.5 because
n <= 100 makes the Bonferroni-corrected exact UCB too wide (n = 16 gives UCB > 0.5 even at 3 failures).
The curve is in-sample on the official set (the same ids fix the UCBs), so "empirical <= certified"
is guaranteed there and is not a calibration check.

### 3c. Realised vs nominal on the untouched calibration prompts
Base only (student rollouts missing, see 3d). The 60 calibration prompts all fall in
`simple_python`, whose official UCB (0.141) admits deployment at every tau >= 0.2. Realised
prompt-level failure of base on those prompts (majority of 8 samples wrong): **29/60 = 0.483, exact
95 % CI [0.352, 0.616], one-sided 95 % UCB 0.597**; sample-level 0.55 (mean pass rate 0.45; only 1/60
prompts is passed by all 8 samples). Realised 0.48 exceeds the nominal tau at tau = 0.2, 0.3, 0.4 and
is within it only at tau = 0.5. **The official-category certificate does not transfer to the generated
calibration prompts**: the generated `simple_python` variants are ~5x harder for base than the
official ones (0.095 vs 0.483), so a category-level certificate calibrated on official prompts is
miscalibrated on the distribution the acquisition pipeline actually produces. (Decoding differs too:
official = single near-greedy reply, gate = 8 sampled replies at the miner temperature; majority-fail is
the closest prompt-level analogue.) Per brief §11, no formal selective-deployment certificate is
claimed; the numbers above are the residual diagnostic.

### 3d. What could not be certified and why
* **Student calibration error, McNemar student-vs-base on the calibration set, student realised-vs-nominal**:
  no rollouts of `crcd_r3_union_t_s0` on the 60 calibration prompts exist anywhere (they are generated
  prompts, absent from every official result/score file). Needed: k-sample rollouts of the checkpoint on
  exactly these 60 prompts (`tools/bfcl_event_mine_single.py --rates-key prompt --only-ids ...`
  against the served checkpoint, or any script writing
  `data/bfcl_sft/gate_v2/rates_crcd_r3_union_t_s0.json` keyed by the same hashes). `tools/certify_bfcl.py run`
  picks that file up automatically and fills in the student Clopper–Pearson bound, the paired McNemar
  and the realised-vs-nominal rows. This is a GPU job (not permitted for T10).
* **Final-policy atom certificate**: needs one re-scoring pass of the 302 events under the saved
  mirror adapter (GPU); the present certificate bounds the training-pass average.
* **Acceptance score from atom residual / assignment uncertainty / margin / support distance (§6.4)**:
  no per-prompt atom loadings exist for official or calibration prompts (fingerprints are only
  computed for mined events), so the risk–coverage curve is category-level, not score-level.
* **Confidence-interval coverage across seeds (§6.4)**: single seed by user decision (2026-09-04).
* A single-category calibration set (60 generated `simple_python` prompts) cannot check calibration
  of any other category; a future calibration split should be stratified over categories and include
  official ids.

## 3e. Stratified official calibration set v2 (follow-up, same day)

**Construction** (`tools/certify_bfcl.py calib-v2`, `data/bfcl_sft/calibration_ids_official_v2.json`).
Universe = the 3641 official single-turn ids. Excluded: every id occurring as `task_id` / `_traj` / `id`
(split on `#`) in any `data/bfcl_sft/pool_*.jsonl`, `events*.jsonl` or `gen*.jsonl` row, every
`_seed_task` / `seed_task` of those rows, the seed official id of every generated id (prefix stripped),
and the seed/demand/calibration lists of `configs/bfcl_support_split.json` — 62 official ids in
total. Then 5 ids per category with `numpy.random.default_rng(0)` over the sorted unused ids:
**n = 65, 5 x 13 categories** (unused pool per category: simple_python 393, simple_java 98,
simple_javascript 48, multiple 196, parallel 196, parallel_multiple 196, irrelevance 237,
live_simple 254, live_multiple 1041, live_parallel 13, live_parallel_multiple 21, live_irrelevance 873,
live_relevance 13). Reserved: never train, mine or select hyper-parameters on these ids.

**Certificate on the held-out ids** (per-id correctness from the official score files; the
category-level UCB certificate is refit on the official ids *minus* these 65, so the check below is
genuinely out-of-sample):

| | fail / n | error rate | exact 95 % CI | one-sided 95 % UCB |
|---|---|---|---|---|
| crcd_r3_union_t_s0 | 9 / 65 | **0.138** | [0.065, 0.247] | **0.229** |
| base | 11 / 65 | 0.169 | [0.088, 0.283] | 0.265 |

Paired: student-only-correct b = 2 (parallel_multiple, live_relevance), base-only-correct c = 0,
difference +0.031 [-0.011, +0.073], McNemar exact p = 0.50 (mid-p 0.25) — not significant at n = 65.
Per-category failures (student / base): simple_java 2/2, simple_javascript 1/1, live_simple 2/2,
live_multiple 2/2, live_parallel 1/1, live_relevance 1/2, parallel_multiple 0/1, all others 0/0.

**Realised vs nominal** (deploy categories whose refit UCB <= tau; realised = failures on the
calibration ids of the deployed categories, exact two-sided 95 % CI):

| tau | student: deployed / coverage / certified / realised | base: deployed / coverage / certified / realised |
|---|---|---|
| 0.2 | 2 cats / 0.165 / 0.139 / 0/10 = 0.000 [0, 0.308] within | 2 / 0.165 / 0.142 / 0/10 = 0.000 within |
| 0.3 | 7 cats / 0.704 / 0.219 / 4/35 = 0.114 [0.032, 0.267] within | 6 / 0.824 / 0.223 / 3/30 = 0.100 within |
| 0.4 | 8 cats / 0.949 / 0.246 / 4/40 = 0.100 [0.028, 0.237] within | 8 / 0.949 / 0.237 / 5/40 = 0.125 within |

On official prompts the category-level certificate is calibrated for both models at every tau
(realised risk below nominal, and the realised CI upper end below the certified pooled risk at
tau = 0.3/0.4); with 5 ids per category the check has little power (a realised 0/10 has UCB 0.31),
so this is a pass of a coarse test, not a tight one. Contrast with 3c: the same certificate fails on
the generated calibration prompts. Both statements stand: the certificate holds in-distribution
(official prompts) and does not transfer to the synthesised task distribution.

Per-category refit UCBs and all numbers: `results/analysis/certify_bfcl_r3t.json` →
`calibration_official_v2`.

## Reproduction
```
PYTHONPATH=src:tools envs/bfcl-venv/bin/python tools/certify_bfcl.py hashes      # one-off hash map
.venv/bin/python tools/certify_bfcl.py calib-v2                                  # stratified official calibration ids (seed 0)
.venv/bin/python tools/certify_bfcl.py run --student crcd_r3_union_t_s0 --base base_hpg
.venv/bin/python -m pytest tests/test_certify.py -q
```
