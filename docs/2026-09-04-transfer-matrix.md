# Single-event transfer matrix on BFCL r2 (T6, 2026-09-04)

Block I of the unified CRCD method predicts transfer between intervention events from gradient
fingerprints (first order: P_ij = η z_j^T U^T U z_i, or ψ_i·ψ_j). The two earlier transfer gates
(`atoms_transfer_gate_v2*.json`) scored these predictions against 8 cluster arms × 8-sample pass rates,
a target with 43% exact zeros. This note measures a dense, directly observed target instead: the effect
of a controlled single-event LoRA update on the margin of every other event.

Files: `tools/bfcl_transfer_matrix.py` (measurement), `tools/bfcl_transfer_matrix_eval.py` (predictors,
statistics), `tests/test_transfer_matrix.py` (11 CPU tests, pass), `results/analysis/transfer_matrix_bfcl_r2.npz`,
`results/analysis/transfer_matrix_bfcl_r2_eval.json`, `logs/t6_transfer_matrix.log`.

## 1. Protocol

Pool: `data/bfcl_sft/pool_events_pref_v2.jsonl`, 210 events (160 unique prompts; 188 call / 22 abstain
teacher continuations). Student Qwen/Qwen3.5-4B, bf16, LoRA r=8 / α=16 / dropout 0 / seed 0 on the
trainer's seven target modules (the fingerprint configuration of `bfcl_r2_v1_meta.npz`; the trainer itself
uses r=16, α=32). Encoding = `src/appworld_train.encode` with `AW_MAX_PROMPT_TOKENS=2048`, tail
truncation, response ≤ 512 tokens + EOS.

For each updated event i:

1. reset the adapter to its initialisation (A = init draw, B = 0, so the policy equals the base),
2. 3 AdamW steps (torch defaults) of the trainer's `AW_DISTILL=ddpo` pair loss on event i alone,
   `-logsigmoid(β (policy_margin − base_margin))`, β = 0.1, summed log-probs, batch = one event,
3. re-score all 210 events: m_j = mean-per-token log π(y^T_j|s_j) − mean-per-token log π(y^S_j|s_j);
   `T_mean[i, j] = m_j(after i) − m_j(base)`; `T_sum` is the summed-log-prob version; the per-side
   log-probs after each update are stored too (`logp_T_after`, `logp_S_after`).

Measurement passes are batched (length-sorted, 12k padded tokens per batch, response positions only
through the LM head, no grad). The batch composition is fixed, so before/after differ only by the adapter
delta; a reset-and-remeasure null check reproduces the base values to 1e-5 nats (float32 storage).

**Step size (deviation from the brief).** The brief asked for the trainer's lr (5e-6). At that size
three Adam steps move each LoRA-B coordinate by ~1.5e-5 and the resulting margin changes sit at the bf16
forward-pass rounding floor: a random-sign B perturbation of the same size changes per-token margins with
sd 0.0064–0.0073 *independently of its size* (5e-5 … 2e-4 tested), and the two smoke rows at 5e-6 gave
T_ii = +0.014 and −0.005 with off-diagonal sd 0.006–0.007, i.e. unresolvable (their off-diagonal rows
correlate 0.04–0.25 with the same rows at larger lr). The sign of the loss is the trainer's; the
problem is resolution, not sign. Smoke scan on the same two rows (events 6, 147):

| lr | T_ii | final pair loss | off-diag sd | corr with the 2e-4 row |
|---|---|---|---|---|
| 5e-6 | +0.014 / −0.005 | 0.667 / 0.688 | 0.006 / 0.007 | 0.09 / 0.04 |
| 5e-5 | +0.197 / +0.386 | 0.340 / 0.529 | 0.049 / 0.011 | 0.97 / 0.76 |
| 1e-4 (used) | +0.418 / +0.585 | 0.135 / 0.284 | 0.088 / 0.030 | – |
| 2e-4 | +0.787 / +0.733 | 0.046 / 0.202 | 0.145 / 0.065 | 1 |

The full run used lr = 1e-4 (20× the trainer's per-step lr; three steps ≈ 3e-4 per coordinate, about
2× the cumulative per-coordinate move of one ddpo epoch of an arm, 26 steps × 5e-6). Noise floor stored
in the npz (`noise_check_delta`, sd 0.0073 per token).

**Smoke check.** All measured self-effects are positive: T_ii ∈ [0.18, 2.01] per token, median 0.84,
120/120 rows (> 24× the noise floor). Row 6 / 147 / 97 (first three rows): +0.418 / +0.585 / +0.930.

**Cost.** 67 s per row on the RTX 6000 Ada (3 train steps + 420 forward passes ≈ 312k tokens), 54 s base
pass, ~18–33 GB. The 3-row projection was 3.96 h for 210 rows > 3 h budget, so the run was cut to the
first 120 rows of the seed-0 permutation (all 210 columns kept): 25,080 off-diagonal pairs, 7,140
symmetric pairs, 2.24 h of row time. (A loop bug let the job continue past row 120; it was stopped at the
120-row checkpoint and the bug is fixed; row 121 was discarded.)

## 2. What the matrix looks like

- Off-diagonal T_mean: mean +0.142, median +0.029, sd 0.306, 75% positive, 73% of |T_ij| above 2× the
  noise floor, 119/120 rows with off-diagonal sd > 2× noise. Spearman(T_mean, T_sum) = 0.82.
- By stratum (different-prompt pairs): same category +0.703 vs different category +0.025; same side
  +0.178 vs different side −0.010; same prompt (62 pairs) +0.468.
- **The matrix is a student-side push-down.** Splitting T = ΔlogpT/n − ΔlogpS/n: corr(T, −ΔS) = 0.94,
  var(ΔS) = 0.082 vs var(ΔT) = 0.010 (var T = 0.094); off-diagonal mean ΔT = +0.055 (83% positive),
  ΔS = −0.086. The pairwise loss mostly lowers the student's own continuations everywhere.
- **Low rank and symmetric.** Rank-1 carries 86% of the Frobenius norm of the 120×120 block (rank 5: 97%;
  a norm-matched Gaussian: 3% / 15%); participation ratio 1.34; σ = 35.5, 9.4, 6.5, 4.2, …. Symmetric part
  = 95% of the norm, Pearson(T_ij, T_ji) = 0.87, Spearman 0.67, sign agreement 74%. An additive row+column
  model explains only 35% (R² row-only 0.17, column-only 0.17): the rank-1 mode is multiplicative, and it is
  the `simple_python` block — the leading left/right singular vectors average 0.149 / 0.157 on
  simple_python events and ≤ 0.011 elsewhere; mean row transfer 0.316 from simple_python rows vs
  0.02–0.10 from every other category. simple_python student continuations are the longest (median 184
  tokens) and 97% of those rows drive the pair loss below 0.01.
- **Saturation.** Final pair loss quantiles 1e-4 / 1.1e-3 / 6.4e-3 / 0.029 / 0.118 (10/25/50/75/90%):
  60% of rows end below 0.01, only 21 rows end above 0.05. The per-row update size is therefore
  heterogeneous; row-centred metrics remove the scale, the "non-saturated" subset is reported below.

## 3. Predictors vs measured off-diagonal transfer (T_mean, 120 rows × 209 columns)

CIs: two-way group bootstrap by prompt (1000; rows and columns drawn from the same prompt resample).
Perm p: one event permutation applied to both axes of the predictor, 500 draws (p = 0.002 is the floor).
`dbl` = double-centred Spearman (row and column main effects removed). `diffP` = different-prompt pairs only.

| predictor | ρ off-diag [95% CI] | ρ row-c | ρ col-c | dbl [95% CI] | sign AUROC [CI] | NDCG@5 [CI] | perm p | diffP ρ |
|---|---|---|---|---|---|---|---|---|
| ψ·ψ (Fisher-whitened) | 0.674 [0.604, 0.734] | 0.825 | 0.827 | 0.857 [0.809, 0.879] | 0.774 [0.734, 0.810] | 0.799 [0.765, 0.843] | 0.002 | 0.672 |
| raw sketch dot | 0.677 [0.609, 0.739] | 0.810 | 0.827 | 0.835 [0.782, 0.864] | 0.797 [0.763, 0.832] | 0.792 | 0.002 | 0.675 |
| sparse dict z_j^T U^T U z_i (K=32, λ=0.02) | **0.686** [0.618, 0.744] | 0.833 | 0.836 | **0.863** [0.819, 0.885] | 0.782 [0.743, 0.819] | 0.762 [0.730, 0.817] | 0.002 | 0.684 |
| PCA-32 cosine | 0.527 [0.440, 0.630] | 0.710 | 0.763 | 0.861 [0.818, 0.885] | 0.674 [0.628, 0.724] | 0.780 | 0.002 | 0.524 |
| k-means(8) same cluster | 0.559 [0.458, 0.660] | 0.686 | 0.765 | 0.814 [0.747, 0.865] | 0.603 [0.576, 0.639] | 0.577 | 0.002 | 0.556 |
| gate-v2 same cluster | 0.589 [0.497, 0.679] | 0.687 | 0.772 | 0.820 [0.782, 0.875] | 0.610 [0.583, 0.645] | 0.592 [0.558, 0.674] | 0.002 | 0.588 |
| same category | 0.608 [0.525, 0.690] | 0.772 | 0.828 | 0.858 [0.810, 0.892] | 0.605 [0.577, 0.640] | 0.629 [0.580, 0.695] | 0.002 | 0.605 |
| same side (call/abstain) | 0.304 [0.186, 0.409] | 0.124 | 0.124 | 0.035 [−0.014, 0.113] | 0.612 [0.553, 0.669] | 0.268 | 0.002 | 0.303 |
| same prompt | 0.062 | −0.084 | −0.063 | −0.002 [−0.045, 0.073] | 0.502 | 0.321 | 0.002 | – |
| student length of j (column only) | 0.319 [0.237, 0.389] | 0.267 | – | −0.010 | 0.689 [0.632, 0.744] | 0.108 | 0.002 | 0.321 |
| −base margin of j (column only) | −0.127 [−0.257, 0.006] | −0.197 | 0.023 | 0.002 | 0.461 | 0.160 | 1.0 | −0.128 |
| random | 0.001 [−0.020, 0.022] | −0.001 | 0.001 | 0.004 [−0.017, 0.025] | 0.498 [0.484, 0.511] | 0.226 | 0.41 | 0.001 |

Paired bootstrap differences (same resamples, 500):

| difference | Δρ off-diag [95% CI] | Δ dbl [95% CI] | Δ sign AUROC [95% CI] |
|---|---|---|---|
| ψ·ψ − same category | +0.065 [+0.015, +0.110] | −0.004 [−0.038, +0.024] | +0.169 [+0.131, +0.201] |
| sparse dict − same category | +0.077 [+0.024, +0.126] | +0.003 [−0.026, +0.030] | +0.177 [+0.136, +0.213] |
| ψ·ψ − same side | +0.372 [+0.264, +0.495] | +0.813 [+0.720, +0.885] | +0.164 [+0.106, +0.229] |
| ψ·ψ − same prompt | +0.587 [+0.510, +0.677] | +0.835 [+0.756, +0.893] | +0.273 [+0.235, +0.318] |
| sparse dict − ψ·ψ | +0.012 [+0.005, +0.020] | +0.007 [+0.001, +0.014] | +0.008 [+0.003, +0.013] |
| ψ·ψ − raw sketch | −0.002 [−0.030, +0.025] | +0.022 [+0.010, +0.040] | −0.022 [−0.042, −0.003] |
| PCA-32 cosine − ψ·ψ | −0.145 [−0.211, −0.084] | +0.007 [−0.009, +0.030] | −0.100 [−0.144, −0.058] |

Within strata (continuous predictors only):

| stratum (pairs) | ψ·ψ ρ / AUROC | sparse dict | PCA-32 | random |
|---|---|---|---|---|
| different prompt, same category (4,265) | 0.441 / 0.972 | 0.419 / 0.973 | 0.237 / 0.961 | −0.002 / 0.508 |
| different category (20,753) | 0.471 / 0.714 | 0.492 / 0.724 | 0.223 / 0.587 | 0.002 / 0.497 |
| different prompt, same side (20,054) | 0.720 / 0.790 | 0.731 / 0.799 | 0.588 / 0.682 | 0.004 / 0.501 |

Non-saturated rows only (final pair loss > 0.05; 21 rows, 4,389 pairs, no CIs): ρ / dbl / AUROC / NDCG@5 =
ψ·ψ 0.32 / 0.49 / 0.58 / 0.74; sparse dict 0.33 / 0.52 / 0.58 / 0.71; same category 0.45 / 0.58 / 0.56 / 0.64;
gate cluster 0.29 / 0.34 / 0.54 / 0.47; random 0.03 / 0.00 / 0.52 / 0.16.

Targets split by side (no CIs): on the teacher-side change ΔlogpT/n, ψ·ψ ρ = 0.446 (dbl 0.500, AUROC 0.654)
vs same category 0.387 (dbl 0.519, AUROC 0.556); on the student-side change −ΔlogpS/n, ψ·ψ 0.434 (dbl
0.802) vs same category 0.542 (dbl 0.787). The teacher-side part is less low-rank (rank-1 = 56% of norm vs
94% for the student side) and less symmetric (Pearson 0.64 vs 0.83).

## 4. Cluster aggregate vs the gate-v2 arm gains

For each gate-v2 cluster c and prompt p: A[c, p] = mean over measured rows i ∈ c and events j with prompt
p (i ≠ j) of T_ij; G[c, p] = pass-rate(arm c) − pass-rate(base) from `data/bfcl_sft/gate_v2`
(1,192 pairs, 36% exact-zero gains; measured rows per cluster 40/4/10/12/16/9/17/12).

- All pairs: Spearman 0.23, Pearson 0.59, column-centred 0.39, row-centred 0.30, AUROC(gain > 0) 0.60.
- Off-cluster pairs only (prompt has no event in c; 1,027 pairs): Spearman 0.07, column-centred 0.10,
  AUROC 0.52. Per cluster: c0 0.26, c1 −0.07, c2 0.03, c3 0.03, c4 0.11, c5 0.14, c6 0.12, c7 0.03.
- Cluster level (8 points): mean single-event transfer vs mean arm gain, Spearman 0.90 — driven by cluster 0
  (simple_python: transfer 0.40 / gain +0.31) against seven clusters with transfer 0.004–0.098 and gains
  −0.006 … +0.024.

The single-event matrix reproduces the coarse "cluster 0 helps, the rest barely move" picture and the
in-cluster gains (c0 all-pairs ρ = 0.79); it does not reproduce which *other* prompts an arm helps
(off-cluster ρ ≈ 0.07–0.10). Either the off-cluster arm gains are mostly noise at 8 samples (the earlier
gates' complaint), or a 3-step single-event margin change does not carry over to pass rates after a
full arm; this measurement cannot tell the two apart.

## 5. Honest read

1. **ψ and the dictionary predict measured transfer, and better than the label baselines on the raw
   pair ranking and on the sign.** ρ 0.67–0.69 vs category 0.61 (paired Δ +0.07, CI excludes 0), vs side
   0.30 and prompt 0.06; sign AUROC 0.77–0.80 vs 0.61 for every indicator (Δ +0.17 [0.13, 0.20]);
   NDCG@5 0.80 vs 0.63. Within a category (different prompts) ψ·ψ still ranks pairs at ρ = 0.44 and
   across categories at 0.47, so the fingerprints carry information beyond the category label. All
   permutation p-values are at the floor (0.002) for every structured predictor, including the labels.
2. **After removing row and column main effects, the fingerprint advantage over the category label
   disappears** (dbl 0.857 vs 0.858, Δ −0.004 [−0.038, +0.024]; PCA-32 cosine 0.861, k-means 0.81).
   The residual interaction structure of T is a category block structure that every reasonable
   predictor captures; what ψ adds over the label is the magnitude ordering and the sign of pairs,
   which live in the (multiplicative) main effects.
3. **Sparse dictionary ≈ ψ·ψ** (+0.012 [+0.005, +0.020] on ρ; consistent but tiny). Fisher whitening
   does not help ρ (−0.002 vs raw), helps dbl by +0.02, hurts sign AUROC by −0.02. PCA-32 cosine is
   clearly worse on ρ and AUROC (−0.14 / −0.10) but equal after double centring.
4. **The target is dominated by a rank-1, symmetric, student-side mode** (ψ·ψ vs −ΔS: 0.94 correlation
   with T; simple_python block = 86% of the norm). The first-order theory predicts a symmetric
   Gram-matrix-like T, and that is what is observed (95% symmetric), but the measured mode is largely
   "pushing on one simple_python pair pushes down every simple_python student continuation", which a
   category label predicts about as well. On the teacher-side part (the capability-relevant half),
   ψ·ψ ρ = 0.45 vs category 0.39, AUROC 0.65 vs 0.56 — a smaller but present edge.
5. **Step-size caveats.** (a) 1e-4 is 20× the trainer's per-step lr, forced by the bf16 measurement floor;
   (b) 60% of rows saturate the pair loss, so per-row update sizes vary; on the 21 non-saturated rows
   the category label beats ψ on ρ (0.45 vs 0.32) while ψ keeps the NDCG edge (0.74 vs 0.64) — n is
   too small for CIs. A replicate at 5e-5 on the same rows (≈ 45 min for 40 rows) would settle whether
   the ranking of predictors depends on the regime; the 2-row scan says the *rows* are stable
   (0.76–0.97 correlation between 5e-5 and 2e-4).
6. **Bootstrap caveat for indicator predictors:** the two-way prompt bootstrap duplicates events, which
   adds same-prompt pairs; the same-prompt CI [0.064, 0.111] therefore sits above its point estimate
   (0.062). Continuous predictors are unaffected.
7. Only 120 of 210 rows were measured (budget rule); the remaining 90 rows are resumable
   (`tools/bfcl_transfer_matrix.py --no-subset` continues from the checkpoint at 67 s/row ≈ 1.7 h).

## 6. Open issues

- Replicate at lr 5e-5 (and ideally 2e-4) on a fixed 40-row subset to test regime dependence of the
  predictor ranking and the saturation effect.
- Evaluate predictors on the teacher-side matrix with CIs (only point estimates above), and on a
  span-only (`AW_DDPO_SPAN_ONLY=1`) update, which should remove the student-side common mode.
- The 90 unmeasured rows (all 210 columns exist); a full square matrix would also allow a
  leave-row-out low-rank reconstruction test of the first-order model.
- The cluster-aggregate comparison needs the arm gains at more than 8 samples per prompt before the
  off-cluster mismatch (ρ ≈ 0.07) can be attributed to the matrix rather than to the gains.

## 7. Follow-up A — regime check: lr 5e-5 vs 1e-4 on the same 40 rows

`results/analysis/transfer_matrix_bfcl_r2_lr5e-5.npz`: the first 40 rows of the seed-0 permutation
re-measured at lr 5e-5 (3 steps, same protocol, 0.74 h, noise floor sd 0.0064). Evaluation restricted
to those 40 rows for both matrices (8,360 off-diagonal pairs each; bootstrap 500, permutation 200,
floor p = 0.005). JSON keys `regime_lr5e-5_rows40`, `regime_lr1e-4_rows40`, `regime_comparison_rows40`.

Saturation: at 5e-5 **0/40 rows** end with pair loss < 0.01 (5/40 ≤ 0.05; median final loss 0.164);
at 1e-4 the same rows give 24/40 < 0.01 (32/40 ≤ 0.05; median 0.0065). Self-effect median 0.334 vs 0.826
(min 0.126 vs 0.209), all 40/40 positive at both. Off-diagonal sd is 3.5× larger at 1e-4; 57% (5e-5) vs
73% (1e-4) of |T_ij| exceed 2× the noise floor. The two matrices agree pair-by-pair: Pearson 0.92,
Spearman 0.95 over the 8,360 pairs; per-row Pearson median 0.97, min 0.80. Symmetry 0.90 (5e-5) vs 0.84;
rank-1 share 0.80 vs 0.92.

| predictor | lr 5e-5: ρ [CI] | dbl | AUROC [CI] | NDCG@5 [CI] | lr 1e-4: ρ [CI] | dbl | AUROC [CI] | NDCG@5 [CI] |
|---|---|---|---|---|---|---|---|---|
| ψ·ψ | 0.682 [0.591, 0.761] | 0.861 | 0.765 [0.722, 0.813] | 0.762 [0.683, 0.837] | 0.702 [0.608, 0.772] | 0.840 | 0.787 [0.738, 0.836] | 0.799 [0.725, 0.858] |
| raw sketch | 0.687 [0.593, 0.765] | 0.842 | 0.795 [0.753, 0.835] | 0.766 | 0.702 [0.603, 0.775] | 0.812 | 0.808 [0.762, 0.850] | 0.800 |
| sparse dict | 0.687 [0.595, 0.767] | 0.866 | 0.770 [0.726, 0.818] | 0.722 [0.647, 0.808] | 0.710 [0.615, 0.780] | 0.845 | 0.793 [0.742, 0.841] | 0.758 [0.684, 0.833] |
| PCA-32 cos | 0.544 [0.395, 0.677] | 0.861 | 0.680 [0.609, 0.747] | 0.755 | 0.541 [0.383, 0.672] | 0.840 | 0.681 [0.602, 0.758] | 0.792 |
| gate cluster | 0.621 [0.489, 0.737] | 0.820 | 0.620 [0.578, 0.672] | 0.611 | 0.626 [0.498, 0.739] | 0.815 | 0.624 [0.583, 0.675] | 0.670 |
| same category | 0.652 [0.537, 0.754] | 0.871 | 0.621 [0.578, 0.672] | 0.648 [0.573, 0.714] | 0.652 [0.541, 0.752] | 0.856 | 0.624 [0.582, 0.676] | 0.682 [0.599, 0.743] |
| same side | 0.252 [0.143, 0.379] | 0.011 | 0.574 | 0.243 | 0.280 [0.163, 0.409] | 0.021 | 0.588 | 0.275 |
| random | 0.006 [−0.032, 0.049] | 0.010 | 0.502 | 0.228 | 0.007 [−0.029, 0.050] | 0.013 | 0.499 | 0.243 |

Paired ψ·ψ − category (same resamples): Δρ +0.029 [−0.044, +0.105] at 5e-5 vs +0.049 [−0.024, +0.123]
at 1e-4 (both CIs include 0 at 40 rows; the 120-row CI excluded 0); Δ dbl −0.019 [−0.059, +0.019] vs
−0.021 [−0.057, +0.009]; Δ AUROC +0.144 [+0.089, +0.196] vs +0.163 [+0.106, +0.213]. Sparse dict −
category: Δρ +0.035 [−0.043, +0.113] vs +0.057 [−0.017, +0.135]; Δ AUROC +0.149 vs +0.169.

Read: the ψ-vs-category ordering is **not regime-dependent** within this range. The non-saturated
regime (lr 5e-5, no row below 0.01) gives the same ranking — ψ / dictionary ahead on ρ by a small,
CI-crossing margin, clearly ahead on sign AUROC (+0.14) and NDCG@5, level on double-centred ρ — and the
matrices themselves are nearly proportional (Spearman 0.95). The earlier "non-saturated subset" result
(§3, 21 rows where category beat ψ on ρ) was a row-selection effect, not a regime effect.

## 8. Follow-up B — teacher-side and student-side targets (120 rows)

T^T_ij = per-token Δ log p(y^T_j | s_j) (teacher continuation only) and T^S_ij = −per-token
Δ log p(y^S_j | s_j) (student continuation only); T_mean = T^T + T^S. Same 120-row matrix, full
protocol (bootstrap 1000, permutation 500). JSON keys `side_teacher`, `side_student`.

Descriptives: T^T off-diagonal mean +0.055, sd 0.101, 83% positive, self-effect median 0.25 (94% positive);
symmetric part Pearson 0.64, rank-1 share 0.56. T^S: mean +0.086, sd 0.287, 49% positive, self-effect
median 0.47 (98% positive); symmetry 0.83, rank-1 share 0.94.

| predictor | teacher T^T: ρ [CI] | dbl [CI] | AUROC [CI] | NDCG@5 [CI] | student T^S: ρ [CI] | dbl [CI] | AUROC [CI] | NDCG@5 [CI] |
|---|---|---|---|---|---|---|---|---|
| ψ·ψ | 0.446 [0.367, 0.515] | 0.500 [0.391, 0.599] | 0.654 [0.591, 0.708] | 0.713 [0.658, 0.774] | 0.434 [0.324, 0.543] | 0.802 [0.754, 0.821] | 0.674 [0.622, 0.727] | 0.658 [0.602, 0.726] |
| raw sketch | 0.425 [0.345, 0.502] | 0.490 [0.382, 0.589] | 0.663 [0.605, 0.716] | 0.702 | 0.482 [0.385, 0.587] | 0.783 [0.734, 0.808] | 0.708 [0.659, 0.760] | 0.665 |
| sparse dict | 0.457 [0.378, 0.527] | 0.504 [0.393, 0.606] | 0.659 [0.596, 0.712] | 0.699 [0.646, 0.761] | 0.433 [0.325, 0.544] | 0.807 [0.761, 0.826] | 0.674 [0.620, 0.727] | 0.614 [0.565, 0.687] |
| PCA-32 cos | 0.397 [0.321, 0.480] | 0.494 [0.376, 0.596] | 0.604 [0.556, 0.655] | 0.694 | 0.420 [0.317, 0.525] | 0.807 [0.762, 0.833] | 0.708 [0.659, 0.760] | 0.655 |
| k-means(8) | 0.328 [0.238, 0.423] | 0.407 [0.285, 0.565] | 0.545 [0.513, 0.578] | 0.565 | 0.517 [0.401, 0.628] | 0.782 [0.716, 0.812] | 0.650 [0.608, 0.701] | 0.470 |
| gate cluster | 0.349 [0.257, 0.441] | 0.410 [0.286, 0.572] | 0.557 [0.523, 0.588] | 0.535 | 0.552 [0.449, 0.654] | 0.789 [0.750, 0.821] | 0.659 [0.619, 0.707] | 0.471 |
| same category | 0.387 [0.300, 0.485] | 0.519 [0.404, 0.639] | 0.556 [0.523, 0.588] | 0.646 [0.597, 0.711] | 0.542 [0.430, 0.644] | 0.787 [0.735, 0.817] | 0.650 [0.608, 0.700] | 0.491 [0.438, 0.556] |
| same side | 0.303 [0.181, 0.407] | 0.047 [−0.072, 0.202] | 0.680 [0.603, 0.745] | 0.310 | 0.101 [0.027, 0.190] | 0.030 [−0.000, 0.068] | 0.522 | 0.189 |
| same prompt | 0.072 | 0.104 [0.026, 0.229] | 0.501 | 0.376 | 0.023 | −0.031 | 0.501 | 0.244 |
| student length of j | 0.002 [−0.077, 0.078] | −0.011 | 0.506 | 0.171 | 0.416 [0.359, 0.472] | −0.010 | 0.683 [0.642, 0.721] | 0.059 |
| random | −0.003 [−0.024, 0.019] | −0.003 | 0.498 | 0.254 | −0.003 [−0.024, 0.019] | 0.004 | 0.499 | 0.172 |

Paired ψ·ψ − category: teacher side Δρ +0.054 [−0.020, +0.125], Δ dbl −0.021 [−0.115, +0.042],
Δ AUROC +0.097 [+0.052, +0.140]; student side Δρ **−0.102 [−0.149, −0.063]**, Δ dbl +0.014 [−0.014, +0.044],
Δ AUROC +0.024 [−0.003, +0.050]. Sparse dict − category: teacher Δρ +0.065 [−0.012, +0.137], Δ AUROC
+0.102 [+0.057, +0.146]; student Δρ −0.103 [−0.153, −0.063]. Sparse dict − ψ·ψ: teacher +0.011 [+0.004,
+0.019], student −0.001 [−0.007, +0.005].

Read: the two halves behave differently. The **student-side** matrix (the dominant, rank-1, 94%-of-norm
part) is a category block plus a column effect of student length (ρ 0.42 for length alone): the category
label beats ψ on raw ρ there (Δ −0.10, CI excludes 0), and everything ties after double centring. The
**teacher-side** matrix — the half that says whether the teacher's continuation becomes more likely on
other events — is harder to predict (ρ ≈ 0.45, dbl ≈ 0.50 for the best predictors), is not driven by
length (ρ 0.00), and is where ψ / the dictionary hold their edge: sign AUROC 0.65–0.66 vs 0.56 for
category (Δ +0.10, CI excludes 0), NDCG@5 0.70–0.71 vs 0.65, ρ +0.05 (CI crosses 0). On T^T the same-side
indicator has the best sign AUROC of all (0.68) because 83% of teacher-side pairs are positive and the
negatives concentrate on cross-side pairs; it has no ranking power (ρ 0.30, dbl 0.05).

## 9. Full 210 × 210 matrix (T9, 2026-09-04)

The 90 unmeasured rows were resumed with `tools/bfcl_transfer_matrix.py --no-subset --lr 1e-4` (the flag now lifts
the stored subset on resume; the 120 measured rows were kept untouched, `transfer_matrix_bfcl_r2_rows120_backup.npz`
is the pre-resume checkpoint). 90 rows × 64 s = 1.62 h; total row time 3.88 h. The new rows look like the old ones:
off-diagonal mean +0.165 / sd 0.342 / 74 % positive (old 120: +0.142 / 0.306 / 75 %), median T_ii 0.82 (0.84),
min T_ii 0.27, 60 % of rows end below pair loss 0.01 in both halves; the new half has more simple_python rows
(44 % vs 33 %). T_ii > 0 for **210/210** rows (min 0.178, median 0.833, max 2.01). Evaluation files:
`transfer_matrix_bfcl_r2_full210_eval.json` (T_mean), `…_T_teacher.json`, `…_T_student.json` (bootstrap 1000,
permutation 500, floor p = 0.002). Two label rows were added to the table by the extended evaluator (same task id —
BFCL task ids group several events of one task —, and the extra label rows of §10 where the pool carries them);
everything else is the §3 protocol.

**Matrix (43,890 off-diagonal pairs, 21,945 symmetric pairs).** Off-diagonal mean +0.152, median +0.026, sd 0.322,
75 % positive, 71 % of |T_ij| above 2× the noise floor, 209/210 rows with off-diagonal sd > 2× noise. Strata: same
category / different prompt +0.714, different category +0.023, same prompt (102 pairs) +0.531. Symmetry: Pearson
(T_ij, T_ji) 0.87, Spearman 0.71, sign agreement 75 %, symmetric part 95 % of the norm. Spectrum of the full square:
rank-1 = **91 %** of the Frobenius norm (rank 2/3/5/10: 94/96/98/99 %; Gaussian reference 2/4/5/9/16 %), participation
ratio 1.21, σ = 71.1, 13.8, 9.5, 7.5, 5.8; additive row+column R² 0.42 (row-only 0.22, column-only 0.20); the rank-1
product a_i b_j correlates 0.95 with the off-diagonal entries. The leading singular vectors are still the
simple_python block (mean |u|, |v| = 0.104 / 0.110 on its 80 events, ≤ 0.026 elsewhere; mean row transfer 0.314 from
simple_python rows vs −0.07 … +0.09 from every other category). Spearman(T_mean, T_sum) = 0.84.

| predictor | ρ off-diag [95% CI] | ρ row-c | ρ col-c | dbl [95% CI] | sign AUROC [CI] | NDCG@5 [CI] | perm p | diffP ρ |
|---|---|---|---|---|---|---|---|---|
| ψ·ψ (Fisher-whitened) | 0.686 [0.625, 0.743] | 0.849 | 0.830 | 0.848 [0.801, 0.869] | 0.772 [0.741, 0.805] | 0.806 [0.783, 0.842] | 0.002 | 0.684 |
| raw sketch dot | 0.694 [0.636, 0.749] | 0.833 | 0.830 | 0.828 [0.778, 0.856] | 0.797 [0.769, 0.827] | 0.808 [0.783, 0.847] | 0.002 | 0.692 |
| sparse dict z_j^T U^T U z_i | **0.697** [0.638, 0.752] | 0.857 | 0.836 | **0.853** [0.804, 0.876] | 0.779 [0.748, 0.811] | 0.778 [0.756, 0.822] | 0.002 | 0.696 |
| PCA-32 cosine | 0.548 [0.467, 0.638] | 0.727 | 0.777 | 0.846 [0.791, 0.872] | 0.680 [0.644, 0.722] | 0.786 [0.762, 0.827] | 0.002 | 0.545 |
| k-means(8) same cluster | 0.587 [0.492, 0.676] | 0.723 | 0.731 | 0.793 [0.759, 0.836] | 0.614 [0.586, 0.649] | 0.583 [0.545, 0.640] | 0.002 | 0.584 |
| gate-v2 same cluster | 0.615 [0.523, 0.697] | 0.728 | 0.752 | 0.796 [0.760, 0.842] | 0.621 [0.594, 0.656] | 0.619 [0.579, 0.676] | 0.002 | 0.613 |
| same category | 0.632 [0.553, 0.708] | 0.806 | 0.819 | 0.837 [0.786, 0.873] | 0.616 [0.589, 0.651] | 0.622 [0.593, 0.684] | 0.002 | 0.630 |
| same side (call/abstain) | 0.304 [0.202, 0.402] | 0.133 | 0.157 | 0.049 [−0.018, 0.103] | 0.609 [0.558, 0.661] | 0.256 [0.227, 0.318] | 0.002 | 0.303 |
| same prompt | 0.060 | −0.067 | −0.056 | −0.002 [−0.026, 0.071] | 0.501 | 0.309 | 0.002 | – |
| same task id | 0.243 [0.221, 0.278] | 0.308 | 0.319 | 0.304 [0.208, 0.390] | 0.519 [0.516, 0.528] | 0.557 [0.536, 0.629] | 0.002 | 0.235 |
| student length of j (column only) | 0.322 [0.256, 0.379] | 0.284 | – | 0.001 | 0.675 [0.626, 0.717] | 0.095 | 0.002 | 0.323 |
| −base margin of j (column only) | −0.166 [−0.279, −0.046] | −0.260 | 0.033 | −0.005 | 0.453 | 0.154 | 1.0 | −0.167 |
| random | 0.001 [−0.015, 0.016] | −0.002 | −0.001 | −0.001 [−0.017, 0.015] | 0.501 [0.491, 0.512] | 0.209 | 0.45 | 0.001 |

Paired bootstrap differences (500):

| difference | Δρ off-diag [95% CI] | Δ dbl [95% CI] | Δ sign AUROC [95% CI] |
|---|---|---|---|
| ψ·ψ − same category | +0.054 [+0.005, +0.098] | +0.008 [−0.021, +0.034] | +0.156 [+0.120, +0.187] |
| sparse dict − same category | +0.065 [+0.017, +0.110] | +0.014 [−0.011, +0.038] | +0.163 [+0.126, +0.195] |
| ψ·ψ − k-means(8) | +0.099 [+0.035, +0.167] | +0.044 [+0.015, +0.068] | +0.158 [+0.122, +0.191] |
| ψ·ψ − same side | +0.384 [+0.291, +0.486] | +0.811 [+0.723, +0.884] | +0.164 [+0.117, +0.216] |
| ψ·ψ − same prompt | +0.600 [+0.528, +0.681] | +0.814 [+0.756, +0.873] | +0.270 [+0.238, +0.307] |
| ψ·ψ − same task id | +0.438 [+0.387, +0.496] | +0.541 [+0.449, +0.622] | +0.252 [+0.220, +0.286] |
| sparse dict − ψ·ψ | +0.011 [+0.006, +0.017] | +0.006 [+0.002, +0.012] | +0.007 [+0.002, +0.011] |
| ψ·ψ − raw sketch | −0.007 [−0.032, +0.016] | +0.020 [+0.010, +0.034] | −0.024 [−0.044, −0.007] |
| PCA-32 cosine − ψ·ψ | −0.137 [−0.190, −0.087] | +0.000 [−0.012, +0.015] | −0.092 [−0.126, −0.059] |

Within strata (ρ / sign AUROC): different prompt, same category (8,082 pairs) ψ·ψ 0.413 / 0.973, sparse dict
0.401 / 0.973, PCA-32 0.237 / 0.953; different category (35,706) ψ·ψ 0.459 / 0.703, sparse dict 0.480 / 0.712;
different prompt, same task id (1,206) ψ·ψ 0.292 / 0.891; different task (42,582) ψ·ψ 0.660 / 0.763. Non-saturated
rows (final loss > 0.05; 41 rows, no CIs): ρ / dbl / AUROC / NDCG@5 = ψ·ψ 0.36 / 0.57 / 0.60 / 0.78, sparse dict
0.37 / 0.60 / 0.59 / 0.76, same category 0.40 / 0.57 / 0.55 / 0.55, random 0.01 / 0.00 / 0.51 / 0.14 — as in §7, the
category label edges ψ on ρ in the saturation-selected subset, ψ keeps sign and top-5 ranking.

Cluster aggregate vs gate-v2 arm gains (all 210 rows; measured rows per cluster 80/11/18/19/19/13/32/18): all pairs
Spearman 0.23, column-centred 0.40, AUROC(gain > 0) 0.60; off-cluster pairs 0.07 / 0.09 / 0.52; cluster level
Spearman 0.86 — unchanged from §4.

**Side decomposition on the full matrix** (T^T = per-token Δ log p(y^T_j), T^S = −per-token Δ log p(y^S_j)). T^T:
off-diagonal mean +0.050, 81 % positive, self-effect median 0.17 (93 % positive), symmetry Pearson 0.55, rank-1 share
0.56, participation ratio 2.8, additive R² 0.36 (row-only 0.29). T^S: mean +0.102, median 0.000, 51 % positive,
self-effect median 0.48 (99 % positive), symmetry 0.84, rank-1 share 0.96, participation ratio 1.09.

| predictor | teacher T^T: ρ [CI] | dbl [CI] | AUROC [CI] | NDCG@5 [CI] | student T^S: ρ [CI] | dbl [CI] | AUROC [CI] | NDCG@5 [CI] |
|---|---|---|---|---|---|---|---|---|
| ψ·ψ | 0.444 [0.378, 0.509] | 0.512 [0.419, 0.598] | 0.639 [0.589, 0.683] | 0.700 [0.662, 0.753] | 0.486 [0.399, 0.578] | 0.789 [0.741, 0.817] | 0.701 [0.661, 0.745] | 0.677 [0.638, 0.735] |
| raw sketch | 0.432 [0.366, 0.496] | 0.505 [0.416, 0.593] | 0.657 [0.609, 0.701] | 0.696 | 0.519 [0.432, 0.614] | 0.771 [0.719, 0.804] | 0.722 [0.683, 0.766] | 0.690 |
| sparse dict | 0.455 [0.388, 0.516] | 0.516 [0.423, 0.603] | 0.643 [0.592, 0.687] | 0.696 | 0.487 [0.399, 0.579] | 0.793 [0.745, 0.822] | 0.701 [0.660, 0.746] | 0.643 |
| PCA-32 cos | 0.391 [0.329, 0.466] | 0.507 [0.412, 0.597] | 0.590 [0.545, 0.636] | 0.686 | 0.464 [0.371, 0.558] | 0.787 [0.733, 0.823] | 0.717 [0.673, 0.762] | 0.672 |
| k-means(8) | 0.345 [0.269, 0.432] | 0.399 [0.310, 0.552] | 0.551 [0.519, 0.583] | 0.572 | 0.545 [0.433, 0.645] | 0.759 [0.713, 0.790] | 0.655 [0.613, 0.699] | 0.481 |
| gate cluster | 0.367 [0.290, 0.451] | 0.410 [0.307, 0.564] | 0.562 [0.531, 0.592] | 0.564 | 0.575 [0.471, 0.672] | 0.764 [0.716, 0.799] | 0.663 [0.624, 0.709] | 0.506 |
| same category | 0.395 [0.313, 0.478] | 0.526 [0.407, 0.623] | 0.559 [0.529, 0.591] | 0.633 | 0.569 [0.462, 0.664] | 0.768 [0.718, 0.803] | 0.654 [0.612, 0.700] | 0.494 |
| same side | 0.313 [0.222, 0.390] | 0.101 [−0.051, 0.215] | 0.671 [0.617, 0.720] | 0.291 | 0.087 [0.009, 0.174] | 0.028 [−0.020, 0.054] | 0.516 | 0.189 |
| same task id | 0.148 [0.117, 0.199] | 0.045 [−0.017, 0.127] | 0.509 | 0.541 | 0.222 [0.181, 0.260] | 0.322 [0.229, 0.378] | 0.526 | 0.463 |
| student length of j | 0.019 [−0.052, 0.085] | 0.004 | 0.505 | 0.166 | 0.411 [0.365, 0.454] | 0.000 | 0.689 [0.659, 0.719] | 0.051 |
| random | −0.002 [−0.018, 0.015] | −0.002 | 0.502 | 0.240 | −0.005 [−0.020, 0.011] | −0.002 | 0.499 | 0.167 |

Paired ψ·ψ − category: teacher side Δρ +0.046 [−0.009, +0.101], Δ dbl −0.007 [−0.071, +0.055], Δ AUROC **+0.080
[+0.044, +0.115]**; student side Δρ **−0.079 [−0.115, −0.039]**, Δ dbl +0.018 [−0.004, +0.040], Δ AUROC +0.047
[+0.025, +0.068]. Sparse dict − category: teacher Δρ +0.056 [−0.000, +0.112], Δ AUROC +0.083 [+0.046, +0.120];
student Δρ −0.077 [−0.115, −0.035].

**Do the conclusions change?** No. Every number moved by at most a few hundredths in the direction of the 120-row
estimate's CI centre, and every qualitative statement of §5 and §8 survives with tighter CIs:

1. ψ vs category on the raw pair ranking: ρ 0.686 vs 0.632, paired Δ +0.054 [+0.005, +0.098] (120 rows: +0.065
   [+0.015, +0.110]) — still positive, still CI-excluding-zero, still small. Sparse dict Δ +0.065 [+0.017, +0.110].
2. Sign AUROC: ψ 0.772 / raw 0.797 / sparse dict 0.779 vs 0.61–0.62 for every label (Δ +0.156 [+0.120, +0.187]).
   NDCG@5 0.78–0.81 vs 0.62.
3. Double-centred: ψ 0.848, sparse dict 0.853, PCA-32 0.846, category 0.837 — tie (Δ +0.008 [−0.021, +0.034]).
   The fingerprint's advantage lives in the main effects (magnitude / sign), not in the residual interaction.
4. Sparse dict ≈ ψ·ψ (+0.011 [+0.006, +0.017]); raw sketch matches ψ on ρ, beats it on sign AUROC by 0.02 and loses
   0.02 on dbl; PCA-32 cosine loses 0.14 on ρ and 0.09 on AUROC, ties after double centring.
5. Rank-1, symmetric, student-side, simple_python-block mode: rank-1 share rises from 86 % (120-row block) to 91 %
   on the full square, symmetric part 95 %, T^S rank-1 96 %.
6. Teacher side vs student side: on T^S the category label beats ψ on raw ρ (Δ −0.079, CI excludes 0) and student
   length alone reaches ρ 0.41; on T^T length is 0.02, category and ψ tie on ρ (Δ +0.046, CI crosses 0), and ψ /
   sparse dict / raw sketch keep the sign-AUROC edge (0.64–0.66 vs 0.56, Δ +0.080 [+0.044, +0.115]) and the NDCG@5
   edge (0.70 vs 0.63). The teacher-side matrix is also the less low-rank half (rank-1 56 %, PR 2.8 vs 1.1).
