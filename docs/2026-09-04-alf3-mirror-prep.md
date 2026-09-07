# ALF-3 mirror-distillation arms on ALFWorld — preparation (T8, 2026-09-04)

Capability-constrained mirror distillation (METHOD §5) on the ALFWorld consequential pool, prepared
exactly as T7 did for BFCL (`docs/2026-09-04-bf3-mirror-prep.md`: same loadings format, same unique
event key `_traj|sha1(prompt)[:16]|sha1(_rejected)[:16]`, same hyper-parameters as
`configs/mirror_bf3_rho1.json`). The trainer `tools/crcd_mirror_train.py` is already patched (event_key
lookup, prompt hash in the cache key); nothing in `src/` or the trainer was touched.

| arm | tag | pool | steps | ρ_k | ε | loadings | config |
|---|---|---|---|---|---|---|---|
| every atom mastered | `crcd_alf3_C_rho1_s0` | C (85) | 44 | 1.0 | 0.05 | K=32 | `configs/mirror_alf3_C_rho1.json` |
| relative target | `crcd_alf3_C_rhoJ0_s0` | C (85) | 44 | clip(J₀+0.3) | 0 | K=32 | `configs/mirror_alf3_C_rhoJ0.json` |
| global K=1 baseline | `crcd_alf3_C_k1_s0` | C (85) | 44 | 1.0 | 0.05 | none, k_random 1 | `configs/mirror_alf3_C_k1.json` |
| all divergences | `crcd_alf3_A_rho1_s0` | A (385) | 49 | 1.0 | 0.05 | K=32 | `configs/mirror_alf3_A_rho1.json` |

Anchors (valid_seen / valid_unseen): base 7.14 / 8.96; CE on pool C, 44 steps 77.86 / 74.63
(`alfabl_CE_conseq_s0`); pairwise on pool C, 55 steps 27.14 / 28.36 (`alfctl_C_56step_s0`); CE on
pool A, 49 steps —/78.36; pairwise A 48 steps —/18.66. Steps: 44 = the CE anchor (85 rows × 4 ep at
grad_accum 8 = 4 × ceil(85/8)); the trainer draws 8 events per step from a running permutation, so
44 steps = 352 draws = 4.14 passes over C, 49 steps = 392 draws = 1.02 passes over A.

Shared settings (= `mirror_bf3_rho1`): student Qwen/Qwen3.5-4B, LoRA as `appworld_train`, lr 1e-4,
grad_accum 8, dual every 4 (10 updates at steps 4…44 for C; 12 at 4…48 for A), η 0.25, η_dual 2.0,
λ_init 0.5, κ 5 (d_k = 1), j_mode best_mass, utility_mode auto (→ **stored** utilities for every
ALFWorld row: `teacher = demo_replay`, u⁺ < 1), γ_probe 0.1 on 4 probe rows, γ_⊥ 1e-3 decoupled on the
full LoRA-B displacement (F = 1, U = None, as in BF-3), `--save merged`, shared base cache
`results/analysis/mirror_alf3_base_scores.json`. **Environment: `AW_MAX_PROMPT_TOKENS=2048
AW_TRUNCATE_SIDE=tail`** as in `scripts/alf_ablation_hpg.slurm` / `alf_valseen_hpg.slurm` (the trainer's
own default is 4096; the base cache key carries the cap, so run every arm under the same env or the
base pass is recomputed). No ALFWorld prompt exceeds the cap (max 1134 tokens in C, 1263 over all 385).

## 1. Files produced

| file | what |
|---|---|
| `tools/atoms_loadings.py` | extended: `--events <jsonl>` (fingerprint source file → exact event-key alignment), `--check-pool` (trainer-lookup coverage), `match_pool()`; BFCL behaviour unchanged |
| `data/atoms/alfworld_v1_K32.npz` | K=32 loadings for all 385 events (serves both pools, see §2) |
| `data/alf_sft/probe_alf_base_correct_v1.jsonl` | 4 probe rows (base-correct trajectories, §3) — not on the allowed-files list but required by the configs |
| `configs/mirror_alf3_{C_rho1,C_rhoJ0,C_k1,A_rho1}.json` | the four arms |
| `tests/test_atoms_loadings.py` | 10 tests (6 BFCL + 4 new: synthetic `--events` subset/re-order/shared-prompt alignment, mismatched-events rejection, real ALF alignment + C-pool lookup, trainer math on both pools); all pass (`logs/t8_tests.log`) |
| `logs/t8_loadings.log`, `logs/t8_smoke_{C_rho1,C_rhoJ0,C_k1,A_rho1}.log`, `logs/t8_smoke_C_rho1_hubmerge.log`, `logs/t8_smoke_driver.log` | build / smoke logs |
| `results/analysis/mirror_alf3_*_smoke_trace{.jsonl,_summary.json}`, `results/analysis/mirror_alf3_base_scores.json` (770 keys = all 385 events × 2 candidates) | smoke traces, base cache (reusable by the full runs) |
| `results/appworld_students/crcd_alf3_C_rho1_smoke/{adapter,hub_merged}` (17 GB, movable to `_trash/`), `crcd_alf3_C_{rhoJ0,k1}_smoke/adapter` (LoRA only) | smoke exports |

## 2. Data: pools, alignment, loadings

* **Pool paths.** The task text named `data/alf_events/abl_*_s0.jsonl`; that directory does not exist.
  The training pools are `data/alf_sft/pool_C_conseq.jsonl` (85 rows, byte-identical to
  `pool_events_pref_v1.jsonl` and `pool_D_weighted.jsonl`; the pool of the CE and pairwise anchors) and
  `data/alf_sft/pool_A_all.jsonl` (385 rows, byte-identical to `data/alf_sft/events_v1.jsonl`).
  `results/alf_records/abl_*_s0.jsonl` are the anchors' evaluation records, not pools.
* **Fingerprint alignment (verified).** `alfworld_v1_v1.index.jsonl` row r ↔ `events_v1.jsonl` row r:
  sha1(prompt) = stored `state_hash` for 385/385; `data/events_unified/alfworld_v1.jsonl` row r agrees
  on state_hash, state_text = prompt, student_continuation = `_rejected`, teacher_continuation =
  `response` for 385/385.
* **Pool ↔ events.** Every C row is an exact events_v1 row (same `_traj`, prompt, `_rejected`, response;
  85/85), likewise B 107/107, A = events_v1. `_traj` **is unique** on ALFWorld (385 distinct; the
  `#p<k>` probe index is per demo), unlike v3t; 4 prompts are shared by two events each (381 distinct
  prompts: events 103/107, 104/108, 171/175, 333/337 — two demos of the same game reaching the same
  state with different y^S), so prompt-hash alignment alone would be ambiguous for those; the event key
  separates them. `pool_E_negteacher` (93) matches 85/93: its 8 `#neg` rows swap response/_rejected
  and are not fingerprinted.
* **One npz for both pools.** `alfworld_v1_K32.npz` is built on `pool_A_all` with `--events
  data/alf_sft/events_v1.jsonl` (exact key alignment, 385/385, 0 fallbacks, 0 missing). Because the
  trainer looks rows up by event key, the same file covers the C pool: **trainer lookup matches
  385/385 (A) and 85/85 (C)** (`[loadings][check]` lines in `logs/t8_loadings.log`; asserted in the tests).
  Row order in the npz = events_v1 order; `pool_row`/`fingerprint_row` = that index.
* **Basis / loading definition** as BF-3: PCA-32 = uncentred SVD of the 385 × 256 whitened,
  L2-normalised ψ; z̄_ki = |z_ki| / Σ_j |z_ji| (rows sum to 1). Stats: PC1 explains **0.694** of the
  variance (PC2 0.051, PC3 0.025, PC32 0.002), energy captured by 32 components 0.903; mean loading
  on PC1 0.347 (max 0.53), PC2 0.077, all others 0.013–0.041; top-atom share per event mean 0.36;
  atom-weighted effective n 67 (PC3) – 352 (PC1). ALFWorld fingerprints are far more one-dimensional
  than BFCL (PC1 31 % there): λ₁ acts as a near-global pressure and the K=1 arm is the natural
  control (exp_log `atoms_fit_alfworld_fisher`: K=4 already covers 76 %).
* Keys in the npz: as BF-3 (`Z`, `state_hash` = event key, `event_key`, `traj`, `pool_row`,
  `fingerprint_row`, `fingerprint_state_hash`, `has_fingerprint`, `Z_signed`, `U`,
  `explained_variance_ratio`, …) plus `events`, `alignment = "event_key"`.

## 3. Probe rows (`data/alf_sft/probe_alf_base_correct_v1.jsonl`)

No `self_anchor` / base-correct rows exist under `data/alf_sft` or `data/alf_events`. The base-correct
trajectory pool is `results/bfas/alfworld/ours_s0/pool.jsonl` (4864 turn-rows, `teacher = self`, from
205 successful self rollouts on 68 **train** games; `#u<N>` = unguided = plain base, `#g<N>` = guided).
Chosen: 4 unguided (base-only) successful rows, one trajectory each, on 4 train games that are **not**
among the 107 demo games of events_v1 (no overlap with any training state), 3 categories, prompt
584–1126 tokens, response 193–256 tokens, each ending in `ACTION:`; `teacher` relabelled
`self_base_correct`, `_source` records the origin. Base success on these games (collection p̂):
0.5, 0.8, 0.2, 0.2. Alternative if a stricter "mastered" criterion is wanted: the 32 events with
ΔU < 0 (`data/alf_sft/events_negative.jsonl`, student branch better than the demo) with
response := `_rejected` — not used because those states are in pool A.

## 4. Smoke results (rai **GPU 3**, A100 80 GB; 2 optimizer steps, dual every 2)

GPU 1 was occupied by a labmate's job (79 GB, 100 % util) at launch, so the smoke used the free,
unassigned GPU 3 instead; GPUs 0/2/4 were not touched. PyTorch 2.13 cu130, env of §0.

* **Base pass**: C 170 sequences ≈ 60 s after model load (~3 seq/s); A's remaining 600 sequences
  ≈ 3 min. Cache `results/analysis/mirror_alf3_base_scores.json`, 770 keys. The three C arms share it
  (rhoJ0 and k1: `0 newly computed`).
* **Base scores on C** (length-normalised log π₀): y^T mean −2.85 nats/token (short `ACTION: …`
  lines, 37 tokens mean) vs y^S mean −0.43 (the student's own 250-token reasoning, truncated at
  256 during mining). Restricted base mass on y^T: mean 0.131, median 0.064; only 2/85 events have
  p₀(T) > 0.5.
* **J(π₀) per atom** — C: min 0.078 (PC1: 0.078), mean ≈ 0.16, max 0.284 (PC3); K=1 global 0.131.
  A: min 0.132, max 0.301, mean ≈ 0.18. Compare BFCL 0.37–0.49: on ALFWorld the base almost never
  prefers the demo action, uniformly across atoms.
  → ρ_rho1 = 1 (gap ≈ 0.7–0.9 on every atom); ρ_rhoJ0 = J₀ + 0.3 ∈ [0.378, 0.584].
* **Steps** (all arms, peak allocator memory 10.6 GB, 11.6 GB for A): C_rho1 33.1 / 33.9 s,
  C_rhoJ0 34.8 / 33.3 s, C_k1 38.0 / 35.0 s, A_rho1 34.0 / 34.5 s → ≈ 34 s/step at grad_accum 8 on
  the A100, i.e. 44 steps ≈ 25 min, 49 steps ≈ 28 min on rai (B200 faster). cmd loss 0.0016 → 0.096,
  probe KL 0 → 0.0063, ⊥ penalty 0 → 0.13 → 0.18, `log_odds_max_abs_err` ≤ 4e-16.
  Λ_mean = 0.163 / 0.198 at λ = 0.5 (= 0.5·32/85 ± window); mean teacher target mass 0.07 → 0.125.
* **First dual update (step 2, window 16 events)**:
  - C_rho1: r = 0.95 − J ∈ [0.754, 0.863] → λ 0.5 → **[1.907, 2.127]**, ξ = 0, 0/32 satisfied.
  - C_rhoJ0: r ∈ [0.223, 0.380] → λ 0.5 → **[0.946, 1.260]** (largest on PC3, atom 3).
  - C_k1: r = 0.840 → λ 0.5 → **2.081**.
  - A_rho1: r ∈ [0.643, 0.884] → λ 0.5 → **[1.686, 2.167]**.
  At η_dual = 2 the rho1/k1 duals reach the cap 5 after ~3 updates (step 12 of 44) unless J rises by
  > 0.5, which it cannot (see §6.1): these arms run at maximum uniform pressure for ~30 of 44 steps.
* **Export chain verified** on C_rho1: `--save merged` → `adapter/model.safetensors` (8.4 GB), then
  `tools/bfcl_hub_merge_export.py --verify` → `hub_merged/` (426 tensors overlaid, 312 untouched,
  2 shards, verification passed, 17 s on rai). Runtimes: C_rho1 127 s (load + base pass + 2 steps),
  C_rhoJ0 81 s, C_k1 85 s, A_rho1 252 s.

## 5. Full-run commands (cluster)

Copy (code via the sync script; data explicitly — `data/` and `results/` are excluded from rsync):

```bash
H=hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
bash scripts/sync_to_hpg.sh                                       # tools/ (incl. atoms_loadings, patched trainer), configs/, src/, tests/
ssh hpg "mkdir -p /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/atoms /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/results/analysis"
scp data/atoms/alfworld_v1_K32.npz                    $H/data/atoms/
scp data/alf_sft/pool_C_conseq.jsonl data/alf_sft/pool_A_all.jsonl $H/data/alf_sft/   # already on hpg from alfabl; harmless
scp data/alf_sft/probe_alf_base_correct_v1.jsonl      $H/data/alf_sft/
scp configs/mirror_alf3_*.json                        $H/configs/                     # in case sync_to_hpg.sh excludes new files
scp results/analysis/mirror_alf3_base_scores.json     $H/results/analysis/            # optional: skips the base pass (cap 2048 tail keys)
```

Per arm, in a slurm script with the preamble of `scripts/alf_ablation_hpg.slurm` (dept account,
`hpg-b200`, 1 GPU, 14 cpus, 100 GB, `--time=03:00:00`, HF/vllm caches on /blue, `source
.venv/bin/activate`, `export PATH=$PWD/envs/vllm/.venv/bin:$PATH`, `AW_MAX_PROMPT_TOKENS=2048
AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) and
`export BFAS_ALFWORLD_EVAL_SPLIT=valid_seen` (development split, as `scripts/alf_valseen_hpg.slurm`):

```bash
ARMS=(C_rho1 C_rhoJ0 C_k1 A_rho1); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; tag=crcd_alf3_${ARM}_s0
# 1. train (44 steps; A_rho1 49) + merged export -> results/appworld_students/$tag/adapter/model.safetensors
PYTHONPATH=src .venv/bin/python tools/crcd_mirror_train.py --config configs/mirror_alf3_${ARM}.json
# 2. Hub-faithful merge -> $tag/hub_merged
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
    --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
# 3. ALFWorld valid_seen evaluation (same call as alf_valseen_hpg.slurm; 140 games, ~5 min on B200)
P=$((8960 + SLURM_ARRAY_TASK_ID))
rm -rf "results/bfas/alfworld/$tag"
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$P VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/$tag" --gpu 0 --port $P --seed 0
[ -f "results/bfas/alfworld/$tag/eval_metrics.json" ] || { echo "[alf3] $tag NO SCORE"; exit 1; }
echo "[alf3] $tag DONE"
```

Traces `results/analysis/mirror_alf3_<ARM>_trace.jsonl` + `_summary.json`; manifest
`results/appworld_students/<tag>/mirror_manifest.json`. Arms sharing the base cache on different nodes
each recompute it if the file is absent (no locking, harmless; ~1 min for C, ~4 min for A). Projected
per arm on B200: train 15–25 min + merge ~5 min + eval ~5 min → well inside 3 h. Compare against the
valid_seen anchors 77.86 (CE-C) / 27.14 (pairwise-C56) / 7.14 (base); valid_unseen only after the arm
is frozen.

## 6. Open decisions

1. **The ρ = 1 constraint is unreachable by construction, and the pressure is capped.** The mirror
   target is q ∝ exp(score₀ + Λ Q/η) with stored utilities: the base's score gap is ≈ 2.4 nats/token
   in favour of y^S and the utility gap only ΔU ∈ {0.2, 0.4, 0.6} (mean 0.28 on C). Even with every λ_k
   at the cap κ = 5, Λ_mean = 5·32/85 = 1.88 and the log-odds shift Λ ΔU/η ≈ 1.88·0.28/0.25 ≈ 2.1 nats
   — the target barely reaches parity between y^T and y^S (BFCL: ΔQ ≈ 0.83, shift ≈ 1.8 nats over a
   gap of ~0.4). The rho1/k1 arms therefore behave as "λ = κ on every atom after step ~12" and the
   arms differ mainly through rhoJ0. Options if the arms are meant to be able to *install* the demo
   action like CE does: `utility_mode = gt_teacher` (Q^T = 1, i.e. treat the demo action as ground
   truth as `demo_replay` did in the CE anchor), larger κ (e.g. 20), or smaller η. Not changed here
   because the brief asked for the BF-3 hyper-parameters; the smoke numbers above quantify the issue.
2. **Loadings are near-global.** PC1 carries 69 % of the variance and 35 % of every event's loading,
   so K=32 ≈ one global pressure plus 31 small modulations; `C_k1` is the control that tests whether
   per-atom duals matter at all. Splitting PC1 by sign (K=64) or using `--power 2` (energy shares:
   PC1 would then dominate even more) are the alternatives.
3. **Y_i on pool A.** For the 268 ΔU = 0 rows best_index ties resolve to the teacher, and for the 32
   ΔU < 0 rows the *student* is the best candidate (J rewards mass on y^S there). Pool A is the
   "all divergences" arm as in the ablation; if that is not wanted, run pool C only.
4. **Truncated y^S.** Student continuations were cut at 256 tokens during mining (mean 250; most lack
   the final `ACTION:` line), so y^S is scored as an unfinished reasoning prefix. This is the same
   candidate the pairwise anchor used, so the comparison is fair, but it inflates score₀(y^S) relative
   to a complete answer.
5. **Probe rows.** 4 unguided base-success rows on non-demo train games (§3); γ_probe = 0.1 as BF-3.
   The KL after two steps (0.006) is 3× the BFCL value (0.002) — the ⊥ term and the probe are the only
   protection of the 7–9 % base competence.
6. **Steps.** 44 (C) / 49 (A) match the CE anchors; the pairwise anchor's best dose was 55 (peak
   28.4 vs 17.2 at 44), so a 55-step variant is the first follow-up if the mirror arms land near the
   pairwise anchor rather than near CE.
7. **GPU deviation.** Smoke ran on GPU 3, not GPU 1 (occupied by a labmate); no other GPU was used.
