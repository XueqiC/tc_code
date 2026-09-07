# BF-3 mirror-distillation arms on BFCL — preparation (T7, 2026-09-04)

Two capability-constrained mirror-distillation arms (METHOD §5, brief §9.2 BF-3) on the
teacher-consistent pool `pool_events_pref_v3t.jsonl` (302 rows, the pool of the headline
pairwise anchor `crcd_r3_union_t_s0`, 38 steps, OVERALL 46.74), with K = 32 atom loadings
from the Fisher-whitened fingerprints `bfcl_r3t_v1`.

| arm | tag | ρ_k | ε_k | config |
|---|---|---|---|---|
| every atom must be mastered | `crcd_bf3_rho1_s0` | 1.0 for all k | 0.05 | `configs/mirror_bf3_rho1.json` |
| relative target | `crcd_bf3_rhoJ0_s0` | clip(J_k(π0) + 0.3, 0, 1) | 0 | `configs/mirror_bf3_rhoJ0.json` |

Shared settings (from `configs/mirror_smoke.json`, T3): student Qwen/Qwen3.5-4B, LoRA as
`appworld_train`, lr 1e-4, grad_accum 8, **steps 38** (= ceil(302/8) × 1 epoch, the pairwise
anchor's step count; the trainer draws a fresh permutation whenever fewer than 8 indices remain,
so 38 steps = one pass), dual update every 4 steps (9 updates at steps 4…36), η = 0.25,
η_dual = 2.0, λ_init = 0.5, κ = 5 (dual cap 5·d_k, d_k = 1), j_mode best_mass,
utility_mode auto (288/302 rows are turn-0 gt_teacher rows → Q^T = 1), γ_probe = 0.1 on 4
probe rows of `anchors_single_base.jsonl`, γ_⊥ = 1e-3 decoupled on the full LoRA-B displacement
(F = 1, U = None — see open decisions), `--save merged`, shared base-score cache
`results/analysis/mirror_bf3_base_scores.json`.

## 1. Files produced

| file | what |
|---|---|
| `tools/atoms_loadings.py` | fingerprints (+ pool) → loadings npz for `--loadings` |
| `data/atoms/bfcl_r3t_K32.npz` | the K = 32 loadings for pool v3t (keys below) |
| `configs/mirror_bf3_rho1.json`, `configs/mirror_bf3_rhoJ0.json` | the two arms |
| `tests/test_atoms_loadings.py` | 6 tests: synthetic PCA recovery + key/alignment, subset fallback, duplicate-key rejection, real-file alignment, trainer math (all pass) |
| `logs/t7_loadings.log`, `logs/t7_tests.log`, `logs/t7_smoke_rho1.log`, `logs/t7_smoke_rhoJ0.log` | build / test / smoke logs |
| `results/analysis/mirror_bf3_{rho1,rhoJ0}_smoke_trace*.{jsonl,json}` | smoke traces |

### Loadings (`data/atoms/bfcl_r3t_K32.npz`)

* Basis: **PCA-32 = uncentred SVD of the 302 × 256 whitened, L2-normalised fingerprints ψ**
  (`atoms.fit_pca(center=False)`, the same call `tools/atoms_fit.py` uses for its PCA baseline).
  The sparse dictionary was not refit on r3t: on bfcl_r2 it matched PCA-32 in held-out coverage
  (0.5536 vs 0.5529) and in transfer prediction, and no r3t fit existed, so the whitened gradient
  subspace is used directly (deterministic, no λ_z). `--method sparse --lambda-z 0.02` is
  implemented if the sparse fit is wanted later.
* Loading: **z̄_ki = |z_ki| / Σ_j |z_ji|** per event (rows sum to 1; `--power 2` gives energy
  shares instead). |z| is the spec's atom relevance (`atoms.atom_relevance`); the per-event
  normalisation equalises events, and the trainer's own column normalisation
  W_ik = z̄_ki / Σ_j z̄_kj makes Λ_i = Σ_k λ_k W_ik. Consequence: mean_i Λ_i = Σ_k λ_k / n, i.e.
  with n = 302 and K = 32, λ ≡ 1 gives mean Λ = 0.106 (the 24-row K=2 smoke had 0.11 at
  λ = [0, 2.6]); λ at the cap 5 on all atoms gives mean Λ = 0.53, i.e. log-odds shift
  Λ ΔQ/η ≈ 0.53·0.83/0.25 ≈ 1.8 nats in favour of y^T. Pressure per event scales as 1/n by
  design (METHOD §5); the dual step η_dual = 2 was kept from the smoke.
* Stats: explained variance of PC1 = 0.311 (PC2 0.049, …, PC32 0.006); ψ energy captured by
  the 32 components = 0.759 on average; mean loading on PC1 = 0.171 (max 0.52), all other atoms
  0.02–0.045; top-atom share per event mean 0.245; atom-weighted effective n per atom 72–195.
* Keys: `Z` (302 × 32), `state_hash` (= the **unique event key**, see §2), `event_key` (alias),
  `traj` (`_traj`, not unique), `pool_row`, `fingerprint_row`, `fingerprint_state_hash`
  (sha1(prompt)), `has_fingerprint`, `Z_signed`, `U` (256 × 32), `explained_variance_ratio`,
  `fingerprint_version`, `fisher_id`, `method`, `power`, `definition`.
* Alignment: fingerprint row i ↔ pool row i, accepted only when sha1(prompt_i) equals the stored
  `state_hash` (302/302 positional, 0 hash fallbacks, 0 missing). All 302 rows have a
  fingerprint: v3t contains **no `self_anchor` rows** (teacher ∈ {teacher_authored_gt 249,
  teacher_authored_abstain 38, deepseek-v4-pro-FC 15}), so Y_i is built for every row and the
  "exclude or zero" question does not arise here (the tool zero-fills and warns if it ever does;
  `--require-all` makes it an error).

## 2. Deviations from the trainer's assumptions (must be handled before the full run)

1. **`_traj` is not a unique event id in v3t.** 302 rows carry only 80 distinct `_traj`
   values (the same task/event index re-mined in rounds 2–4 with different states and different
   y^S; 198 distinct prompts). `crcd_mirror_train.load_loadings` builds `{state_hash: row}` and
   looks up `Event.event_id` (= `_traj`), so with `_traj`-keyed loadings 271 rows would silently
   receive another row's loadings. The npz therefore stores under `state_hash` the unique key
   `_traj|sha1(prompt)[:16]|sha1(_rejected)[:16]`, which the trainer must build from its `Event`
   the same way. With the unpatched trainer every row misses → all-zero Z → J0 = NaN → the run
   fails at the first dual update (`ValueError: J has NaN entries`), i.e. loudly, not silently.
2. **Base-score cache key collides across prompts.** `cache_key` = `student|cap|event_id|owner:sha1(text)`;
   7 of the 498 (id, candidate) keys in v3t occur with two different prompts (same `_traj`,
   same y^T, different state), so one of each pair would read the other's log π0. Adding
   sha1(prompt)[:16] to the key fixes it.

Both were applied in the smoke through a wrapper that monkey-patches the two functions
(scratchpad `mirror_train_patched.py`; identical CLI). **Patch for `tools/crcd_mirror_train.py`
(T3's file, not edited by T7):**

```python
import hashlib
def _sha(t): return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]

def event_key(ev):                       # == tools/atoms_loadings.event_key_for_event
    y_s = ev.candidates[ev.index_of(mirror.STUDENT)].text
    return f"{ev.event_id}|{_sha(ev.prompt)}|{_sha(y_s)}"

# in load_loadings(): key by event_key when the npz carries one
        by_key = "event_key" in data.files
        ...
            j = lookup.get(event_key(ev) if by_key else ev.event_id)

# cache_key(): include the prompt
    return f"{cfg.student}|cap{cap}{side}|{ev.event_id}|{_sha(ev.prompt)}|{cand.key()}"
```

(The base cache written by the smoke, `results/analysis/mirror_bf3_base_scores.json`, already
uses the patched key; with the unpatched key the trainer just recomputes it, ~2.5 min.)

Other things the trainer assumes that this arm does not provide (kept as in the smoke):

* `--fisher` expects `{lora_B param name: diag}` for the r = 16 training LoRA; the available
  `fisher_bfcl_r3t_v1.npz` is a flat vector over the r = 8 fingerprint LoRA → F = 1.
* `--U` expects a (256 × K) basis in the trainer's count-sketch of the LoRA-B displacement;
  the loadings' `U` lives in the fingerprint tool's JL projection of the r = 8 gradients — a
  different sketch, so it cannot be passed. The ⊥ penalty therefore acts on the *full*
  displacement (U = None), i.e. it is a plain proximal pull to θ0, not "off-atom" only.
* Trace `event_ids` are `_traj` (non-unique); use `row` (position in the pool) to join.

## 3. Smoke results (rai GPU 0, 2 optimizer steps, dual every 2, otherwise the arm's config)

Both runs used the patched wrapper (§2), `--steps 2 --dual-every 2`, GPU 0 shared with a
labmate process (10.5 GB), PyTorch 2.13 cu130. Logs `logs/t7_smoke_rho1.log`,
`logs/t7_smoke_rhoJ0.log`, `logs/t7_smoke_rho1_hubmerge.log`.

* **Loadings**: `[mirror][patch] loadings keyed by event_key; matched 302/302`, K = 32.
* **Base-scoring pass** (302 events × 2 candidates = 604 sequences, cached to
  `results/analysis/mirror_bf3_base_scores.json`, 505 keys = 498 (id, candidate) + the 7
  prompt-collisions of §2): ≈ 2.5 min after model load (≈ 4 seq/s), process memory 12.6 GB
  (nvidia-smi), allocator peak ≤ 10.9 GB. The second smoke hit the cache (`0 newly computed`).
* **J(π0) per atom**: min 0.371, mean 0.417, max 0.492 (atom 1 = PC1: 0.492). Under best_mass this
  is the atom-weighted mass the base puts on y^T inside {y^T, y^S}: the base prefers its own
  continuation on ~58 % of the (loading-weighted) events, uniformly across atoms.
  → ρ_rho1 = 1 (gap 0.95 − J0 ≈ 0.46–0.58 everywhere); ρ_rhoJ0 = J0 + 0.3 ∈ [0.671, 0.792].
* **Steps**: step 1 30.7 s / 35.7 s (peak 10.9 GB), step 2 47.1 s / 49.3 s (peak 17.4 GB; the
  step time follows prompt length, 4096-token cap). Mean over the two smokes ≈ 40 s/step at
  grad_accum 8 → 38 steps ≈ 26 min on rai. cmd loss 0.0032 → 0.0027, probe KL 0 → 0.0021
  (identical in both smokes, as expected: the duals are the same until the first update).
  Closed-form log-odds check `log_odds_max_abs_err` = 2e-16. Λ_mean = 0.051 / 0.060 at
  λ = 0.5 (= 0.5·32/302 ± window), mean teacher target mass 0.51 / 0.39.
* **First dual update (step 2, window 16 events)**:
  - rho1: gap = 0.95 − J ∈ [0.438, 0.587] on every atom → λ 0.5 → **[1.376, 1.675]**, ξ = 0,
    0/32 satisfied. Expected direction (all violated). At this rate λ hits the cap 5 after ~5 dual
    updates (step ~20 of 38) unless J rises; λ_max spread 1.38–1.68 tracks 1/J0.
  - rhoJ0: gap = J0 + 0.3 − J ∈ [0.226, 0.345] → λ 0.5 → **[0.952, 1.190]**, 0/32 satisfied
    (J after 2 steps 0.36–0.51 is still ≈ J0; the window estimate is noisy at 16 events).
  - ⊥ penalty 0 → 0.124 → 0.275 (decoupled step, same magnitude as the T3 smoke: 0.115 → 0.243).
* **Export**: rho1 `--save merged` → `results/appworld_students/crcd_bf3_rho1_smoke/adapter/model.safetensors`
  (8.4 GB, ≈ 15 s after the summary); then
  `tools/bfcl_hub_merge_export.py --adapter …/adapter --out …/hub_merged` → `hub_merged/`
  (426 tensors overlaid, 312 untouched, 2 shards; ≈ 3 min on CPU). The rhoJ0 smoke used
  `--save adapter` (101 MB) to save disk (the root disk is at 98 %; the two smoke dirs hold
  17 GB and can be moved to `_trash/`).
* Total wall-clock: rho1 275 s (load + base pass + 2 steps) + export; rhoJ0 98 s (cache hit).
* Not meaningful yet: `Lambda_vs_teacher_mass_corr` = −0.47 after 2 steps — Λ varies by
  ±0.01 across events while the target mass is set by score0 differences; this statistic only
  becomes informative once λ differentiates across atoms.

## 4. Full-run commands (cluster)

The cluster copy has the same repo (`bash scripts/sync_to_hpg.sh` syncs code only; it excludes
`data/` and `results/`). Copy explicitly (the pool is already on hpg from the `bfclT` run; the
others are new):

```bash
H=hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
bash scripts/sync_to_hpg.sh                         # tools/, configs/, src/ (incl. the trainer patch)
scp data/atoms/bfcl_r3t_K32.npz               $H/data/atoms/            # ssh hpg mkdir -p .../data/atoms first
scp data/bfcl_sft/pool_events_pref_v3t.jsonl  $H/data/bfcl_sft/         # already there; harmless
scp data/bfcl_sft/anchors_single_base.jsonl   $H/data/bfcl_sft/         # probe rows
scp results/analysis/mirror_bf3_base_scores.json $H/results/analysis/   # optional: skips the 604-sequence base pass
```

Per arm, inside the slurm script (same preamble as `scripts/bfcl_teacher_hpg.slurm`: dept
account, 1×B200, 14 cpus, 200 GB, HF/vllm caches on /blue, `source .venv/bin/activate`,
`AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`):

```bash
ARMS=(rho1 rhoJ0); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; tag=crcd_bf3_${ARM}_s0
# 1. train (38 steps) + merged export -> results/appworld_students/$tag/adapter/model.safetensors
PYTHONPATH=src python tools/crcd_mirror_train.py --config configs/mirror_bf3_${ARM}.json
# 2. Hub-faithful merge -> $tag/hub_merged   (bfcl_std_campaign.sh also does this if hub_merged is absent)
python tools/bfcl_hub_merge_export.py --adapter results/appworld_students/$tag/adapter \
    --out results/appworld_students/$tag/hub_merged --model Qwen/Qwen3.5-4B
# 3. official evaluation (same call as the anchor)
bash tools/bfcl_std_campaign.sh 0 $((9125 + SLURM_ARRAY_TASK_ID)) "$tag"
[ -f results/bfcl_std/$tag/data_overall.csv ] || { echo "[bf3] $tag NO SCORE"; exit 1; }
```

The two arms share the base-score cache; if they run concurrently on different nodes each
computes it independently (the trainer writes the file after scoring, no locking) — harmless.
Traces: `results/analysis/mirror_bf3_{rho1,rhoJ0}_trace.jsonl` + `_summary.json`; manifest
`results/appworld_students/<tag>/mirror_manifest.json`.

Projected time per arm (B200 ≈ 1.5–2× rai's card on this workload; rai numbers in §3):
base pass ~2.5 min (skip with the cache) + 38 × ~40 s ≈ 26 min train (rai numbers; B200
likely 15–20 min) + <1 min merged export + ~5 min hub merge + ~3 h official eval → request
`--time=05:00:00` as for `bfclT`. Peak GPU memory in the smoke: 17.4 GB (fits any card).

## 5. Open decisions

1. **ρ.** ρ = 1 with ε = 0.05 demands J_k ≥ 0.95 for every atom: J0 ∈ [0.371, 0.492], so every
   atom starts violated by ≈0.5 and λ_k grows by ≈ η_dual·0.5 = 1.0 per dual update until the
   cap 5 (≈ 4–5 updates) unless J rises — this is close to "maximum uniform pressure" and is the
   stress arm. ρ = J0 + 0.3 (ε = 0) asks for a relative gain; atoms whose J rises by 0.3 lose
   their pressure. Neither is the certified-ρ of METHOD §3.5 (LCB of atom-weighted Q^T); that
   needs the outcome posterior and is not implemented in the trainer.
2. **K = 32 PCA vs sparse dictionary.** PCA-32 was chosen because the sparse fit gave no
   measurable advantage on r2 and no r3t fit exists; PC1 (31 % variance, mean loading 0.17) is a
   shared direction that every event loads on, so λ_1 behaves like a global (K = 1) pressure and
   the remaining 31 atoms modulate it. If a K = 1 global-mirror baseline is wanted, run the same
   config with `"loadings": null, "k_random": 1`. Signed loadings are folded by |z|; splitting
   each PC into ± half-atoms (K = 64) is the alternative if direction matters.
3. **Y_i membership.** All 302 rows (no self_anchor rows in v3t); Y_i = {y^T, y^S} only (no
   `_extra_candidates` in the pool). Rows from calibration ids were removed upstream.
4. **Regularisers.** F = 1 and U = None (see §2), so the ⊥ term is a global proximal pull;
   γ_⊥ = 1e-3 decoupled as in the smoke. γ_probe = 0.1 on 4 base-anchor rows.
5. **Steps.** 38 = one pass, matched to the pairwise anchor; with 9 dual updates the duals
   have little time to settle. A 2-epoch variant (76 steps) would be the first dose follow-up
   (BF-5).
