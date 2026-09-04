# CRCD — Canonical Method Document (skeleton v0.1, 2026-09-04)

Status legend: **[impl]** implemented in this repo · **[prop]** proposed, not yet implemented · **[val]** empirically validated (official protocol) · **[neg]** tested and falsified / downgraded.

## 1. Problem
Input (S_few, π0, T, V, B): few-shot task spec, pretrained small LM (Qwen3.5-4B, LoRA r=8) under a fixed agent scaffold, black-box text-only teacher T, executable verifier V with bounded trajectory utility, teacher-output-token budget B. Objective: maximize expected trajectory utility of the scaffolded student subject to TeacherTokens ≤ B; output also a certificate (representation coverage, atom-wise residual risk, selective-deployment risk). The student deploys without the teacher.

## 2. Notation (authoritative)
x_i task instance · s_i full decision state · y_i^S / y_i^T student / teacher continuation at s_i · Q_i^S, Q_i^T posterior expected terminal utility after the respective intervention · ψ_i Fisher-whitened teacher–student differential gradient fingerprint · u_k atom, U = [u_1..u_K] dictionary, z_i sparse composition, span(U) capability subspace · J_k(π) atom-weighted probe performance · ρ_k required level · r_k = [ρ_k − J_k]_+ residual. Ownership superscripts T/S only (never y+/y−); teacher behaviour is not assumed better.

## 3. Block I — Capability atom discovery
- Event e_i = (s_i, y_i^S, y_i^T, O_i) with matched continuation outcomes. **[impl]** BFCL single-turn / stateful miners, ALFWorld demo-replay miner (K continuations per branch), unified schema **[impl, T1]**.
- Fingerprint: length-normalised ∇ log π(y|s) at θ0 for y^S and y^T over LoRA params, diagonal empirical Fisher from student rollouts, ψ̃ = F^{-1/2}(g^T − g^S), fixed random sketch, unit-normalised, versioned. **[impl, T1]** Baselines A1/A2/A3 (Adam-whitened, unnormalised) **[impl]**.
- Sparse dictionary min ½n⁻¹‖Ψ − UZ‖² + λ_z‖Z‖₁, ‖u_k‖ = 1 (soft coding; PCA / k-means / semantic / category / random baselines). **[impl, T2]**
- Transfer prediction: continuous first-order η z_j^T U^T U z_i vs measured held-out change (Spearman off-diag, centred variants, NDCG, calibrated MSE, sign AUROC, bootstrap/permutation, partial correlation). **[impl, T2]** Prior evidence: hard-cluster gate v2 gave column-centred Spearman 0.43 (p=0.004) vs category 0.32 — conditional only **[val-partial]**.
- Atom inference q_φ(z|s, y^S), atom performance J_k, coverage, LCB, residual r_k **[prop]**.

## 4. Block II — Budgeted acquisition
Query q = (s, L) (continuation length as a decision variable); counterfactual outcome posterior with matched continuations and shared randomness **[impl for ALFWorld/BFCL branch outcomes]**; ΔU = Q^T − Q^S is a derived random variable only — no hard ΔU>0 filter, no raw-ΔU weighting, ΔU=0 not discarded, ΔU<0 not reversed **[neg: all four tested and rejected, see ledger]**. Acquisition objective: expected posterior residual reduction per teacher-output token; offline replay first **[prop]**. Sequential unlocking: reachability must be re-estimated under the current student **[val: ALFWorld depth-1 reachability 0.03 → 0.68 after acquisition]**.

## 5. Block III — Capability-constrained mirror distillation
Constrained objective: min E_s KL(π_θ‖π0) + κ Σ d_k ξ_k s.t. J_k(π_θ) ≥ ρ_k − ε_k − ξ_k; primal-dual with logged λ_k, J_k, r_k, ξ_k, mastered-state KL. Mirror target q_i(y) ∝ π0(y|s_i) exp(Λ_i Q_i(y)/η) over a candidate set with length-normalised scores; Λ_i = Σ_k λ_k z̄_ki / Σ_j z̄_kj; loss Σ_i KL(q_i‖π_θ) + γ_⊥‖(I − UU⁺)F^{1/2}Δθ‖² + γ_probe E KL(π_θ‖π0). **[impl, T3 — smoke only]**. Required limiting behaviours: log q(y^T)/q(y^S) = log π0(y^T)/π0(y^S) + (Λ_i/η)(Q^T − Q^S); large unsatisfied atom → teacher-like target; satisfied atom → vanishing dual pressure. Baselines: CE/SFT, base-centred pairwise, reference-free pairwise, CE→pairwise, KL-regularised CE, global mirror (K=1), ablations **[impl for CE/pairwise/ref-free/CE→pair; others prop]**.

## 6. Block IV — Certification **[prop]**
Representation coverage on held-out fingerprints; atom-wise (d_k, LCB[J_k], ρ_k, UCB[r_k], n_k) with simultaneous bounds; acquisition stopping with shadow price ν; selective deployment (risk–coverage, realised vs nominal risk).

## 7. Empirical anchors already established (official; regression suite, not targets)
- ALFWorld valid_unseen (134; used repeatedly — disclose): base 8.96; first-round CE on consequential events 71.1±2.2 (3 seeds; 74.63 single run); best pairwise 25.4±1.7; CE on first divergences 24.9±3.2; CE on zero-ΔU 50.0±5.4; reversing negative-teacher events harmful (8.2); from the competent CE model, repeated CE 55.2 vs pairwise 67.9, event-matched low dose both ≈ 74.6; reachability of depth ≥1 states 0.03 → 0.68 after acquisition.
- BFCL official: base 46.06; corrective CE 8–19 (mastered-prompt KL ≈ 0.4); base-centred pairwise KL ≈ 0.005, cumulative retraining 47.37±0.23 (C3 balanced anchors 47.91±0.36, adaptive anchors 47.26±0.32 with Irrel +2.6), saturating at round 3; call/abstain is one coupled boundary; raw ΔU magnitude tracks both target gain and boundary damage.

## 8. Stale claims (marked, do not reuse)
relative correction universally better than CE **[neg]**; task-family success selects the operator **[neg]**; raw ΔU magnitude as weight **[neg]**; reverse negative-teacher events **[neg]**; first disagreement as reliable target **[neg]**; fingerprint clusters = validated atoms **[not established]**; event efficiency = teacher-token efficiency **[not established; BFCL uses GT at zero teacher cost]**.
