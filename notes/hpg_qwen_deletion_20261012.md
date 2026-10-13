# hpg /blue deletion record — 2026-10-12 (CDT)

Instruction (user, tc channel, 2026-10-12 21:46 CDT, msg 1559396762014318683): "tc上面所有和qwen相关的全部删掉，垃圾缓存全部删掉".
Context: fsu-compsci-dept /blue at 18.24T/20T, group unable to submit jobs; cleanup coordinated with the control plane (hq).

Scope: tc-owned directories on /blue/fsu-compsci-dept/xc25.fsu/hq only. Not touched: scaling-down-law, wm, rai, yd24f.

Deleted (full list with sizes, file counts and the evidence for each: `hpg_qwen_deletion_20261012.tsv`; raw inventory `hpg_qwen_inventory_20261012.tsv`):
- Qwen3.5-4B artifacts, about 1,705 GiB in 290 items. They come from the Sept TVD/CRCD/RTD era (all AppWorld/ALFWorld/BFCL students on Qwen/Qwen3.5-4B @851bf6e). The items are:
  - tc-alignment/results: appworld_students (106 students), e3_tier1_*, behavior_atom_v1, bfcl_std, rtd_v1, rtd_baselines, bfas, appworld evals of those students, and figs.
  - tc-alignment-v11 and v11b: rtd_v1_1 control windows, students and bfcl_std.
  - tc-alignment-table1: appworld_students and evals.
  - tc-alignment-alf: c26f RTD runs, appworld_students, bfas, alf_records and analysis.
  - Evidence for each item comes from model_type, student and model_path fields; items with no model string are eval records of those students, or empty/lock-only directories.
- Garbage (_trash of tc dirs, incl. paper-1 superseded dirs 47.5 GiB) and caches (vllm, uv, triton, inductor, xdg, envs caches), about 177 GiB.

Kept on purpose: tc-alignment/results/appworld/gemma4_base_dev40 (Gemma); every live dependency, namely:
- tc-alignment/envs (vllm venv in envs/vllm/.venv, ALFWorld data in envs/alfworld);
- tc-alignment-vllm;
- tools/hf-cache (Gemma) and tools/uv-pythons;
- ICML27_* (paper 1, Gemma);
- all Gemma results: tc-alf-taskeq, tc-alf-kcurve, hqk32, hqk32-tvd and tc-alignment-vllm/adapters.

Consequences:
- The deleted Qwen student weights were the only copies. The Sept numbers survive only in exp_log / PROJECT_STATE / paper drafts.
- The Qwen base model is still on rai (~/.cache/huggingface, shared). This record does not cover rai.
- Seven students were trained on another student's hub_merged as their base (alfabl_CE_conseq_s0, crcd_hyb_stage1_s0, crcd_r2_b3pref_s0). Those downstream students were deleted as well, so no orphaned dependency remains.
