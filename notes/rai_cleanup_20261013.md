# rai cleanup 2026-10-13 (user, tc channel 12:55 CDT: "1" = delete group 1 of the 12:4x CDT list)

Deleted (rm, user-ordered; rai / at 91% → 89%, avail 1,580,947,861,504 → 1,677,532,217,344 bytes):
- results/pi1_k32/merged — 95,807,011,066 bytes: 4 merged Gemma-4-12B students (seed0/1 × pass3/10).
  Restore: tools/alf_merge_lora.py (untracked in this tree, on disk) on results/pi1_k32/seed-{0,1}/pass-{3,10}/lora
  (kept) with base google/gemma-4-12B-it @707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.
- _trash/ — 698,183,139 bytes (rai R0/R1 OOM and smoke leftovers, report drafts, hpg_qwen_relocated notes).
- ../ICML27_agent_distillation/_trash/ — 93,506,122 bytes (aborted smoke runs, old framework page backups, the
  10-06 hpg-root rsync accident copy). Empty _trash/ dirs recreated.

Not deleted (awaiting the user's group-2 decision): results/rtd_v1 (6.1G, 9 Qwen3.5-4B RTD students);
tc-alignment-table1/_trash 124G (stale/legacy Qwen3.5-4B BFCL sft/bbopd students), tc-alignment-v11prod/_trash 23G and
tc-alignment-v11prod2/_trash 20G (V0/V2 pre-batch runs, Qwen3.5-4B, 2026-09-08); HF cache Qwen3.5 family + Qwen2.5-1.5B
(~41G, shared cache → via hq). Qwen3 0.6B–32B in the HF cache may be sdl's: not ours to decide.

## 13:01 CDT: all tc trash emptied (user, tc channel: "trash都可以删除")
Contents of every projects/tc-alignment*/_trash removed (dirs kept, empty): 178,696,688,785 bytes in total, of which
tc-alignment-table1 132,930,667,073 (Qwen3.5-4B BFCL sft/bbopd stale/legacy students), tc-alignment-v11prod 23,848,793,591
and tc-alignment-v11prod2 21,119,745,507 (V0/V2 pre-batch runs, Qwen3.5-4B, 09-08), v11run 682,907,458, the rest < 60 MB each.
rai avail 1,741,993,345,024 → 1,920,690,847,744 bytes (87%). Other projects' _trash (sdl, soma, read, ...) not touched.
Still pending (not trash): results/rtd_v1 (6.1G Qwen3.5-4B), HF cache Qwen3.5 family + Qwen2.5-1.5B (~41G, via hq).
