# Table 1 permanent archive

Per the 2026-09-13 directive: these cells are never re-run. Each directory keeps the
small, verifiable artifacts of one cell — metrics, official evaluator output, purchase
manifest, exposure schedule, and gzipped train/eval logs. Adapter weights and rollout
dumps stay under `results/paper_baselines/<cell>/` on rai (gitignored) and on LONI at
`/work/xueqic/hq/tc-alignment/results/paper_baselines/<cell>/`.

Cell name = `table1_<benchmark>_<method>[_s<seed>]`; no suffix means seed 0.
`metrics.json` carries `overall_accuracy_percent`, the absolute budget `B`,
`teacher_tokens_charged`, and the checkpoint and export SHA-256 digests.
