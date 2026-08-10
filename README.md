# tc-alignment

Layout:
- `src/` code
- `configs/` one config per experiment
- `scripts/` run_local.sh (rai) · sync_to_hpg.sh · job_hpg.slurm · fetch_results.sh
- `logs/` run logs (gitignored)
- `data/` datasets (gitignored)
- `results/<exp_id>/` metrics + artifacts; `results/figs/` plots (gitignored)
- `notes/exp_log.md` one line per experiment
- `paper/` LaTeX (ICLR style) — created when writing starts
- `PROJECT_STATE.md` living state file — Claude reads this first
