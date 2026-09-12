# tc-alignment code

Code, data pipelines, experiment configs, results bookkeeping and notes for the
budgeted few-shot black-box agent distillation project (RTD).

- Paper source lives in a separate repository, <https://github.com/XueqiC/tc-alignment>,
  which is the one synced with Overleaf. Nothing but `paper/` belongs there.
- Project memory: `PROJECT_STATE.md` (status, running jobs, restart checklist) and
  `notes/exp_log.md` (one line per experiment).
- Worktrees on the lab server hold parallel lines of work (`-uni` RTD engine, `-base`
  paper baselines, `-mech`/`-mech2` mechanism validation, `-ws` adapters, `-g4` BFCL harness).
