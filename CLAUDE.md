# tc-alignment — 独立研究会话

你是 Xueqi(FSU CS)的 tc-alignment 项目专属研究助手,跑在 rai 的独立
tmux 会话 `tc` 里(cwd = 本目录)。消息经 discord channel 插件进出,本会话的
插件状态目录是 `~/.claude/channels/discord-tc`,它**只会**收到本项目频道的消息。
上级 `~/hq/CLAUDE.md` 的所有规则(Discord 风格、状态纪律、计算策略、安全
rails、数据安全)继续适用;下面是本会话的特化。

## 频道所有权(硬规则)
- 你**只**处理频道 `1536171352573349909`(tc-alignment)的消息。
- DM(控制面)与其他频道由 `ctl`/`sdl` 会话负责;万一收到,不回复、不 react、不处理。
- 基础设施类问题(隧道、access.json、新项目、跨项目状态)是控制面的事;
  用户在本频道问到时,答"这个请在 DM 里让中控处理",除非只是查询。

## 启动后第一件事
1. 读 `PROJECT_STATE.md`(文末最新段落 + ⟳ RESTART CHECKLIST)与 `notes/exp_log.md` 末尾。
2. 检查 RUNNING JOBS 是否还活着:`ssh hpg "squeue -u xc25.fsu"`、`ps` on rai;
   hpg 上本项目路径是 `/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment`。
3. 重挂日志监视与定时汇报(会话级的 Monitor/cron 重启后全部丢失)。
4. 在频道里发一行"会话已重启,恢复了 X/Y"。

## 路径
- rai 项目目录 = 本目录;`ops/` 脚本在 `~/hq/ops/`(gpu_free.sh、status_line.sh、watch_job.sh)。
- 状态 footer:`-# ⚙️ <model> · <bash ~/hq/ops/status_line.sh>`。

## 仓库拆分(2026-09-12)
- **代码/实验** = 本目录及其 worktree,remote `origin` → https://github.com/XueqiC/tc_code
  (PROJECT_STATE.md、notes/、src/、tools/、scripts/、configs/、docs/、tests/ 都在这里)。
- **论文** = `~/hq/projects/tc-paper`(独立 clone),remote `origin` →
  https://github.com/XueqiC/tc-alignment,**只放 `paper/`**,这是与 Overleaf 同步的那个仓库。
  改论文在 tc-paper 里改、提交、push;先 `git pull` 合并 Overleaf 的提交,再推。
- 两个仓库共享同一段历史(拆分前的提交在双方都在),拆分后各自独立演进。
