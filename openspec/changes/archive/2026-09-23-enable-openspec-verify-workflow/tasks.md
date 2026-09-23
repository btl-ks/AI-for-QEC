# Tasks

## 1. Workflow 配置

- [x] 1.1 使用 `openspec config profile` 把全局 profile 改为 `custom`，保留 `propose`、`explore`、`apply`、`update`、`sync`、`archive` 并加入 `verify`，保持 `delivery: both`；用 `openspec config list` 验证精确 workflow 集合。
- [x] 1.2 在 QEC 项目应用 profile 更新并生成 workflow 文件；用文件清单和内容检查验证 `verify` 与六个既有 workflows 均已物化，`openspec doctor --json` 报告健康，且 `git check-ignore -v .agents/skills/openspec-verify-change/SKILL.md` 证明生成状态不进入项目源码。

## 2. 验证与归档

- [x] 2.1 运行 `git diff --check`、`openspec validate enable-openspec-verify-workflow --strict --no-interactive` 和 `openspec validate --all --strict --no-interactive`，并使用新生成的 verify workflow 按 completeness、correctness、coherence 检查本 change。
- [x] 2.2 从 `openspec/templates/verification.md` 创建 `verification.md`，记录全局配置路径与摘要、项目生成文件、精确命令和限制；确认任务与证据完整后归档。
