# AI-QEC

这是一个 AI for QEC 研究平台。当前有两条可执行路径：project-local toy synthetic plumbing baseline，以及用于 Torlai–Melko (2017) 独立复现的 toric-code code-capacity RBM smoke 路径。后者验证真实错误链、RBM 训练、Gibbs 解码与小规模精确 MWPM 对照的连通性；其 smoke 结果不构成论文性能复现或阈值结论。

```text
toy synthetic event-slot data → ridge baseline → plumbing benchmarks
toric code-capacity data → joint RBM → Gibbs decoder / exact MWPM → logical-failure benchmark
```

论文 Notebook 在配置单元格中定义默认 smoke 参数，使用 PyTorch RBM 与单链 Gibbs。将同一单元格中的 `training.decoder.parallel_chains` 改为 64 可另行运行并行链平台实验；两种结果需分别解释。

给 Coding Agent / AI 助手的固定入口是 `AGENTS.md`，其中包含强约束：盖棺定论的学术结论必须引用论文，且不得用空文件或占位数组冒充生成数据。

## 目录结构

| 目录 | 职责 |
| --- | --- |
| `ai_qec/` | 唯一的 Python 实现包；包含 QEC、数据、模型、训练、评估和运行时能力 |
| `configs/` | 受版本控制的实验参数与执行流程声明，不保存运行结果 |
| `scripts/` | 可执行命令入口及实验、导出编排；领域实现调用 `ai_qec/` |
| `tests/` | 当前的单元与 smoke 自动测试，不保存实验结果 |
| `docs/` | 项目设计、研究主题、计划、协议和工作区规则 |
| `paper/` | 论文原文、实验 Notebook、复现模板和本地导出 release |
| `datasets/` | 可跨 run 复用的不可变数据集，由 manifest 和 generation hash 标识 |
| `runs/` | 每次执行的配置快照、日志、指标、checkpoint、预测和图表 |

隐藏目录 `.git/`、`.codex/`、`.agents/`、`.vscode/` 属于版本控制或本地工具状态，不作为项目源码目录维护。Git、写入者与产物的详细规则见 [当前架构](docs/PROJECT_DESIGN.md)。

## 文档索引

| 文档 | 回答的问题 |
|---|---|
| [当前架构](docs/PROJECT_DESIGN.md) | 代码如何分层、文件放在哪里、哪些路径可执行？ |
| [技术栈决策](docs/TECHNOLOGY_STACK.md) | Stim、Qiskit、Sinter、PyMatching、PyTorch、GPU backend 和 Python 包管理如何接入？ |
| [研究主题](docs/RESEARCH_TOPICS.md) | D1–D5 各研究什么？ |
| [优化计划](docs/OPTIMIZATION_PLAN.md) | 下一步任务、依赖与验收条件是什么？ |
| [论文复现设计](docs/PAPER_REPRODUCTION_DESIGN.md) | 论文 Notebook 与主项目实现如何协作？ |
| [Torlai–Melko 复现报告](paper/summary/TORLAI_MELKO_2017_REPORT/TORLAI_MELKO_2017_REPORT.md) | RBM 解码器复现的设计、问题、结果与局限是什么？ |

开发和架构审阅先读 `AGENTS.md` 与当前架构；选择工具时读技术栈决策，执行任务时读优化计划中的对应验收条件。

1. `scripts/run_experiment.py`
   读取 YAML，一键按 `flow` 执行实验；自动建立 run、日志和 manifest。

2. `scripts/export_paper.py`
   读取同一份 YAML，以显式 allowlist 方式导出论文最小可复现代码包，不复制整个私有仓库。

3. `configs/experiment.example.yaml`
   一个可读、可执行、可导出的 D2.2 弱 crosstalk 学习实验模板。

## 依赖

```bash
# 安装 uv 后，在仓库根目录执行
uv sync --extra plot --extra notebook

# Torlai–Melko PyTorch RBM 路径需要额外安装
uv sync --extra torch --extra plot --extra notebook
```

常用环境组合如下：`uv sync --extra sim` 用于 Stim/PyMatching/Sinter，`uv sync --extra torch` 用于 PyTorch，`uv sync --group dev` 用于测试与静态检查。论文 Notebook 需要 `torch`、`plot` 和 `notebook`；从已安装环境启动时可使用 `uv run --extra torch --extra plot --extra notebook jupyter lab`。

`pyproject.toml` 是依赖的唯一声明来源，`uv.lock` 是解析后的版本锁定来源。正式运行和论文导出使用 `uv run --locked ...` 或 `uv sync --locked`；不要再维护单独的 `requirements.txt`。

## 选择 Conda 执行环境

配置可选地指定所有实验步骤使用的 Conda 环境。例如本机的 `quantum` 环境：

```yaml
execution:
  conda_env: quantum
```

将该段放入实验 YAML 后，仍使用通常的 `python scripts/run_experiment.py ...` 命令。runner 会先确认环境存在，再通过该环境的 Python 执行每个 `flow` 步骤；`run_manifest.json` 会记录环境名、解释器路径和关键包版本，`environment.json` 会记录完整依赖快照。环境不存在时 runner 会在创建 `runs/` 或 `datasets/` 前失败。未声明 `execution` 的配置继续使用启动 runner 的 Python。

## 快速 smoke run

`configs/experiment.smoke.yaml` 使用小样本，适合验证项目是否完整：

```bash
python scripts/run_experiment.py \
  --config configs/experiment.smoke.yaml \
  --project-root .
```

运行产物会写入：

- `datasets/<dataset_id>-<generation-hash>/`（不可变、内容寻址）
- `runs/<run_id>/`

目录边界与产物规则见 [当前架构](docs/PROJECT_DESIGN.md)。

## 预览完整实验执行流

```bash
python scripts/run_experiment.py \
  --config configs/experiment.example.yaml \
  --project-root . \
  --dry-run
```

## 正式执行

```bash
python scripts/run_experiment.py \
  --config configs/experiment.example.yaml \
  --project-root .
```

正式配置设置了 `reproducibility.require_clean_worktree: true`：git 工作区有未提交改动时 runner 会拒绝运行，保证 `run_manifest.json` 中记录的 commit 精确对应代码。确需临时运行时加 `--allow-dirty`，改动会以 `git_diff.patch` 保存到 run 目录。

注意：示例配置同样只运行 project-local toy synthetic baseline。多 seed、Stim、Transformer、paired generation、adaptation、实时/硬件运行时与模型导出会明确失败，等待后续 Phase 实现。

## 导出论文最小复现包

```bash
python scripts/export_paper.py \
  --config configs/experiment.smoke.yaml \
  --project-root . \
  --run-dir runs/<run_id>
```

导出器只会复制 `paper_export.include` 中显式列出的闭包文件，并在 `paper/releases/` 下创建不可变 release；可用 `python scripts/export_paper.py --verify <release-dir>` 验证其内容索引。
