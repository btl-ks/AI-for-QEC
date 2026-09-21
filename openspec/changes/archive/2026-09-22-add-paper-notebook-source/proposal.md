# Proposal

## Why

仓库目前没有承载论文实验源码的目录，也没有把目标 Notebook 编排顺序表示成可检查的接口。需要新增一份有效的 `.ipynb` 源码模板，在不伪造执行能力的前提下固定 Configure、恢复、数据、训练、科学评估、Accuracy Gate、性能评估和收尾顺序。

## What Changes

- 新增 `paper/srcs/ai_for_qec_workflow.ipynb` 和 paper 目录说明。
- 新增 `NotebookPlatform`、`NotebookExperiment`、`NotebookRun` Protocol，表达目标高层编排接口但不提供实现。
- Notebook 使用合法 Python 配置语法，并把 Accuracy Gate 从科学评估结果中显式分离。
- Notebook 默认只定义 `run_paper(runtime)`，不会调用不存在的 runtime，因此 clean `Run All` SHALL 成功。
- 不实现 Stim、CUDA、MWPM、训练、恢复、性能测试、绘图或 Artifact 持久化。

## Capabilities

### New Capabilities

- `research/paper-reproduction`: 定义论文 Notebook 源码的目录、公共编排边界、执行顺序和 contracts-only 安全行为。

### Modified Capabilities

无。

## Impact

- 新增 `paper/` 目录与一份 Notebook 源码。
- 新增 `ai_qec.paper` 接口模块，并从 `ai_qec.notebook_api` 导出三个 Protocol。
- capability registry 新增一项 `planned/contracts-only` 能力。
- 增加 Notebook JSON、cell 执行和公共导入测试，不增加运行时依赖。
