# AI for QEC Research Platform

面向 AI for Quantum Error Correction 的可复现研究平台。

> 当前状态：**contract scaffold only**。仓库只定义公共数据类型、抽象协议、正式需求和验证规则；数据生成、训练、解码、恢复、Accuracy Gate 与性能后端均未实现。

## 研究主路径

```text
Configure
  → Dataset Resolve / Reuse
  → AI Training
  → Scientific Evaluation + Classical Baseline
  → Accuracy Gate
  → Performance Research
  → Accuracy Revalidation
  → Trade-off Report
```

项目坚持 accuracy first、performance second。正式性能研究必须在科学 Accuracy Gate 通过之后进行。

## 仓库结构

- `openspec/`：正式需求、active changes、能力状态与验收依据。
- `ai_qec/`：不绑定具体执行框架的公共 contract。
- `configs/`：仅用于说明配置边界的示例，不代表已有执行器。
- `paper/srcs/`：论文 Notebook 源码；当前模板只定义依赖注入式编排，不执行研究任务。
- `tests/`：只验证 contract 的导入、构造和不可变性。
- `docs/AI_for_QEC_*.md`：架构背景，不能替代 OpenSpec。

Notebook 和外部调用方的稳定导入入口是：

```python
import ai_qec.notebook_api as qec
```

当前没有 `create_experiment()` 或训练入口。未来实现必须先经过 OpenSpec change，再由实现对象满足 `ExperimentFactory`、`DatasetResolver`、`Trainer` 等 Protocol。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q ai_qec tests
openspec validate --all --strict --no-interactive
```

## 开发规则

先阅读 `AGENTS.md` 和相关 OpenSpec。接口存在不等于 capability 可用；以 `openspec/capabilities.yaml` 的状态和 evidence 为准。
