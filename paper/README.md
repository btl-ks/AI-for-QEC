# Paper notebooks

`srcs/` 保存论文实验的 `.ipynb` 源码；Notebook 是实验配置、阶段编排、可视化入口和研究解释，不承载数据生成、训练、解码、Artifact 或恢复实现。

## 当前状态

当前仓库仍是 **contracts-only**：

- `ai_for_qec_workflow.ipynb` 可以在干净内核中 `Run All`。
- 默认执行只导入 contract、创建配置并定义 `run_paper(runtime)`。
- Notebook 不会主动创建 runtime，也不会生成 dataset、model、metrics 或图表。
- 真正运行研究流程前，必须由后续 OpenSpec change 提供满足 `qec.NotebookPlatform` 的实现。

目标调用方式：

```python
result = run_paper(runtime)
```

其中 `runtime` 由项目实现层注入，不由 Notebook 自行构造。
