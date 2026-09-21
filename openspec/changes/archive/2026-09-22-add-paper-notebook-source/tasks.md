# Tasks

## 1. 编排接口

- [x] 1.1 新增 NotebookPlatform/Experiment/Run Protocol，并通过公共 `__all__` 导入测试验证
- [x] 1.2 保持评估、Gate 和性能输出边界分离，并通过类型构造测试验证

## 2. Paper 源码

- [x] 2.1 创建 `paper/README.md` 和 `paper/srcs/ai_for_qec_workflow.ipynb`，并用 JSON 解析验证文件有效
- [x] 2.2 在 Notebook 中实现有效 CONFIG 和依赖注入式 `run_paper(runtime)` 编排，并通过 code-cell compile 验证
- [x] 2.3 执行全部 Notebook code cells，验证 contracts-only 模式不调用 runtime 且不产生 Artifact

## 3. 验证与治理

- [x] 3.1 更新 capability registry 为 `research/paper-reproduction: planned/contracts-only` 并通过 schema 验证
- [x] 3.2 运行全部单元测试、compileall 和 OpenSpec strict validation，并保存 verification evidence
- [x] 3.3 构建 wheel 并验证新增 paper Protocol 可从安装后的公共 facade 导入
