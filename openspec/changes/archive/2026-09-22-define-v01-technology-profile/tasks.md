# Tasks

## 1. 技术目录

- [x] 1.1 创建 technology catalog 与 JSON Schema，并通过 schema validation 验证全部 primary/alternative/status 字段
- [x] 1.2 更新 capability registry 与 README 技术基线，并确认执行 capability 仍为 `planned/contracts-only`

## 2. QEC 与数据接口

- [x] 2.1 定义 circuit adapter 与 noise compiler Protocol，并通过公共导入和泛型结果构造测试验证
- [x] 2.2 扩展 syndrome generator descriptor 与 QECBatch BatchLayout，并通过 CPU/GPU representation 构造测试验证
- [x] 2.3 定义 CPU→GPU 与 GPU→GPU pipeline spec/Protocol，并验证 pinned/non-blocking 和 zero-copy/host-staging policy 可表达

## 3. Registry 与配置预检

- [x] 3.1 实现通用 Registry 的装饰器注册、重复拒绝、未知 key 报错和 build，并通过局部 Registry 单元测试验证
- [x] 3.2 建立完整字段路径到 Registry 的集中映射，并验证 code/noise/generator/model/training/runtime 字段命名唯一
- [x] 3.3 实现递归 unresolved 检测、精确 allow 和已注册选项预检，并验证所有错误发生在工厂调用前

## 4. 训练与解码选择

- [x] 4.1 扩展 ExecutionSpec/ResolvedExecutionPlan 以记录 trainer framework，并通过 PyTorch/CUDA 构造测试验证
- [x] 4.2 扩展 decoder provenance contract 以记录 technology/device/version，并通过 PyMatching CPU 与 PyTorch GPU 结果构造测试验证

## 5. 验证与归档

- [x] 5.1 运行全部单元测试、compileall、catalog schema 与 OpenSpec strict validation，并保存 verification evidence
- [x] 5.2 构建 wheel，在仓库外隔离环境验证新增接口可从 `ai_qec.notebook_api` 导入
