# Tasks

## 1. 仓库与治理

- [x] 1.1 创建 Python 项目、README、AGENTS、忽略规则和配置样例，并通过文件清单检查验证
- [x] 1.2 创建 capability registry 与 schema，并确认领域能力均保持 `planned/contracts-only`

## 2. 公共契约

- [x] 2.1 定义 QEC、Noise、QECBatch 与数据集三层对象，并通过公共构造测试验证
- [x] 2.2 定义 Artifact、Experiment、Attempt、Stage、恢复和随机流接口，并通过不可变性测试验证
- [x] 2.3 定义 TrainingSpec、ExecutionSpec 和两类 checkpoint，并通过类型构造测试验证
- [x] 2.4 定义 DecodeRequest、DecodeResult、科学评估与 Accuracy Gate 接口，并通过公共导入测试验证
- [x] 2.5 从 `ai_qec.notebook_api` 导出全部受支持的公共契约，并验证 `__all__` 可导入

## 3. 验证与收尾

- [x] 3.1 运行标准库单元测试和 compileall，确认接口包可执行导入且不依赖第三方运行库
- [x] 3.2 运行 `openspec validate --all --strict` 并保存 verification evidence
- [x] 3.3 初始化 Git 仓库并确认工作树只包含预期的 scaffold 文件
