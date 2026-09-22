# Proposal

## Why

项目只需要 setuptools 隐式 namespace package 和零 `__init__.py`，不需要以安装隔离换取 `src` 目录层级。将源码恢复到仓库根目录可减少路径迁移与开发入口复杂度，同时继续用受限 package discovery 保证 wheel 只收录 `ai_qec`。

## What Changes

- 将可安装源码从 `src/ai_qec/` 迁回仓库根目录 `ai_qec/`。
- 配置 setuptools 从仓库根目录发现 namespace package，并以 `include = ["ai_qec*"]` 限制发现范围。
- 继续保持源码与测试树中零 `__init__.py`，保留叶子模块导入和公共 facade。
- 保持 `import ai_qec.notebook_api as qec`、130 个公共导出、wheel 内容和 runtime 行为不变。
- 更新 README、架构背景、测试、capability evidence 和验证命令中的物理路径。
- 不恢复包级便捷聚合导入，不新增 runtime backend、GPU、分布式、FPGA、ASIC 或 QPU 能力。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `governance/python-packaging`: 将强制 `src` layout 改为受限发现的仓库根目录 namespace layout，并相应调整零初始化文件的检查路径。

## Impact

影响 `pyproject.toml`、源码物理路径、packaging contract 测试、README、架构背景和 capability evidence。仓库根目录将再次允许未安装源码被直接导入，因此不再提供 `src` layout 的安装隔离；隔离 wheel 安装仍作为发布内容验证门槛。
