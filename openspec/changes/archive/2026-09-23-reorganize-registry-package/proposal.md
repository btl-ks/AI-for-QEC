# Proposal

## Why

Registry 核心、集中声明、构造前配置校验和内置实现加载目前散落在 `ai_qec` 顶层，模块边界与其共同承担的配置注册职责不够清晰。将这些内部模块收拢到独立 namespace 可降低顶层噪声，并使依赖方向更容易审计，同时保持公共 Notebook API 和运行行为不变。

## What Changes

- 新建无 `__init__.py` 的 `ai_qec/registry/` 隐式 namespace。
- 将 Registry 核心、Registry 集中目录、配置预检和内置实现加载迁移为明确叶子模块。
- 更新实现模块、runtime、测试和文档中的内部导入路径。
- 保持 `ai_qec.notebook_api` 的公共导出集合以及可选 runtime 的延迟加载行为不变。
- 保留 `ai_qec/technology.py` 的独立边界；本 change 不改变技术选型、Registry key 或 capability 状态。

## Capabilities

### New Capabilities

无。本 change 是内部包结构重构，不引入新行为。

### Modified Capabilities

无。既有 `governance/configuration-registry` 与 `governance/python-packaging` 要求保持不变。

## Impact

- 影响 `ai_qec/registry.py`、`ai_qec/registries.py`、`ai_qec/config_validation.py`、`ai_qec/implementations.py` 及其内部调用方和测试。
- 非公共内部模块路径将改为 `ai_qec.registry.*` 明确叶子模块；公共入口仍为 `ai_qec.notebook_api`。
- 不新增依赖，不改变配置格式、构造语义、已注册实现或 runtime 后端。
